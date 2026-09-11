# Agent knowledge-base strategy — ML method note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (—)

## The question and the decision it serves

`architect` is drafting `claude/docs/plans/agent-knowledge-base-strategy.md` for a GraphRAG-backed
retrieval layer over distilled agent knowledge (`:KnowledgeEntry` nodes in `kaizen_team`, one node
per rule/technique, retrieved by embedding similarity — FR-4 / AC-2 in
`claude/docs/requirements/agent-knowledge-base-strategy.md`). The embedding model choice
(Qwen3-Embedding-0.6B @ 1024-dim via LM Studio, reusing `falkor-chat`'s proven stack) is **settled
and not reopened here**. Four method questions are mine to answer so the plan can commit rather
than carry them as open questions: (1) does the corpus mismatch (chat messages vs. distilled
technical prose) break that model choice, (2) node granularity, (3) retrieval parameters
(top-K, score floor, traversal), (4) how to validate AC-2 before/after migration.

**Bottom line, quotable:** keep Qwen3-Embedding-0.6B, but use its own documented **asymmetric
query-instruction convention** (free, and this corpus needs it more than `falkor-chat`'s did);
chunk at **one distinct claim per node, not one markdown heading per node** — the existing
`review-techniques.md` file already violates the "50-400 words, one technique per bullet"
assumption the requirements doc states, and that gap is a real, present-tense retrieval-quality
risk, not a hypothetical; retrieve with **top-K=5 and a calibrated cosine-distance floor**, no
conversation-style traversal, only a `familyId` sibling link where §2's split applies; and gate
AC-2 on a **~40-pair independently-authored golden set with Wilson-interval recall@K**, not a bare
percentage.

## Findings from the real system

- **`falkor-chat`'s model stack** (`falkor-chat/docs/DESIGN.md` §1.3): Qwen3-Embedding-0.6B
  GGUF Q8_0, 1024-dim (native), cosine distance, `CREATE VECTOR INDEX` DDL, read via
  `db.idx.vector.queryNodes`. Chosen for small-model MTEB quality, 100+ languages, ~0.6 GB
  resident. Upgrade path is documented as **re-embed only, no schema change**
  (`qwen3-embedding:4b`, same family, same 1024-dim MRL) — a real, cheap escape hatch if this
  corpus ever needs it.
- **`falkor-chat`'s actual embedding call is unprefixed** (`server/falkorchat/embedding.py`,
  `OpenAICompatibleEmbedder.embed`): raw `text` goes straight into `{"model": ..., "input": text}`
  for both the write path (message text) and, by the same client, any query embedding. That's
  defensible there — chat-message-to-chat-message similarity is roughly symmetric in register —
  but it is **not** what Qwen3-Embedding's own model card documents as the intended usage.
  Verified against the model card (`huggingface.co/Qwen/Qwen3-Embedding-0.6B`, fetched today): the
  model is trained for an **asymmetric** convention — queries get
  `"Instruct: {task_description}\nQuery:{query}"`, documents get no prefix at all ("No need to add
  instruction for retrieval documents"). 32k token context, far more than any entry needs.
- **The requirements doc's granularity assumption doesn't hold against the actual file it cites.**
  I read the full 1,277-line `claude/analyst/review-techniques.md` (the file the brief names as
  the "one bullet = one self-contained technique, 50-400 words" model). Most headings do match
  that shape. But several do not: e.g. *"A guard derived from the artifact it guards is blind
  along the derivation axis…"* is one `## ` heading containing at least five distinct,
  independently-verified sub-claims (parametrized-test deletion blindness, AST-alias blindness,
  a two-axis coverage gap, a name-vs-site allowlist gap, a semantic-vs-syntactic docstring gap),
  each with its own "Origin:" and its own worked measurement, running well past 1,500 words total.
  *"Two checks for a multi-shape authorization/security-gate function"* similarly bundles two
  independently actionable checks under one heading. This is exactly the failure mode that makes
  granularity a real design question rather than a formality (detail in Recommendation 2).
- **`falkor-chat`'s hybrid retrieval** (`DESIGN.md` §8, `QUERIES.md` §6): `db.idx.vector.queryNodes`
  seeds $k neighbors (example: 10), then `MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed)` plus an
  `OPTIONAL MATCH` through shared `:Entity` `MENTIONS` pulls up to 5 related messages, all in one
  query, `ORDER BY score ASC` (cosine distance, 0 = identical), served via `GRAPH.RO_QUERY`. The
  traversal half exists because a chat message's *meaning* is genuinely thread- and
  entity-contextual — a reply reads differently without its parent.
- `cypher-mcp`'s read path is unrestricted (`mcp__cypher__query` reads need no `agent` argument),
  so an agent can issue a `db.idx.vector.queryNodes` call against `kaizen_team` today with no new
  write-authorization shape — that question is orthogonal to the four here and stays with
  `architect`/`graph-dba` per the requirements doc's own Open question #1/#3.

## Recommendation 1 — corpus mismatch: keep the model, change the calling convention

**Verdict: Qwen3-Embedding-0.6B is a sound choice for this corpus. The one concrete, low-cost
change worth making is adopting the model's documented asymmetric instruction convention on the
query side** — something `falkor-chat` never needed and therefore never built, but this corpus
needs more than `falkor-chat`'s did.

Why the corpus difference doesn't break the model:
- The English-only realism (no PT-BR need) is a pure non-issue — a multilingual model handles a
  monolingual corpus at least as well as a narrower one; nothing about capability is lost.
- Density/jargon and embedded code/regex/paths are a real but *bounded* risk. Qwen3's training
  mix is documented as including code, and MTEB's suite includes a Code-retrieval category the
  model is evaluated against — but I could not pull the specific per-category (CoIR) sub-score
  from the model card in this session, only the aggregate multilingual mean (64.33). **I am not
  going to assert a code-retrieval number I didn't verify.** Treat "does this encoder handle
  code/regex-dense entries as well as prose entries" as an open empirical question the golden set
  (Recommendation 4) answers directly, by stratifying it into the query set — not as something to
  assume either way before evidence.
- The one difference that *is* actionable now, for free: **falkor-chat's retrieval is
  near-symmetric (message ↔ message); this system's retrieval is sharply asymmetric** (a short,
  informal situation sentence — *"I'm testing a guard whose subject is other code's text"* — versus
  a long, formal, technique-dense passage). That is precisely the case Qwen3-Embedding's
  query/document prefix asymmetry was built for. Concretely: embed every stored `:KnowledgeEntry`
  unprefixed (as `falkor-chat` already does), but embed every **query** as
  `"Instruct: Given a coding agent's description of its current situation, retrieve the distilled
  technique or rule that applies to it.\nQuery: {situation text}"`. This costs nothing extra to
  implement (it's a string template at the query-embedding call site, not a pipeline change) and
  is the single highest-leverage free lever available before reaching for a different model.

**What would change my mind:** if the golden-set eval (Recommendation 4), once stratified by
code/regex-heavy vs. prose-only entries, shows materially lower recall@K on the code-heavy subset,
that's the trigger to try `qwen3-embedding:4b` (documented re-embed-only upgrade, no schema
change) or a code-aware embedder — not to abandon the family speculatively now.

## Recommendation 2 — node granularity: one claim per node, not one heading per node

**Verdict: per-technique is the right target granularity, but "one technique" must be defined by
independent actionability, not by the markdown `## ` boundary cobb happens to have used.** The
brief's own working assumption — "each bullet is already one self-contained technique" — is falsified
by the real file (Findings, above). Migrating mechanically at "one node per `## ` heading" will
silently create a handful of oversized, multi-topic nodes precisely in the file that most needs
fine retrieval (`review-techniques.md` is the densest, most heterogeneous KB in the roster).

Why this matters for retrieval quality specifically, not just tidiness: an embedding of a
multi-topic block is a single vector near the *centroid* of everything the block discusses. A
query about one narrow sub-topic (e.g. "an AST reader only checking one alias spelling") will score
lower against that diluted centroid than it would against a vector representing only that
sub-claim — even though the sub-claim is present verbatim in the node's text. This is the standard
RAG chunking failure (topic dilution), not a corpus-specific guess: it is exactly why chunk
coherence is a first-class lever in retrieval design generally.

**Recommendation, concretely:**
- Default unit = the existing convention (one `## `-headed technique = one node) wherever a
  heading genuinely holds one claim, one example, one origin — the common case, unchanged.
- **Split trigger:** a heading becomes N nodes when it contains N independently-attributed
  sub-claims — operationally, N distinct bolded sub-headers/enumerated checks each carrying their
  own "Origin:"/worked-example, or a heading materially longer than ~500 words spanning more than
  one verifiable claim. `review-techniques.md`'s "A guard derived from the artifact it guards…"
  and "Two checks for a multi-shape authorization…" both trip this rule today.
- **Do not go finer than one claim.** Splitting to sentence-level breaks self-containment: the
  worked example/counter-example is what makes a rule *actionable* rather than a bare assertion,
  and stripping it defeats the reason these files are written the way they are. The right node is
  the smallest unit that stands alone if someone reads only it.
- **Reject the coarser fallback (whole-file-as-chunk with per-technique as a secondary layer).**
  It directly reopens FR-4 — the entire point is not reading the whole file to find the relevant
  part — so it needs no evidence to reject, only the requirement's own text.
- **Migration process implication:** this makes the migration a **curator judgment pass, not a
  mechanical script**, at least for the pre-existing six KBs (AC-4). A splitting heuristic can
  flag candidates (length + multiple bolded sub-claims + multiple "Origin:" mentions), but the
  actual split boundary is an editorial call — route it through `cobb`, the same agent who already
  owns distillation judgment, not through an unsupervised script. This is a real (if small) piece
  of migration effort `architect` should size into the plan rather than assume away.
- **Sibling linkage for split entries.** When one heading becomes several nodes, give them a
  shared `familyId` (or a `(:KnowledgeEntry)-[:SAME_FAMILY]->(:KnowledgeEntry)` edge) so a query
  that surfaces one sub-claim can optionally pull its siblings — see Recommendation 3's traversal
  answer, this is the one traversal edge that earns its complexity here.

## Recommendation 3 — retrieval parameters

**Top-K:** **5**, not `falkor-chat`'s 10. Rationale: `falkor-chat` seeds 10 ANN neighbors *before*
traversal expansion and topic filtering narrows the final answer set — 10 is a seed budget, not
the number of things shown to the reader. Here there is no traversal expansion narrowing the
result, so top-K is directly what lands in the querying agent's context. An on-demand,
rare-path consult (the whole reason this isn't just the always-loaded prompt) wants a small,
precise handful, not a discovery-style spread — 5 keeps context cost low and reduces the chance of
the agent synthesizing across two only-loosely-related techniques as if they were one.

**Score floor: yes, mandatory, and it must be calibrated, not asserted.** The whole point of FR-4
is "the right entry, worded differently" — the failure mode this system exists to avoid is *not*
"nothing came back," it's "an agent trusted an irrelevant top-1." I am **not** going to hand you a
made-up cosine-distance constant (e.g. "0.3") — the right value depends on the exact instruction
prefix used (Recommendation 1 changes the query vector's distribution versus an unprefixed
baseline) and on this specific corpus, neither of which I can compute without running the pipeline.
**Derive the floor from the golden set in Recommendation 4**: compute the distance distribution
for true-positive (paraphrased-situation → correct-entry) pairs versus the distance distribution
for known-irrelevant pairs (any golden query against a random other entry), and set the floor near
the crossover — the point that maximizes separating true hits from noise on the measured data.
Ship the threshold as a config value derived from that run, documented with the run that produced
it, re-derived whenever the corpus or the prefix convention changes materially.

**Traversal: your instinct is correct — confirmed, with one addition.** A `:KnowledgeEntry` is not
part of a conversation thread, so `falkor-chat`'s `REPLY_TO`/`MENTIONS`-following pattern has no
real analogue — there's no "prior turn" or "mentioned entity" that adds context the way a reply
adds context to a chat answer. What *does* earn a traversal:
1. **`familyId` siblings** (Recommendation 2) — the one case where "the rest of this discussion"
   is genuinely part of the same original claim, split only for retrieval precision.
2. **`topic` as a filterable property, not necessarily an edge** — a plain indexed property is
   enough for "other entries on this topic"; don't build a graph edge for something a property
   filter already answers.
3. **Owning-agent attribution** — reuse the kaizen graph's existing
   `(:Agent)-[:PRODUCED]->(:KaizenEntry)` shape (or an equivalent `[:FOR_AGENT]`/`[:AUTHORED_BY]`
   edge for `:KnowledgeEntry`) so a query can optionally scope to "my own KB" versus "the whole
   team's" — a real, low-cost affordance, but it's a filter, not a hop that adds retrieved context
   the way `falkor-chat`'s entity-mention traversal does.

Do not build anything resembling entity extraction/`MENTIONS`-fusion over this corpus — that
machinery exists in `falkor-chat` to link chat messages through shared real-world entities, and
there is no equivalent need here; it would be complexity borrowed from a different problem.

## Recommendation 4 — evaluation design for AC-2

**Design: a ~40-pair, independently-authored golden set, scored as recall@K with a Wilson-interval
CI, not a bare pass/fail percentage.**

- **Size.** AC-2 is fundamentally a binary check aggregated into a rate (is the correct entry in
  the top-K, yes/no, over many pairs) — the same shape this lab already has a standing convention
  for (Wilson score interval, not Clopper-Pearson or rule-of-three, per this repo's established
  practice for small-n pass/fail bounds). At n≈40, a Wilson 95% CI around a recall point estimate
  in the 0.80-0.90 range runs roughly ±10-12pp — tight enough to catch a gross regression (e.g.
  recall dropping into the 60s) and loose enough to be honest about what a corpus this size can
  actually support. Don't report a bare "34/40 = 85%" — report the interval alongside it, and
  don't gate on the point estimate crossing a threshold without checking whether the CI already
  contains it (this lab's own established discipline against false-precision small-n gates).
  40 pairs against a ~100-250-entry eventual corpus is proportionate to "small internal tool, not a
  production RAG system" — a full one-query-per-entry set would be the gold standard but is not
  worth the authoring cost here.
- **Stratification, not random sampling.** Distribute the 40 across: (a) every existing/interim
  agent's KB, weighted roughly by its entry count, so no single dense file (like
  `review-techniques.md`) silently dominates or is silently absent; (b) a deliberate code/regex/
  shell-heavy subset versus a prose-only subset, sized enough to separately report recall on each
  — this is what actually answers Recommendation 1's open question about domain fit, rather than
  leaving it a guess; (c) 4-6 explicit **negative** queries — genuine situations with no matching
  entry in the corpus — to validate the score floor rejects rather than force-feeds (this is the
  half of AC-2's spirit that a naive "does the right thing come back" set never tests); (d) a
  handful of **near-duplicate stress pairs** — two entries that are topically close but
  meaningfully distinct — to confirm the embedding discriminates rather than just clusters broadly.
- **Author it independently of `cobb`.** `cobb` writes the stored entries; if `cobb` also writes
  the paraphrased eval queries, the paraphrase will echo the entry's own vocabulary more than a
  genuinely different agent's situation-description would, understating the real AC-2 gap this
  requirement exists to close. Prefer **real** situation descriptions pulled retrospectively from
  session transcripts or raw `kaizen_team` entries that actually triggered a KB read, over
  synthetic paraphrases, wherever a real one exists — better external validity, same cost to
  collect (it's a query, already exists in text form) as writing one from scratch.
- **Metric:** recall@K primary (K=5, matching Recommendation 3); MRR secondary, to see whether
  correct hits land at rank 1 or rank 5 — useful for tuning K/floor, not itself a gate.
- **Sequencing:** run a small pilot (~15-20 pairs, one dense agent's KB) once the pipeline plus a
  first embedded slice exists, to sanity-check the mechanism and produce the first calibrated score
  floor (Recommendation 3). Run the full ~40-pair set as the AC-2 regression gate once migration
  covers both the four interim KBs (AC-1/AC-5) and the five/six pre-existing KBs (AC-4) — and
  re-run it, cheaply, any time a bulk migration tranche lands or the embedding model/prefix
  convention changes (Qwen3-Embedding-0.6B → 4B upgrade, or adopting Recommendation 1's prefix on
  an already-embedded corpus, both invalidate a previously-calibrated floor).

## Risks & open questions (mine to flag, not mine to resolve)

- **Recommendation 2's split-migration effort is real, not zero**, and should be sized into
  `architect`'s plan rather than assumed to be a mechanical re-embed — this is the concrete shape
  of FR-7's "no more rework than the pre-existing KBs cost" concern, landing specifically on
  `review-techniques.md`.
- **The code-retrieval quality question (Recommendation 1) stays genuinely open until the
  stratified golden-set run produces a number** — I looked for a citable Qwen3-Embedding
  code-retrieval sub-score and could not verify one in this session; don't let this note's overall
  "keep the model" verdict be read as having closed that specific sub-question with evidence it
  doesn't have.
- **The score-floor value and the top-K=5 default are both provisional until Recommendation 4's
  pilot run** — I'm recommending the *procedure* that produces the number, not the number itself,
  deliberately: any cosine-distance constant I supplied without running the pipeline would be
  exactly the kind of unmeasured claim this lab's own conventions (and my own guardrails) rule out.
- Which graph is the substrate (`kaizen_team` directly, or `falkor-chat`'s graph via ingestion) is
  explicitly `architect`'s call per the requirements doc's Open question #1 — every recommendation
  above is substrate-agnostic by design (model, chunking, retrieval params, and eval all transfer
  unchanged to either answer).
