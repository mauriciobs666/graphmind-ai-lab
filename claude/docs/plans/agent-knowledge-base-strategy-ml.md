# Agent knowledge-base strategy — ML method note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`) · **Version:** 2

**Revision note (2026-09-17).** Revised in place, not forked (`AGENTS.md` collision rule 5 — this
note has never itself been independently reviewed or executed against; only the plan's Stage 0,
whose content didn't depend on this note, has shipped) to answer `architect`'s two explicit asks
in the plan's 2026-09-17 substrate revision and to retarget terminology from the former
`kaizen_team`/`:KnowledgeEntry` framing to Option B's falkor-chat `Document`/`Chunk` inside
`ws:agent-team` — see the response section below for the two answers, stated explicitly, and the
recommendation sections for what changed in each one's technical detail.

## The question and the decision it serves

`claude/docs/plans/agent-knowledge-base-strategy.md` (Status: active, `architect`) designs a
GraphRAG-backed retrieval layer over distilled agent knowledge — resolved, as of the 2026-09-17
revision, to **Option B**: falkor-chat's own `Document`/`Chunk` shape inside a new `ws:agent-team`
workspace (Track 2, FR-2–FR-7), reached via falkor-chat's already-shipped `search_documents` MCP
tool directly, no bespoke retrieval server. The embedding model choice (Qwen3-Embedding-0.6B @
1024-dim via LM Studio, reusing `falkor-chat`'s proven stack) stays **settled and not reopened
here**. This note's original four method questions — corpus fit, node granularity, retrieval
parameters, AC-2 validation — are answered below, confirmed **substrate-agnostic** as both this
note and the plan's closing section assert. This revision additionally answers two
realization-level questions the plan's revision raised explicitly: whether a client-side calling
convention (rather than server code) still delivers the query-instruction-prefix/score-floor
methodology intact, and whether losing an automatic `familyId` sibling-pull traversal is a real
retrieval-quality risk.

**Bottom line, quotable:** keep Qwen3-Embedding-0.6B, but use its own documented **asymmetric
query-instruction convention** (free, and this corpus needs it more than `falkor-chat`'s did);
chunk at **one distinct claim per `Document`, not one markdown heading per `Document`** — the
existing `review-techniques.md` file already violates the "50-400 words, one technique per
bullet" assumption the requirements doc states, and that gap is a real, present-tense
retrieval-quality risk, not a hypothetical; retrieve with **top-K=5 and a calibrated cosine-distance
floor**, both now applied by a **documented client-side calling convention** around
`search_documents`, not server code (confirmed sound, one condition attached — response section
below); no conversation-style traversal, and no automatic sibling-pull (a title-prefix convention
substitutes — real, but bounded, quality loss, empirically checked rather than assumed, response
section below); and gate AC-2 on a **~40-pair independently-authored golden set with
Wilson-interval recall@K**, now including a small multi-facet stratum scored by set-recall, not a
bare percentage.

## Response to `architect`'s two explicit asks (2026-09-17 plan revision)

### 1. Client-side realization of the prefix/floor — confirmed, on one condition the plan should add

**Answer: yes, this still satisfies the original methodology intent — provided the template and
the floor live in exactly one shared, cited artifact, never duplicated per-agent.** I verified
architect's own finding directly (`falkor-chat/server/falkorchat/services.py:1309-1354`):
`search_documents` calls `q_vec = embedder.embed(query)` on the raw string with no template, no
floor parameter, over-fetching `k = limit * SEARCH_DOCUMENTS_OVERFETCH` and returning whatever
`limit` ranks by `score` (cosine distance) ascending — exactly the "verbatim embed, no
server-side shaping" the plan describes. Moving the prefix and floor to a documented client-side
calling convention preserves the two things that actually make the method valid: the write side
(`ingest_document`) stays unprefixed, matching Qwen3-Embedding's own asymmetric convention
(documents get no prefix), and the floor stays a number *derived from the golden set*, not
asserted — neither of those properties depends on which side of the wire applies them.

**The condition:** a calling convention realized as **prose repeated into N agents' own prompts**
is a different, and strictly worse, risk profile than the same convention as **server code**,
because server code cannot be skipped or mistyped per-caller and prompt text can. Concretely
recommend, resolving Track 2 Stage 7 item (c) rather than leaving it open: land the template and
the calibrated floor as a **single canonical artifact** — a small `skills/`-style package (e.g.
`skills/agent-kb-retrieval/`) holding the exact prefix string, the `limit=5` value, and the
current floor with the run/date that calibrated it, each **as a fenced literal / a small tested
constant**, not prose paraphrase — and have every consuming agent's prompt cite that one artifact
by a single line, per this repo's own "point to the source, don't duplicate" convention
(`claude/AGENTS.md`), never reproduce the template inline. This also directly answers what the
plan's §7 unit test should target: a constant/function stored in that artifact, asserting the
exact string for a given `situation` — catching **drift in the stored definition** (e.g. a future
edit silently dropping the `Instruct:`/`Query:` framing), exactly as §7 intends.

**A gap the plan's §7 test does not close, worth naming rather than assuming away:** that test
guards the *definition*, not *compliance* — whether a given agent's actual `search_documents` call
at runtime used the documented template. This lab's agents call MCP tools directly from their own
prompt-driven reasoning, with no shared wrapper code every call is forced through, so there is no
code path a pytest-style test can hook to catch one agent silently omitting the prefix on one
call. That failure mode degrades quietly — the call still returns *some* ranked results, just
against a floor calibrated for a different (prefixed) embedding distribution, so a skipped prefix
doesn't error, it just quietly demotes true positives below the floor or admits false ones above
it. Accept this as a residual, name it in Risks below, and treat Stage 8's golden-set regression
run as the actual backstop (it re-validates end-to-end recall periodically, which would catch an
aggregate compliance drift even without attributing it to a specific agent) — do not promise a
unit test can catch what only runtime behavior can produce.

**One more discipline this condition surfaces, not previously stated:** the floor and the prefix
are calibrated **together** — a floor computed from the distance distribution of prefixed-query
pairs is not valid against an unprefixed or differently-worded query. The canonical artifact
should carry an explicit dependency note ("this floor value is valid only for this exact prefix
string; re-derive on any prefix change"), so a future edit to one doesn't silently invalidate the
other.

### 2. Losing automatic `familyId` sibling-pull — not a material loss for the primary use case; test the one case it might be

**Answer: probably not material for the retrieval task this system exists to serve, but this is
an empirical question the golden set should be made to answer, not one either of us should
assert.** Splitting one dense heading into several claim-`Document`s (Recommendation 2) already
solves the problem sibling-pull was invented to patch around: each sub-claim gets its own
precise embedding, so a query about one narrow sub-topic (e.g. "an AST reader only checking one
alias spelling") is now directly, independently retrievable on its own merits — that was always
the actual fix for topic dilution; `familyId` traversal was a *convenience* on top of it ("also
show me its siblings"), not the mechanism that made the split sub-claim findable at all. For the
**primary use case** — one querying agent, one current situation, wanting the one applicable rule
— losing the automatic pull doesn't degrade what gets found; it only removes a one-hop
convenience for a case the primary use case doesn't need.

**Where it could matter, and doesn't currently have a test:** a situation that genuinely implicates
*more than one* sibling claim at once (e.g. a guard-design situation touching two of
`review-techniques.md`'s five split sub-claims simultaneously). Without auto-pull, `top-K=5` and
the score floor can each independently cut off a lower-ranked-but-still-relevant sibling, and the
agent has no signal that it's missing a second relevant rule — silently incomplete is worse than
visibly empty. This is architect's own instinct that the loss is probably fine, which I share for
the primary case, but I do not share it with confidence for the multi-facet case, because nothing
in the plan or this note currently measures it — it's a guess either way without the stratum
below.

**So: yes, the ~40-pair golden set needs a dedicated stratum for this**, rather than leaving it a
guess. Concretely: carve out **4-6 pairs** (same weight class as the existing negative-query
stratum, within the existing ~40-pair budget, not additive to it) drawn specifically from families
Stage 6 migration actually splits (so the true sibling set is known), with situations authored —
independently, same rule as the rest of the set — to plausibly need more than one sibling from
that family. Score this stratum with a **different metric than the headline recall@5**: per-pair
**set-recall** (the fraction of the true sibling set present in the top-5), reported **separately**,
never blended into the single-answer recall@5 number — blending a "was the one right answer in
top-K" construct with a "were all the right answers in top-K" construct under one metric would
itself be a validity error, since a high blended score could hide a real completeness gap. Note
this is a genuinely different check than the existing near-duplicate-stress-pair stratum (§4,
Recommendation 4): near-duplicates test whether the embedding *discriminates* between similar
things (false-positive avoidance), while this stratum tests whether it *recovers everything
relevant* (completeness) — the plan's §8 risk bullet currently reads as if the near-duplicate
pairs already cover this; they don't, and the two should stay reported separately.

**Decision rule from the result, so this doesn't sit open forever:** high set-recall (most
siblings recovered without help) closes the risk — the title-prefix manual-recovery convention is
sufficient. Materially low set-recall is the trigger for a fix that, notably, needs **no new
falkor-chat capability and no schema change** — the same client-side-calling-convention pattern
already used for the prefix/floor extends naturally: after a hit whose title carries a
family-slug, the calling convention issues one extra `list_documents` call filtered by that
title-prefix and merges the results locally. Name this as the evidence-triggered fallback, not
something to build speculatively now.

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
- **Option B's `search_documents` (2026-09-17, self-verified, not just taken on architect's word):**
  `Services.search_documents` (`falkor-chat/server/falkorchat/services.py:1309-1354`) embeds
  `query` verbatim (`q_vec = embedder.embed(query)`) with no template, over-fetches
  `k = limit * SEARCH_DOCUMENTS_OVERFETCH` against `repository.search_chunks`, and returns the
  top `limit` rows ordered by `score` (cosine distance) ascending — no floor parameter, no
  traversal (`repository.search_chunks` is `Chunk`-ANN only, confirmed no `Entity`/sibling
  expansion in its Cypher). This is the exact substrate the response section below reasons
  against — grounds both answers directly, not by inference from the plan's prose.

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
  query/document prefix asymmetry was built for. Concretely: embed every stored claim-`Document`
  unprefixed (as `falkor-chat`'s `ingest_document` already does — write side unchanged, §2 of the
  plan), but embed every **query** as `"Instruct: Given a coding agent's description of its
  current situation, retrieve the distilled technique or rule that applies to it.\nQuery: {situation
  text}"`. This costs nothing extra to implement — a string template at the query-embedding call
  site, not a pipeline change — and is the single highest-leverage free lever available before
  reaching for a different model.

**Realization under Option B (2026-09-17): confirmed sound as a client-side calling convention
around `search_documents`, not server code — full reasoning and the one condition attached (a
single canonical, cited artifact for the template + floor, not per-agent prose) in the response
section above.** Nothing here changes; only where the template lives and how it's guarded against
drift does.

**What would change my mind:** if the golden-set eval (Recommendation 4), once stratified by
code/regex-heavy vs. prose-only entries, shows materially lower recall@K on the code-heavy subset,
that's the trigger to try `qwen3-embedding:4b` (documented re-embed-only upgrade, no schema
change) or a code-aware embedder — not to abandon the family speculatively now.

## Recommendation 2 — node granularity: one claim per `Document`, not one heading per `Document`

**Verdict: per-technique is the right target granularity, but "one technique" must be defined by
independent actionability, not by the markdown `## ` boundary cobb happens to have used.** The
brief's own working assumption — "each bullet is already one self-contained technique" — is falsified
by the real file (Findings, above). Migrating mechanically at "one `Document` per `## ` heading"
will silently create a handful of oversized, multi-topic documents precisely in the file that most
needs fine retrieval (`review-techniques.md` is the densest, most heterogeneous KB in the roster).

Why this matters for retrieval quality specifically, not just tidiness: an embedding of a
multi-topic block is a single vector near the *centroid* of everything the block discusses. A
query about one narrow sub-topic (e.g. "an AST reader only checking one alias spelling") will score
lower against that diluted centroid than it would against a vector representing only that
sub-claim — even though the sub-claim is present verbatim in the document's text. This is the
standard RAG chunking failure (topic dilution), not a corpus-specific guess: it is exactly why
chunk coherence is a first-class lever in retrieval design generally.

**Recommendation, concretely:**
- Default unit = the existing convention (one `## `-headed technique = one `Document`) wherever a
  heading genuinely holds one claim, one example, one origin — the common case, unchanged.
- **Split trigger:** a heading becomes N documents when it contains N independently-attributed
  sub-claims — operationally, N distinct bolded sub-headers/enumerated checks each carrying their
  own "Origin:"/worked-example, or a heading materially longer than ~500 words spanning more than
  one verifiable claim. `review-techniques.md`'s "A guard derived from the artifact it guards…"
  and "Two checks for a multi-shape authorization…" both trip this rule today.
- **Do not go finer than one claim.** Splitting to sentence-level breaks self-containment: the
  worked example/counter-example is what makes a rule *actionable* rather than a bare assertion,
  and stripping it defeats the reason these files are written the way they are. The right document
  is the smallest unit that stands alone if someone reads only it.
- **Reject the coarser fallback (whole-file-as-chunk with per-technique as a secondary layer).**
  It directly reopens FR-4 — the entire point is not reading the whole file to find the relevant
  part — so it needs no evidence to reject, only the requirement's own text.
- **Migration process implication:** this makes the migration a **curator judgment pass, not a
  mechanical script**, at least for the pre-existing six KBs (AC-4). A splitting heuristic can
  flag candidates (length + multiple bolded sub-claims + multiple "Origin:" mentions), but the
  actual split boundary is an editorial call — route it through `cobb`, the same agent who already
  owns distillation judgment, not through an unsupervised script. Realized in the plan (§6) exactly
  this way. This is a real (if small) piece of migration effort, sized into the plan rather than
  assumed away.
- **Sibling linkage for split entries — degraded under Option B, not carried forward as a graph
  edge (2026-09-17).** The original design here gave siblings split from one heading a shared
  `familyId` property/edge and a Cypher-side pull. `Document` carries no such property and
  `search_documents` does no traversal at all (confirmed, Findings above) — mitigated instead by a
  shared **title-prefix** naming convention (`"<family-slug> — <claim-title>"`, plan §2),
  manually recoverable via `list_documents`/`get_document` but with no automatic "pull my
  siblings" step in retrieval. **Whether this loss is material, and what the golden set should do
  about it, is answered in full in the response section above — not reasserted here.**

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
it, re-derived whenever the corpus or the prefix convention changes materially. **Realization under
Option B (2026-09-17):** `search_documents` has no floor parameter (confirmed, Findings above) —
the floor is applied client-side by the same calling convention as the prefix (Recommendation 1),
against the returned `score`; same derivation, same "re-derive on any material change" discipline,
now covering the prefix template too, since the two are calibrated jointly (response section
above).

**Traversal: your instinct is correct — confirmed, with one revision (2026-09-17).** A `Document`
is not part of a conversation thread, so `falkor-chat`'s `REPLY_TO`/`MENTIONS`-following pattern
has no real analogue — there's no "prior turn" or "mentioned entity" that adds context the way a
reply adds context to a chat answer. Of the three traversal candidates the original design named:
1. **`familyId` siblings (Recommendation 2) — not available under Option B.** `search_documents`
   does no traversal at all (Findings above); this is now a title-prefix naming convention, not a
   graph hop. **Whether this loss is material is answered in the response section above, with a
   golden-set stratum to test it rather than a guess** — not reasserted here.
2. **`topic` as a filterable property** — not part of Option B's content model (plan §2 defines no
   `topic` property on `Document`). Not built, and not obviously needed: `search_documents`'s
   semantic ranking already does topic-adjacent retrieval implicitly; treat as a nice-to-have, not
   a gap worth flagging in Risks.
3. **Owning-agent attribution — now satisfied for free, not a separate build.** The plan's §4.1
   `produced_by` extension makes `INGESTED_BY` resolve to the actual writing `Agent`
   (`(a:Agent {agentId: $producedBy})`) for every `ws:agent-team` write — exactly the filter this
   recommendation asked for ("scope to my own KB vs. the whole team's"), arrived at independently
   as a prerequisite for correct attribution (plan §1), not because this recommendation asked for
   it. No further design needed here.

Do not build anything resembling entity extraction/`MENTIONS`-fusion over this corpus — that
machinery exists in `falkor-chat` to link chat messages through shared real-world entities, and
there is no equivalent need here; it would be complexity borrowed from a different problem.
(Under Option B this happens automatically and inertly anyway — every `ingest_document` call
fires unconditional extraction into `ws:agent-team`, confirmed unused by `search_documents`, plan
§1/§4.4 — accepted noise, not something this recommendation needs to guard against.)

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
- **Stratification, not random sampling.** Distribute the ~40 across: (a) every existing/interim
  agent's KB, weighted roughly by its entry count, so no single dense file (like
  `review-techniques.md`) silently dominates or is silently absent; (b) a deliberate code/regex/
  shell-heavy subset versus a prose-only subset, sized enough to separately report recall on each
  — this is what actually answers Recommendation 1's open question about domain fit, rather than
  leaving it a guess; (c) 4-6 explicit **negative** queries — genuine situations with no matching
  entry in the corpus — to validate the score floor rejects rather than force-feeds (this is the
  half of AC-2's spirit that a naive "does the right thing come back" set never tests); (d) a
  handful of **near-duplicate stress pairs** — two entries that are topically close but
  meaningfully distinct — to confirm the embedding discriminates (false-positive avoidance) rather
  than just clusters broadly; **(e) 4-6 multi-facet/sibling-claim pairs (added 2026-09-17, within
  this same ~40-pair budget, not additive to it)** — drawn from families Stage 6 migration
  actually splits, situations authored to plausibly need more than one sibling from that family at
  once, scored by **per-pair set-recall** (fraction of the true sibling set in top-5), reported
  **separately** from the headline recall@5 rather than blended into it — a genuinely different
  construct from (d)'s discrimination check: this one tests completeness, not false-positive
  avoidance. This stratum is what settles whether losing automatic `familyId` sibling-pull
  (Recommendation 2, response section above) is a retrieval-quality problem empirically, rather
  than by architectural argument alone.
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
  `review-techniques.md`. Confirmed realized in the plan's §6 (Track 2 Stage 6), unchanged.
- **The code-retrieval quality question (Recommendation 1) stays genuinely open until the
  stratified golden-set run produces a number** — I looked for a citable Qwen3-Embedding
  code-retrieval sub-score and could not verify one in this session; don't let this note's overall
  "keep the model" verdict be read as having closed that specific sub-question with evidence it
  doesn't have.
- **The score-floor value and the top-K=5 default are both provisional until Recommendation 4's
  pilot run** — I'm recommending the *procedure* that produces the number, not the number itself,
  deliberately: any cosine-distance constant I supplied without running the pipeline would be
  exactly the kind of unmeasured claim this lab's own conventions (and my own guardrails) rule out.
- **New (2026-09-17): client-side calling-convention compliance is not unit-testable, only
  aggregate-observable.** The plan's §7 unit test guards the stored template's *definition*
  against drift; it cannot catch one agent's runtime call silently omitting the prefix or floor,
  since this lab's agents call `search_documents` directly from prompt-driven reasoning with no
  shared wrapper code forcing every call through the canonical artifact. The failure mode is
  quiet (a skipped prefix demotes true positives below a floor calibrated for the prefixed
  distribution rather than erroring) — accepted, with Stage 8's recurring golden-set run as the
  only backstop that would notice an aggregate compliance drift (not attribute it to one agent).
  Full reasoning in the response section above.
- **New (2026-09-17): the multi-facet/sibling-claim golden-set stratum (Recommendation 4(e)) is
  the concrete instrument that resolves whether losing `familyId` sibling-pull matters** —
  currently unmeasured; my own architectural read (aligned with `architect`'s) is that it probably
  doesn't for the primary single-rule use case, but that read is provisional until this stratum
  runs, exactly like the score-floor/top-K values above.
- **Superseded (2026-09-17):** "which graph is the substrate" is no longer open — the requirements
  doc's decision log settled it as Option B (falkor-chat ingestion, `ws:agent-team`), confirmed by
  `architect` reading the current source directly (plan §1). Every recommendation above is
  confirmed to transfer unchanged in substance, exactly as this note asserted substrate-agnostically
  before the choice was made — only the realization mechanism for the prefix/floor (server code →
  client-side convention) and the sibling-linkage mechanism (`familyId` edge → title-prefix
  convention) changed, both addressed above.
