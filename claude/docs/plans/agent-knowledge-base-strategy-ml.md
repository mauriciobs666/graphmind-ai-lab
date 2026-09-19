# Agent knowledge-base strategy — ML method note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`) · **Version:** 3

**Revision note (2026-09-19).** Revised in place, not forked (`AGENTS.md` collision rule 5 — still
never independently reviewed or gated as of this pass) to execute Stage 8 Phase 1: the AC-2
golden-set design Recommendation 4 specified, plus a pilot run against the now-fully-migrated
corpus (all 13 KB files, 327/332 claims `ready` in `ws:agent-team`) that produces the first
calibrated score floor. New section "Stage 8 Phase 1" below; the "Risks & open questions" section
is updated at its end to close what this resolves and name what stays open for the qa-engineer-run
full-set gate (Stage 8 Phase 2). Nothing above this point changed in substance.

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

## Stage 8 Phase 1 — golden-set design, pilot execution, score-floor calibration (2026-09-19)

**The question this phase answers:** Recommendation 4 designed the AC-2 golden set and its
sequencing (pilot → full set) before any corpus existed to test it against. Migration (Stage 6)
is now closed — 327/332 claims `ready` across all 13 KB files (5 permanently `failed`, content
byte-exact but not search-retrievable — `docs/reviews/agent-knowledge-base-strategy4-stage6.md`).
This phase (a) lands the full ~40-pair golden set as a reusable, structured artifact so a later
`qa-engineer` unit doesn't redesign it, and (b) executes a weighted pilot subset against the real
`search_documents` tool to sanity-check the mechanism and produce `skills/agent-kb-retrieval/
SKILL.md`'s first calibrated score floor, closing that file's "provisional, currently disabled"
section. **Sequencing call, diverging slightly from the letter of Recommendation 4's own
condition:** that recommendation gated the *full* 40-pair execution on migration covering both
Track 0's interim KBs and the pre-existing ones — now true — but I judge the pilot-then-full split
still the right call even so: a fresh calibration needs a first real number before a `qa-engineer`
unit spends a much larger query budget against a floor that hasn't been sanity-checked at all, and
the dispatching brief asked for exactly this split. The **full 45-pair table below is the
design**, all queries and expected answers fixed; **16 of the 45 are executed here** (the pilot);
the remaining 29 are ready for Stage 8 Phase 2 to run as-is, with no re-derivation needed — and,
notably, **none of the 29 were used to derive the floor below**, so Phase 2's run is genuine
out-of-sample validation of it, not a re-run of the same evidence.

### Golden-set design — 45 pairs, not exactly ~40

Sized to 45 rather than exactly 40 because covering **every one of the 13 migrated KBs with a
floor of at least 1 pair**, plus a full 6-family stratum-(e) set (Stage 6 review Appendix A names
7 real split families; I used 6 plus one held in reserve) and a 6-pair negative stratum, added up
to slightly more than "~40" — I did not force a cut to hit the number exactly, since every row
earns its place under one of Recommendation 4's five axes and the ML note's own "~40" was
explicitly approximate, not a hard budget. Weighted roughly by each KB's migrated-claim count
(`claude/cobb/scripts/kb-claim-manifest.json`, cross-checked against the coordination ledger's
per-file totals, 332 claims total): `review-techniques.md` (81 claims, 24%) and
`falkordb-quirks.md` (86 claims, 26%) get 9 rows each; `coordination-techniques.md` (39, 12%) gets
5; the remaining ten KBs get 1-2 each (floor of 1, `estimator-test-fixtures.md` at only 2 claims
total). All expected `documentId`s below were confirmed live against `ws:agent-team`
(`mcp__cypher__query`) before being fixed in this table — not asserted from the source `.md`
files' headings alone.

**Query-authoring discipline (per Recommendation 4's "author independently of `cobb`"):** every
query below is my own situation-description, grounded in the claim's real `Origin:`/worked-example
text (a genuine historical incident, `falkor-chat` K-028, `cypher-mcp`'s `authorize_write()`, a
real WSL2/LM-Studio session, etc. — read directly from the source `.md` files, not paraphrased
from the stored claim's own title/fact wording) rather than a synthetic rewording of the stored
claim itself. This is deliberately a harder test than echoing the claim's vocabulary — it costs
some recall (see the pilot's one clean miss, below) by design, not by accident.

Legend: **Tags** — a=KB-weighted, b-code=code/regex/shell-heavy, b-prose=prose-only,
c=negative, d=near-duplicate stress, e=multi-facet/sibling stratum (scored by set-recall, not
headline recall@5). **Pilot** = executed 2026-09-19 (✓) vs. design-only, ready for Phase 2.

#### `claude/analyst/review-techniques.md` (9 rows)

| # | Tags | Query (situation) | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| R1 | a,e | "Reviewing a fix to falkor-chat's workflow-timer state machine (the K-028 v2-to-v3 change): the new version makes every wait step carry a mandatory default fallback transition so the state machine always makes forward progress. Two earlier review passes already signed off on the fix's own reasoning. What am I supposed to check beyond confirming the fix's internal logic holds together?" | `d41d743e5c344eccbf46715f54cbab44` + `5c2ef4055ff34ea6a1ba5f7eb329ae4c` (family h13) | ✓ |
| R2 | a,e | "I'm doing a Pass-2 review of a fix to cypher-mcp's authorize_write() function, which checks an incoming Cypher write statement against a sequence of recognized authorized shapes before allowing it through. The fix author supplied two concrete attack reproductions that are now both blocked. What should I check beyond re-running those two named attacks?" | `0158a990da6d46b58b0eea3f32047669` + `ba1e9b45d83a443c8fde5f94dbe51daa` (family h21) | ✓ |
| R3 | a,e | "A document under review says three flagged passages have each since had their own certifying sentence corrected on review. I don't want to just read the diffs between versions — how do I actually verify a document's claim about its own revision history, and does the same concern apply to a header's blanket promise about what every entry underneath it does?" | `5a6763497f2740e8aa21efac72b3a1c6` + `ccefd08d853247cd941de116946cda5b` (family h39) | ✓ |
| R4 | a,e | "A plan claims a NULL-backfill decision surfaced during a specific earlier investigation, and separately a pasted grep result from a few days ago said a schema had zero RELATIONSHIP-type constraints. Are both of those still safe to cite as-is?" | `138ec31887af425ca7f0bd3eb87817c4` + `7fbe0fe06e824180a6bf4937bfe88542` (family h14) | — |
| R5 | a,e | "I'm trying to establish where some data materialized in a live graph actually came from — can git history settle that, and if two derived snapshots agree with each other, does that prove they're both correct?" | `f0a93c99dace45029538dd2bbb78a390` + `af28885869224aab85048555903c3c36` (family h22) | — |
| R6 | a,e | "A review's suggested fix told the implementer to capture `$?` right inside an `if ! VAR=\"$(cmd)\"` block to route a helper's tri-state return — and separately, a static guard has been closed three times against a fixed list of shapes and keeps getting beaten by a form nobody listed. Are both of those remedies actually sound?" | `af9ffb191cbd4659a77dbf3a1139c289` + `5b1b477ff67e4e3b81c57f14899bbe48` (family h40) | — |
| R7 | a,d | "I grepped the codebase for a function name and got three hits, so I'm ready to call a 'does this helper already exist' finding confirmed. Is counting those grep hits actually strong enough evidence that the function is defined, not just referenced somewhere?" | `d3c4ef7117c343bfaf003d1ad7fc44b8` (distractor, must not outrank/replace: `4fe63fb109bd4a04ba479389a29c8e08`) | ✓ |
| R8 | a,b-code | "I'm reviewing a piece of code that clamps a computed value between a min and a max before using it downstream. Is there a known failure mode where that kind of clamp can quietly let a NaN through, or where swapping the order of the min/max calls changes the result?" | `297b3ed4acf44f07bfed46f7de76e49d` | ✓ |
| R9 | a,b-prose | "A repo-wide lint/smell sweep flagged a violation in a file my current diff never touched. Should I treat that as something my change introduced, or could it have been sitting there before my diff even started?" | `6d88fecb435547dea1d93d67b8995356` | ✓ |

#### `claude/graph-dba/falkordb-quirks.md` (9 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| F1 | a,d | "I need to store a KaizenEntry.fact string that contains an apostrophe, like a contraction, as an inline Cypher literal. Is the SQL-style doubled single-quote trick for escaping an embedded apostrophe going to work here?" | `17731d3d4d9d4f04a15d139dd42e9bad` (distractor: `719fc4c69d1845ad9298b32c0adecdc6`) | ✓ |
| F2 | a,b-code | "A load test against FalkorDB is reporting a peak row count that never exceeds exactly 10000, no matter how large the matching set actually is, and a query with an explicit LIMIT 50000 is coming back short too. Is there a silent server-side cap I should know about, and how would a caller tell a capped reply apart from a genuinely complete one?" | `5f04f062d95b4ace82d3c5b5059dbc8c` | ✓ |
| F3 | a,b-code | "I wrapped a redis-cli GRAPH.RO_QUERY call in a bash script with set -e and it just silently continued past what I'm pretty sure was a malformed query, no error caught. Does redis-cli's exit code actually reflect a rejected query, or do I need a different way to detect that?" | `f38c0ed774494b88afa6a5e93ad37010` | ✓ |
| F4 | a,b-code | "I ran a targeted property update against one node and the reply said one property was set AND one was removed, but I didn't remove anything and every other property is still there. What's going on?" | `684e4ea5762c4f62abad5b8dfbbf9040` | — |
| F5 | a,b-code | "A repository function projects a property that a UNIQUE constraint is supposed to guarantee is present, but one row came back with that key as null and nothing raised. Is a UNIQUE constraint here also a NOT NULL guarantee?" | `72a1e5e1cb4144bfa325f3a41713a6a3` | — |
| F6 | a,b-code | "I need to rename a graph key without doubling RAM or risking a window where the data doesn't exist under either name. Is COPY-then-DELETE the only way, or is there something more direct?" | `fae4b822cca4460a8c8bde6c0b8c35a0` | — |
| F7 | a,b-prose | "A colleague says a destructive Redis op run from inside a wrapper script used to slip past our command-pattern guard because the literal command text never appeared in the outer shell invocation. Is that still an open gap?" | `801c1bb16d4d457abafe132db1419257` | — |
| F8 | a,b-code | "I want to non-destructively check whether a graph key already holds data before writing to it, without accidentally creating an empty graph as a side effect of the probe itself." | `be7d7c5755b64419b9ed6394226299d2` | — |
| F9 | a,b-code | "A query returns one scalar column alongside a collect(DISTINCT ...) aggregate from an OPTIONAL MATCH fan-out, and I assumed the scalar would be constant across every row of the result. Is that actually guaranteed?" | `74ce458645d247dda774d950c001ccd3` | — |

#### `claude/teco/coordination-techniques.md` (5 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| C1 | a,b-prose | "A delegated subagent reported finishing its file edits and handed back cleanly, no crash, no error — but when I went to verify, the edits weren't actually in the working tree, and there's no git history showing they were ever made and then reverted. How is that even possible, and how would I confirm this happened rather than assume I mis-read the diff?" | `9870c48f9840491d94e4f1d62ca7a3c5` | ✓ |
| C2 | a,b-prose | "When mutation-testing a set of tests that are already green on arrival, is deleting the implementation the only mutant worth running, or should a rejected design alternative also count as one?" | `ef4c32cc11c44747994bb60e62bb5940` | — |
| C3 | a,b-prose | "Two concurrent sessions are both about to commit into the same working tree. If our changed files don't overlap, is a plain `git commit -a`-style commit still safe, or do I need to do something more careful?" | `1b0addf5b8354bacaeb0e938d09d7ab6` | — |
| C4 | a,b-prose | "I couldn't find evidence of something in one place I checked — is that enough to conclude it doesn't exist anywhere, or could it just be that I looked in the wrong place?" | `506f08975c4641508da50fe9b9c585a2` | — |
| C5 | a,b-prose | "Two independent-seeming checks in a gate both agree with each other. Does agreement between them mean the underlying claim is actually verified, or could my own checking instrument be the thing that's wrong for both?" | `1200232e42074349a6bd6c0cc45e290f` | — |

#### `claude/qa-engineer/qa-testing-techniques.md` (2 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| Q1 | a,e | "model-bench's run command just refused to proceed with a stale-attestation error, even though I re-ran attest a minute ago non-interactively with --set for every field and none of the operator-attested values actually changed. What's causing the refusal, and what's the right way to re-attest from a script rather than an interactive shell in the first place?" | `781c055d0be5468f98dbe6143d5ac551` + `1034d23fe5b04503b90c53128e7cfb17` (family) | ✓ |
| Q2 | a,b-prose | "A model shows up in the /v1/models listing, but I need to know whether it's actually loaded with enough context length for my prompt, not just that it's present at all." | `93fcdd089eb94f1d8b29ce388622c982` | — |

#### `claude/devops/ops-quirks.md` (2 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| O1 | a,b-prose | "A bash script I'm reviewing shells out to jq to pull one specific key out of a JSON config file for an exact-match lookup. Is jq something I can assume is installed on every dev box here, or is there a more portable way to do that lookup?" | `2fdbec98873c4ce38a49d72f01c2a95a` | ✓ |
| O2 | a,b-prose | "I edited .mcp.json to add or reconfigure an MCP server. Is there a way to confirm the launch shape actually works from a plain shell, or do I have to restart the whole harness to find out?" | `91b9f5bb09e54db3a798f05f8a55bd40` | — |

#### `claude/architect/plan-authoring-techniques.md` (2 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| P1 | a,b-prose | "A plan states a completeness claim ('every X now does Y') but I only have the author's word for it. What would actually make that check able to fail, rather than just being transcribed as true?" | `08a422aeca974749a5a4cf1dab2164e5` | — |
| P2 | a,b-prose | "A permission or allow-list design I'm reviewing technically enforces its rule today, but I suspect a slightly different input shape would sail right past it. Is a tighter pattern the right fix, or does this need something more structural?" | `15c71ec82ff249a79a3b9045268e0c3f` | — |

#### `claude/data-scientist/statistical-method-techniques.md` (1 row)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| S1 | a,b-prose | "Two groups' 95% confidence intervals don't overlap, and the sample is small and near a ceiling value. Is that non-overlap itself a valid difference test?" | `f12fb25f754e4d4ca58acc163c9c422a` | — |

#### `claude/frontend-engineer/frontend-quirks.md` (1 row)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| E1 | a,b-code | "A fresh Vite + TypeScript scaffold is failing to compile because it can't resolve a JSON import. Is that a tsconfig option I need to turn on myself?" | `6891424b924a404c9691203415b848ba` | — |

#### `claude/tdd-engineer/test-design-techniques.md` (1 row)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| T1 | a,b-prose | "A mutation-testing pass keeps deferring one mutant as 'needs a lucky random input to trigger.' Is there a way to force it without relying on Monte-Carlo luck?" | `aa72815433924cfea2760062daa22b40` | — |

#### `claude/tdd-engineer/estimator-test-fixtures.md` (1 row)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| X1 | a,b-prose | "A test fixture I'm using has zero variance on the dimension the rule under test actually cares about. Could that be silently hiding whether the rule works at all?" | `75c0e47a01244de5b4373a193887293c` | — |

#### `claude/tdd-engineer/guard-testing-techniques.md` (2 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| G1 | a,b-prose | "A mutation-testing pass on a guard came back all green. Does that mean a plain coverage probe over the same guard would find nothing new?" | `2a467d9fef184191b5fadfee14b6839d` | — |
| G2 | a,b-prose | "I hand-wrote a resolver that decides 'which object is this' from a set of clues. Testing it feels like it has two different independent things that could go wrong — is that right, and are both equally testable?" | `f29bddad2cf14479ba6bafd9bd0c4212` | — |

#### `claude/graph-dba/falkordb-reference.md` (2 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| D1 | a,b-code | "Is there a meaningful difference between GRAPH.QUERY and GRAPH.RO_QUERY beyond just read-vs-write, and should I always be parameterizing?" | `e69ff7c592a14a02b9f6eed78f716cf8` | — |
| D2 | a,b-code | "I want to run a graph algorithm like betweenness or label propagation against FalkorDB. Do I need something like Neo4j's GDS library, or APOC?" | `1de0ff37ffa3494bbc832e5c9fd26cd6` | — |

#### `claude/data-scientist/lm-studio-model-notes.md` (2 rows)

| # | Tags | Query | Expected `documentId`(s) | Pilot |
|---|---|---|---|---|
| L1 | a,b-prose | "Between a small Mistral-family model and a Qwen3 model of similar size, which one is more likely to emit a real structured tool call rather than plain prose describing what it would call?" | `0181f4c0350d47458e3800e76a36d05a` | — |
| L2 | a,b-prose | "Qwen3-Embedding's own docs describe some kind of asymmetric convention between how queries and documents should be embedded. If I just embed both the same way, how bad is that really?" | `089f80e62c0f458992f086fae82ef557` | — |

#### Negative queries (6 rows — no matching entry in the corpus)

| # | Tags | Query | Expected | Pilot |
|---|---|---|---|---|
| N1 | c | "I need to rotate the TLS server certificate on our production Postgres database without any client-facing downtime — what's the safe sequence for that?" | none | ✓ |
| N2 | c | "I want to set up a canary deployment for a newly retrained ML model in production, with an automatic rollback if p99 latency regresses past a threshold — how should that pipeline be structured?" | none | ✓ |
| N3 | c | "How do I configure GPU passthrough so LM Studio can use an NVIDIA card for inference on this Linux host?" | none | ✓ |
| N4 | c | "How should I structure a React Server Components data-fetching boundary to avoid request waterfalls in a Next.js app?" | none | ✓ |
| N5 | c | "What's the correct way to configure the OAuth2 Authorization Code flow with PKCE for a public single-page app that has no backend session store?" | none | — |
| N6 | c | "How do I configure a Kubernetes readiness probe for a FastAPI container whose startup depends on a slow-starting database connection?" | none | — |

### Pilot execution (2026-09-19, run against `ws:agent-team`)

Method: for each of the 16 ✓-tagged rows above, built the exact prefixed string per
`skills/agent-kb-retrieval/SKILL.md`'s fenced template (`"Instruct: Given a coding agent's
description of its current situation, retrieve the distilled technique or rule that applies to
it.\nQuery: {situation}"`), called `search_documents(query=<prefixed>, limit=5)` against
`ws:agent-team`, and judged each `documentId` returned (`score` = cosine distance, ascending,
lower = more similar) against this row's fixed expected answer(s). Full per-query score tables are
not reproduced here (already run and recorded in this session's tool trace); the results below are
the derived judgments.

**Hits/misses, single-answer pool (R7, R8, R9, F1, F2, F3, O1, C1 — 8 queries):**

| Query | Rank of correct answer | Best score of correct answer |
|---|---|---|
| R7 (near-dup) | 1 | 0.294 |
| R8 | 1 | 0.179 |
| R9 | 2 | 0.340 |
| F1 (near-dup) | 1 | 0.200 |
| F2 | 1 | 0.267 |
| F3 | 1 | 0.165 |
| O1 | 1 | 0.305 |
| C1 | **miss — not in top 5** | — |

7/8 found → **recall@5 = 0.875**, **Wilson 95% CI [0.529, 0.978]** (z=1.96, n=8 — a wide interval
by construction at this n, exactly the honesty the Wilson convention is for; this is not the
full-set gate). **MRR = 0.8125** (six rank-1 hits, one rank-2 at 0.5, one miss at 0).

Both near-dup rows (R7, F1) passed their discrimination check cleanly: in both cases the intended
distractor sibling (`4fe63fb1…`, `719fc4c6…`) did **not** appear anywhere in the top 5 — the
embedding didn't just avoid ranking it above the correct answer, it didn't surface it as a
plausible alternative at all. Small n (2), but a clean directional result for Recommendation
4(d)'s discrimination question.

**Set-recall, stratum (e) (R1, R2, R3, Q1 — 4 families, 8 sibling documents):**

| Family | Siblings found / total | Set-recall |
|---|---|---|
| R1 (h13) | 2/2 | 1.0 |
| R2 (h21) | 2/2 | 1.0 |
| R3 (h39) | 1/2 | **0.5 — second sibling (`ccefd08d…`) never appeared in top 5** |
| Q1 (model-bench) | 2/2 | 1.0 |

Pooled: 7/8 siblings found (0.875 — numerically identical to the headline recall@5 figure above,
a coincidence of this specific n=8-vs-n=8 pilot, reported here as its own construct per
Recommendation 4's explicit instruction not to blend the two). **The R3 miss is a genuine finding,
not floor-related** (see below): the top 5 for that query held 3 duplicate chunks of the *other*
sibling document (`5a676349…`, at seq0/1/2) crowding out room the true second sibling needed —
concrete, measured evidence for the `familyId`-sibling-pull risk the 2026-09-17 revision flagged
as needing empirical evidence rather than architectural argument. On this one case, the loss is
real: no floor value would have fixed it, because the document never reached the top 5 at all.

**Negative stratum (N1-N4 — 4 queries, no correct answer exists):**

| Query | Closest (top-1) returned score |
|---|---|
| N1 (Postgres TLS) | 0.647 |
| N2 (ML canary deploy) | 0.562 |
| N3 (LM Studio GPU) | 0.552 |
| N4 (React Server Components) | **0.446 — closest of the four, a real near-neighbor by topic (LM Studio/React infra) without answering the actual question** |

### Score-floor derivation

Pooled every *found* true-positive document's best (lowest) returned score across all 16 pilot
queries (14 found-document events: 7 stratum-e siblings + 2 near-dup primaries + 5 plain
positives; the R3/C1 misses contribute no score, they are recall losses regardless of any floor)
against the 4 negative queries' closest (lowest, i.e. most dangerous) returned score:

- **Worst surviving true positive: 0.409** (`5c2ef405…`, the h13 second sibling).
- **Closest negative-query false match: 0.446** (N4, React Server Components → a React-18
  batching claim).
- Gap: **0.037** — real, but thin.

**Calibrated floor: 0.42** (cosine distance; reject any hit with `score > 0.42`), sitting in the
gap with a margin of 0.011 above the worst surviving true positive and 0.026 below the closest
negative false match. At this floor: **0 of the 14 found true-positive documents are wrongly
dropped** (max 0.409 < 0.42), and **all 4 pilot negative queries are correctly rejected** (all four
top scores > 0.42).

**Explicit limitation, not glossed over: this floor is fit and validated on the same 4 negative
queries and same 14 positive hits — there is no held-out validation in this pilot.** The margin is
thin (0.037) and n is small; a different negative query, or a different sibling-claim pair, could
plausibly land on either side of 0.42. The right mitigation already exists in the design: **the 29
design-only rows above (including N5/N6, 2 more negatives) were never used to derive this floor**,
so Stage 8 Phase 2's full run against them is genuine out-of-sample validation, not a re-run of the
same evidence — treat 0.42 as this pilot's first, provisional, conservative floor, to be
**confirmed or re-derived** once Phase 2 lands, exactly as Recommendation 3's original "re-derive
whenever the corpus or the prefix convention changes materially" discipline already anticipated
(a full-set run is itself such a change in effective sample size, if not in corpus/prefix).

**Run/date of record:** 2026-09-19, `data-scientist`, pilot n=16 of 45 designed pairs, against
`ws:agent-team` post-Stage-6-migration (327/332 claims `ready`). `skills/agent-kb-retrieval/
SKILL.md`'s floor section is updated with this value and this run/date below.

### A secondary, unplanned finding worth flagging: code/regex-heavy retrieval looked *strong*, not weak

Recommendation 1 named "does this encoder handle code/regex-dense entries as well as prose
entries" as the corpus's open empirical question. This pilot is far too small to close it (5
code/regex-heavy queries vs. 3 prose-only ones in the clean single-answer pool), but the
*direction* is the opposite of the worry: every code/regex-heavy query (R7, R8, F1, F2, F3) landed
its correct answer at rank 1, scores clustered tightly (0.165-0.294); the one clean miss (C1) and
the one rank-2 (R9) were both prose/process narratives, not code. Not a claim the risk is closed —
Phase 2's larger, more balanced sample is what would actually settle it — but worth carrying
forward as a specific thing to watch rather than assuming the original worry direction holds.

## Risks & open questions (mine to flag, not mine to resolve)

- **Recommendation 2's split-migration effort is real, not zero**, and should be sized into
  `architect`'s plan rather than assumed to be a mechanical re-embed — this is the concrete shape
  of FR-7's "no more rework than the pre-existing KBs cost" concern, landing specifically on
  `review-techniques.md`. Confirmed realized in the plan's §6 (Track 2 Stage 6), unchanged.
- **Partially resolved (2026-09-19), still not closed with confidence.** The code-retrieval
  quality question (Recommendation 1) got a first, small, *directional* data point from the Stage
  8 Phase 1 pilot — every code/regex-heavy query retrieved cleanly at rank 1, no sign of the
  originally-worried underperformance — but n=5 code-heavy queries cannot close this; a citable
  Qwen3-Embedding code-retrieval sub-score is still unverified. Treat the pilot's direction as a
  reason for lower urgency, not as evidence the question is settled — Phase 2's larger, more
  balanced sample is what would actually close it.
- **Resolved (2026-09-19).** The score-floor value and top-K=5 default were provisional pending
  Recommendation 4's pilot run — that run happened (Stage 8 Phase 1, above): **top-K=5 is
  confirmed sufficient** (every found true positive landed at rank 1-2, well inside K=5, no
  evidence for widening it), and **the score floor is calibrated at 0.42** (cosine distance),
  landed in `skills/agent-kb-retrieval/SKILL.md`. Carried forward as a new, narrower open item
  below: the floor's margin is thin (0.037) and derived on a small, non-held-out sample.
- **New (2026-09-17): client-side calling-convention compliance is not unit-testable, only
  aggregate-observable.** The plan's §7 unit test guards the stored template's *definition*
  against drift; it cannot catch one agent's runtime call silently omitting the prefix or floor,
  since this lab's agents call `search_documents` directly from prompt-driven reasoning with no
  shared wrapper code forcing every call through the canonical artifact. The failure mode is
  quiet (a skipped prefix demotes true positives below a floor calibrated for the prefixed
  distribution rather than erroring) — accepted, with Stage 8's recurring golden-set run as the
  only backstop that would notice an aggregate compliance drift (not attribute it to one agent).
  Full reasoning in the response section above.
- **Partially resolved (2026-09-19), and the answer is more mixed than either of us guessed.**
  The multi-facet/sibling-claim golden-set stratum (Recommendation 4(e)) ran on 4 families in the
  pilot: 3 of 4 recovered both siblings cleanly (set-recall 1.0), but the 4th (family h39, R3
  above) recovered only one of two — the missing sibling never reached the top 5 at all, crowded
  out by three duplicate chunks of the *other*, already-found sibling document. **This is real,
  measured evidence the `familyId` sibling-pull loss is material in at least one case**, not just
  a hypothetical the architectural argument could wave off — and no score-floor value could have
  fixed it (the document wasn't retrieved, so a floor never got a chance to reject or admit it).
  Genuinely open for Phase 2: is this systematic (a specific chunk-duplication pathology worth
  fixing — e.g. de-duplicating same-document chunks before truncating to top-K) or a one-off on
  this specific family/query pairing? The remaining 2 stratum-(e) families in the full design
  (R4/h14, R5/h22, R6/h40) are exactly what would tell.
- **New (2026-09-19): the calibrated floor (0.42) is fit and validated on the same small sample
  — thin margin (0.037), no held-out negatives in the pilot.** Not a defect in the derivation (the
  method — max surviving true positive vs. min negative false match — is the right one, and it's
  disclosed in the same section, not hidden), but a genuine reason not to treat 0.42 as final.
  Stage 8 Phase 2's run against the 29 not-yet-executed design-only rows (including 2 more
  negatives, N5/N6) is real out-of-sample validation of this exact number — if it holds up there,
  it's confirmed; if a Phase 2 negative scores below 0.42 or a Phase 2 positive scores above it,
  re-derive using the pooled Phase 1 + Phase 2 evidence rather than patching the single number.
- **New (2026-09-19): the one clean single-answer miss (C1, `coordination-techniques.md`'s
  "a delegate's edits can vanish mid-run" claim) is unexplained, not just unlucky.** The returned
  top 5 were all topically adjacent (verification-distrust/provenance claims) but none was the
  actual target — a genuine embedding-distance gap between this real, independently-authored query
  and its correct answer, not a floor issue (nothing scored close enough for a floor to matter) and
  not an artifact of duplicate chunks crowding the page (all 5 returned were distinct documents).
  Whether this is one hard query or a real weak spot for narrative/process claims (as opposed to
  the concrete code/config claims that all retrieved cleanly) is exactly what Phase 2's larger,
  more prose-heavy sample would surface — flagged, not diagnosed further here.
- **Superseded (2026-09-17):** "which graph is the substrate" is no longer open — the requirements
  doc's decision log settled it as Option B (falkor-chat ingestion, `ws:agent-team`), confirmed by
  `architect` reading the current source directly (plan §1). Every recommendation above is
  confirmed to transfer unchanged in substance, exactly as this note asserted substrate-agnostically
  before the choice was made — only the realization mechanism for the prefix/floor (server code →
  client-side convention) and the sibling-linkage mechanism (`familyId` edge → title-prefix
  convention) changed, both addressed above.
