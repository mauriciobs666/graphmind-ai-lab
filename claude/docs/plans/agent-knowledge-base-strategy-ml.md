# Agent knowledge-base strategy — ML method note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`) · **Version:** 7

**Revision note (2026-09-19, fourth pass, U4 of the follow-up batch coordination,
`claude/docs/plans/agent-knowledge-base-strategy5-coordination.md`).** Revised in place — corrects
the "Stage 8 Phase 2 addendum" section's ranked root-cause below, not a new section: `devops` (U3,
same coordination) directly falsified that addendum's leading candidate (a discrete LM Studio
backend-state change between `qa-engineer`'s and `analyst`'s sessions) with process-uptime and
model-load-log evidence — no restart/reload anywhere near the 2026-09-19 14:00-15:00 window, self-
reconfirmed against the same logs. "A third data point" and "1. Likely cause, ranked" below are
corrected accordingly; items 2-4's recommendations (keep the fixed 0.43 floor, the tight-clustering
standing practice, Stage 8's acceptance) are unchanged and now say so explicitly.

**Revision note (2026-09-19, third pass, U2 of the follow-up batch coordination,
`claude/docs/plans/agent-knowledge-base-strategy5-coordination.md`).** Revised in place, not
forked (`AGENTS.md` collision rule 5 — new section, not a revision of previously-gated content) to
diagnose Stage 8 Phase 2's DEF-1 finding (prose/narrative queries missing their correct document at
a materially higher rate than code/config queries,
`claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`). New section "DEF-1
diagnosis — prose/narrative retrieval quality" appended at the end, after "Stage 8 Phase 2
addendum." Nothing above it changed in substance.

**Revision note (2026-09-19, second pass, U8).** Revised in place, not forked (`AGENTS.md`
collision rule 5 — the Stage 8 Phase 1 section below was reviewed/gated `approve with suggestions`
by `analyst`, but this addendum is a new section, not a revision of that gated content) to answer
U6b/U6c's escalation on R6/h40's cross-session score instability. New section "Stage 8 Phase 2
addendum" appended at the end, after "Risks & open questions." Nothing above it changed in
substance.

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
- **Standing practice, added 2026-09-19 (Stage 8 Phase 2 addendum, R6/h40 score-instability
  consult, below): record each row's floor-relevant score gap on every future full-set/regression
  run.** For each row, alongside the existing hit/miss and score table, record the distance
  between the floor-relevant document's score and its nearest competing candidate in the same
  top-5. **Flag any row where that gap is < 0.025 as floor-unstable-risk** — do not treat that
  row's floor-applied verdict as settled calibration evidence from a single run; reproduce the
  query in at least one additional, independently-timed session before trusting it. This is a
  mechanical, cheap check (no new queries needed beyond what a full-set run already executes) that
  closes a gap discovered the hard way: R6/h40's second sibling scored 0.4201/rank 3 in one session
  and 0.4405/rank 5 in two later ones, caught only because `analyst`'s independent review happened
  to re-run the exact query — the gap check below would have flagged this row as at-risk before
  its score was used as the sole input recalibrating the floor, regardless of which reading came
  first. Full diagnosis and reasoning: "Stage 8 Phase 2 addendum" section below.

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
floor of at least 1 pair**, plus a full 6-family stratum-(e) set (Stage 6 review Appendix A names 7 real split families; all 7 appear as design rows (R1-R6 + Q1), of which 4 (R1, R2, R3, Q1) were executed in this pilot.) and a 6-pair negative stratum, added up
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

## Stage 8 Phase 2 addendum — R6/h40 score-instability consult (2026-09-19)

**The question this addendum answers.** `qa-engineer` (Stage 8 Phase 2, U6/U6b) and `analyst`
(U6a/U6c) jointly found and confirmed that R6/h40's second sibling
(`5b1b477ff67e4e3b81c57f14899bbe48`) scores **0.4201 at rank 3** reproducibly (3/3) in
`qa-engineer`'s session and **0.4405 at rank 5** reproducibly (3/3, across two separate review
passes) in `analyst`'s — genuine, session-internally-stable, cross-session-inconsistent behavior,
not transcription error or ordinary per-call jitter. Both explicitly declined to pick a fresh
point-value floor themselves and escalated four questions here: (1) likely cause, (2) whether this
is isolated to this one row or a corpus-wide property, (3) whether a fixed-point floor is even the
right mechanism, (4) whether Stage 8's `approve` gate should stand. `skills/agent-kb-retrieval/
SKILL.md`'s floor section (0.43, interim) is the artifact any recommendation here must be concrete
enough to hand to a follow-up dispatch against, per the brief — I do not edit it myself.

### A third data point, collected live for this consult

I ran R6's exact prefixed query (`ml.md`:450, the string `skills/agent-kb-retrieval/SKILL.md`'s
template produces) against `ws:agent-team` once. Result: top 5 —
`af9ffb191c...` 0.278255462646484 (rank 1, the correct first sibling), `b09fca8c9a...`
0.421908736228943 (rank 2), `5fabd6dce3...` 0.428452491760254 (rank 3), `64eed8bab7...`
0.428490340709254 (rank 4), **`5b1b477ff6...` 0.440489292144775 (rank 5)** — matching `analyst`'s
reading exactly, not `qa-engineer`'s. Two things about this third reading matter beyond "one more
vote for 0.4405":

- **Byte-exact match at 15 decimal places** (`0.440489292144775`) against `analyst`'s Pass 2 number,
  from a session with no shared process state with `analyst`'s (independent MCP client connection,
  hours apart, no cache I have access to). Continuous floating-point non-associativity (e.g.
  different summation order from request batching/concurrency) does not reproduce to 15 decimal
  places by chance across independent sessions — that degree of match is the signature of a
  **discrete, quantized backend state** (two distinct compute paths each individually deterministic),
  not continuous per-call jitter.
- **The *other* four scores also match `analyst`'s reading closely** (`b09fca8c9a` 0.4219,
  `5fabd6dce3`/`64eed8bab7` both ≈0.4285 in both readings), while `qa-engineer`'s session's values
  for those same competing documents were never recorded in the report (only the sibling's own
  0.4201/rank-3 was). This means the discrepancy is not "one document's score moved while
  everything else held" — it's consistent with the whole query's embedding vector differing
  between the two backend states, which happens to be *invisible* everywhere except this query's
  own top-5, because R6's top-5 sits inside an unusually tight ~0.16-point band (0.278 outlier
  aside, the next four cluster within 0.019 of each other) where a small vector-level shift is
  enough to reorder ranks and cross the floor. F4/N4/R4-h14, all confirmed stable, have no
  competing candidate within that band of the reported target score.
- **Timing pattern worth naming, not provable from 3 points alone:** `qa-engineer`'s run (U6/U6b)
  is chronologically first; `analyst`'s two passes and my own run, all later, agree with each
  other and disagree with `qa-engineer`. Three independent sessions landing 2-1 with the two
  *later* ones matching is more consistent with **a single state change that happened once and
  persisted** than with **live, per-session random routing between two parallel replicas** — the
  latter would predict a roughly even mix across three independent sessions, not a clean
  before/after split. **Corrected 2026-09-19 (U4): the specific mechanism this bullet originally
  named for that "single state change" — a restart or reload of the LM Studio backend between the
  two sessions — is now falsified by `devops`'s direct evidence (see "1. Likely cause, ranked"
  below).** The chronological 2-1 split itself still stands as observed (three points cannot rule
  out coincidence, as this bullet already said), but it no longer licenses a backend-reload
  explanation; what could still produce a state change with no reload is addressed below.

### 1. Likely cause, ranked

**Corrected 2026-09-19 (U4).** I confirmed one structural fact that changes the shape of the
answer versus the report's own framing: `ws:agent-team` is served by **one single, always-on
falkor-chat server process** (`start_agent_team.sh`, default port 8200) — not a load-balanced pool
the tool caller reaches differently per session. Its embedding-backend `base_url` is resolved
**once, at process startup** (the script's documented 3-tier fallback: explicit env var →
`opencode.local.json` → the shared `opencode.json`), not per-request. That rules out "different
sessions transparently load-balance across live replicas serving different weights" as a *literal*
mechanism — there is one process, one resolved `base_url`, for every calling session unless that
process itself restarted in between.

**The original ranking's leading candidate is now disproven, not merely unconfirmed.** `devops`
(U3, same coordination) ran exactly the check this section's original "What would confirm this"
line asked for, and reported: the `falkor-chat-agent-team` server process has been running
continuously since 2026-09-18 23:12:50 with no restart anywhere near 2026-09-19 14:00-15:00
(`ps -eo lstart`); LM Studio's own server log shows exactly 5 model-unload events all day
(02:02:31, 02:03:15, 09:31:15, 09:32:10, 15:44:47), none inside the 14:00-15:00 window — the
embedding model was continuously resident across both `qa-engineer`'s run (~14:27) and `analyst`'s
first review pass (~14:42); and no LM Studio application-process restart on the Windows side in
that window either. I independently re-confirmed the process-start timestamp and the exact
unload-event list myself via direct `ps`/log reads — both match `devops`'s report exactly. This
falsifies "a discrete backend restart/reload happened once between the two sessions" as the
mechanism, not just as a specific timing guess.

Ranked, corrected:

1. **Now most likely by elimination — live floating-point non-associativity from concurrent
   request batching on an otherwise-unchanged, continuously-resident backend**, i.e. this query
   happened to land in a different batch composition each time `search_documents` reached the
   embedding server, and llama.cpp's batched matmul reduction order shifted enough to move the
   embedding measurably. This was originally weighed second because the byte-exact 15-decimal match
   across two independent sessions is a poor fit for "different batch composition every call" (that
   predicts a *spread* of close-but-not-identical values across many calls, not two clean,
   exactly-repeating clusters) — that objection still stands, and I am not overstating this as a
   confirmed mechanism, only as the leading one once the discrete-reload candidate is eliminated. A
   session-scoped variant of this (e.g. each session's own parallel tool-call pattern forming an
   internally-stable batch shape that differs session-to-session, without any backend restart) is
   consistent with both the byte-exact-within-session and disagree-across-session pattern and with
   `devops`'s findings. **What would confirm this — not investigated by `devops` or by me, no log
   surface exists for LM Studio's internal batch scheduling:** running R6 several more times
   *within* a single session while deliberately varying concurrent load (e.g. firing several other
   `search_documents` calls simultaneously vs. serially) — if the score changes with concurrency
   pattern inside one session with the backend process continuously up (now independently
   confirmed, not just assumed), that is direct, positive evidence for this mechanism. This would
   need either instrumenting concurrent load against the shared service or a bounded follow-up
   investigation — named as an open item below, not required.
2. **Second, unchanged in its own reasoning — FalkorDB's own `db.idx.vector.queryNodes` (HNSW-based
   ANN) returning different approximate results run-to-run.** Still weighed lowest for the same
   reason as before: HNSW's approximation nondeterminism affects *which* candidates get
   exact-distance-scored and returned, not the *value* of a distance computation for a document
   that **is** returned in both readings — cosine distance between two fixed, already-embedded
   vectors is deterministic once both vectors exist, so if the *stored* document vector is
   unchanged (confirmed — corpus unchanged since Stage 6) and the *query* vector were unchanged,
   the reported score could not differ. This candidate only survives if the query-side embedding is
   what's actually changing (folding it back into (1)), so I still don't treat it as a distinct
   cause, only as the tool-layer at which (1) would surface.
3. **Eliminated (2026-09-19, U4) — a discrete state change in the shared LM Studio backend (a
   reload, restart, or a different compute-kernel path selected on reload) happening once between
   the two sessions.** This was the original ranking's leading candidate. `devops`'s process-uptime
   and model-load-log evidence above directly falsifies it: no restart, no reload, no unload event
   anywhere near the window in question, on either side (falkor-chat process or LM Studio itself).
   Kept here, demoted rather than deleted, so a future reader sees what was ruled out and why,
   rather than wondering whether it was simply never considered.

**Why this correction does not reopen sections 2-4 below.** Every recommendation in this addendum
(the tight-clustering flag as standing practice, keeping a single fixed 0.43 floor with the
residual risk named as a class rather than a single row, and Stage 8's acceptance standing
unchanged) was reasoned from the **observed pattern** — two internally-consistent,
mutually-contradicting readings for one borderline document — not from *which* backend mechanism
produced that pattern. Falsifying the discrete-reload candidate changes which mechanism is likely,
not whether the pattern is real, whether it is worth a standing detection practice, or whether a
fixed floor with a named residual risk is still the right response to it — none of sections 2-4's
reasoning below cites the discrete-reload hypothesis as a premise. Confirmed unaffected by rereading
each; only the "currently live" framing in section 4 below (which did lean on the reload timeline)
is corrected there.

### 2. Is this isolated to one row, and is tight score-clustering a good risk predictor?

**Judgment: not proven to be corpus-wide, but tight top-5 clustering is a real, checkable risk
predictor and should be used as one, independent of ever fully closing the root-cause question.**
The report's own spot-checks (F4, N4, R4/h14 — all confirmed byte-stable to the 3rd-4th decimal
across both sessions) show the effect is not universal; every other measured row is fine. But the
*mechanism* argued for above — a small, usually-invisible embedding-computation difference that
only becomes rank/floor-relevant when several competing scores sit within a narrow band — predicts
exactly the pattern observed: instability shows up precisely on the one row whose top-5 (excluding
the clean rank-1 hit) clusters within ~0.019 of each other, and nowhere else spot-checked. That is
a testable, mechanistic reason to expect tight clustering to correlate with instability, not just a
post-hoc pattern-match on n=1.

**This is worth checking systematically, and cheaply — no new queries needed for the historical
run.** Recommend a **mechanical, one-pass addition to any future full-set gate** (Stage 8's own
periodic re-run, per Recommendation 4's "re-run cheaply after any bulk migration or corpus growth"
clause): when recording each row's top-5, also record the **score gap between the floor-relevant
document (the worst-scoring true positive expected to pass, or the best-scoring non-match near the
floor) and its nearest neighbor in the returned list.** Flag any row where that gap is smaller than
the confirmed observed instability magnitude (~0.02, rounding up for safety to **0.025** as a
detection threshold) as **floor-unstable-risk** — not a defect, a flag that this row's floor-applied
verdict should not be trusted from a single run and should be spot-reproduced across at least two
independent sessions before being used as calibration evidence. This is exactly what would have
caught R6 before it became the sole input recalibrating the whole floor, and it generalizes: it
would catch the *next* tightly-clustered row too, whatever backend state happens to be live when it's
first measured, without needing the root cause resolved first.

### 3. Is a fixed-point score floor the right mechanism? — recommendation

**Recommendation: keep a single fixed-point floor as the standing mechanism (option (a), not (b) or
(c)) — but change what "calibrated" means going forward: a floor value is not treated as final
until every row within ~0.025 of it has been reproduced across at least two independent sessions.
Document R6/h40's second sibling's admission as an explicit, named, accepted residual risk in
`SKILL.md`, not silently absorbed into a new point value.** Reasoning, against each alternative
named in the brief:

- **(b) — widen the floor's required margin as a matter of policy: rejected, and demonstrably not
  achievable here, not just undesirable.** The report's own arithmetic already proves this: the
  worst *other* true positive is 0.4140 (C4) and the closest stable negative is 0.446 (N4) — a
  0.032 natural window. R6/h40's second sibling's *worse* observed reading (0.4405) leaves only
  0.0055 to the negative boundary. A margin-widening policy applied uniformly cannot admit 0.4405
  and preserve any working margin at the same time — the report already showed this is a hard
  arithmetic wall, not a policy choice avoided out of caution. Widening margin *only* around this
  one row, rather than globally, degenerates into option (a) below (a named exception), not a
  distinct mechanism.
- **(c) — a re-query-and-average/re-query-and-take-best convention for borderline scores: rejected
  as ineffective for the failure mode actually observed, not merely unnecessary.** This only helps
  if the instability is *per-call* random within a session — average/best-of-N over repeated calls
  samples across that randomness. What's actually been observed is **session-scoped, not per-call**:
  `qa-engineer` got 0.4201 three times in a row in one session; `analyst` and I each got 0.4405
  three-for-three across our own calls. A calling agent re-querying 3× *within its own session*
  would get the same session-pinned value three times over — false confidence, not real averaging,
  because whatever determines the state (per the corrected diagnosis above, most likely a
  session-scoped concurrent-load/batching pattern rather than a backend reload) does not change
  between calls inside one session. Recommending this
  convention would ship a mechanism that looks like it addresses the problem while measurably not
  doing so against the one case with real evidence.
- **(a) — keep a fixed floor, document the residual risk as an accepted, bounded limitation:
  recommended, with one addition beyond the current interim state.** 0.43 is correct for every
  other measured true positive (worst 0.4140) and every measured negative (closest 0.446) — the
  method (max surviving true positive vs. min negative false match) is sound, only this one row's
  input is contested, and no single point value can be "more correct" than another given two
  equally-reproducible, mutually contradicting readings. The addition: don't just accept the risk
  as static — **adopt the tight-clustering flag from item 2 above as standing practice** for every
  future full-set/regression run, so a future row landing in this same situation is caught by
  process rather than by a lucky independent-review re-run (which is how this one was actually
  caught — not by design).
- **(d) — something else, specifically "characterize and pin the backend to remove the
  nondeterminism at its source":** **narrowed 2026-09-19 (U4).** `devops`'s check (item 1 above)
  already ruled out the premise this originally targeted — the backend process and the embedding
  model were both continuously up and unreloaded across the discrepancy window, so there is no
  discrete reload event left to "pin against." If concurrent-batching (the now-leading candidate)
  is confirmed by the bounded follow-up named under item 1, the equivalent fix would be about
  controlling concurrent load during a regression-gate run (e.g. running it without competing
  concurrent traffic against the shared service), not backend version-pinning — still a
  `devops`/`graph-dba`-level action, not a methodology change to the floor mechanism, and still not
  something I can execute or verify from here. See "actionable now vs. follow-up" below.

**Concrete `SKILL.md` change this implies (not applied by me — for a follow-up dispatch):** in the
"Score floor" section, keep **0.43** as the operative value (no change to the number), but revise
the framing from "one known unresolved risk" (implying a single contested row awaiting resolution)
to a standing **class** of risk: *"a hit whose score sits within ~0.025 of a competing candidate's
score, near the floor, should not be trusted from a single `search_documents` call — this convention
has one confirmed instance (R6/h40's second sibling, 0.4201–0.4405 depending on backend state) and
may have others not yet identified; the mitigation is corpus-level (Stage 8's periodic regression
gate re-runs each row's floor-relevant gap and flags any below 0.025 for multi-session
reproduction), not a per-call client-side retry."* This is a **wording/framing change**, not a
number change — 0.43 stays, R6/h40 stays a named exception, but the file should read as describing
a checkable *pattern* rather than a single closed-out anomaly, so `cobb`/whoever next runs the
regression gate knows to apply the item-2 check rather than treating R6/h40 as the only row that
will ever need it.

### 4. Does this change Stage 8's acceptance?

**No — Stage 8 stands as accepted (`analyst`'s U6c `approve`), and this diagnosis does not surface
anything that would reopen it.** Two independent reasons:

- **The headline gate metrics are unaffected either way.** Pooled recall@5 (0.875, Wilson
  [0.719, 0.950]) and set-recall (13/14 = 0.929) are both computed on **raw retrieval** (was the
  document present anywhere in the top 5), and R6/h40's second sibling **was found in the top 5
  under both observed readings** (rank 3 and rank 5) — only its *floor-admission* status differs,
  and the report already correctly reports floor-applied figures both ways (0.857 vs. 0.929 for
  the affected sub-metric) rather than picking one. Nothing here changes what was actually
  measured.
- **Escalating instead of resolving was the methodologically correct call, and remains so.** Given
  two equally-reproducible, mutually-contradicting readings with no principled way to prefer one
  as "the" true value from either `qa-engineer`'s or `analyst`'s vantage point alone, picking either
  0.4201 or 0.4405 as a fresh floor-deriving input would have been a false-precision overclaim —
  exactly the failure mode `analyst`'s Pass 2 named and declined to commit. **Corrected 2026-09-19
  (U4): my own third data point's original framing ("favoring 0.4405 as the currently-live state,
  per the timing-pattern reasoning") leaned on the now-falsified discrete-reload hypothesis and is
  withdrawn — there is no "currently-live backend state" to favor once no reload occurred.** What
  the third data point still establishes, unaffected: a third, independent, differently-timed
  session reproduced `analyst`'s 0.4405 exactly rather than `qa-engineer`'s 0.4201, which remains
  evidence the 2-1 split is not coincidence, just not evidence of *which* mechanism drives it. This
  does not change the underlying judgment that neither reading was "provably more correct" from the
  position `qa-engineer`/`analyst` were each in — it only became more informative once a third,
  differently-timed session was available, which is itself an argument for the process fix in item
  3 (multi-session reproduction as standing practice) rather than a retroactive complaint about the
  original escalation.

### Actionable now vs. follow-up

**Actionable now (no further investigation needed):**
- `SKILL.md`'s floor-section framing change described under item 3 (0.43 unchanged; reframe R6/h40
  from a single resolved anomaly to a named instance of a checkable class) — cheap, mechanical,
  ready for a follow-up dispatch.
- Add the tight-clustering flag (item 2: record floor-relevant score gaps, flag <0.025, require
  multi-session reproduction before trusting) to whatever process document governs Stage 8's
  periodic re-run (the test plan, or a note in this file's Recommendation 4) — also cheap and ready
  now.

**Completed (2026-09-19, U3/U4):**
- **`devops` follow-up: done.** Checked `ws:agent-team`'s server process uptime/restart history
  and LM Studio's own model-load/reload log around `qa-engineer`'s U6/U6b run timestamp versus
  `analyst`'s U6a first-pass timestamp — **falsified** the "one discrete backend-state change
  happened in between" hypothesis (continuous uptime, no reload in the 14:00-15:00 window on
  either side), independently re-confirmed against the same logs. No operational fix follows from
  this branch (there was no reload to keep stable or re-run after).

**Open, bounded, not required — my own call to name, not to execute:**
- The trigger this file's own original "Lower priority, only if the above is inconclusive" bullet
  set is now met: the restart/reload check came back negative, so a `devops`/`graph-dba` check of
  whether **concurrent request load** measurably changes R6's score within one session (the
  corrected ranking's item 1, "What would confirm this") is the next thing that would actually
  distinguish a mechanism, if this is worth pursuing further. It's a more expensive, noisier
  experiment to design well (no existing log surface for LM Studio's internal batch scheduling),
  and none of sections 2-4's recommendations depend on resolving it — so this stays a named,
  bounded open item, not a blocking follow-up.
- Not recommended as a use of further investigation time: chasing a "third" score value or treating
  n=3 as enough to fully characterize the distribution. Three points establish reproducible
  bimodality convincingly enough to act on; a fourth or fifth reproduction would sharpen confidence
  marginally, not change the recommendation.

## DEF-1 diagnosis — prose/narrative retrieval quality (2026-09-19)

**The question this section answers.** Stage 8 Phase 2's full 45-pair gate confirmed DEF-1 as
real, not a one-off: `b-prose`-tagged queries miss their correct document 21% of the time (15/19,
Wilson CI [0.567, 0.915]) versus 0% for `b-code`-tagged queries (11/11, CI [0.741, 1.000]) — all 4
misses across the full set (C1, P1, X1, G2) are `b-prose`. The report named two candidate next
steps, neither executed there: (a) a prefix-wording or per-claim keyword tweak, (b) a title/
family-slug differentiation fix for X1's cross-KB confusion. This diagnoses the actual mechanism
and judges both candidates against it, plus what else is worth trying.

### Method

Read all 4 misses' expected documents' stored text/title via `get_document`, then re-ran all 4
exact prefixed queries live against `ws:agent-team` at `limit=20` (not the standard `limit=5`) to
see how far each expected document actually sits from its query — a `limit=5` miss alone cannot
distinguish "just outside the cutoff" from "genuinely far away," and that distinction changes what
a fix should look like.

### Finding 1 — this is not a top-K or floor-tuning problem; the gap is largely real embedding
distance

| Query | Expected doc found at `limit=20`? | Rank | Score |
|---|---|---|---|
| C1 | **No** — absent even at rank 20 | — | — |
| P1 | Yes | 9 | 0.4487 |
| X1 | Yes | ~11 | 0.4632 |
| G2 | **No** — absent even at rank 20 | — | — |

Two of the four (C1, G2) don't surface even 4x past the operative top-K; the other two (P1, X1) do
surface, but both at scores already above the 0.43 floor — so even a top-K widen to 10-15 would not
have made either of them admissible without a separate floor change, and a floor change large
enough to admit 0.4487/0.4632 was already shown unworkable in the Phase 2 addendum above (it would
leave no margin to the closest negative, 0.446). **Widening top-K is not a viable mitigation for
any of the 4** — ruled out by measurement, not by assumption.

### Finding 2 — document length/dilution explains half the misses, not the common cause

G2's expected document (`f29bddad2cf14479ba6bafd9bd0c4212`, `guard-testing-techniques.md`) is a
~370-word entry with a worked router-reach-guard case study (`TARGET`/`VALUE` axis terminology, AST
alias analysis) — despite its title being a near-paraphrase of the query ("hand-written… resolver…
two axes" vs. the query's "hand-wrote a resolver… two different independent things"), the dense,
jargon-specific worked example pulls its embedding away from the query's plainer framing (topic
dilution, exactly the mechanism Recommendation 2 named for `review-techniques.md`, here showing up
in a different KB). X1's expected document (`75c0e47a01244de5b4373a193887293c`,
`estimator-test-fixtures.md`) is longer still (~750 words, a numeric worked table) — same
mechanism.

But **C1 (~110 words) and P1 (~95 words) are shorter than the `b-code` documents that hit cleanly
at rank 1** (e.g. R8's clamp/NaN entry, ~230 words; F2's `RESULTSET_SIZE` entry, ~230 words) — so
length/dilution is a real, partial mechanism (2 of 4), not the root cause common to all four.

### Finding 3 — the common cause is corpus-neighborhood density, not query wording

Two of the four misses (P1, G2) have **striking exact-phrase overlap** between the query and their
own expected document's title — P1's query says "completeness claim… transcribed… check… fail"
against a title that reads "A completeness claim must be derived, not transcribed — and its check
must be able to fail"; G2's query says "hand-wrote a resolver… two… independent things" against "A
hand-written… resolver has two axes." **High lexical overlap did not save either from missing** —
both still lost to several *other* documents scoring lower (closer). This rules out "the query is
badly worded relative to its target" as the explanation and points at the embedding model's
fine-grained discrimination capacity within a crowded region, not the query-document pair's own
wording.

The structural asymmetry: the `b-code` KBs (`falkordb-quirks.md`, `frontend-quirks.md`,
`falkordb-reference.md`) each document a narrow, largely non-overlapping operational fact, with
unique low-frequency tokens (`RESULTSET_SIZE`, `GRAPH.RO_QUERY`, exact measured numbers) giving
each claim a sharp, near-lookup embedding signature and few-to-no topically-adjacent neighbors
anywhere else in the 332-claim corpus. The `b-prose` KBs — `review-techniques.md` (81 claims),
`coordination-techniques.md` (39), plus `plan-authoring-techniques.md`,
`guard-testing-techniques.md`, `test-design-techniques.md`, `estimator-test-fixtures.md`,
`qa-testing-techniques.md`, `statistical-method-techniques.md` — collectively form one broad,
stylistically homogeneous semantic neighborhood (every entry is cobb's own "principle stated once +
worked instance + Origin:" voice, on the shared theme of verifying/reviewing/testing AI-agent work)
holding on the order of 150-200 of the corpus's 332 claims. C1's `limit=20` result list is the
clearest direct evidence of this: **~15 distinct documents, all from this same neighborhood, all
about "verifying a delegate's/reviewer's claim via git,"** none of them the expected one, filled
every slot up to rank 20. A 0.6B-parameter embedding model discriminating inside a dense,
stylistically homogeneous cluster of ~150-200 near-synonymous short claims is a materially harder
task than discriminating a handful of lexically-unique operational facts — this is consistent with
a known general property of smaller embedding models (fine-grained semantic-textual-similarity/
paraphrase discrimination scales with model capacity in the broader embedding literature), but **I
have not benchmarked this specific effect on this corpus against a larger model** — naming it as
the expected mechanism, not a verified number. Do not cite this as a settled benchmark claim without
running Recommendation 1 below.

### Judging the two candidate fixes named in the AC-2 report

**(a) Prefix-wording or per-claim keyword tweak — rejected as a fix for this gap.** The prefix's
job is to shift the query vector toward a "retrieval task" framing; it has no mechanism to add
local discriminative power inside an already-dense semantic cluster. Finding 3 is the direct
evidence against it: P1 and G2 already have strong lexical/semantic alignment with their own
correct target and still lose — the failure is at the embedding model's fine-discrimination layer,
a layer a prefix string cannot reach. A keyword-augmentation variant (appending distinguishing
terms to a claim's stored text) is a more invasive version of the same idea and carries the same
objection, plus a new risk: it would need to be applied consistently across ~150-200 claims to
avoid arbitrarily helping the ones cobb happens to touch first.

**(b) Title/family-slug differentiation for the `estimator-test-fixtures.md`/
`test-design-techniques.md` X1 case — narrower than the report frames it, but worth a scoped
version.** Recommendation 2's `familyId`→title-prefix convention exists for documents **split from
one shared heading during migration** — true siblings. X1's and T1's documents were never split
from a shared heading; they are two independently-authored, topically-overlapping claims in
different files (X1: a percentile-bootstrap/clamp mechanism with a numeric worked table; T1: the
general "vary the position/dimension the rule anchors to" principle). There is no shared family to
prefix, so a family-slug convention does not literally apply here. What **would** help this one
already-discovered pair: a `cobb` curation pass that either merges the two into one canonical entry
cross-referenced from both files, or sharpens each one's opening sentence to foreground its own
distinguishing evidence rather than the shared abstract framing both currently lead with. Cheap,
low-risk, worth doing — but scope it honestly: **it only fixes this one discovered collision.** C1
and G2 have no single identifiable "near-duplicate twin" to differentiate against; each is
outcompeted by many different, only-loosely-related documents from the same crowded neighborhood,
not confused with one specific sibling claim. A title-differentiation pass cannot touch that
pattern.

### Recommendation — what's actually worth trying, and what's out of scope for a documentation fix

**Honest bottom line: neither candidate in the report closes this gap, and I am not going to
manufacture a cheap fix that the evidence above doesn't support.** Ranked:

1. **Recommended first step (moderate cost, bounded, low-risk): a targeted held-out trial of the
   documented upgrade path, `qwen3-embedding:4b`** (same family, same 1024-dim MRL, re-embed-only,
   no schema change — the escape hatch Recommendation 1 already named). Re-embed a small,
   representative slice (the 150-200-claim `b-prose` neighborhood plus a `b-code` control sample)
   and re-run these 4 queries plus 5-10 already-hitting `b-prose`/`b-code` controls under the same
   prefix/floor convention, re-derived for the new model. **Concrete acceptance criterion:** if at
   least 3 of these 4 queries' expected documents land in top-5, the model-capacity hypothesis is
   supported and a fuller corpus re-embed is justified; if fewer (especially if C1/G2 still miss
   even at `limit=20` under the 4B model), the hypothesis is falsified and effort should move to
   item 2. This is the cheapest way to get real evidence on the model-capacity question before
   committing to a full corpus re-embed or a structural retrieval change.
2. **Named, but out of scope for me to design or execute — route to `graph-dba`/`architect`:
   hybrid lexical (BM25/full-text) + semantic score fusion.** This is the single most promising
   structural lever specifically for P1 and G2, where the query and its correct document share
   strong exact-phrase overlap a lexical signal would reward directly, independent of the embedding
   model's fine-discrimination ceiling. Not something a documentation-convention fix can deliver;
   flagging the direction and these two queries as the concrete motivating evidence for a future
   design consult, not proposing a build here.
3. **Apply now, cheap, scoped correctly:** the `cobb` curation edit for the X1/T1 pair described
   under candidate (b) above — merge or differentiate the two claims' opening text. Land it as a
   normal content edit, not a schema/naming-convention change.
4. **Do not apply:** a prefix-wording tweak or blanket keyword augmentation (candidate (a)) — named
   and rejected above, not a partial recommendation.
5. **If neither 1 nor 2 is pursued, name the residual honestly rather than closing it silently:**
   recall on this corpus's broad review/verification/test-methodology neighborhood (~150-200 of 332
   claims) will likely stay materially below the corpus's other, sparser neighborhoods for as long
   as retrieval is single-model dense-vector-only. This is a structural property of a small
   embedding model over a densely homogeneous corpus region — not a defect in any one KB file's
   formatting, and not something a prefix or title-convention change can be expected to fix.

**Traceability.** Diagnosed against `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`'s
DEF-1/"C1-pattern finding" sections; live re-verification (`search_documents`/`get_document` at
`limit=20`) run 2026-09-19 against `ws:agent-team`, same corpus state as that report (no migration
activity between). Coordination: `claude/docs/plans/agent-knowledge-base-strategy5-coordination.md`,
U2.

## DEF-1 Recommendation 1 execution — `qwen3-embedding-4b` held-out trial (2026-09-19)

**The question this section answers.** The diagnosis above named the corpus-neighborhood-density
hypothesis (a 0.6B-parameter encoder's fine-discrimination capacity is the bottleneck inside the
~150-200-claim `b-prose` neighborhood) as the leading, but unverified, explanation for DEF-1, and
Recommendation 1 specified a bounded held-out trial as the cheapest way to get real evidence before
committing to a fuller corpus re-embed. This section executes that trial, exactly as scoped in
`claude/docs/plans/agent-knowledge-base-strategy6-coordination.md` (U2), and judges the result
against my own prior acceptance criterion.

### Environment note

`devops` resolved the recommendation's literal string (`qwen3-embedding:4b`, an Ollama-style tag
that does not exist) to the real artifact — **`Qwen/Qwen3-Embedding-4B-GGUF`, Q4_K_M quantization**
— loaded in LM Studio under API id `text-embedding-qwen3-embedding-4b`. I independently
re-confirmed it live before running anything (a fresh `/v1/embeddings` probe call returned a
correct 2560-dimension, non-degenerate vector, ~0.4s). Per the flagged VRAM ceiling (~357-360 MiB
free with the 4B model resident — no headroom for a second concurrent model load), this trial does
**not** load the 0.6B model alongside it; the 0.6B comparison numbers used throughout are the
already-recorded Stage 8 Phase 1/Phase 2 figures, not a fresh live run.

### Method

**No write to `ws:agent-team`.** Every read against it in this trial was `GRAPH.RO_QUERY` (via the
`cypher` MCP tool, and — for bulk full-text fetch, see below — directly against the same
`falkordb-dev` container on its published host port, same read-only command). No `ingest_document`
call was made against `ws:agent-team` at any point.

**Corpus slice reconstruction.** The diagnosis's named "~150-200-claim `b-prose` neighborhood" is
the union of 8 KB files: `review-techniques.md`, `coordination-techniques.md`,
`plan-authoring-techniques.md`, `guard-testing-techniques.md`, `test-design-techniques.md`,
`estimator-test-fixtures.md`, `qa-testing-techniques.md`, `statistical-method-techniques.md`. I
reconstructed the exact per-file `documentId` sets from `claude/cobb/scripts/kb-claim-manifest.json`
(the same manifest Stage 6 migration itself produced and verified byte-exact against
`ws:agent-team`) rather than re-deriving them by search, and dropped the one `documentId` recorded
there as permanently `status:failed` (1 in `guard-testing-techniques.md` — not search-retrievable
in production either, so excluding it is the correct like-for-like comparison). This yielded
**188 `b-prose` documents** — inside the diagnosis's own named 150-200 range. For the **`b-code`
control sample**, I pulled every `b-code` KB file present in the same manifest structure —
`falkordb-quirks.md` (86, the source of the `F2` control and of most `b-code`-tagged golden-set
queries), `falkordb-reference.md` (19), `frontend-quirks.md` (9) — rather than hand-picking a
smaller subsample, once the fetch mechanism was built for the `b-prose` side anyway: **114 `b-code`
documents. Total corpus: 302 documents** (188 + 114), all confirmed `status:'ready'` on fetch (no
`failed` document leaked into the slice). This is wider than "just `F2`'s source file" as first
scoped, but it is still a bounded, non-production held-out sample, and it does not change any
result below — none of the 4 target/control queries' outcomes depend on which extra `b-code`
documents sit in the corpus, since none of the extras land within reach of any target query's top
ranks.

**Text fetch.** `get_document` returns full untruncated text but is one call per document;
`mcp__cypher__query`'s own display layer truncates long `text` values (confirmed: a 680-character
field was cut to ~340 chars + a "(+N chars)" marker) — unusable for full-fidelity re-embedding.
Fetched all 302 documents' full `title`/`status`/`text` via direct `GRAPH.RO_QUERY` calls against
the `falkordb-dev` container (`localhost:6379`, the same instance `ws:agent-team` lives in, reached
over its published host port per `cypher-mcp/README.md`'s own documented access pattern) — strictly
read-only, one `MATCH (d:Document {documentId:$id}) RETURN ...` per document, no write. All 302
fetched cleanly, `status:'ready'`, no parse errors; spot-checked word counts against the diagnosis's
own earlier estimates (e.g. X1's document: diagnosis said "~750 words," this fetch measured 555 —
the diagnosis's figure was itself an eyeballed approximation, not a discrepancy).

**Query set.** The 4 originally-missed queries (C1, P1, X1, G2) plus **8 controls** (6 `b-prose`:
C2, C4, Q2, P2, T1, G1; 2 `b-code`: R8, F2) drawn from Stage 8 Phase 1/Phase 2's already-hitting
rows, plus **1 negative** (N4, the closest false match under the 0.6B model, 0.446) as a floor
sanity check not required by the brief but cheap to add. All 13 query situation strings and expected
`documentId`s were copied verbatim from `claude/docs/plans/agent-knowledge-base-strategy-ml.md`'s
own Stage 8 Phase 1 table and `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`
— not reworded.

**Prefix/floor convention, re-derived for the new model.** The asymmetric query/document
instruction convention itself (documents unprefixed, queries wrapped in `"Instruct: Given a coding
agent's description of its current situation, retrieve the distilled technique or rule that applies
to it.\nQuery: {situation}"`) is a property of the Qwen3-Embedding **architecture/training recipe**,
documented as shared across the whole model-size family (0.6B/4B/8B) — I reused it unchanged rather
than inventing a new template, since nothing about model size implies a different instruction
format. The **floor**, by contrast, is a property of this specific model's score distribution and
had to be measured fresh — see "Floor observation" below; it is not carried over from the 0.6B
model's 0.43.

**Ranking.** Embedded all 302 corpus documents unprefixed and all 13 queries prefixed, via batched
calls to `http://localhost:1234/v1/embeddings` (`text-embedding-qwen3-embedding-4b`, 2560-dim,
batches of 20; 302 corpus + 13 query embeddings completed in 38s total). Computed cosine distance
(`1 - cosine_similarity`, L2-normalized vectors) between every query and every corpus document in
plain Python (no FalkorDB vector index involved — brute-force exact ranking over the 302-document
slice, which is *more* accurate than an ANN index would be, so this is if anything a favorable,
not unfavorable, comparison for the 4B model). Reproduced the three most consequential numbers
(C1, X1, G2's target-document scores) via a second, independent embedding call each — all three
reproduced to 4 decimal places exactly, so this is not a repeat of the 0.6B model's documented
cross-session score-instability finding (R6/h40) — at least not within-session; no cross-session
reproduction was attempted, an explicit limitation, see below.

### Results

**The 4 target queries:**

| Query | Expected doc | Rank in top-302 | Score | In top-5? | Comparison to 0.6B (from DEF-1 diagnosis) |
|---|---|---|---|---|---|
| C1 | `9870c48f98…` | 14 | 0.4437 | **No** | 0.6B: absent even at rank 20. 4B: now surfaces, but at rank 14, still a top-5 miss. |
| P1 | `08a422aeca…` | **1** | 0.3940 | **Yes** | 0.6B: rank 9, score 0.4487 (above its own floor). 4B: rank 1 — clear improvement. |
| X1 | `75c0e47a01…` | **4** | 0.3870 | **Yes** | 0.6B: rank ~11, score 0.4632. 4B: rank 4 — clear improvement, but a thin margin (rank-5 competitor scored 0.3899, only 0.0029 apart — reproduced twice, stable). |
| G2 | `f29bddad2c…` | 53 | 0.5678 | **No** | 0.6B: absent even at rank 20. 4B: still absent well past rank 20 — no material improvement, if anything relatively worse. |

**2 of 4 target documents land in top-5 (P1, X1); 2 do not (C1, G2).**

**The 8 controls — all 8 hit at rank 1** (scores 0.1371–0.3940, well inside a sane floor):

| Query | Tag | Rank | Score |
|---|---|---|---|
| C2 | b-prose | 1 | 0.3188 |
| C4 | b-prose | 1 | 0.3423 |
| Q2 | b-prose | 1 | 0.1371 |
| P2 | b-prose | 1 | 0.2510 |
| T1 | b-prose | 1 | 0.3489 |
| G1 | b-prose | 1 | 0.2553 |
| R8 | b-code | 1 | 0.1715 |
| F2 | b-code | 1 | 0.2200 |

This confirms the embedding/ranking mechanism itself is sound under the 4B model (no pipeline bug
depressing scores generally) — the 2 remaining target misses are a genuine retrieval-quality
result, not an artifact of a broken trial.

**Negative control (N4, "React Server Components data-fetching boundary"):** closest match in the
302-document corpus scored 0.5233 (a `frontend-quirks.md` "React Router" entry) — correctly
rejected under any plausible floor, and notably farther from the boundary than the 0.6B model's
0.446 on the same query. **Floor observation, heavily caveated:** if a floor were fit against only
this trial's data (worst surviving top-5-hit score 0.3940 [P1] vs. this one negative's 0.5233), the
gap (0.129) is far wider than the 0.6B model's calibrated 0.037 — a promising secondary signal for
general (non-crowded-neighborhood) discrimination, but **n=1 negative query is not a floor
derivation**, only a single data point; do not treat 0.129 as a re-derived floor or cite it as if it
were Recommendation 3's actual full-negative-stratum exercise. A real re-derivation needs the same
~6-negative-query discipline Stage 8 Phase 1 used, not something this bounded trial was scoped to
redo.

### Verdict against the acceptance criterion

**My own prior criterion, verbatim: "if at least 3 of these 4 queries' expected documents land in
top-5, the model-capacity hypothesis is supported and a fuller corpus re-embed is justified; if
fewer (especially if C1/G2 still miss even at `limit=20` under the 4B model), the hypothesis is
falsified and effort should move to item 2."**

**Result: 2 of 4 (P1, X1) land in top-5 — below the 3-of-4 bar. Hypothesis falsified**, and the
named "especially" trigger is substantially met: G2 misses cleanly even at a limit far wider than
20 (rank 53 of 302); C1 no longer misses at `limit=20` (it now appears at rank 14, an improvement
over "absent even at rank 20" under 0.6B) but still misses the operative `limit=5` cutoff the
production pipeline actually uses. **Per the criterion's own instruction, effort should move to
item 2** (hybrid lexical/semantic score fusion, routed to `graph-dba`/`architect`) rather than a
fuller corpus re-embed under this model on model-capacity grounds alone.

The result is not a uniform null, though, and that nuance matters for anyone deciding what to do
next: **model capacity is a real, partial lever** (P1 and X1 both moved from top-5 misses to clean
hits, and C1 moved from "not even in the top 20" to "in the top 20, still not top 5") — but it is
**not sufficient on its own** to close the gap for the two hardest cases (C1, and especially G2,
which shows no material improvement at all). This is consistent with Finding 3 in the diagnosis
above: P1 and G2 both had strong lexical/exact-phrase overlap with their own correct document and
still lost — a symptom a lexical-fusion signal (item 2) would attack directly, and a symptom pure
embedding-capacity scaling only partially fixes (it fixed P1, it did not fix G2, its sibling case in
the same diagnostic pattern).

### What this trial does and does not tell you

**Does tell you:** on a corpus of this shape and this exact document set, a larger embedding model
in the same family measurably improves ranking for at least some `b-prose` queries inside the dense
neighborhood, but does not close the gap for the specific queries whose expected document has
*many* other topically-adjacent competitors rather than one identifiable near-duplicate (G2's
pattern, per the diagnosis's own Finding 3) — that residual looks structural to dense-vector-only
retrieval, not a model-size problem alone, reinforcing rather than undercutting item 2's case.

**Does not tell you, and would need separate validation before committing to a fuller corpus
re-embed under this model:**
- **Held-out ≠ full-corpus.** This ran on 302 of the corpus's 332 claims (excluding
  `ops-quirks.md`, `lm-studio-model-notes.md`, and 1 permanently-failed document) — a genuinely
  representative slice of the two neighborhoods in question, but not the whole corpus, and not a
  test of whether re-embedding interacts with the excluded files or with cross-neighborhood
  queries the diagnosis didn't stratify into this set.
- **No cross-session score-stability check.** All reproduction here was **within this one session**
  (three targets' scores reproduced exactly, twice each) — this corpus's own precedent (R6/h40)
  found a real, unexplained cross-session score drift on the 0.6B model for one borderline document.
  Whether the 4B model is more or less prone to that same instability is untested; a fuller
  commitment should re-run at least the borderline cases (X1's thin rank-4/rank-5 margin is exactly
  the kind of case that instability would flip) from a fresh session before trusting them as final.
- **Floor is not re-derived, only illustrated.** As stated above, n=1 negative is not a
  calibration — a real floor for this model needs Stage 8 Phase 1's full ~6-negative-query
  discipline repeated fresh, not inferred from this trial.
- **Cost/latency of the 4B model in production was not measured here.** Embedding 302 documents +
  13 queries took ~38s total in this trial (batches of 20, local GPU) — a real number, but from a
  cold, idle, uncontended model; it says nothing about the 4B model's behavior under this lab's
  actual query load or about the VRAM contention already flagged (only ~357-360 MiB free with this
  model resident, no headroom for the 0.6B model to also be loaded) — a real operational constraint
  a "fuller corpus re-embed" decision would have to resolve (does production stop loading the 0.6B
  model at all, or contend for the same thin margin?), not addressed by this note.
- **This is one embedding pass, not a regression suite.** A genuine "fuller corpus re-embed"
  decision should re-run Stage 8's actual 45-pair golden set (all `b-prose`/`b-code`/stratum-(e)/
  negative rows, not just this trial's 12 queries) under the 4B model before shipping — this trial
  is deliberately narrower, exactly as scoped.

**Bottom line for the decision this serves:** do not commit to a fuller corpus re-embed under
`qwen3-embedding-4b` on the strength of this trial. The acceptance criterion I set myself is not
met. The next move, per my own prior ranking, is item 2 (hybrid lexical+semantic fusion) — this
trial's own result (P1 and G2 are the same diagnostic pattern, lexical overlap without a
discriminative embedding win, and model-capacity scaling fixed only one of the two) is itself
additional evidence for that direction, not just a fallback because item 1 failed.

**Traceability.** Corpus/documentId reconstruction: `claude/cobb/scripts/kb-claim-manifest.json`
cross-checked against direct `GRAPH.RO_QUERY` reads (`status:'ready'` confirmed for all 302).
Embeddings: `http://localhost:1234/v1/embeddings`, model id `text-embedding-qwen3-embedding-4b`,
run 2026-09-19. Query/expected-answer source: this file's own Stage 8 Phase 1 table (§"Stage 8
Phase 1") and `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`. Coordination:
`claude/docs/plans/agent-knowledge-base-strategy6-coordination.md`, U2.

## Item 2 — hybrid lexical+semantic score fusion method (2026-09-20)

**The question this section answers.** `claude/docs/plans/agent-knowledge-base-strategy7-coordination.md`
(U2, `teco`) executes item 2 of my own DEF-1 recommendation above: hybrid lexical+semantic score
fusion for `search_documents`/`search_chunks` over `ws:agent-team`, motivated specifically by P1
and G2 — the two DEF-1 misses whose query and expected document share strong exact-phrase/lexical
overlap the embedding didn't reward. `graph-dba` (U1, `claude/docs/plans/
agent-knowledge-base-strategy-graph.md`) confirmed feasibility (a full-text index on `Chunk.text`
costs ≈2.9 MiB, favorable scaling, no migration hazard) and measured the actual score
characteristics of `db.idx.fulltext.queryNodes` this design has to reconcile with: unbounded,
TF-IDF-family, magnitude tracking term rarity not fixed relevance (0.026–4.00 across real probes),
higher-is-better/`DESC`, opposite direction from the vector signal's cosine-distance/`ASC`; and
confirmed exact-phrase queries enforce strict contiguous adjacency (1 of 3 tested phrases matched).
This section decides the fusion formula, the lexical query shape, the acceptance bar, and U5's test
design — method only, nothing here implemented or trial-run (per the brief; U3/U4 build it, U5 runs
it).

### 1. Fusion formula — Reciprocal Rank Fusion, not magnitude normalization

**Decision: Reciprocal Rank Fusion (RRF), not per-query min-max normalization onto `[0,1]`.**

RRF's fused score for a chunk `d` is `Σ_{s ∈ {vector, lexical}} w_s / (k + rank_s(d))`, where
`rank_s(d)` is `d`'s 1-indexed position within signal `s`'s own returned, ranked list, and a chunk
absent from a signal's list contributes 0 for that signal — no special-casing needed, this is
plain RRF, not a variant I'm inventing. **`k = 60`** (the standard default from the RRF literature,
Cormack/Clarke/Buettcher 2009) and **`w_vector = w_lexical = 1`** (equal weight — no evidence yet
justifies asymmetric weighting; see Risks). Each signal is over-fetched to **`K = 20`** before
fusion (vector: `db.idx.vector.queryNodes` top 20 by cosine distance ascending; lexical:
`db.idx.fulltext.queryNodes` top 20 by RediSearch score descending) — 20 matches the depth this
file's own DEF-1 diagnosis and the Recommendation-1-execution trial already probed to (`limit=20`),
so results stay comparable to that existing evidence, and it gives RRF real room: a candidate at
vector rank 9 under the 0.6B model (P1), or absent from the 0.6B model's vector top-20 entirely
(C1, G2 both — the DEF-1 diagnosis's own Finding 1 table, not the separate 4B-trial numbers, which
found C1 at rank 14 only under the unadopted 4B model), still has a chance to be pulled in if it
ranks well on the lexical side.

**Why RRF over normalize-then-weighted-sum, against U1's actual numbers, not in the abstract.** A
min-max approach has to solve U1's "magnitude tracks term rarity, not relevance" problem itself —
concretely, min-max-ing lexical scores *within one query's own returned set* doesn't fix this: a
query whose only real lexical signal is a common word (U1's `"the"`, capping at 1.00 despite
95%-of-corpus incidence) would still get rescaled to a full `1.0` for its top hit, indistinguishable
from a query whose top hit scored `4.00` on a genuinely rare, high-value term (`"falkordb"`) — the
distortion U1 measured doesn't disappear under per-query rescaling, it just becomes invisible.
Min-max is also degenerate whenever a list has 0 or 1 members (exactly what a lexical query returns
routinely at this corpus's scale — see §3's OR-term design, which usually returns *something*, but
a pathological single-hit case still needs a special-cased scale). **RRF sidesteps all of this by
never reading raw magnitude at all** — only ordinal position, which is comparable across queries by
construction regardless of what drove a given score's absolute value. This is exactly U1's own
framing of RRF's appeal (§3, "a rank-based method that never looks at raw magnitude"), and I am
adopting it as the concrete choice rather than leaving it a stated option.

**What RRF gives up, named honestly, not hidden.** RRF discards *margin* information a
magnitude-aware method would keep — a vector hit at cosine distance 0.15 (a strong, confident
match) and one at 0.40 (barely inside the floor) both count as "rank 1" if each is its own query's
top vector hit, so RRF cannot express "the vector signal was much more confident here than there."
This is a real, accepted cost of picking RRF, not a free upgrade — a document that is a genuinely
excellent vector match but shares literally no vocabulary with the query could, in principle, be
outranked by some other document that is only a mediocre vector match but a strong lexical one.
I judge this an acceptable trade against the magnitude-comparability problem RRF avoids, because
(a) this corpus's own measured pattern (DEF-1 diagnosis, Finding 3) shows `b-code` queries tend to
score well on *both* signals at once (their correct document's unique low-frequency tokens —
`RESULTSET_SIZE`, `GRAPH.RO_QUERY` — are exactly what both a sharp vector embedding and a
high-IDF lexical match reward), so the risk case (strong-vector/zero-lexical-overlap losing to
weak-vector/strong-lexical) is not the corpus's dominant pattern; and (b) it is a testable risk,
not an assumed-safe one — U5's regression check (§4 below) is specifically designed to catch it if
it happens on the already-passing rows.

### 2. Lexical query shape — OR-term for ranking, a rank-based admissibility gate, phrase held in reserve

**Decision: run the RediSearch default OR-term query as the lexical side's sole contribution to
the fused ranking; do not run the whole query string as a quoted phrase, and do not build a
phrase-first/OR-fallback pipeline.** Reasoning against U1's own measurement, not restated in the
abstract: U1's phrase-match test used a *favorable, artificial* condition — a contiguous 3-word run
copied verbatim from a real chunk's own text as the query — as "a stand-in for a situation
description that shares vocabulary with its correct answer" (U1's own words). That stand-in
overstates what a real situation-style golden-set query does: P1's and G2's own queries are
free-worded paraphrases (`"A plan states a completeness claim… but I only have the author's word
for it. What would actually make that check able to fail, rather than just being transcribed as
true?"` vs. the stored title `"A completeness claim must be derived, not transcribed — and its
check must be able to fail"`) — real overlapping technical terms (`completeness claim`, `able to
fail`, `transcribed`), but scattered across a differently-structured sentence, not a single
contiguous run spanning the query. A whole-query phrase search against text shaped like this has a
near-zero expected hit rate on a genuinely paraphrased query — querying it as a phrase would
reproduce U1's own "2 of 3 tested phrases returned 0 rows" finding as the *common* case here, not
the exception, defeating the entire lexical signal on exactly the queries item 2 exists to fix.
OR-term, by contrast, sums TF-IDF across whatever terms *do* overlap regardless of adjacency or
sentence structure — the realistic overlap pattern P1/G2 actually exhibit — and it does so without
needing a fallback branch, since it isn't the thing that returns 0 rows in the common case. **This
also directly answers the brief's fallback question: a lexical signal contributing 0 for a query is
the correct, unremarkable degenerate case of RRF (that query's fused score reduces to the vector
signal alone) — not a defect needing a rescue path** — it just shouldn't be the *common* case, which
is why OR-term, not phrase, is the primary shape.

**The admissibility gate — a second, separate design decision from the ranking formula, needed
because RRF alone reopens the exact risk the vector floor exists to close.** The vector floor
(**0.43**, `skills/agent-kb-retrieval/SKILL.md`'s current operative value, landed by Stage 8 Phase
2's full-set gate — not this file's own earlier 0.42 Stage 8 Phase 1 provisional figure, which that
phase's own text already flagged as due for refinement and which Phase 2 superseded) exists because
"the failure mode this system exists to avoid is not nothing came back, it's an agent trusted an
irrelevant top-1" (Recommendation 3, above). RRF's fused ranking has no floor of its own — a
document with a poor vector score (nowhere near 0.43) but a decent OR-term lexical rank could be
pulled into the fused top-5 purely on loose term co-occurrence (U1 measured `"the"` matching 95% of
the corpus — an extreme case, but it shows how permissive default OR-scoring can be). Decision: a
candidate is eligible for the returned top-5 only if **vector cosine distance ≤ 0.43 (the current
`SKILL.md` floor, unchanged by this design) OR the candidate ranks in the lexical OR-term list's
own top 2.** Both halves of this gate are
rank/threshold-based on a *calibrated* signal (the floor) or *rank position* (never raw lexical
magnitude) — deliberately consistent with §1's core reason for choosing RRF over magnitude
normalization: nothing in this design ever compares a raw lexical score against a fixed number,
because U1 already showed that number has no stable meaning across queries. Ranking (RRF, over the
union of both top-20 lists) and admissibility (this gate) are applied in that order — rank first,
then walk down the fused list keeping the first 5 that pass the gate, backfilling from further down
the list for any that don't — mirroring `Services.search_documents`'s own existing over-fetch-then-
filter idiom (Findings, above), not a new pattern. If fewer than 5 pass, return fewer — the same
graceful-degradation behavior the floor already produces today for a negative query.

**Phrase queries stay available, unused by default — an evidence-triggered fallback, not built
now.** The same full-text index serves both query shapes at zero extra migration or schema cost
(RediSearch query syntax is a call-site choice, not an index property) — so nothing is lost by not
using phrase matching in v1. If U5's negative-stratum re-run (§4) shows the rank-≤2 OR-term gate
admitting a false positive a stricter check would have caught, the concrete, already-designed
fallback is to require a genuine contiguous-phrase match (via `db.idx.fulltext.queryNodes` in
phrase mode) as an additional condition for any candidate admitted through the lexical-rank half of
the gate specifically — named here so a follow-up dispatch doesn't have to redesign it, not applied
speculatively before evidence shows it's needed.

**One implementation-facing correctness note for U3/U4, worth stating explicitly since it's an easy
mistake:** the lexical query must run against the **raw situation text only**, never the
`"Instruct: …\nQuery: {situation}"` wrapper (Recommendation 1). That wrapper is a property of
Qwen3-Embedding's trained query/document asymmetry — it means nothing to RediSearch's lexical
scorer, and feeding it in would inject the same boilerplate terms (`retrieve`, `situation`,
`technique`) into every single lexical query, polluting the TF-IDF signal with a constant,
query-independent term set. Stored documents are already unprefixed (Recommendation 1's write side);
the lexical query side should match that convention, stripped of the vector-only wrapper.

### 3. Acceptance criterion — retire the inherited "≥3 of 4," replace with a mechanism-specific bar

**Decision: the strategy6 bar ("≥3 of 4 target queries land in top-5") does not transfer to item 2,
and I am not silently inheriting it.** That bar was designed for item 1, a uniform lever (more
model capacity) tested against 4 queries treated as interchangeable instances of one hypothesis.
Item 2 is not that: the DEF-1 diagnosis itself (Finding 3, above) split the 4 into two different
mechanisms — P1/G2 (lexical-overlap-without-embedding-win, fusion's actual target) and C1/X1
(crowded-neighborhood dilution and document-length dilution respectively, mechanisms fusion was
never expected to fix) — treating all 4 as fungible inputs to one fraction was already a
simplification strategy6 made for a different lever; carrying it forward for a structural lever
whose own motivating diagnosis explicitly names only 2 of the 4 as its target would be adopting a
bar shaped for a different hypothesis, not re-deriving one for this one.

**Reconciling item 1's result — what's actually left for fusion to prove.** Item 1's held-out trial
(`qwen3-embedding-4b`, above) showed P1 and X1 flip to top-5 hits and C1/G2 stay misses — but
**that trial was never adopted**: its own verdict explicitly declined to commit to a fuller corpus
re-embed ("do not commit… The acceptance criterion I set myself is not met"), so **production is
unchanged, still the 0.6B model**, and under production settings today **all 4 of C1, P1, X1, G2
still fail** — nothing item 1 found has actually shipped. So item 2 is not chasing 2 already-solved
queries; it is being layered onto a baseline where none of the 4 currently succeed. What item 1's
result *does* change is which subset item 2 should expect to move, and how much weight a given
outcome should carry:
- **G2 is the single necessary target.** It is the one query in the lexical-overlap bucket where
  model capacity gave *zero* improvement (rank 53 of 302 under 4B, worse if anything than "absent at
  limit 20" under 0.6B) — the cleanest evidence available that this specific failure needs a
  structural, not capacity, fix. If fusion cannot move G2, the mechanism this unit exists to test
  has not been demonstrated, regardless of what happens to the other three.
- **P1 is an expected, confirmatory win, not the deciding signal.** It sits in the same
  lexical-overlap bucket as G2 and should respond to the same mechanism (OR-term rewarding
  `completeness claim`/`transcribed`/`able to fail` overlap) — but because item 1 already showed
  P1 is *also* fixable by capacity alone, a P1-only win (with G2 unmoved) would show fusion adds
  nothing this corpus doesn't already have a documented, simpler alternative for, not that fusion
  earns its complexity.
- **C1 and X1 are out of scope for this lever — not fungible "misses" fusion must also flip.** C1's
  diagnosis (Finding 3) is explicitly *not* a lexical-overlap case — it's outcompeted by ~15
  topically-adjacent, only-loosely-related documents from the same dense neighborhood, with no
  single lexical trigger to reward. X1's is document-length/dilution (Finding 2), the same
  mechanism model capacity partially addressed, not an overlap gap. Neither should be scored
  pass/fail against fusion; record what happens to both (informative, and a real regression check —
  next bullet — but not a required win).

**The replacement bar, concrete and gradeable, four conditions:**
1. **G2 lands in top-5 under fusion.** Necessary condition — without this, judge the unit "hybrid
   didn't help" regardless of the other three.
2. **No regression on the other 45-pair rows already recorded as hits** (Stage 8 Phase 1/Phase 2):
   every row that lands in top-5 today under vector-only must still land in top-5 under fusion — a
   rank/inclusion check per row, not a re-derivation of the whole set from scratch.
3. **No regression on the negative stratum (N1–N6):** every negative query must still return either
   nothing, or nothing that passes the admissibility gate — this is the concrete, mechanism-specific
   check for the new failure mode fusion introduces (§2's gate risk), and it did not exist as a
   concern under vector-only.
4. **P1, C1, X1 outcomes are reported, not graded.** State whether each moved, and if so how, but
   don't let a P1-only win read as "3 of 4," and don't let a C1/X1 non-win read as "hybrid failed."

**"Hybrid helped" for this unit's own bottom-line verdict = condition 1 holds AND conditions 2–3
hold.** "Hybrid didn't help" = condition 1 fails, regardless of 2–3 (a regression-free result that
still misses the one thing this lever was built to fix is not a positive result for item 2, even
though it would still be worth shipping-neutral information about the mechanism's safety).

### 4. Evaluation plan for U5

**Sequencing:** run only after U3's plan and U4's implementation land against the real
`ws:agent-team` (per the coordination doc's own note — never a probe graph for this final check,
since it's validating the actual shipped mechanism, not feasibility).

- **Queries:** the full 45-pair golden-set design (Stage 8 Phase 1's table, above) — not just the 4
  DEF-1 misses. Fusion changes the retrieval mechanism for every query, not only the failing ones,
  so a full re-run is the only way to catch condition 2/3's regression checks; at 45 pairs against a
  ~558-chunk corpus this is cheap, and it's the scope Stage 8 already established as this KB's
  standing regression gate.
- **Corpus:** production `ws:agent-team`, post-U4 (real full-text index landed via
  `bootstrap_schema.sh`, real fusion code path), not a probe graph — U1's probe work already closed
  the feasibility question; U5 is validating shipped behavior.
- **Baseline:** the already-recorded vector-only Stage 8 Phase 1 (16 rows) + Phase 2 (remaining 29,
  `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`) per-row ranks/scores — do
  **not** re-run vector-only fresh; pull the existing recorded figures as the comparison baseline,
  the same discipline U1 used citing existing evidence rather than rederiving it.
- **Per-query "helped" vs. "didn't help," specifically for the 4 target queries:**
  - **G2** — helped: expected document in top-5. Partially-helped, worth recording but not passing:
    now appears within the fused top-20 (an improvement over "absent at limit 20" even if it misses
    top-5). Didn't help: still absent from the fused top-20 equivalent.
  - **P1** — helped (expected/confirmatory): top-5. Didn't help: still misses — this outcome would
    be a genuine negative finding worth its own flag (it would mean OR-term didn't catch overlap the
    diagnosis itself identified as strong — check whether the OR-term query actually fired on the
    right terms before concluding the mechanism itself is unsound).
  - **C1, X1** — record rank/score movement either way; do not gate the unit's verdict on either.
    An unexpected win on either is a bonus to note, not a claim to build the mechanism's case on.
- **Regression checks (conditions 2–3 above), concretely:** recompute recall@5 across all strata
  (`b-code`, `b-prose`, negative, near-dup stress `(d)`, multi-facet `(e)`) and compare to the
  recorded baseline per stratum — flag any stratum whose recall@5 drops, not just the aggregate
  figure, since a stratum-level regression could hide inside an unchanged aggregate. Pay particular
  attention to the near-dup stress pairs `(d)` (R7, F1) and the negative stratum (N1–N6): both are
  exactly where §1's named RRF trade-off (magnitude-blind ranking) and §2's named gate risk (loose
  OR-term admission) would first show up as a real defect rather than a theoretical one.
- **What to compare against, one more time, explicitly:** the comparison is fusion-over-0.6B vs.
  vector-only-0.6B (both on the model actually in production) — **not** vs. item 1's 4B trial
  numbers. The 4B figures are cited above only to reconcile which subset fusion targets (§3); they
  are not a baseline U5 measures against, since that model was never adopted.

## Risks & open questions (mine to flag, not mine to resolve) — item 2 addendum

- **RRF's magnitude-blindness (§1) is an accepted, not a resolved, trade-off.** A vector-confident
  match with zero lexical overlap could in principle be outranked by a lexical-strong/vector-weak
  document. Judged low-risk given this corpus's own measured pattern (unique-token `b-code` claims
  tend to score well on both signals at once), but that judgment is a prediction from the existing
  DEF-1 evidence, not a fresh measurement — U5's stratified regression check (§4) is what would
  actually catch it if the prediction is wrong.
- **G2's rescue depends on an unverified premise I cannot check in this unit:** that G2's expected
  document scores well enough on the OR-term lexical query to either pass the admissibility gate or
  contribute enough RRF weight to be pulled into the fused top-5. Nothing in U1's or my own work
  measured G2's actual lexical score — the diagnosis's "striking exact-phrase overlap" language
  described the *title's* wording relative to the query, not a measured RediSearch score against
  the full stored document text. This is squarely what U5 tests; I am not asserting the mechanism
  will work, only that this design gives it the best-available structural chance to.
- **The `k=60` RRF constant is the literature default, not re-derived for this corpus's shallow
  (top-20) list depth.** At this depth the difference between lexical/vector rank 1 and rank 20 is
  fairly narrow (`1/61` vs. `1/80`) — deliberately gentle, so no single signal at a weak rank
  dominates, but untested against a smaller `k` (sharper preference for near-top ranks) on this
  specific corpus. Not blocking U3/U4 — ship `k=60` as the default, documented as a single named
  constant (same "one canonical, cited artifact" discipline as the prefix/floor,
  `skills/agent-kb-retrieval/SKILL.md`), and treat a `k` retune as the first lever to pull if U5's
  result is close but not clean, before redesigning the formula itself.
- **The admissibility gate's rank-≤2 lexical threshold is a judgment call, not a calibrated number**
  — unlike the vector floor (calibrated against a measured true-positive/negative score gap), this
  threshold has no equivalent calibration exercise behind it, because U1 already showed raw lexical
  magnitude isn't stable enough to calibrate a threshold against. Rank-2 vs. rank-1 vs. rank-3 is a
  reasonable-but-unverified choice; U5's negative-stratum results are the first real evidence either
  way, and the phrase-match fallback named in §2 is the designed response if rank-2 proves too loose.
- **Item 1's own open item (code/regex-heavy retrieval strength, and the "code/regex encoder
  quality" question named in Recommendation 1) is unaffected by this design** — fusion changes
  ranking, not the underlying vector signal's own quality, and none of this section's decisions
  depend on that question being resolved either way.

**Traceability.** Inputs: `claude/docs/plans/agent-knowledge-base-strategy-graph.md` (U1,
`graph-dba`, 2026-09-20) for feasibility and score characterization; this file's own DEF-1
diagnosis and Recommendation-1-execution sections (above) for the target-query reconciliation.
Coordination: `claude/docs/plans/agent-knowledge-base-strategy7-coordination.md`, U2. No trial run
in this unit — every number cited above is either U1's measurement or this file's own prior,
already-recorded pilot/trial results; nothing new was executed here.
