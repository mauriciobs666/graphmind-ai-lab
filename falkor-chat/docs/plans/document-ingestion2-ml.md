# Document Update Detection — Method Note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** —

## 1. The question and the decision it serves

`document-ingestion2.md` (Ready for design) extends the shipped `document-ingestion.md`/
`document-ingestion-ml.md`/`document-ingestion-graph.md` set with real update/delete. FR-2 requires
the system to **automatically** detect that newly-submitted content is an edited version of an
already-ingested document, at the same confidence-tiered pattern already shipped for entity fusion
(FR-8/FR-9/FR-10 of the original feature): a very-high-confidence match auto-supersedes
(`decidedBy='system'`, no review); a lower-confidence match becomes a pending suggestion requiring
confirm/reject. OQ-2 (detection technique) and part of OQ-1 (confidence tiering) are delegated here,
exactly as OQ-1 (entity-match technique) was delegated to `document-ingestion-ml.md` — that note is
the load-bearing precedent this one carries forward, not a fresh design.

**Decision this serves:** whether `architect`'s plan for `document-ingestion2` builds detection
against a deterministic default (mirroring `document-ingestion-ml.md`'s posture on entity matching)
or needs an ML signal (embeddings/LLM) at some tier, and where in the pipeline detection runs given
`Chunk.embedding` is async.

**Top-line answer:** the same posture as the entity precedent, applied with equal or greater force —
**deterministic, uncalibrated-threshold-free auto tier; deterministic-but-inherently-continuous
candidate generation for the suggested tier; embeddings and LLM comparison both deferred to a scoped
v2, not a v1 precondition.** Concretely: auto tier = **exact content-identity** (a normalized-text
hash equality, computed synchronously off already-available raw text, no title requirement); suggested
tier = **shingled-Jaccard content-overlap** over a cheaply-narrowed candidate shortlist, with the raw
ratio stored as an audited, explicitly-non-probabilistic `SUPERSEDES.confidence`. Detection needs
**no wait** for chunk embeddings — both tiers run off `Document.title`/`Document.text` alone, which
is a genuine advantage over an embedding-based design, not just a fallback. The auto tier's
correctness bar is, if anything, **stricter** than the entity case (§4.2 explains why), which
reinforces rather than weakens the "no numeric threshold" conclusion.

## 2. Findings from the real system

**F1 — `document-ingestion-ml.md`'s posture is the direct precedent, and its own reasoning
transfers.** That note (§4.1) argued the auto-merge tier is the *one* fusion action with zero
human/agent review, so its criterion carries all the correctness risk in the design, and an
uncalibrated numeric threshold has no place gating an unreviewed action absent calibration data —
which this pipeline still has none of (confirmed: no golden set, no production traffic yet, per
`document-ingestion.md` §7's still-open risk item). Nothing about moving from entity- to
document-granularity changes that argument; if anything it sharpens it (§4.2).

**F2 — `Chunk.embedding` is computed strictly out-of-band, after the synchronous write returns, and
per-chunk, never per-document.** `EmbeddingWorker.embed_chunk` (`server/falkorchat/embedding.py:230-247`)
is called from `background._safe_embed_chunk`, scheduled alongside (not before) `IngestionPipeline`'s
own background extraction step — both are peers scheduled right after `services.ingest_document`'s
synchronous `repository.create_document` call returns (`document-ingestion.md` §2.2/§4 Stage 2/3).
There is **no whole-`Document` embedding anywhere in the shipped schema or code** — the only vector
index that exists is `Chunk`'s (`bootstrap_schema.sh:239-240`, confirmed live in
`document-ingestion-graph.md` and unchanged since). Any technique that needs an embedding — whether a
new `Document.embedding` or an aggregate over `Chunk.embedding`s — is unavailable until every chunk's
embedding job has completed, which for a large document (up to the plan's `MAX_DOCUMENT_CHARS =
500,000`, ≈500 chunks at the shipped 1,000-char chunk size, `chunking.py:21`) can be the slowest of
the three async peers (embedding, extraction, and now detection), not the fastest.

**F3 — `Document.text` and `Document.title` are both available synchronously, at write time, with
no I/O.** `repository.create_document` (`repository.py:1013-1057`) writes `Document.text` verbatim
and `Document.title` (falling back to `source_label` or `""` when neither is given —
`services.py:1114`) in the same atomic `GRAPH.QUERY` as the chunks themselves. Both are already
committed before either background peer (embed, extract) is even scheduled. **Any detection
technique built on these two fields alone needs no async wait at all** — this is the structural
argument for Q4 below, independent of the calibration argument for Q1/Q2.

**F4 — `Document.title` is frequently absent or generic, weakening any title-dependent criterion.**
`title` defaults to `""` when the caller supplies neither `title` nor `source_label`
(`services.py:1081-1095`), and nothing constrains it to be distinctive (two unrelated documents can
share a common title like `"README"` or `""` itself). A criterion that leans on title equality alone
would both under-match (an edit that also changes the title) and over-match (two empty titles are
trivially "equal"). Title is therefore a *weak, optional booster*, not a load-bearing identity
signal — the analog of relying on `Entity.name` alone without `Entity.type` in the entity case, which
`document-ingestion-ml.md` §3.1 already rejected for exactly this "not enough to be an identity
check" reason.

**F5 — the exact-tier concurrency fix precedent transfers directly.** `document-ingestion-graph.md`/
`document-ingestion.md` §3.4's concurrency note closed a check-then-act race for
`create_entity_with_auto_match` by folding the candidate lookup, the new node's creation, and its
conditional auto-link into one atomic `GRAPH.QUERY` — because two concurrent writes (a batch,
`ingest_documents`, or two concurrent MCP calls) could otherwise each miss the other's
not-yet-committed sibling, silently defeating the auto tier's "no confirmation needed" guarantee. The
identical race shape exists for document supersession: a batch resubmitting the same content twice,
or two concurrent `ingest_document` calls for the same file (e.g. a double-triggered sync job), can
each run an exact-match lookup before either sibling's `Document` commits. The fix is the same
mechanism, one hop over: fold the content-hash lookup, the new `Document`'s creation, and its
conditional auto-supersede edge into one atomic write. This is `graph-dba`'s Cypher to design, not
mine, but the *method* requirement (atomicity at the auto tier only, not the suggested tier — same
asymmetry reasoning as the entity case, F1) is the same conclusion transplanted.

**F6 — `SAME_AS`'s edge shape and tiering vocabulary is exactly what the requirements doc says to
reuse for `SUPERSEDES`** (`{matchId, status, confidence, technique, createdAt, decidedAt, decidedBy,
resuggestCount, lastResuggestedAt}`, `document-ingestion-graph.md` §1.5, `document-ingestion.md`
§3.4). Nothing in this note proposes a different property set — only what drives `technique` and
`confidence` at document granularity.

**F7 — a false auto-decision has a larger blast radius for documents than for entities.** Entity
fusion never hides or destroys anything: a spurious `SAME_AS{status:'confirmed'}` edge is a link
sitting alongside two still-independently-reachable nodes (`document-ingestion-ml.md` §4.1's own
framing of the residual risk as low-severity for exactly this reason). Document supersession is
different by requirement: AC-4 says default search must surface **only** the current version once
supersession happens — a false auto-supersede actively **removes an unrelated, independent document
from the default view**, not merely adds a spurious link. FR-1's "retained, not destroyed" caveat
means the content isn't gone (recoverable via history, AC-5), but it is wrongly hidden until someone
notices and reverses it — a real, if recoverable, correctness harm with a strictly larger footprint
than the entity case's spurious-edge risk. This asymmetry is addressed directly in §4.2.

## 3. Recommendation — signal, tiers, and thresholds (Q1/Q2)

**Verdict: no embedding, no LLM, and no numeric similarity threshold anywhere in the auto tier.**
The suggested tier uses a deterministic content-overlap ratio, stored as an audited, explicitly
non-probabilistic value — same posture `document-ingestion-ml.md` §4.3 set for `SAME_AS.confidence`.

### 3.1 Auto tier ("very-high confidence", FR-2/AC-1) — exact content identity, no threshold

**Criterion: `Document.textNormalizedHash` (a hash of the case-folded, whitespace-collapsed full
text) equality against an existing `currentVersion` document — title plays no role.** This is a
*definitional* identity check, the direct document-granularity analog of the entity tier's
"normalized-name + identical-type" — not a probabilistic judgment, and it needs no calibration data
for the same reason that one didn't: two documents whose full text is identical modulo whitespace/
case are, for any real corpus, the same content, full stop. `confidence` is stored as `1.0` (a
deterministic identity flag, not a score — same convention as `SAME_AS`'s exact tier), `technique =
'exact_normalized_text_hash'`.

- **Why hash-equality, not a title match, and not "title + near-duplicate":** F4 rules out title as
  a load-bearing signal. A design that required title equality *in addition to* content match would
  actually be **wrong**, not merely redundant: consider an automated re-sync of an unedited file
  whose caller happens to pass a timestamped title each run (`"Notes — 2026-09-01"` vs `"Notes —
  2026-09-10"`) — the content is byte-identical, so it is unambiguously the same document, and
  requiring title equality too would wrongly push this into the suggested tier for no safety
  benefit. Conversely, a design that allowed title-match-plus-a-high-but-not-perfect content-overlap
  *ratio* into the auto tier reintroduces exactly the "guess wearing a decimal point" anti-pattern
  `document-ingestion-ml.md` §4.2 named — there is no calibration data to justify *any* cutoff below
  1.0, however close to 1.0 it looks intuitively.
- **This deliberately does not catch real edits.** A single-character change anywhere in the
  document flips the hash entirely, so genuinely edited content — the case FR-2 exists for — never
  auto-supersedes; it is picked up by the suggested tier (§3.2) or missed entirely if it also falls
  below that tier's candidate net (§4). This mirrors the entity precedent's own posture exactly: the
  auto tier is deliberately narrow and boring; the tier that does the interesting work is the
  reviewed one. Given F7's larger blast radius for documents, an auto tier that is at least as
  conservative as the entity one is the right default, not a compromise.
- **Free side benefit:** this closes, as a side effect, `document-ingestion.md` §7's own
  still-open risk ("non-idempotent document creation on retry... a retried `ingest_document` call
  mints a second `Document`") for the specific case of an unchanged retry — a retried call now
  auto-supersedes its own accidental duplicate rather than leaving two independent, byte-identical
  `Document`s in the graph forever. Not the only fix for that risk (a client-side idempotency key
  would be more direct), but a real, unplanned-for improvement worth noting to `architect`.

### 3.2 Suggested tier ("plausible but not very-confident", FR-2/AC-2, FR-9 mirror) — deterministic content-overlap, no calibrated gate

**Candidate generation:** narrow the workspace-wide comparison set cheaply *before* computing
anything content-overlap-shaped — never a raw O(n) pairwise scan against every `currentVersion`
document (§4 below covers scoping and cost in full). **Overlap metric: shingled Jaccard over
word n-grams (e.g. 5-word shingles), not `difflib.SequenceMatcher.ratio()`.** Both are
deterministic, no-ML options; the choice between them is a real, decidable trade-off, not a coin
flip:

- `SequenceMatcher.ratio()` is well-suited to short strings (it's already exactly what candidate-name
  fuzzy-matching conceptually approximates for entities, modulo RediSearch doing the heavy lifting
  there) but its underlying algorithm has worse-than-linear behavior on long, dissimilar inputs
  before its `autojunk` heuristic kicks in — a real risk at this pipeline's own `MAX_DOCUMENT_CHARS =
  500,000` ceiling, computed pairwise against a candidate shortlist under bulk ingestion (FR-11, up
  to `MAX_BATCH_SIZE = 20` documents per call).
- **Shingled Jaccard scales the way this comparison shape needs to:** building a document's shingle
  set is O(length), and comparing two documents' shingle sets (as hashed sets) is bounded by the
  smaller set's size — the standard technique for large-document near-duplicate detection (the same
  family of algorithm behind web-scale dedup/plagiarism tooling), which is exactly this problem
  shape. Recommended default: 5-word shingles, case-folded/whitespace-collapsed first (reuse the same
  normalization as §3.1's hash, not a second normalizer).
- `confidence = |A ∩ B| / |A ∪ B|` — stored on `SUPERSEDES` for the suggested tier, `technique =
  'shingled_jaccard_overlap'`.

**No numeric ratio gates promotion to the auto tier — ever, in v1.** A document that is 99% but not
100% content-identical stays in the suggested tier, full stop; there is no "close enough to auto"
carve-out, for the same reason §3.1 rejected one. The suggested tier's ratio is a *ranking and
noise-filtering* signal, not a second confidence gate on top of an already-decided identity check —
mirroring `document-ingestion-ml.md` §4.3's framing of `SAME_AS.confidence` precisely: **do** apply a
non-ML noise-floor cutoff (an implementer-tunable minimum ratio below which a candidate never reaches
the pending queue at all — a usability floor against alert fatigue, not a correctness gate: FR-9/AC-2
require *plausible* matches to surface, not every weak one) and **do** store the raw ratio for
audit/UI, documented explicitly as "a content-overlap measurement, not a calibrated probability that
this is the same document."

### 3.3 Why embeddings and LLM comparison are both out of v1, at any tier

- **Embeddings (candidate (c)):** ruled out for the auto tier for the calibration reason above (any
  cosine cutoff is an uncalibrated guess), and ruled out for the suggested tier's *v1* candidate
  generation for two independent reasons, not one: (i) the same missing-calibration-data argument
  applies to "is this candidate similar enough to bother surfacing" exactly as it did for entity
  matching (`document-ingestion-ml.md` §4.2); (ii) **structurally**, per F2, an embedding-dependent
  signal cannot even run until every chunk's embedding job finishes — for the largest documents this
  pipeline permits, that is the slowest of the background peers, not a neutral tradeoff. Both
  arguments point the same direction; neither alone is load-bearing enough to skip stating both.
- **LLM comparison (candidate (d)):** rejected for the same reason `document-ingestion-ml.md` §4.2
  rejected "LLM-confirmation matching" for entities — a binary "is this an edit of that" judgment
  from an LLM needs the same false-positive/recall calibration as embeddings, which this pipeline
  doesn't have, plus a real throughput cost that is *worse* here than in the entity case: comparing
  whole documents (up to 500,000 chars) rather than short entity-name pairs means either an
  expensive full-document prompt or an extra unvalidated summarization step, per candidate, under
  bulk ingestion. Nothing about the document case makes this cheaper or more valid than the entity
  case already found it to be; it is dominated by shingled Jaccard on cost and by "deferred to v2
  with a real eval" on validity. Not recommended even as a v2 first move — see §5.
- **What I am not claiming:** that shingled Jaccan-based lexical overlap catches every real
  edit — a document rewritten heavily enough (new wording, same underlying subject) can fall below
  any reasonable overlap floor while still being "the same document, edited," the direct
  document-level analog of the entity note's "IBM"/"International Business Machines" non-lexical-synonym
  gap. That gap is real, named, and exactly what would justify climbing to embeddings in a v2 (§5) —
  not dismissed, just not solvable with zero calibration data today.

## 4. Candidate scope and pipeline placement (Q3/Q4)

### 4.1 Candidate scope

- **Scope to `currentVersion` documents only, workspace-wide — never compare against an
  already-superseded document.** This is not optional: comparing against a superseded document could
  reopen or duplicate a supersession chain that already resolved, the direct analog of the entity
  case's own posture (a `SAME_AS` edge is reused/reopened, never blindly re-created against a
  candidate that has since been superseded itself). Whatever boolean/pointer field `graph-dba`
  designs to mark a document non-current is the natural index for this filter — not this note's
  schema call to finalize.
- **Do not scope by `ingestedBy`/actor.** Restricting candidates to the same actor would miss the
  motivating real-world case squarely: an agent-team knowledge-base file ingested by one actor and
  later re-ingested (edited) by a different human or agent is exactly the shape
  `agent-knowledge-base-strategy.md` surfaced this feature for in the first place. Actor identity is
  a reasonable optional tie-break/ranking booster (same actor + high overlap ranks slightly above the
  same overlap from a different actor) but must not be a hard filter.
- **The real cost lever is candidate-generation narrowing, not actor/scope filtering** — a
  workspace can have many documents, and a document-vs-every-document shingle comparison is O(n),
  unacceptable at scale. Recommend a two-step "cast a broad net cheaply, then apply the exact metric
  only to the shortlist" design, the same shape `fusion.py`'s RediSearch-fuzzy-then-classify already
  uses for entities:
  1. **Auto tier:** an indexed equality lookup on `Document.textNormalizedHash` — O(1) via a RANGE
     index (no uniqueness constraint: two genuinely different documents can, in principle if
     vanishingly unlikely, share a hash before any is superseded), not a scan. This is cheap enough
     to run inline, synchronously (§4.2).
  2. **Suggested tier:** narrow first via a cheap indexed signal — a full-text query against
     `Document.text` (or a fingerprint/shingle-based index, if `graph-dba` judges a raw full-text
     index on up to 500,000-char documents too RAM-heavy — this specific indexing choice, and its
     RAM cost per rule 6, is graph-dba's call, not mine) to get a bounded shortlist (e.g. limit 5,
     mirroring the entity tier's own candidate cap), **then** compute the shingled-Jaccard ratio only
     against that shortlist in Python. Title-fuzzy match (RediSearch on `Document.title`, when
     non-empty) is a cheap additional/complementary candidate-generation signal, not a replacement
     for the text-based one — F4 already established title alone is too weak and too often empty to
     carry candidate generation by itself.
- **Named, accepted v1 gap:** an edit that also substantially changes the title, in a way that
  drops it below both the title-fuzzy net and the content-fingerprint net's discrimination, will not
  surface as a candidate at all — it silently becomes a new, unrelated `Document`, exactly today's
  behavior. This is the direct analog of `document-ingestion-ml.md` §4.1's "same-name-same-type-
  different-entity" residual risk, mirrored in the opposite direction (a miss, not a false-positive) —
  name it in `qa-engineer`'s eventual test plan as a known v1 limitation, not a defect, rather than
  building a full O(n) pairwise scan to close it pre-emptively against an unmeasured problem.

### 4.2 Where detection runs — plainly, per tier

**The auto tier runs synchronously, folded into the existing document-creation write — it does
not need to wait for anything.** Recommend a new atomic repository call,
`create_document_with_auto_supersede(ws, *, document_id, title, text, text_normalized_hash,
source_format, ingested_by, created_at, chunks, match_id) -> dict`, replacing today's
`create_document` at the `services.ingest_document` call site — mirroring `create_entity_with_
auto_match`'s shape exactly (one round trip: look up an existing `currentVersion` document with the
same `textNormalizedHash`, create the new `Document`+`Chunk`s, and — only if a candidate existed —
write the `SUPERSEDES{status:'confirmed', decidedBy:'system', confidence:1.0,
technique:'exact_normalized_text_hash'}` edge and flip the candidate's `currentVersion` off, all in
one `GRAPH.QUERY`). Because `Document.text`/`Document.title` are both already committed at this
point (F3) and the hash lookup is an indexed equality check, there is no calibration-independent
reason to defer this to a background job. **AC-1 is therefore satisfiable synchronously, at the same
"ingestion completes" moment `ingest_document`'s response returns** — the one part of FR-2 that can
be, since it needs neither an LLM call nor an embedding.

**The suggested tier runs asynchronously, as a background job scheduled alongside (not blocking)
extraction/embedding — not because it depends on them, but for cost/latency reasons of its own.**
Recommend a new peer, e.g. `_safe_detect_update` (mirroring `_safe_extract`/`_safe_embed_chunk`'s
try/except-log-never-raise discipline), scheduled right after the synchronous document write returns,
alongside the other per-document background jobs (`document-ingestion.md` §2.2). The reason to defer
it is not F2's async-embedding constraint — this tier doesn't touch embeddings either — it is that
candidate-shortlist fetch + Jaccard computation over up to 500,000-char texts is unbounded-ish Python
work that has no place on the synchronous request path, especially under bulk ingestion (`FR-11`, up
to 20 documents/call). **AC-2 is therefore satisfied *eventually*, not synchronously** — the same
eventual-consistency posture `Document.status` already has for extraction/embedding, and the
requirements doc's own "when ingestion completes" phrasing should be read the same way it already is
read for `Document.status` reaching `'ready'`.

**Net effect worth naming to `architect`:** because neither tier needs `Chunk.embedding`, a pending
update-suggestion can appear well *before* a document's chunks are embedded or its entities extracted
— a genuine advantage of the deterministic design over any embedding-based alternative, not merely a
consolation for lacking calibration data. `Document.status`'s existing state machine
(`processing`/`ready`/`failed`, `repository.py:1080-1144`) tracks embedding+extraction completion;
detection's suggested-tier completion is a separate signal (the presence/absence of a pending
`SUPERSEDES` edge) and should not be folded into `pendingJobs`/`status` — a detection-job failure is a
soft failure (the document is still valid and searchable standalone) and must not flip the whole
document to `'failed'` the way an extraction/embedding failure does. This is an explicit scope note
for whoever designs the exact wiring, not a schema decision this note is making.

**Batch semantics (AC-8 analog):** `ingest_documents` loops the single-document path per item
(`document-ingestion.md` §3.6); as long as items are processed sequentially within one call, each
subsequent item's auto-tier check sees prior items' already-committed `Document`s, so within-batch
duplicates auto-supersede each other correctly with no special batch-aware logic — same reasoning
`document-ingestion.md` §3.6 already gives for entity fusion across a batch. If items are ever
processed concurrently, the same race F5 names applies and is closed the same way (atomicity, not a
processing-order guarantee).

## 5. `SUPERSEDES.confidence` (Q5)

Same framing as `document-ingestion-ml.md` §4.3 for `SAME_AS.confidence`, restated at document
granularity:

- **Auto tier:** `confidence = 1.0` — a deterministic identity flag, not a score. Nothing to gate.
- **Suggested tier:** `confidence` stores the raw shingled-Jaccard ratio, for audit/UI purposes,
  documented at the point of use (docstring/API doc) as "a content-overlap measurement between 0 and
  1, not a calibrated probability that this is the same document" — so a future reader doesn't
  mistake it for the auto tier's identity guarantee, and doesn't promote it to a silent auto-gate
  later without the calibration work in §6.
- **A non-ML precision floor is appropriate and recommended** — a minimum Jaccard ratio below which
  a candidate never reaches the pending queue at all, framed explicitly as an implementer-tunable
  noise-floor/UX cutoff (mirrors `MAX_DOCUMENT_CHARS`/`MAX_BATCH_SIZE`'s "implementer-tunable, not
  load-bearing" framing, `document-ingestion.md` §3.5), not a validated match-confidence threshold.
  Its exact value is a routine implementer judgment call, same posture the entity note already set
  for the fuzzy-tier floor.

## 6. Evaluation design — for the deferred embedding/LLM rung, if/when climbed

Not required for v1 (§3.3) — recorded now, mirroring `document-ingestion-ml.md` §5's own minimal-
viable sizing, so a future numeric threshold on this path isn't shipped as a guess either.

- **Golden set:** document **pairs**, labeled `{same-document-edited-version, different-document}`,
  stratified into: (1) trivial edits (typo/whitespace fixes) — sanity check, likely already caught by
  the shipped shingled-Jaccard floor, useful mainly to confirm the deterministic tier isn't
  regressing; (2) **substantive rewrites** — the real target this rung exists for: same underlying
  subject/purpose, materially different wording/structure (the document-level analog of the entity
  note's "IBM"/"International Business Machines" non-lexical-synonym case) — lexical overlap alone
  is expected to miss these, which is exactly why they'd justify climbing to embeddings; (3) **hard
  negatives** — two genuinely different documents with high lexical overlap (a shared template,
  heavy boilerplate/legal text, two unrelated documents built from the same starting draft) — the
  class most likely to blow up a false-supersede rate if a raw similarity score were trusted
  uncritically, and per F7 the class that matters most here precisely because a false document-level
  positive hides content from default view, not just adds a spurious link. **Minimal viable size,
  same order as the entity precedent:** n≈20-30 hand-labeled pairs for a one-sided screen, honestly
  reported with Wilson-interval-aware small-n caveats, not a certification; grow it from real
  post-ingestion `SUPERSEDES`-suggestion confirm/reject outcomes once the suggested tier has been live
  long enough to accumulate them — production data outranks a hand-built set here too.
- **Metric — asymmetric, gated on the costlier error, not symmetric accuracy/κ:** the costly error is
  a **false supersede** (the auto tier declaring two independent documents "the same edited-version
  pair," silently hiding one from default search) — report **false-supersede rate** (of pairs the
  technique would auto-supersede, the fraction that are actually independent documents) as the
  primary gate and **edit-recall** (of true edit-pairs, the fraction detected at all, at minimum at
  the suggested tier) as the secondary one. Report any symmetric accuracy/κ figure as a diagnostic
  only — an always-suggest, never-auto-supersede detector scores a trivially perfect false-supersede
  rate by construction, the same collapse-onto-the-safe-class failure mode
  `document-ingestion-ml.md` §5 already named for entity matching, K-027 item 3's guard-judge
  precedent before that.
- **Given F7, the acceptable false-supersede rate for documents should be set at least as strict as
  whatever bar a future entity-matching v2 sets for false-merge** — a document false-positive's blast
  radius (hiding independent content from default view) is larger than an entity false-positive's
  (a spurious but non-hiding link), so this is not a case where "the same number transfers"; if
  anything the document threshold, once ever set from real data, should be tighter. Not fixing a
  number here — flagging the asymmetry so whoever runs this calibration doesn't default to copying
  the entity number unexamined.
- **No threshold number is proposed here**, for the same reason `document-ingestion-ml.md` §5 gives —
  set after the golden set exists and the calibration is run, never reasoned backward from a target.
- **Where semantic matching would slot in, if this rung is climbed:** for **candidate generation
  only** at the suggested tier, never auto-supersede directly off a cosine score even a calibrated
  one. Two genuinely different implementation paths exist, with a real trade-off worth naming to
  whoever designs v2: (a) a **new whole-`Document`-level `Document.embedding`** computed directly off
  `Document.text` — avoids depending on chunk-embedding completion (so it keeps the "detection
  doesn't need to wait" property this note's v1 design has) but is a wholly new vector index with its
  own RAM cost, the same `document-ingestion-ml.md` §4.2 argument against `Entity.embedding`; (b) an
  **aggregate/derived signal off the existing `Chunk.embedding`s** (centroid, or best-chunk-pair
  cosine) — reuses an index that already exists, avoiding new RAM, but reintroduces exactly F2's
  async-completion dependency this note's v1 design deliberately avoided, meaning a v2 built this way
  would regress the "suggestion can appear before embeddings finish" property named in §4.2. Neither
  is obviously right; it is a genuine RAM-vs-latency trade `graph-dba`/`architect` should weigh
  explicitly when this rung is picked up, not inherited by default from the entity case's answer.
  LLM-based comparison, if ever pursued, should use summaries rather than full documents to bound
  cost, but is not recommended as v2's first move — embeddings dominate it on cost per §3.3's
  reasoning, unchanged at this rung.

## 7. Risks & open questions

- **The suggested tier's content-overlap net has a named, accepted miss** (§4.1): a substantially
  rewritten edit that also changes the title enough to escape both candidate-generation signals will
  not be detected at all in v1 — silently becomes an independent `Document`, today's exact behavior.
  Worth one line in `qa-engineer`'s eventual test plan as a known v1 limitation.
- **`Document.textNormalizedHash`'s exact indexing (RANGE, no constraint) and the suggested tier's
  candidate-generation index (full-text on `Document.text` vs. a lighter fingerprint scheme) are
  schema/RAM decisions for `graph-dba` to finalize**, not this note's to make — flagged in §4.1/§4.2
  as the specific open points, mirroring how `document-ingestion-ml.md` flagged `Entity.nameNormalized`'s
  exact DDL to the same owner.
- **A false auto-supersede is more consequential here than a false entity auto-merge (F7)** — hides
  independent content from default search, not just adds a spurious link. This is already reflected
  in the recommendation (an even narrower, hash-identity-only auto tier than the entity precedent's
  name+type match) and in §6's calibration-asymmetry note; naming it again here so it isn't lost as a
  one-off observation buried in the findings section.
- **`SUPERSEDES` confirm/reject/recheck/list surface should mirror `SAME_AS`'s exactly** — FR-3 of
  `document-ingestion2.md` already requires confirm-or-reject symmetry with FR-10, and OQ-3's
  "rejected, not permanent" resolution for entities (auto-reopen on corroboration + manual recheck)
  is a design pattern worth reusing verbatim for documents rather than re-derived, since nothing about
  document granularity changes that reasoning. This is `architect`'s wiring call, not an ML question,
  flagged here only because it falls directly out of reusing the `SAME_AS` edge shape (F6).
- **Extraction/fusion's own entity-level consequences on update/delete (FR-7, deferred by the
  requirements doc to whoever designs this) are explicitly out of scope for this note** — this note
  answers only FR-2's detection question; the fate of entities extracted from a document that gets
  superseded or deleted is `architect`'s call to make, weighed against the shipped fusion mechanics,
  not an ML-methodology question this note is positioned to resolve.
