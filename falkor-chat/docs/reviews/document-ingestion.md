# Ingestion Pipeline & Entity Fusion — plan-gate review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-050 (M5)

**CPG:** considered, not relevant — this is a pre-implementation design-consistency review across
three freshly-written documents for new-code work in `falkor-chat/server`, not an impact analysis
against an existing call graph. The brief explicitly directed skipping the CPG check; no relevant
CPG was consulted.

## Scope

Static plan-gate review of the design set for K-050 (Ingestion Pipeline & Entity Fusion, M5),
before any implementation code is written:

- `falkor-chat/docs/requirements/document-ingestion.md` (`tico`, Ready for design — the contract)
- `falkor-chat/docs/plans/document-ingestion.md` (`architect`, reconciled after `graph-dba`'s note)
- `falkor-chat/docs/plans/document-ingestion-graph.md` (`graph-dba`, schema/Cypher companion)
- `falkor-chat/docs/plans/document-ingestion-ml.md` (`data-scientist`, extraction/matching method note)

Checked against `falkor-chat/docs/DESIGN.md` (§5.1, §5.4, §7.1, §9, §10–§10.3, §11, §13.3, §14) and
`falkor-chat/docs/QUERIES.md` (§10–§10.3, §13.3), `scripts/bootstrap_schema.sh`, and the existing
test estate (`server/tests/test_provenance.py`, `server/falkorchat/repository.py`,
`server/falkorchat/modelconfig.py`, `server/falkorchat/background.py`, `server/falkorchat/embedding.py`).
Every factual/grounding claim cited below was checked against the live file, not taken on the
documents' word.

**Verdict: needs changes.** One blocker and four major findings, all fixable without a structural
redesign — routing detail is in each finding below so the fixes can be split across
`architect`/`graph-dba`/`data-scientist` without a second blind read.

---

## Findings

### BLOCKER — Fusion's exact-tier auto-merge has an unguarded check-then-act race across concurrent ingestion

**Where:** `docs/plans/document-ingestion.md` §3.4/§4 Stage 4, `docs/plans/document-ingestion-graph.md`
§1.6/§2.3 (`find_entity_candidates` + `create_or_reopen_match` as two separate round trips).

**Evidence.** `find_exact_candidate` (a plain `MATCH (e:Entity {nameNormalized, type})` read) and
`create_entity`/`create_or_reopen_match` (separate `CREATE`/write calls) are three independent
`GRAPH.QUERY` round trips per extracted entity — nothing binds them into one atomic operation the
way `create_or_reopen_match` itself binds "check existing `SAME_AS`, then create-or-reopen" into a
single guarded query (graph note §1.6). The codebase's own established background-scheduling
pattern is fire-and-forget, not serialized: `background.py`'s `_safe_embed` is scheduled per posted
message via FastAPI `BackgroundTasks` (concurrent per request) or a bare `threading.Thread` for MCP
(`background.py:1-11`) — there is no single queue/worker draining one item at a time. The plan's own
§2.2 explicitly models `IngestionPipeline` on this same pattern ("mirroring the existing 'embed
messages: async worker, decoupled from the write path' row"), and `ingest_documents` (FR-11) is a
first-class batch entry point.

**Why it matters.** If two entities that should exact-match (same normalized name + type) are
extracted around the same wall-clock time — either from two documents in the same `ingest_documents`
batch (AC-8's own explicit scenario: "both mention the same entity... a SAME_AS edge links their
extracted entities") or from two independent MCP calls by two connected agents (FR-5's own stated
"persistent memory" use case) — both entities' `find_exact_candidate` reads can run before either
sibling's `create_entity`/`create_or_reopen_match` write commits. Neither read sees the other's
not-yet-committed entity, so both entities are created with no `SAME_AS` edge between them at all —
silently defeating FR-8's "very-high confidence… linked automatically, no confirmation required"
guarantee for exactly the pair it was built to catch. This is the one fusion action with **zero**
human/agent review (`decidedBy='system'`), so there is no downstream safety net that would catch the
miss — contrast with every other race-prone write in this codebase (`Thread` HEAD/TAIL, member-ensure,
model-override CAS), which is deliberately guarded in one atomic query per `falkor-chat/AGENTS.md`
rule 4. Fusion's exact tier breaks that same discipline without comment anywhere in the three
documents.

**Suggested fix.** Either (a) fold the exact-tier candidate lookup into the same atomic query as
entity creation + fuse-decision — i.e., extend `create_or_reopen_match`'s own guarded-`CREATE`
pattern one level earlier so "does an exact candidate already exist" and "create this entity, link
or not" happen inside one `GRAPH.QUERY`, closing the race at the database layer the way every other
correctness-critical write in this codebase already does; or (b) state and enforce an explicit
sequencing guarantee (e.g., extraction+fusion for a batch/document processes strictly one entity at a
time, never concurrently, unlike embedding which has no such requirement) and say so in the plan.
Route: `architect` (design decision) + `graph-dba` (the atomic-query alternative, if chosen). Flag to
`teco` before Stage 4 starts — this changes the shape of `create_or_reopen_match`'s call site, not
just its internals.

---

### MAJOR — `docs/plans/document-ingestion.md` §3.7 has a stale, unreconciled reference

**Where:** `docs/plans/document-ingestion.md:344` (§3.7, FR-2 paragraph).

**Evidence.** §0's delegation table states the plan "has been reconciled against" both notes and
specifically credits `graph-dba` with settling "the generalized two-label `EMITTED` seed resolution
for FR-2 (§3.7)." But §3.7's own body text was never rewritten: it still reads *"the response
contract (`kind: 'message'|'chunk'` per seed, or a single namespaced id scheme) is exactly the kind
of decision this plan defers rather than inventing ad hoc."* Both options it names were **rejected**
by `graph-dba`'s actual resolution (`document-ingestion-graph.md` §3.1: neither a `kind` field nor a
namespaced id scheme — a bare-id `coalesce` against both `Message.msgId` and `Chunk.chunkId`
directly, reusing the existing author/mention-resolution idiom). This is the exact "stray leftover
reference to the superseded design" pattern the plan-gate brief asked me to check for — confirmed by
grep: it is the only surviving `must design`/`defer` phrasing left in the plan; every other such
phrase (e.g., the v2 embedding-matching deferral) is a genuinely still-open item, not a stale one.

**Why it matters.** Stage 5's file list correctly points implementers at `graph-dba`'s note for the
actual Cypher, so this alone probably won't misdirect an attentive implementer — but it directly
contradicts §0's "reconciled" claim, and a reader of §3.7 in isolation (which is exactly how the
FR-2/AC-5 story is told) would believe the id-scheme question is still unresolved.

**Suggested fix.** One paragraph rewrite in §3.7: replace the "defers" sentence with the actual
resolution (bare-id `coalesce`, cite `document-ingestion-graph.md` §3.1/§3.2/§3.3). Cheap, in-place
edit — route to `architect`.

---

### MAJOR — `docs/plans/document-ingestion-ml.md` never updated its "MatchSuggestion" terminology after the schema reconciliation

**Where:** `docs/plans/document-ingestion-ml.md:15,87,230,270,292,314`.

**Evidence.** Six occurrences of `MatchSuggestion`/`MatchSuggestion.confidence` remain in the
data-scientist note (the question framing in §1, finding F6, §4.1's residual-risk bullet, §4.3's
heading and its "store the raw score" bullet, and §5's golden-set-sourcing bullet) — all written
against the plan's **original**, pre-reconciliation node model (`MatchSuggestion` node +
`CANDIDATE_A`/`CANDIDATE_B` edges). The finalized schema, adopted throughout the main plan and fully
specified by `graph-dba`, has no `MatchSuggestion` node at all — the field in question is
`SAME_AS.confidence` on a property-bearing edge (`document-ingestion-graph.md` §1.5). Unlike the main
plan (which has an explicit, dated "Revision note" at §3.4 walking through exactly this change), the
ML note carries no such note anywhere — nothing marks these six references as superseded.

**Why it matters.** The plan's own §0 delegation table asserts "no schema-affecting divergence" for
this note, which is true of its *recommendations* (deterministic exact-match, precision floor,
deferred v2 embeddings) — but not of its *terminology*, which now names a node that doesn't exist. An
implementer building stage 4 against this note in isolation (as its own file, per the plan's
delegation model) would reasonably look for a `MatchSuggestion` node/label and not find one. This is
exactly the kind of cross-document leftover the brief asked me to check for.

**Suggested fix.** A mechanical terminology pass in `document-ingestion-ml.md`: `MatchSuggestion` →
"the `SAME_AS` edge" / `MatchSuggestion.confidence` → `SAME_AS.confidence` at the six sites above. No
change in substance — the recommendations themselves are schema-shape-agnostic. Route to
`data-scientist`.

---

### MAJOR — FR-8's auto-merge tier has no audit/discovery surface, and the ML note's own follow-up review isn't in the plan's stage table

**Where:** `docs/plans/document-ingestion.md` §3.5 (MCP/REST table), §4 Stage 4;
`docs/plans/document-ingestion-ml.md` §4.1, §6.

**Evidence.** The MCP/REST table (plan §3.5) has exactly one listing tool, `list_pending_matches`,
which (per `document-ingestion-graph.md` §1.7) filters strictly on `{status: 'pending'}`. There is no
tool/route to list or browse `status='confirmed', decidedBy='system'` edges — the auto-merged tier.
The ML note's own residual-risk argument for shipping an uncalibrated exact-match auto-merge default
leans on exactly this being correctable later: *"an incorrect `MatchSuggestion{status:'confirmed'}`
is itself a reviewable, correctable record via `reject_match`/`recheck_match`"* (§4.1) — but both of
those tools take a `match_id` the caller must already know, and nothing in the design gives an
operator a way to discover that id short of a raw Cypher query against the live graph. Separately,
the same note names a "firm follow-up, not optional" (§6): once stage 3 is live, `data-scientist`
should qualitatively review a sample of real extraction output before stage 4 (fusion) is trusted
with it. The plan's own stage table (§4) has no such checkpoint between Stage 3 and Stage 4 — the
only place this requirement exists is a sentence in the ML note pointing at "`teco`'s coordination,"
which is not itself part of this design set's gate.

**Why it matters.** This is the one fusion action with zero built-in human/agent review by design
(FR-8/AC-2). The two gaps compound: if extraction quality turns out to be poor (a real, named,
unmeasured risk per the ML note's own F2/F4), silent auto-merges could accumulate with no stage gate
to catch it early **and** no way to enumerate them for correction after the fact.

**Suggested fix.** (a) Add a `list_matches(status=..., limit=...)` (or equivalent status-filterable
listing) tool/route so `status='confirmed'` rows are discoverable, not just `pending` ones — cheap,
same index (`document-ingestion-graph.md` §1.5's `SAME_AS.status` index already supports it). (b)
Promote the ML note's stage-3→4 qualitative-review checkpoint from an aside into an explicit line in
the plan's own Stage 3/Stage 4 boundary (§4) — even framed as non-blocking/advisory, as the ML note
intends, it should be visible in the actual build sequence a coordinator gates against, not only in a
sibling document. Route (a) to `architect`+`graph-dba`, (b) to `architect` (plan edit) with
`data-scientist` confirming the framing is preserved.

---

### MINOR — a few completeness gaps, not blocking

- **Stage 3/4 file lists don't add a `background.py` failure-isolation wrapper**, unlike Stage 2's
  explicit `_safe_embed_chunk` (`document-ingestion.md` §4 Stage 2 vs. Stage 3/4). The §5 test-
  strategy bullet "Background-job failure isolation" expects the same `_safe_embed`-style
  try/except-log-never-raise discipline for extraction/fusion, but no `_safe_extract`/`_safe_fuse`
  (or equivalent) is named in either stage's file list. Easy to catch by convention, but worth adding
  explicitly since every other stage's file list is exhaustive.
- **`MAX_DOCUMENT_CHARS` (500,000) × `MAX_BATCH_SIZE` (20) compounds into a large background-LLM
  fan-out** — a single `ingest_documents` call at both caps produces on the order of 600 chunks/doc ×
  20 docs ≈ 12,000 extraction LLM calls queued from one MCP/REST call, with no rate-limiting/
  backpressure discussion anywhere (plan §3.5/§7). Each bound is individually reasonable and
  explicitly "implementer-tunable, not load-bearing" — but their product isn't examined, and this is
  exactly the write surface FR-5 opens to any connected agent. Worth one sentence acknowledging the
  compounded ceiling, even if the answer is "acceptable for v1."
- **`document-ingestion-graph.md` §3.3's `QUERIES.md` §10.3 (reverse-read) generalization is
  described in prose only** ("needs the mirror-image change… resolve the anchor via the same
  two-label `coalesce`") — unlike every other Cypher shape in this note, it is not written out or
  live-verified. Should be completed to the same standard before Stage 5 implementation.
- **§5's AC-5 test-strategy row doesn't explicitly call out extending `server/tests/test_provenance.py`.**
  The existing tests assert on `read_provenance`/`read_citing_answers`'s current positional-row-parsed
  dict shape (`{seedMsgId, text, role, score, rank}` — verified at `server/falkorchat/repository.py:581-620`
  and `server/tests/test_provenance.py:56-60` etc.); the generalized read query
  (`document-ingestion-graph.md` §3.3) adds/reorders columns (`seedKind`, `documentId`,
  `documentTitle`), so this file *will* need updating — the existing suite will fail loudly if it
  isn't, which makes this self-enforcing, but Stage 5's own file list (`document-ingestion.md` §4)
  doesn't name `test_provenance.py`, unlike every other stage's exhaustive file list.

---

## What's solid

- **Grounding is unusually rigorous.** Every factual claim I checked against the real files held up
  exactly as stated: the dormant `Document`/`Chunk`/`Entity` schema and its DDL
  (`docs/DESIGN.md` §5.1, `scripts/bootstrap_schema.sh:48-240`), the `Chunk`-"bootstrapped, never
  populated" note (`docs/QUERIES.md:472`), the §5.4 supernode-watch precedent, the K-042 "four closed
  kinds" crosswalk (`server/falkorchat/modelconfig.py:85-102`) and its additive-by-design comment, the
  §11 12.4 KB/vector RAM figure, and the exact `EMITTED` write/read shapes in `docs/QUERIES.md` §10.1-
  §10.3. `graph-dba`'s note goes further and live-verifies every Cypher shape it introduces (minus the
  one §10.3 gap above) against throwaway probe graphs, correctly never against `reference`/`ws:*`.
- **FR/AC coverage is complete and traceable.** Every FR-1..FR-14 and AC-1..AC-10 maps to a specific
  stage and mechanism, and the §5 test-strategy table gives each AC a concrete assertion shape at a
  sensible altitude.
- **The `SAME_AS`-edge-vs-`MatchSuggestion`-node reconciliation itself is a model of how this should
  work**: a real blocker was named in the first draft, resolved with a live measurement rather than an
  assumption (RAM is a wash — cited as measured, not guessed), decided on a defensible non-RAM axis
  (hop count + write-path fit), and the resulting planner trap (§1.4, bare-label-forces-full-scan) is
  threaded consistently through every subsequent query in the note.
- **The ML note's core methodological argument is sound**: refusing to ship an uncalibrated numeric
  threshold on the one unreviewed action (auto-merge), while treating the *deterministic* exact-match
  criterion as legitimately correct on its own terms rather than merely "safe by default," is the
  right framing, and the false-merge/recall asymmetric-metric design for the future v2 rung correctly
  mirrors the K-027 guard-judge precedent rather than reasoning from scratch.
- **Never-merge-nodes (§3.1's two-axes framing)** is a clean, low-risk design choice that makes FR-6
  true by construction and avoids the destructive-graph-surgery problem this codebase has no
  primitive for.

## Open questions (for the caller)

- Is the fusion race (blocker finding) better closed with an atomic combined query or an explicit
  processing-order guarantee? Both are workable; I don't have a strong preference, but it changes
  `create_or_reopen_match`'s call site shape either way, so it's worth `architect`+`graph-dba`
  agreeing before Stage 4 starts rather than discovering the choice mid-implementation.
- Should the stage-3→4 qualitative extraction-quality review (major finding above) become a hard gate
  (blocking `teco`'s Stage 4 kickoff) or stay advisory as `data-scientist` framed it? I've recommended
  making it *visible* in the plan either way; whether it also becomes *blocking* is a coordinator/
  stakeholder call, not mine to make unilaterally.

---

## Pass 2 (2026-08-22) — re-gate after the fix pass

**Scope.** Targeted re-review of the same three-document design set after `architect`/`graph-dba`/
`data-scientist` landed fixes for Pass 1's 1 blocker + 4 major + 4 minor findings. Re-read every
section Pass 1 cited plus the new/changed sections the fix pass added (`document-ingestion.md` §3.4
"Concurrency note" + §4 Stage 3/Stage 4 + the new Stage 3→4 checkpoint + §5's new concurrency-test
bullet; `document-ingestion-graph.md` §1.6's post-review scope note + §1.7's `list_matches` addition
+ new §1.8 + §2.2's post-review note + §2.3's post-review update + §3.4 (new); `document-ingestion-
ml.md` in full for the terminology pass). Grepped all three documents for every stale-reference
pattern Pass 1 named (`MatchSuggestion`, `CANDIDATE_A`/`CANDIDATE_B`, `find_exact_candidate`,
`create_entity(`, `classify(`) to check nothing was left behind or newly introduced. Cross-checked
two of the fix pass's own knowledge-base claims against the real file
(`claude/graph-dba/falkordb-quirks.md:53,249` — both new quirks the fix pass claims to have added
are genuinely there, not just asserted). Did not re-verify grounding claims Pass 1 already checked
and that are unchanged by this fix pass (e.g. `background.py`'s `_safe_embed` shape, the `EMITTED`
write/read baseline) — only the new/changed material.

**Verdict: approve.** All five Pass 1 findings are genuinely closed, not just touched, and the fix
pass — despite being large and coordinated across three documents with two fixes converging on the
same call site — introduced no new inconsistency I could find.

**CPG:** considered, not relevant — same reasoning as Pass 1; the brief again explicitly directed
skipping the CPG check for this design-consistency re-review.

### 1. Blocker — closed, and the closure is real, not a weaker proxy

`document-ingestion-graph.md` §1.8's `create_entity_with_auto_match` folds the candidate lookup, the
new entity's `CREATE`, and the conditional `SAME_AS` link into one `GRAPH.QUERY` — verified by reading
the Cypher itself, not just the prose around it. The ordering that matters (`OPTIONAL MATCH candidate`
→ `WITH candidate ORDER BY createdAt ASC LIMIT 1` → `CREATE (e:Entity {...})` → conditional `FOREACH`
link) forces the candidate row to be collapsed to at most one (real or `null`) *before* the new
entity's `CREATE` executes, which is exactly what closes the self-match risk. Traced through the
concurrency argument by hand: under FalkorDB/Redis's serialized command execution (the same platform
property every other atomic guarded write in this codebase already relies on, `falkor-chat/AGENTS.md`
rule 4 — not a new assumption introduced by this fix), two concurrent calls for the same
`(nameNormalized, type)` key can never interleave; whichever call's `GRAPH.QUERY` executes second
necessarily sees the first's committed `CREATE`. Walking the three-call sequence graph-dba's
behavioral test describes by hand (fresh pair → `exactMatched=false`; same pair again →
`exactMatched=true` pointing at the *first* call's id, never its own; a third entity with two now-
eligible candidates → picks the oldest, never itself or the middle one) confirms it converges to
exactly the star topology AC-2/AC-8 need (one hub entity, every later duplicate `SAME_AS`-linked to
it), not a chain or a missed pair.

On "is this a real check of the specific property that mattered, not a weaker proxy": yes, for the
half of the concern that's testable at design time. The behavioral test targets precisely the
self-visibility question the brief flagged as "the one thing worth not assuming" — does a query's own
`CREATE` leak into its own earlier `MATCH` within the same `GRAPH.QUERY` — and the `GRAPH.PROFILE`
read backs it with a structural argument (candidate resolution is a strictly earlier, separate
pipeline stage than the `Create` operator, not just favorable clause order). What the design-time
verification does *not* (and cannot) test is actual concurrent execution with real overlapping
threads — but that's correctly out of scope for a plan-gate design check; it's deferred to
implementation as its own named test (`document-ingestion.md` §5's new "Exact-tier auto-merge race"
bullet, explicit that it wants "real threads/tasks, not sequential calls dressed up as concurrent").
That's the right split of responsibility, not a gap.

The "no reopen branch" simplification (a brand-new entity can't already carry a `SAME_AS` edge, so
`create_entity_with_auto_match` doesn't need `create_or_reopen_match`'s reopen logic) is sound by
construction — traced it myself against §1.6's reopen branch and confirmed the premise: the reopen
case only exists for `create_or_reopen_match` because *both* its endpoints are pre-existing,
id-resolved entities that could carry history; a node `CREATE`d fresh inside the same query has none.

**FR-8/AC-2, FR-9/AC-3 coverage** (task item 5, spot-checked): AC-2's test row now targets
`create_entity_with_auto_match`'s output shape directly (`SAME_AS{status:'confirmed',
decidedBy:'system'}` immediately, no pending step) and gains a genuine new regression-test sibling for
the concurrent case. AC-3/FR-9 is explicitly and correctly stated as unaffected by the fix (§3.4's FR-9
bullet: "this tier's lookup stays a separate read followed by `create_or_reopen_match`... it does not
need the exact tier's atomicity fix") — traced the reasoning (a missed/duplicated fuzzy suggestion
under concurrent timing still lands in the reviewed `pending` queue either way, never silently defeats
a zero-review guarantee) and it holds: nothing on the fuzzy path is ever auto-confirmed, so there's no
analogous silent-miss failure mode to close. Both hold up.

### 2. The four majors — all genuinely resolved

- **§3.7 stale reference (`document-ingestion.md`):** the "defers" sentence is gone. §3.7 now states
  the actual resolution — bare-id `coalesce` against `Message.msgId`/`Chunk.chunkId`, both rejected
  alternatives (`kind` field, namespaced id scheme) named and reasoned against — and cites
  `document-ingestion-graph.md` §3.1-§3.3 exactly as suggested. Confirmed by direct read (lines
  400-427).
- **ML note's `MatchSuggestion` terminology:** grepped `document-ingestion-ml.md` for
  `MatchSuggestion`/`CANDIDATE_A`/`CANDIDATE_B` — zero hits. All six sites Pass 1 named (§1 framing,
  F6, §4.1, §4.3 heading + body, §5) now read `SAME_AS`/`SAME_AS.confidence`. Confirmed by direct read,
  not just the grep — the surrounding sentences still make sense with the substituted term (e.g. §4.3's
  heading is now "`SAME_AS.confidence` — a precision floor, not a second ML gate").
- **Auto-merge audit surface:** `list_matches(status=None, limit=50)` is in the plan's §3.5 MCP/REST
  table with its own explanatory bullet, and `document-ingestion-graph.md` §1.7 gives it real,
  live-verified Cypher — including catching and fixing a *second* trap along the way (`$status IS
  NULL OR ...` silently defeating the `SAME_AS.status` index even when bound), which I cross-checked
  is genuinely recorded in `claude/graph-dba/falkordb-quirks.md:249-260`, not just asserted in the plan
  doc.
- **Stage 3→4 checkpoint visibility:** `document-ingestion.md` now has a dedicated
  "### Checkpoint — extraction-quality qualitative review (advisory, not blocking)" section between
  Stage 3 and Stage 4 (lines 518-533), explicitly advisory (matching `data-scientist`'s own framing,
  which Pass 1 left as the caller's call, not mine) and explicitly visible in the build sequence a
  coordinator gates against, closing the exact gap Pass 1 named.

### 3. The four minors — all applied, none silently dropped

`_safe_extract`/`_safe_fuse` are now named in Stage 3's and Stage 4's file lists respectively; the
`MAX_DOCUMENT_CHARS`×`MAX_BATCH_SIZE` compounded-ceiling sentence is in §3.5's Bounds bullet; the
graph note's §3.3→§3.4 reverse-read is written out in full and live-verified (including, notably,
checking the one thing Pass 1's minor finding didn't get to — whether the forward-read's planner trap
also bites the reverse-read's unbound `a:Message` endpoint; it doesn't, and the note explains why);
and `server/tests/test_provenance.py` is now named explicitly in Stage 5's file list with the exact
shape delta it needs. All four confirmed by direct read at the cited locations.

### 4. Sweep for fix-pass-introduced inconsistency — none found

- **Stage 3 → Stage 4 handoff at the entity-creation call site (the specific concern the brief
  flagged):** reads coherently, and correctly, as one evolving call site, not a double-creation risk.
  Stage 3's file list wires `IngestionPipeline` to `repository.create_entity` (the plain-create
  primitive). Stage 4's file list is explicit that its wiring change *replaces* that call site with
  `create_entity_with_auto_match` — `document-ingestion.md:546-555`: "this single atomic call replaces
  stage 3's plain `create_entity` at this call site *and* the original `find_exact_candidate`
  pre-check." `document-ingestion-graph.md` §2.2 makes the same point from the other document
  ("`create_entity` ... is no longer the entity-creation call site `IngestionPipeline` uses ... remains
  correct and usable standalone"). No path through either document reads as "call `create_entity`,
  then separately also call `create_entity_with_auto_match`" — it's one call site, described at two
  points in the build timeline.
- **Fuzzy/suggested tier coherence around the changed exact tier:** unaffected as designed.
  `fusion.py`'s `find_fuzzy_candidates` is explicitly called out as "unchanged from the original draft"
  in both documents; the three-way `classify(exact, fuzzy)` helper Pass 1's version implied is
  explicitly retired in favor of a two-way `classify_fuzzy(fuzzy)`, with the collapse reasoned through
  in-line (`document-ingestion.md:541-545`) rather than left implicit. Grepped for a stray `classify(`
  call anywhere else in the three documents — the only hit is the one explaining the retirement, not a
  leftover call site.
- **Stray terminology sweep, full grep across all three documents** (not spot-checked): every
  remaining `MatchSuggestion`/`CANDIDATE_A`/`CANDIDATE_B`/`find_exact_candidate`/"three independent
  round trips" hit is in a passage explicitly narrating history — a revision note, a "post-review"
  section, or `document-ingestion-graph.md` §1's own retained comparison table (which correctly keeps
  the rejected node-model's real name for the reader comparing the two shapes, not a stale claim about
  the current design). None reads as a live claim about the current schema. No new stray reference was
  introduced by the fix pass itself.
- **DDL delta:** `document-ingestion-graph.md` §5 explicitly confirms the fix pass added no new DDL
  (both `create_entity_with_auto_match` and `list_matches` reuse pre-existing indexes) — checked
  against §1.5/§2.3's index list and it's accurate; nothing new needed and nothing new claimed.

### What's solid (unchanged from Pass 1, reconfirmed)

Everything Pass 1's "What's solid" section named still holds — the grounding discipline, the FR/AC
traceability, the `SAME_AS`-vs-node reconciliation process itself, the ML note's methodological
argument, and the never-merge-nodes design. The fix pass adds one more instance of the same discipline
worth naming on its own: `graph-dba`'s §1.8 "one thing worth not assuming" framing (verifying the
same-query `MATCH`-before-`CREATE` ordering live rather than trusting Cypher clause order, given this
build's own history of same-query-ordering surprises) is exactly the right level of paranoia for a
concurrency-critical fix, and it's backed by both a behavioral test and a structural `GRAPH.PROFILE`
argument rather than either alone.

### Open questions — both from Pass 1, now resolved by the fix pass

- Atomic query vs. processing-order guarantee: resolved — atomic query, reasoned through in
  `document-ingestion.md` §3.4's "Concurrency note" (no serialization primitive exists in this
  codebase's background-scheduling model to hang a processing-order guarantee off, so the atomic-query
  option was the only one that didn't require inventing new machinery).
- Hard gate vs. advisory for the stage-3→4 checkpoint: resolved as advisory, matching
  `data-scientist`'s original framing — a legitimate call within the range Pass 1 left open, now
  visible in the plan's own build sequence as asked.

No new open questions from this pass.
