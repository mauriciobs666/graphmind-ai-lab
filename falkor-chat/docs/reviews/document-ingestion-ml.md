# Ingestion Pipeline & Entity Fusion — extraction-quality qualitative review (Stage 3→4 checkpoint)

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** K-050 (M5)

Reviews real Stage 3 (`falkorchat.extraction`/`falkorchat.ingestion.IngestionPipeline`, commit
`dbc5241`) extraction output against the checkpoint I named in `docs/plans/document-ingestion-ml.md`
§6 ("firm follow-up, not optional") and that `docs/plans/document-ingestion.md`'s own Stage
3→4 checkpoint section (around line 518) restates. Advisory, not a gate — Stage 4 (fusion) does not
wait on this. Scope: qualitative under-/over-extraction read, plus an explicit answer to the named
cross-referencing/stub-repair trigger for the one-call-vs-two-stage fallback (§6's "first fallback
point, not a closed decision").

**Verdict up front:** the one-call-combined-entities+relationships design is **still the right v1
default** — do not trigger the two-stage fallback. The extraction call itself (JSON validity,
schema-shape, taxonomy adherence) is solid on real content: 27/28 real calls across this review
parsed and validated cleanly, only one failure and it was a pure transport error, not a model or
parser fault. But the review surfaced a **real, quantifiable, and more consequential problem than
the one the two-stage fallback would fix**: independent per-chunk extraction calls type the *same*
real-world entity inconsistently across mentions often enough to materially blunt FR-8's exact-match
auto-merge tier — 30% of repeated entity names in the README sample alone already carry a type
conflict. This is a genuine Stage 4 readiness data point (detailed in §4) — not a blocker, since
fusion's own correctness doesn't depend on extraction's output quality, but it changes what recall
to *expect* from the exact-match tier once Stage 4 ships, and it points at a cheaper, more targeted
fix than the two-stage call (§5).

## 1. What I ingested and why

Two real documents, 28 chunks reviewed total (target was 20-30):

- **`falkor-chat/README.md`**, ingested whole via the live `POST /documents` → background
  embed+extract pipeline against a locally running server (`FALKORCHAT_WS_ID=acme`) and a local LM
  Studio serving `qwen/qwen3-4b-2507` (the `extraction` kind's configured default,
  `config/models.json`) — **18/18 chunks**, chosen because it's Organization/Product/Concept-heavy
  real technical prose (the plan's own suggested source), plus non-prose content (bash blocks, a
  markdown table, a directory-tree listing) that exercises FR-1's non-prose-format claim at
  chunk-boundary granularity.
- **Wikipedia's "Apollo 11" article** (`en.wikipedia.org/wiki/Apollo_11`, fetched via the plaintext
  extract API, first ~11,000 chars = 14 chunks after `chunking.split_into_chunks`), chosen for real
  Person/Location/Event/Organization density the README alone doesn't provide — **10/14 chunks**
  sampled (seq 0-6, 8, 10, 13), covering the mission summary, Cold War/Space Race background,
  Kennedy's Rice University speech, and prime-crew personnel.

**Method deviation, disclosed:** README ran the full live pipeline (real HTTP ingestion, real
`BackgroundTasks`, real graph writes) end to end — this is the primary "Stage 3 is live
end-to-end" evidence and the source of the type-inconsistency quantification in §4. Apollo 11 also
started this way, but the local LM Studio instance became unstable under **concurrent** background
load from two independently-scheduled per-chunk tasks resolving two different models (`extraction`'s
chat model and `embedding`'s separate embedding model) — HTTP 400 `Engine protocol startup was
aborted`, then `terminated` errors that persisted for several minutes (full trace in the kaizen entry
I filed, `kaizen_team`). Rather than burn further time fighting infra flake, I switched to calling
`falkorchat.extraction.extract()` directly (same prompt, same parser, same model, same `temperature:
0`, same 180s timeout) via `llm.OpenAICompatibleLLM` pointed at LM Studio, one call at a time, with no
concurrent embed calls competing for the engine — same code path, decoupled only from the
FastAPI-background-task/embedding-contention machinery, not from the extraction logic under review.
Every Apollo result below is real model output through the real `extraction.py`/`ingestion.py`
validation and stub-repair code, just not additionally round-tripped through a graph write (the
graph-write mechanics were already proven correct by README's full run). I also had to override
`FALKORCHAT_OPENCODE_CONFIG` to a scratch copy pointing `baseURL` at `http://localhost:1234`
instead of the shared config's `http://192.168.0.69:1234`, which was unreachable in this session —
both this and the concurrent-serving instability are filed as `kaizen_team` entries, not edited into
any shared config file.

## 2. Under-extraction

**Clear pattern: code/config-block-heavy chunks reliably extract nothing, even when they contain
real, taxonomy-eligible entities.** README chunks 1, 3, 6, and 15 all returned valid, empty
`{"entities": [], "relationships": []}` (not parse failures — genuinely empty, valid results). The
cleanest example, chunk 15 (full text, `ws:acme` `Chunk.text`):

> "...It talks to the REST API on the same origin (no CORS). The browser and MCP front doors share
> one `services.py`. The AI responder is **off by default** so imports stay network-free; set
> `FALKORCHAT_ENABLE_AGENT=1` (and have LM Studio up) to wire the live embedder + LLM + `@mention`
> responder..."

This is flowing prose, not a code block, and it names **LM Studio** — a real, unambiguous named
product squarely inside the closed taxonomy's `Product` definition — plus `REST API` and `MCP`.
Zero entities were extracted. Chunks 3 and 6 are more defensible misses (bash command blocks with
env-var flags and script names, e.g. `FALKORCHAT_ENABLE_AGENT=0 ./scripts/start_server.sh`) but show
the same shape: the model reliably treats code/shell content as non-extractable even when it names
real tools. This is a genuine, reproducible under-extraction pattern, not a one-off — 4 of 18 README
chunks (22%) hit it. It matters specifically for this codebase because the plan's own FR-1 rationale
for choosing LLM extraction over NER was format breadth (Markdown, CSV, JSON, Mermaid); this shows
the model's prose bias reasserting itself *within* a mixed-format document, not just across whole-
document formats.

No comparable under-extraction was found in the Apollo prose sample — entity coverage there was
consistently thorough (crew names, module names, dates, locations, organizations all captured; see
§3/§4 for what went wrong instead).

## 3. Over-extraction

Three distinct shapes, not one:

**(a) Rhetorical/analogical language reified as fact.** Apollo chunk 8 is Kennedy's Rice University
speech, including the rhetorical volley "why climb the highest mountain? Why... fly the Atlantic? Why
does Rice play Texas?" The model extracted `"Rice play Texas"` and `"Atlantic flight"` as `Event`
entities and `"highest mountain"` as a `Concept`, then fabricated relationships like `"Rice play
Texas" —is referenced in→ "the Moon mission"` and `"Atlantic flight" —is referenced in→ "the Moon
mission"`. These are rhetorical analogies inside a quoted speech, not facts about the Moon mission —
the model didn't distinguish quoted-speech rhetoric from literal assertion.

**(b) Non-prose structural content over-extracted as low-signal nodes.** README chunk 12 (34
entities from 996 chars) is a directory-tree listing (`scripts/bootstrap_schema.sh  # create indexes
+ constraints...`). Roughly half the extracted entities are real strings from the tree but not
meaningful real-world things: `config`, `constraints`, `db`, `indexes`, `repository`, `seed`,
`services`, `uvicorn`, `venv`. Not hallucinated (every string is literally present) but low-value —
a knowledge graph doesn't benefit from an `Entity` node for the word "seed." This is a distinct
failure mode from (a): faithful transcription of structural content, not fabrication, but a form of
extraction the taxonomy/prompt doesn't currently guard against.

**(c) A garbled duplicate within one call.** Apollo chunk 4 extracted both `"Sputnik 1"` (`Product`,
correct) and `"Sput-1"` (`Other`) as separate entities, linked by the nonsensical relationship
`"Sputnik 1" —caused→ "Sput-1"`. `"Sput-1"` does not appear anywhere in the source text — this is a
low-frequency but real generation artifact (the model contradicting itself within a single reply),
worth naming even though it's rare in this sample (1 instance across 28 chunks).

## 4. Cross-referencing / stub-repair — the named trigger, checked directly

This is the specific question the ML note flagged as the two-stage fallback's trigger (§6: "the
local 4B struggling specifically with cross-referencing... ignoring §3.2's stub-repair"). Direct
finding: **the stub-repair mechanism itself works exactly as designed** — every relationship
endpoint not found in the same call's `entities` list correctly got a stub. The real problem is one
level up, and it's more consequential than stub-repair not firing:

**The model frequently refers to the same real-world entity by two different strings *within the
same call* — full name in `entities`, a shortened/aliased form in a `relationships` endpoint — and
exact-normalized-name matching (by design, per the ML note's own scoping) cannot bridge that, so
stub-repair creates a spurious duplicate instead of resolving to the existing entity:**

- Apollo chunk 5: `entities` lists `"National Aeronautics and Space Administration (NASA)"` typed
  `Organization`. A relationship references `"NASA"` as its object — normalized `"nasa"` does not
  match normalized `"national aeronautics and space administration (nasa)"` — so stub-repair
  synthesizes a **second** entity, `{"name": "NASA", "type": "Other"}`, losing the correct type.
- Apollo chunk 13: `entities` lists `"Jim Lovell"` (`Person`). A later relationship references
  `"Lovell"` as its subject — same exact-match miss — stub-repair synthesizes a duplicate
  `{"name": "Lovell", "type": "Other"}`.

Both are clean, independently-occurring instances of the identical failure shape (not one lucky/
unlucky chunk) — this is the real cross-referencing weakness, not "the model forgets to list an
entity at all" (the case §3.2 designed stub-repair for). A within-call alias mismatch is a
narrower, more mechanical problem than general relationship-quality, and — critically for the
fallback decision — **a two-stage call wouldn't reliably fix it either**: a second call conditioned
on the entities list would still need to name entities as free text, and nothing in that design
forces it to reuse the identical string. See §5 for a cheaper, more targeted fix.

**A separate, real relationship-accuracy problem also showed up in Apollo chunk 13**, distinct from
the alias issue: the source text describes the *initial backup-crew assignment for Apollo 9*
(Armstrong/Lovell/Aldrin), but the model's relationships state `"Jim Lovell" —was assigned as command
module pilot to→ "Apollo 11"` — factually wrong (Apollo 11's actual CMP was Michael Collins; Lovell
never flew on Apollo 11). This looks like the model conflating "this crew was later expected to crew
Apollo 11 under the normal rotation scheme" (stated two sentences later) with the literal Apollo 9
backup-crew assignment sentence, producing a relationship not supported by either sentence read in
isolation. **Caveat, not an excuse:** chunk 13 is truncated mid-sentence at the 1000-char boundary
("Lovell took his place on the Apollo 8 crew, a[...]"), cutting off the disambiguating context the
next chunk would supply (Collins recovering and moving to Apollo 11). Hard character-boundary
chunking measurably compounds this kind of cross-reference error on narratively dense, chronologically
entangled source text — worth a one-line flag for whoever next tunes chunking strategy, not a
verdict on the extraction call itself.

**The consequential finding: entity-type inconsistency across independent per-chunk calls, at scale.**
Because each chunk's extraction call has no memory of any other chunk's call, the same real-world
entity can legitimately get a different `type` label each time it's mentioned — the ML note's F3.1/§4.1
risk, stated as a plausible concern. Checked directly against the live README graph (`ws:acme`, 523
`Entity` nodes from the 18-chunk run, 395 distinct `nameNormalized` values):

| Metric | Value |
|---|---|
| Distinct normalized entity names | 395 |
| Names mentioned more than once (fusion candidates) | 84 |
| Of those, names with **>1 distinct `type` across mentions** | **25 (30%)** |

Concrete examples (`nameNormalized` → `types seen`, mention count): `falkordb` → `[Product,
Organization]` (9 mentions), `lm studio` → `[Organization, Product]` (3), `united states` →
`[Organization, Other]` (4), `nasa` → `[Organization, Other]` (4, the same alias-mismatch mechanism
as §4's Apollo case, now visible at population scale), `kennedy` → `[Person, Other]` (3),
`falkorchat` → `[Organization, Other, ...]` (typed `Organization` for what is this repo's own
project name — a clear misclassification on its own, and doubly so once it collides with a second,
differently-typed mention).

**Why this matters more than the alias/stub issue:** FR-8's exact-match auto-merge tier is
gated on `nameNormalized` **and** identical `type` (plan §3.4, ML note §3.1's whole rationale for a
closed taxonomy). A same-name pair that disagrees on type will **not** auto-merge, and per the
plan's design (§3.4), it also won't fall through to the fuzzy/suggested tier automatically, because
that tier is triggered off `exactMatched=false` from the exact-lookup call, which itself doesn't
condition on type at all in the way I'd want checked — this is a question for `graph-dba`'s Stage 4
implementation, not something I resolved here, but the practical upshot either way is that the exact
tier's *effective* recall on real content will run measurably below what a same-name-only match rate
would suggest. This isn't a new axis the plan hadn't considered (§4.1 already accepted a narrower,
same-name-same-type-different-entity risk) — it's the **mirror-image** gap: same-entity-different-
type pairs that the plan's design assumed would be rare enough not to name, and this sample shows
happening on 30% of repeat mentions.

## 5. Recommendation

**Do not switch to the two-stage entities-then-relationships fallback.** The evidence for triggering
it (§6's named condition) was "the local 4B struggling with cross-referencing, ignoring stub-repair."
Stub-repair is not being ignored — it fires correctly every time it's needed. The actual weak point
(within-call name aliasing) is not clearly fixed by a second call, and the *more* consequential
finding (cross-call type inconsistency) is structurally untouched by the two-stage design entirely —
splitting one chunk's call into two calls does nothing for entities that disagree across *different*
chunks' calls, which is where 100% of the 30% figure lives. Paying 2x background LLM cost per chunk
(FR-11 relevant under bulk ingestion) for a fallback that doesn't address the finding that actually
matters is not a good trade.

**Instead, two cheap, deterministic, non-LLM follow-ups worth `architect`/`graph-dba` scoping for
Stage 4 (not blocking, not sized as a full design here):**

1. **Widen stub-repair's own-call matching from exact-normalized-only to a same-call substring/
   containment check** before synthesizing a stub — e.g., if a relationship endpoint's normalized
   name is a token-bounded substring of (or contains) an existing same-call entity's normalized name,
   resolve to that entity instead of creating a duplicate. This directly fixes both §4 alias
   instances (`"nasa"` ⊂ `"national aeronautics and space administration (nasa)"`; `"lovell"` ⊂
   `"jim lovell"`) with no extra LLM call, no new infrastructure, and no change to the extraction
   prompt. Scope this to *within one chunk's own entities list* only, exactly like today's
   stub-repair — do not widen it to cross-chunk matching, which is fusion's job.
2. **Flag the type-inconsistency finding to whoever builds/tunes Stage 4's exact-match tier** as a
   real, measured recall ceiling (not a hypothetical one) — 30% of repeat-mention names in a modest
   18-chunk sample. This is not something to fix inside extraction (the taxonomy is already as
   guardrailed as F3.1 reasoned it should be for v1); it's information the fuzzy/suggested tier's
   design and any post-launch metrics should account for, since a same-name-different-type pair is
   exactly the kind of case that tier could plausibly catch if it's willing to relax the type-match
   requirement for a *suggestion* (never for auto-merge) — a design question for `graph-dba`/
   `architect`, not a decision I'm making here.

## 6. Does this change Stage 4 readiness?

**No change to whether Stage 4 can proceed** — this was always advisory, and nothing here says
fusion's own logic is unsound; fusion still does exactly what it's designed to do with whatever
extraction hands it. **It does change what recall to expect from Stage 4's headline mechanism once
it ships**: the exact-match auto-merge tier will visibly under-perform a naive "same name = same
type most of the time" intuition, on real content, from day one — not a corner case to discover
later. Whoever owns Stage 4's acceptance criteria (`qa-engineer`/`architect`) should not be surprised
when a real ingested corpus shows a chunk of same-name entities sitting unmerged; that's this
finding showing up in production, not a regression. I'm not proposing a numeric target here — no
labeled data exists to set one, same reasoning as §4.2 of the original method note — just flagging
that the gap is real, measured, and worth a line in the eventual test plan the same way §4.1's
narrower risk already earned one.

## 7. Sample-size honesty

28 chunks, 2 documents, one model (`qwen/qwen3-4b-2507`, `temperature=0`), one qualitative pass —
this is exactly the "quick, not exhaustive" read the checkpoint asked for, not a calibration. The
30% type-inconsistency figure is a real, directly-computed statistic from real graph data, not an
estimate, but it's computed over 84 repeat-name pairs from a single 18-chunk document — it should be
read as "this is a real, non-trivial rate observed in this sample," not as a number to gate anything
on without a larger, more representative pull once real ingestion traffic accumulates. No golden set,
no ground truth, no inter-rater comparison was constructed here — under-/over-extraction judgments
above are mine, made by reading source text against extracted output directly, the same posture the
checkpoint asked for.

## 8. Environment notes (not methodology, but relevant to reproducing this)

- The local server was started with `FALKORCHAT_OPENCODE_CONFIG` overridden to a scratch copy of
  `~/.config/opencode/opencode.json` with `lmstudio.options.baseURL` pointed at `http://localhost:1234`
  instead of the shared file's `http://192.168.0.69:1234`, which was unreachable this session — no
  shared config file was edited. Both this and the concurrent-model-serving instability observed
  under combined embed+extract background load are filed as `kaizen_team` entries for whoever next
  runs a similar live-ingestion session against local LM Studio.
- The `uvicorn` server I started for this review is **stopped** as of this writing (clean shutdown,
  no orphaned process) — the ingested `ws:acme` graph data (README's 18 chunks, 523 `Entity` nodes,
  371 `RELATES_TO` edges) is left in place for anyone who wants to inspect it further or wants it as
  a starting corpus for Stage 4 development/testing. Restart with
  `FALKORCHAT_OPENCODE_CONFIG=<scratch-config> ./scripts/start_server.sh` if `192.168.0.69:1234`
  is still unreachable when next needed.
