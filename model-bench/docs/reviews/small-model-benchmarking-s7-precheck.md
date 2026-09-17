# Small-model-benchmarking S7 — `chat-responder` golden items, agent pre-check (FR-19 process step, mirroring S6)

> **Status:** active · **Owner:** `analyst` · **Tracks:** U171 (S7)

## Scope & verdict

Reviewed `model-bench/packs/chat-responder-grounded-answers/items.jsonl` (30 items) and
`PROVENANCE.md` against `docs/plans/small-model-benchmarking-s7-spec.md` §3.1 (item shape), §2.6
(context-copying rule), §3.7 (`PROVENANCE.md` shape), and `pack.json`'s `format` defaults —
structural precedent from `model-bench/docs/reviews/small-model-benchmarking-s6-precheck.md`.
Not re-derived (already confirmed by `teco`'s U169 mechanical pass, per the brief): `mustContain`
substring presence, question-not-verbatim-in-corpus, `mustAbstain` context genuinely lacking the
fact, `iter_items()`/`validate_pack()` success. I did re-run `validate_pack(load_pack(...))` live
(`[]`, 30 items) and cross-read every one of the 30 items against its own topic's full corpus
thread (all 121 rows, all 12 topics, `packs/embedder-graphrag-retrieval/corpus.jsonl`) and against
`PROVENANCE.md`'s `basedOn` claim, rather than trusting the containment check alone — this is what
the item-quality/fairness judgment below is built on.

I did not author any item and did not touch `provenance.verifiedBy` (still empty on all 30 rows,
confirmed) — that field stays reserved for the stakeholder's own FR-19 sign-off, never this
pre-check's to fill, exactly as `teco` held twice during S6.

**Verdict: approve with suggestions.** 30 of 30 items are sound — no blocker, no major. Two minor
findings (a giveaway risk on two single-value items, an easy-by-construction item) and two nits
(a topic-spread observation, a coincidental duplicate literal). The set's design quality is
genuinely strong: the cross-item value pairing (a value that is the correct answer in one item and
the planted fabrication guard in another) is deliberate and consistent throughout, and every
abstain item is a real confabulation trap (a plausible-sounding fact a model is likely to invent)
rather than a trivial "nothing here" case.

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(checked live via `mcp__cypher__query` `GRAPHS`); this is a code-level task in a component with no
CPG, so "considered, not relevant" applies rather than "not applicable."

## Findings

### Minor — cr-12 and cr-13 carry a single `mustContain` value that is a plausible "textbook default," guessable without reading the passage

- **cr-12**: "Where are permanently-failed push-notification deliveries logged for manual
  review?" → `mustContain: ["dead-letter queue"]`. "Dead-letter queue" is the standard,
  widely-known term for exactly this kind of failure sink in any messaging/retry system — a model
  could answer correctly from general knowledge of the question's shape alone, without reading
  `context`.
- **cr-13**: "How long do access tokens stay valid before they expire, per the OAuth bug report?"
  → `mustContain: ["15 minutes"]`. 15-minute access-token TTL is an extremely common OAuth
  tutorial/default value; a model could pattern-match to it without genuine grounding.

Both are single-value items (no second, less-guessable value in the same item to force real
comprehension, unlike e.g. cr-11's `["4 retries", "24 seconds"]` or cr-25's `["90 days", "91
days"]`, where the second value is a specific/derived figure that is not guessable). Diluted
across a 30-item aggregate `groundingRate`, the distortion risk is small, so this does not rise to
major. **Suggested improvement**: either accept as-is (two items out of 30 is a bounded risk), or
strengthen one or both by adding a second, non-generic required value from the same passage (e.g.
cr-13 could also require the immediately adjacent fact "the refresh call is failing" or similarly
specific wording) so `checklist_pass` needs more than a lucky guess. Not a fix I'd block the stage
on.

### Minor — cr-21's context is a single sentence that already contains all three required literals verbatim

`cr-21` ("Which two initiatives did the team prioritize for Q3, and which one got pushed to
Q4?") uses `context: ["Let's prioritize bulk-import and analytics-export for Q3, and push the
mobile redesign to Q4 with a design-only spike in the last 4 weeks of Q3."]` — one passage that
already names `bulk-import`, `analytics-export`, and `mobile redesign` (all three `mustContain`
values) in a single, near-restatable sentence. Not wrong (the question is still clear and the
fact is genuinely there), but it is the easiest item in the set — closer to "copy this sentence"
than "read two passages and synthesize." **Suggested improvement**: none required; flagged so the
topic/difficulty-spread judgment (§3.1's "reasonable, representative sample") accounts for it —
worth a one-line note in the stage's own test report rather than a re-author.

### Nit — 4 of 12 topics (notification-retry, hr-onboarding, q3-roadmap, on-call-rotation) contribute zero `mustAbstain` items

The 8 abstain items span 8 of 12 topics; the other 4 topics' 2-3 items are all answerable. With
2-3 items per topic this is close to unavoidable (can't split an abstain item across a 2-item
topic without dropping an answerable one), and the pack-level abstain ratio (8/30 ≈ 27%) is a
healthy minority per §3.1's own target — not a real skew, just worth naming so it isn't mistaken
for an oversight later.

### Nit — the literal `"90 days"` is reused as a `mustContain` value in two unrelated items (cr-08, cr-25)

`cr-08` (partition archival) and `cr-25` (Finance's point-in-time-recovery requirement) both
require the exact string `"90 days"` for two different facts in two different topics. This is
harmless — each item is scored independently against its own reply and own context — but worth a
one-line note since a future author extending this pack could accidentally read it as evidence of
a copy-paste error rather than genuine coincidence (both topics independently land on 90 days).

## What's solid

- **Every `mustAbstain: true` item is a real confabulation trap, not a trivial absence.** Each
  asks for something a language model is specifically prone to inventing under this exact
  question shape — a named individual (cr-06), a dollar cost (cr-03, cr-09), a percentage of a
  user population (cr-15), a vendor/company name (cr-24, cr-26), a specific customer identity
  (cr-28), or a second item in an enumerated list the passage never completes (cr-18's "besides
  SOC 2, which other framework"). None are "obviously nothing here" — all are exactly the shape
  a weaker model hallucinates against.
- **The `mustNotContain` distractors are deliberately cross-wired between items, not
  independently invented.** Multiple pairs use the *correct* answer to one item as the *forbidden*
  fabrication guard in a different, topic-adjacent item: `"3%"`/`"2.5%"` (cr-01 ↔ cr-10),
  `"v3.8"`/`"v2.6"` (cr-02 ↔ cr-05), `"15 minutes"`/`"30 minutes"` (cr-13 ↔ cr-17), `"90
  days"`/`"91 days"` and `"5 retries"` (a same-topic superseded proposal never in the item's own
  context, cr-11). This is a materially stronger design than independently-invented distractors —
  every guard is a real value a model could plausibly confuse from an adjacent fact, verified by
  reading each pair's own context directly.
- **cr-07 and cr-25's same-context attribution traps are genuinely fair.** Both place two
  numerically close, topically related values in the item's own `context` (Postgres 42,000 vs.
  DynamoDB 45,000; Finance's 90-day requirement vs. the policy's 91-day actual coverage) and
  require the model to attribute the right figure to the right referent — read directly against
  the corpus, both are unambiguous to a careful reader, so this is a stronger test than a bug.
- **`PROVENANCE.md`'s `basedOn` column checks out.** I cross-read all 30 rows against the actual
  corpus text each item's `context` copies from — every `basedOn` docId genuinely matches what the
  item draws from, and `corpusVersion: "1.1.0"` in every item matches
  `packs/embedder-graphrag-retrieval/pack.json`'s real, current `packVersion`.
  `verifiedBy`/`verifiedAt` are empty throughout, as required at this stage.
  `validate_pack(load_pack(...))` returns `[]` and `iter_items()` yields all 30 rows, re-confirmed
  live.
- **Topic spread matches §3.1's target**: 6 topics contribute 3 items each, 6 contribute 2 each
  (18 + 12 = 30), no single corpus thread dominates.

## Open questions

None — nothing here needs the stakeholder's or `teco`'s input beyond the two minor findings above,
which are suggestions, not blockers.
