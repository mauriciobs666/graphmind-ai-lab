# NL-query-generation second-dataset corpus — content review (U29-gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-055 (M6)

**CPG:** considered, not relevant — this is a content/data review of a ~470-line standalone seed
script and its live-graph output, not an impact analysis across the server codebase; the script's
few call sites into `falkorchat.db`/`Repository` were read directly and are small enough that a
CPG traversal adds nothing a direct read doesn't already give.

## Scope & verdict

Reviewed: `falkor-chat/scripts/seed_nlq_eval_corpus.py` and its `.sh` wrapper, the provenance
sidecar `falkor-chat/server/tests/eval/nlq_corpus_provenance.json`, and the live `ws:nlq-eval`
graph (queried directly via `mcp__cypher__query`), against `docs/plans/workflow-nl-query-generation-ml.md`
§4's stated corpus requirements. This is the content-review gate on U29's corpus, before U29b
authors golden question/answer pairs against it. Did not review U29b (not yet started) or the
`server/tests/eval/` harness modules for the unrelated GraphRAG eval corpus.

**Verdict: approve with suggestions.** All eight checklist items in the brief hold on live
verification (taxonomy variety, predicate variety, the conflicting-fact pair, the absent fact, doc
concreteness, no leakage, idempotency soundness, and the two disclosed garbled entities). One
finding below (#1) surfaces a **third, undisclosed** extraction defect of the same class as the
two the delegate flagged — it does not invalidate the corpus, but it must be handed to U29b
explicitly, not left for that unit to rediscover.

## Findings

### 1. (Major) A third extraction defect exists that the delegate did not disclose — a reversed relationship direction inside one of the two FR-6 conflict documents

Live query: `NLQ-EVAL-06` (`ceb295aa…`) produced `Marlowe Robotics -[acquired]-> Brightline
Systems` (correct — matches the source text "Marlowe Robotics acquired Brightline Systems").
`NLQ-EVAL-08` (`35474077…`) — one of the two deliberate conflicting-fact documents — produced
`Marlowe Robotics -[acquired by]-> Brightline Systems`, which reads as the *opposite* direction
(read literally, "X acquired by Y" means Y acquired X). This is a genuine extraction error, not
just the two disclosed issues (mangled date, inconsistent job-title typing) — and it sits inside
the same document that carries the FR-6 conflicting-fact pair, so a golden-pair author working
that document for a relationship-traversal question ("who acquired Brightline Systems") could
anchor on the reversed edge and produce a wrong or self-contradicting golden answer, or could
correctly ignore it but only by independently noticing the direction issue themselves.

Separately, `NLQ-EVAL-11` (NovaGrid) produced a second garbled date entity beyond the disclosed
one: `NovaGrid -[was released in]-> "September than"` (should read "September 2025"; a correct
`"September 2025"` `Event` entity also exists in the graph, unlinked to this edge) — structurally
identical to the disclosed `"January 1026"` garble in `NLQ-EVAL-05`.

**Suggested action:** before U29b starts, the coordinator should pass forward a corrected,
complete list of known-garbled/unreliable extraction artifacts — not just the two the delegate
self-disclosed. Concretely: *(a)* the acquisition direction — only `NLQ-EVAL-06`'s `acquired` edge
is reliable for a Brightline-acquisition golden pair, never `NLQ-EVAL-08`'s `acquired by`; *(b)*
two garbled date entities, `"January 1026"` (`NLQ-EVAL-05`, Series B) and `"September than"`
(`NLQ-EVAL-11`, NovaGrid release) — both have a correctly-extracted sibling elsewhere in the graph
(`"January 2026"` via `Marlowe Robotics -[received funding in]->`, and the standalone `"September
2025"` `Event` entity) that should be used instead; *(c)* the disclosed `"Chief Technology
Officer"` type split (`Concept` in `NLQ-EVAL-01`, `Other` in `NLQ-EVAL-12`, no `SAME_AS` edge
between them — confirmed live) affects any leadership-succession question ("who is the current
CTO") beyond just "avoid precise pairs against these two nodes."

### 2. (Minor) The script's own conflicting-fact self-check is looser than it needs to be

`main()`'s `conflict_holds` computation (`seed_nlq_eval_corpus.py:411-421`) collects **every**
`RELATES_TO` object where `subject == "Marlowe Robotics"`, not specifically the `"has"`-labeled
edge FR-6 cares about. In this run it happens to be correct because `NLQ-EVAL-08` also carries the
unrelated `acquired by` edge (finding #1) — that edge alone would make `conflict_objects_b` differ
from `conflict_objects_a` even if the actual `"has"` edges had been merged into one. Live-verified
independently that the real `"has"` edges do differ (`62` vs `140 employees`, both present), so the
data is fine, but the check that's supposed to catch a future regression would not reliably catch
one. Suggest narrowing the filter to `r["predicate"] == "has"` (or asserting exactly one such edge
per side) so a future re-run's self-check means what it claims.

### 3. (Nit) No incremental provenance write; unhandled network/HTTP errors mid-loop

`_http_json` (`seed_nlq_eval_corpus.py:242-249`) has no try/except around `urlopen`, and the
provenance sidecar is written once at the very end of `main()` (`:443-444`), after every document
in the loop has been processed. A transient failure on, say, document 8 exits with a raw traceback
and writes no sidecar at all, even though documents 1-7 already reached a terminal state in the
graph. Re-running is cheap (idempotency skips the already-terminal documents), so this is not a
correctness risk, just a rough edge — worth a `try/except` around the per-document body that logs
and continues, or a friendlier error message, if this script gets reused for a larger corpus later.

### 4. (Nit) `'failed'` is treated identically to `'ready'` as a skip condition

`seed_nlq_eval_corpus.py:359` — a document that failed extraction on a prior run is skipped on
every subsequent run exactly like a successfully-`ready` one, silently staying failed forever
unless `FORCE_REINGEST=1`. The end-of-run `WARNING` (`:455-460`) does surface this, so it's
visible, not silent-silent — but consider a louder signal (non-zero exit code) so a CI/automation
context can't mistake "ran without exception" for "corpus complete." Moot for this delivery: the
live run reached 12/12 `ready`.

## What's solid (live-verified, not just read off the provenance file)

- **Entity.type variety**: all 7 taxonomy values represented and matching the provenance file
  exactly on direct query — `Organization 17, Other 11, Location 11, Event 8, Product 7, Person 5,
  Concept 3`.
- **Predicate variety**: 44 `RELATES_TO` edges across 38 distinct predicate labels, live-counted —
  a genuine mix (founded, headquartered in, launched, presented at, partnered with, invested,
  acquired, has, researches, competes with, develops, promoted to, joined as, etc.), not a narrow
  set.
- **Conflicting-fact pair**: `Marlowe Robotics -[has]-> "62"` (`NLQ-EVAL-07`) and `Marlowe Robotics
  -[has]-> "140 employees"` (`NLQ-EVAL-08`) both persist as distinct edges — confirmed by direct
  query, not the JSON's word.
- **Absent fact**: no `Entity.name`, `RELATES_TO.label`, or raw `Chunk.text` anywhere in the live
  graph contains `revenue`/`arr`/`annual recurring` — checked at all three levels (broader than the
  script's own pre-ingestion source-text-only grep, which only covers the corpus's Python string
  literals).
- **Idempotency design**: the title-based existence check, terminal-status skip, and
  `FORCE_REINGEST` escape hatch are sound and clearly documented; the sequential post-then-poll
  loop (the fix for the LM-Studio model-swap-thrashing defect) is correctly implemented — each
  document is polled to a terminal status before the next is posted, confirmed by reading the loop
  body. Kaizen entry for the underlying LM Studio fact is present and accurately described.
- **Concreteness**: real named entities, dates, and numbers throughout ($18M Series B, 800 kg
  payload, 62 vs 140 employees) — enough for a golden-pair author to write precise, unambiguous
  expected answers.
- **No answer-key leakage risk**: `grep -rl nlq server/tests/ scripts/` finds only the corpus
  script and its provenance sidecar — no golden-set file exists anywhere yet, confirming the
  sequencing (corpus before golden pairs) was not violated.
- Script structure/conventions (env-var overrides, `.sh` wrapper health-check-before-exec pattern)
  mirror `seed_eval_corpus.py`/`.sh` closely, as intended.

## Open questions

None that block U29b from starting — finding #1 is guidance to carry forward, not a corpus defect
requiring re-ingestion or a script change.
