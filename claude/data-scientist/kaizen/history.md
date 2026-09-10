# Kaizen — Change History: data-scientist

> Dated log of actual changes to the `data-scientist` agent. Most recent first.

## 2026-09-10 — Distillation U56: a fresh 2-entry chunk, 1 promoted to the knowledge base, 1 discarded as already captured by an in-flight, out-of-scope plan doc

- **What:** `cobb` distilled `data-scientist`'s newly-refilled 2-entry `kaizen_team` chunk
  (`a1f2c3d4…`, `b2e3d4c5…`, both 2026-09-10 — `data-scientist` was fully drained at U50/`40b0daf`
  and this is a distinct, later capture, not a re-run). Both entries were captured while
  `data-scientist` was writing `claude/docs/plans/agent-knowledge-base-strategy-ml.md` (K-030),
  which is explicitly out of scope for this pass to touch (a separate concurrent session is
  progressing it) — that plan doc turned out to already contain a fuller, more rigorous treatment
  of both entries' substance.
- **`a1f2c3d4…` (falkor-chat's `OpenAICompatibleEmbedder.embed()` sends unprefixed text for both
  query and document embedding; Qwen3-Embedding-0.6B's model card documents an asymmetric
  Instruct/Query convention) — re-derived independently, confirmed true, then promoted in
  generalized form.** Read `falkor-chat/server/falkorchat/embedding.py` in full: `embed()` takes
  one `text` argument with no query/document branch, and every call site
  (`embedding.py:207`, `tools.py:357`, `responder.py:103`, `services.py:1253`) calls the same
  method identically for stored content and for retrieval queries — confirmed symmetric, not
  merely as the entry paraphrased it. Independently WebFetched
  `huggingface.co/Qwen/Qwen3-Embedding-0.6B`: confirmed the asymmetric convention
  (`"Instruct: {task}\nQuery:{q}"` for queries, no prefix for documents) and additionally found the
  card's own bound on the effect — "1% to 5%" MTEB improvement from the instruction, not a
  correctness-scale defect. **Judged not a stop-and-ask fork**: `agent-knowledge-base-strategy-ml.md`
  Recommendation 1 had already reached, and stated, the substantive verdict — falkor-chat's own
  symmetric usage is "defensible" because its retrieval is message-to-message (roughly symmetric
  register), and the asymmetric convention is worth adopting only for that plan's own, sharply
  asymmetric corpus (a short query against a long distilled-technique passage). So this is not a
  live, un-triaged production bug that changes scope if acted on wrong — a more careful analysis
  than mine already exists and already declined to flag it as one. **Promoted the durable,
  reusable half** — not the falkor-chat-specific verdict (already owned by the out-of-scope plan
  doc) — as a new dated section in `claude/data-scientist/lm-studio-model-notes.md`: check a
  model's card for a documented asymmetric query/document convention before reusing one embedder
  call symmetrically, weigh the modest (1-5%) benefit against how asymmetric the actual corpus is,
  and don't file symmetric reuse as a bug by default. File went 2,660 → 2,913 words (+253, one new
  section, no restructuring of existing sections).
- **`b2e3d4c5…` (`claude/analyst/review-techniques.md` doesn't uniformly hold its own "one `##`
  heading = one self-contained technique, 50-400 words" convention; cited example "A guard derived
  from the artifact it guards…") — re-derived independently, confirmed true and understated,
  discarded without a file edit.** Recomputed word counts per section
  (`awk '/^## /{...}'`): the cited section is 1,287 words (not "50-400"), and it is not the only
  outlier — two more sections run 1,448 and 2,057 words, plus eight more between 419 and 635; only
  28 of 39 sections fit the file's own stated band. Read the cited section in full: it bundles at
  least seven independently-verified sub-claims (parametrized-test deletion blindness, AST-alias
  blindness, the two-axis coverage-enumeration argument, the `ast.Assign` census, the name-vs-site
  allowlist gap, the docstring-vs-body semantic/syntactic gap, and the shell-harness exit-code
  trap), each with its own "Verified 2026-09-08" citation — confirms, and somewhat exceeds, the
  entry's "5+" count. **Not acted on directly** — `agent-knowledge-base-strategy-ml.md`
  Recommendation 2 already analyzes this exact section by name (citing the same heading) at
  strictly greater depth (RAG topic-dilution reasoning, a concrete split-trigger heuristic, a
  `familyId`/`SAME_FAMILY` sibling-linkage design) and explicitly assigns the actual heading split
  to a **future, sized migration effort** routed through `cobb` — not a standalone edit made now,
  disconnected from that migration's still-open schema questions. Restructuring the file today
  would risk redoing the split to a shape the in-flight K-030 plan hasn't settled yet. Today's
  actual consumption model for this file is whole-file, on-demand load (its own header: "loaded on
  demand… not part of the always-loaded prompt body") — the retrieval-dilution harm the finding
  describes is specific to the future embedding-based consumption K-030 is designing for, so
  deferring costs nothing under the current model. **Discarded as already documented, not
  re-tracked** — the finding is fully captured, at greater depth, in the K-030 plan doc, which is
  itself a tracked, owned effort; opening a duplicate `analyst` plan item would fork the same fact
  across two backlogs with no single owner. `review-techniques.md` untouched (16,264 words / 39
  sections, unchanged — verified via the same `wc -w`/`awk` count before and after).
- **Both entries verified true; net file changes this unit: `lm-studio-model-notes.md` +253 words
  (one section); `review-techniques.md` 0 words (no edit); no `plan.md` item opened for either
  agent** (both already tracked under K-030, owned elsewhere — dedup-checked, neither entryId
  appears anywhere else in `claude/`).

## 2026-09-10 — Distillation U50: 2 promoted (1 generalized, 1 routed to `analyst`), 2 discarded as superseded by the tree's own subsequent development

- **What:** `cobb` distilled `data-scientist`'s 4-entry `kaizen_team` inbox, re-queried fresh at
  dispatch (unchanged from the brief's list: `b3f1c0a4…`, `e09bd084…`, `e3e8ead6…`, `3f9c1e42…`,
  all 2026-09-09/10). **model-bench has an active, unrelated, concurrent coordination running** —
  nothing under `model-bench/` or `docs/plans/small-model-benchmarking*.md` was written; every
  disposition below routed elsewhere or discarded outright, so the collision never had to be
  resolved by keeping anything open.
- **`b3f1c0a4…` (pack has two sizes: analysis-unit count vs. item count) — promoted, generalized,
  and its model-bench-specific form discarded as already published.** The entry's own worked
  example (roles.UNIT_KIND_BY_ROLE tool-caller = 12 vs. `len(run.items)` = 80, and a defect where
  an `-ml` note had substituted one for the other in three places) is superseded: re-read at
  `892433e`, `modelbench/roles.py` now carries a **second**, purpose-built table
  (`ANALYSIS_UNIT_FIELD_BY_ROLE`) and `docs/plans/small-model-benchmarking.md` v1.29 discusses the
  attempt-count-vs-item-count distinction at length (P16-3 et seq.) — the specific confusion the
  entry flagged has long since been built around and re-documented in far more current detail than
  the entry itself carries. The **general** insight underneath is durable and was not yet in this
  agent's own prompt: in any repeated-measures benchmark, the analysis unit (what `n`/DEFF/a
  paired test are computed over) is a different count than the raw per-turn/per-item count (the
  right denominator for a coverage/latency rate), and a benchmark's own terminology invites
  swapping them silently. Folded one clause onto the existing **Experiment design** bullet,
  Classical ML & statistics (`data-scientist.md`, 2,569 → 2,661 w, +92).
- **`e09bd084…` (a bare `file.py:NNN` cite rots; pin to a symbol+count or a sha) — promoted to
  `claude/analyst/review-techniques.md`, not this agent's own knowledge base.** Re-derived: the
  fact is a general review/plan-authoring hygiene practice with no ML-methodology content — it's
  about citing *code*, durably, inside any docs/plans or review note, a concern this pass has
  hit repeatedly (many units' own citations are pinned to a named sha for exactly this reason) but
  had never written down as a rule anywhere. `claude/AGENTS.md`'s own citation convention covers
  citing *other documents* by path, not code line numbers, and is an always-loaded file already at
  its word budget — the wrong shelf for a technique, not a live constraint. Grepped
  `review-techniques.md` clean first (no existing section on citation-drift specifically, despite
  many sections that *use* the pinned-sha technique for a different purpose — verifying old code
  behavior, not citing it in prose). New section, **13,904 → 14,129 w (+225)**, 32 → 33 sections.
  No `MENTIONS` tag added — the entry is fully disposed of in this same pass, not deferred to a
  future `analyst` pass.
- **`e3e8ead6…` (`callCount` fixture collision risk) — discarded, superseded by the tree's own
  development.** The entry's premise — `ItemTiming`/`LatencyBlock` are unbuilt, so `callCount`
  exists only as an unrelated `tests/test_convo.py` fixture stub — is now false: re-derived at
  `892433e`, `modelbench/results.py` now declares a real `callCount` property and field on
  `ItemTiming`/`LatencyBlock` (`:233`, `:277`, `:285-287`, `:299`, `:317`), so grepping `callCount`
  today finds the real field, not only the fixture. The false-positive risk the entry warned about
  no longer exists.
- **`3f9c1e42…` (undispatchable bucket silently charges harness/pack faults to the model) —
  discarded, superseded — the exact hazard flagged has already been fixed in code.** Re-derived at
  `892433e`: `modelbench/convo.py` now declares a distinct `ToolDispatchFailed(RuntimeError)`
  (`:325`) that a raising `ToolEnvironment.dispatch` is re-raised as (`:758`), explicitly separate
  from `_undispatchable_tool_content`'s three model-failure reasons
  (missing-function-name/unparseable-arguments/no-dispatch-record, `:464/:466/:470`) — the class's
  own docstring states the reason this class exists: "so the runner cannot confuse the two." The
  entry's still-true half (the three named reasons *are* model failures) is unchanged and adds
  nothing beyond what the code itself now states inline; nothing worth promoting survives once the
  flagged hazard is fixed.
- **Verified:** `bash claude/scripts/audit-team.sh` clean (not re-run this unit — no hook, catalog,
  or roster change; only two prose edits to existing prompt/KB files).
- **Docs touched:** `claude/data-scientist/{data-scientist.md,kaizen/history.md}` ·
  `claude/analyst/review-techniques.md`.

## 2026-09-10 — `lm-studio-model-notes.md`: an unrecognized `model` id's outcome depends on residency, not the catalog; catalog reordering and single-model residency (inbound promotion from `qa-engineer`'s capture, U47)

- **What:** `cobb`, distilling `qa-engineer`'s single `kaizen_team` entry (unit U47, entry
  `a1c2e9d4-7b3f-4e2a-9c1d-2f6b8a0e5d17`, 2026-09-10, `suggestedHome: project docs` — overridden;
  see below), added three bullets to the existing `/api/v0/` section in `lm-studio-model-notes.md`,
  right after the existing JIT-auto-load bullet. No new section. **2,323 → 2,660 w (+337).**
- **Why here and not `qa-engineer`'s own `qa-testing-techniques.md`** (the entry's producer, and
  the destination its `suggestedHome` implied): all three facts describe LM Studio **server/API
  behavior** — the same class already living in this file's JIT-load bullet — not a testing method
  or environment/tooling workaround. They sit beside their natural companion content rather than
  fragmenting one `/api/v0` loading-mechanics story across two knowledge bases.
- **Re-derived live on this box, 2026-09-10 — and the re-derivation corrected the entry's central
  claim from an absolute to a conditional it never stated.** The entry's own evidence quote read as
  internally contradictory (a model "already loaded by the operator" producing a 400), so it was
  not taken at face value; ran three arms directly instead, via the real `/api/v0` server and the
  `lms.exe` CLI for unload control:
  1. **Unrecognized `model`, nothing resident** (`lms ps --json` → `[]`): `POST
     /api/v0/chat/completions` with a bogus id → HTTP 400 `"No models loaded. Please load a model
     in the developer page or use the 'lms load' command."`, no load attempt. The entry's claim,
     confirmed — for this one precondition.
  2. **Unrecognized `model`, something already resident** (the entry did not test this arm): the
     identical bogus id against a server with `prism-ml/bonsai-27b` loaded returned **HTTP 200**,
     served by `bonsai-27b`, with the response's own `model`/`model_info` reporting what actually
     ran. `/api/v0/chat/completions` performs **no validation of `model` against the catalog** once
     anything is loaded — it answers from whatever is resident regardless of what was asked for.
  3. **Valid catalog `model`, nothing resident:** JIT-loads and serves correctly (matches the
     existing "unloaded model" bullet's mechanism) — confirming fact 1 is not a general "JIT never
     fires" claim, only the cold-start-plus-no-catalog-match case.
  - Promoted with the corrected scope stated explicitly, plus the QA consequence the entry itself
    never drew: a live test asserting "unknown model → 400" holds only from a cold start; from a
    warm one, the identical call **silently succeeds against the wrong model** instead of catching
    the intended validation — assert on the response body's `model`, not just the status code,
    whenever cross-test residency isn't controlled.
  - The other two facts reproduced exactly as captured, no correction needed: reloading a
    different model moved it to index 0 of `GET /api/v0/models`'s `data` array and dropped the
    prior model to `state: not-loaded` with no explicit unload call — catalog order tracks load
    state, not any stable identity, and this config never holds two models resident at once.
- **Checked for prior coverage before writing:** grepped this file and `qa-testing-techniques.md`
  for JIT/400/reorder/index/resident/evict language — none of the three facts existed anywhere in
  the repo before this promotion.
- **`suggestedHome: project docs` overridden to knowledge base.** The fact is durable LM-Studio
  server behavior on this lab's own stack, exactly this file's charter — not something any
  project's docs tree needs (no falkor-chat/model-bench code depends on it today).
- **Graph:** cleared as part of the same disposition as `qa-engineer`'s U47 entry — 1 `PRODUCED`
  (`qa-engineer`) / 0 `MENTIONS` ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`, run only after
  both this entry and `qa-engineer`'s matching history entry landed.
- **Docs touched:** `claude/data-scientist/{lm-studio-model-notes.md,kaizen/history.md}` ·
  `claude/qa-engineer/kaizen/history.md` (producer's disposition record).
- **Plan items:** none opened — fully promoted, no follow-up needed.

## 2026-09-09 — orphan-backlog entry `b7e41c92` cleared: already promoted, accurately, by this agent's own U15 (U31)

- **What:** U31 of `claude/docs/plans/kaizen-distillation2-coordination.md` — the **orphan-backlog** unit, the first shaped by *edge* rather than by producer. The 11 nodes it covers carry **0 `PRODUCED` edges** and are alive only on `MENTIONS`; every earlier unit was organised by producer, so none of them could ever have been reached. `data-scientist` carried one of the 12 edges, tagged by `teco`'s U18.
- **`b7e41c92-3d5a-4f18-9c60-2a8e17d34f5b` (2026-09-03) — already promoted; cleared without re-promoting.** Its content is carried, accurately and in full, by `claude/data-scientist/data-scientist.md`'s bullet *"A seeded percentile bootstrap over a discrete outcome is not a function of the data alone"* — written by **U15** (commit `42d80d7`, this agent's own chunk C) from a sibling entry on the same subject, three days before the `MENTIONS` tag was placed. That bullet states all three of the entry's parts: the bound moves with **row order** at a fixed seed because `random.Random.choice` draws an **index**; the effect is conditional on the target landing within Monte-Carlo error of an atom boundary at `k/n`; and a **smaller `B` makes the flip more common, not less**. Nothing was narrowed, garbled, or over-broadened in the promotion, so re-promoting would have produced a duplicate — the worse of the two outcomes.
- **Verified by re-deriving, not by re-reading either the entry or the bullet.** Executed 2026-09-09 (system `python3` 3.12.3): `random.Random.choice`'s source is an index draw; twenty resamples of one 12-row multiset at a **fixed seed 0** differ between the vector and its reversal (equality test returned `False`); and sweeping 60 row permutations at one fixed seed gives **2** distinct 2.5th-percentile lower bounds at `B=400` but **1** at both `B=2000` and `B=10000` — the counterintuitive half confirmed, with the two higher-`B` runs serving as the passing control that a naive stable check is exactly what the entry warns about.
- **Graph:** 0 `PRODUCED` / 1 `MENTIONS` ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`.
- **Docs touched:** `claude/data-scientist/kaizen/history.md` only — no prompt or knowledge-base change was warranted.
## 2026-09-09 — `data-scientist.md`: a provenance audit is computed before any value-modifying post-transform (U27)

- **What:** `cobb`, distilling `data-scientist`'s single `kaizen_team` entry (unit U27, entry
  `a4f1c6d2-3b7e-4c81-9f0a-6d25e8b7c913`, 2026-09-08, `suggestedHome: knowledge base`), folded one
  sentence pair onto the existing *"The arithmetic of a published statistic is part of the claim"*
  bullet in § *Classical ML & statistics*. No new bullet, no new knowledge base; one line changed,
  +80 words.
- **The promoted rule.** A provenance label is part of the published claim, so the
  which-instrument-produced-this-number audit is computed **before** any value-modifying
  post-transform (support clamp, rounding, normalisation), never after: the transform can collapse
  two distinct candidates onto one printed value, and a tie-break over the transformed pair then
  names a source that produced neither. Moving it is free wherever the transform is bound-wise and
  monotone — it commutes exactly with a bound-by-bound `min`/`max` composition.
- **Why the prompt and not `lm-studio-model-notes.md`.** That knowledge base is LM-Studio
  small-model realism; this is a statistics-reporting rule, and § *Classical ML & statistics*
  already hosts that class (exact index arithmetic; seeded-bootstrap atom instability). A second
  knowledge base for a single entry is premature.
- **Verified by re-derivation, not by re-reading**, against `model-bench/modelbench/stats.py` pinned
  at `d45e5ff` and copied read-only into the session scratchpad. Nothing was written under
  `model-bench/`.
  - **The counts are exact, and were enumerated rather than decomposed.** Tables per `n`:
    455 / 5 456 / 10 660 / 12 341 at `n` = 12 / 30 / 38 / 40 — each equal to `C(n+3,3)` — summing to
    **28 912**; times the **6** design effects `{1.0, 1.2, 1.5, 2.0, 4.0, 7.0}` = **173 472**
    combinations. The multiplier is the six design effects: 28 912 is already the total across all
    four `n`, not a per-`n` figure.
  - **The null result reproduced:** over all 173 472 combinations, **0** differences in the printed
    bounds between compose-then-clamp and clamp-then-compose, **0** point-containment differences,
    **0** zero-exclusion differences.
  - **It is not a vacuous null.** The clamp actually bound at least one arm-bound in **8 180** of
    173 472 combinations (4.7 %), and **1 781** had *both* arms' lower bounds outside support — the
    tie-inducing shape.
  - **Passing controls, run in the same loop over the same data**, so a uniform `0` could not be
    read off a dead comparator: an asymmetric clamp `(-1,1)` vs `(-0.98,0.98)` → **10 056**
    differences; width-normalisation, which is not bound-wise → **167 167**; a non-monotone
    bound-wise `-abs()` → **141 412**; and rounding to 3 dp — monotone and bound-wise, like the
    clamp → **0**, which is the commute theorem holding on a second transform rather than a second
    dead probe.
- **The rule-versus-null-result tension resolves in the entry's favour, and the entry itself says
  so.** The sweep measures **printed numbers**; the rule is about **attribution**. The null result is
  not evidence *for* the rule — it is evidence that adopting the rule is *free*. The evidence *for*
  the rule is the divergence the sweep never counted: recomputing `bound_by` on the clamped arms
  (what the code does at `d45e5ff`) against the unclamped arms disagrees on **3 525** of 173 472
  combinations — 1 776 lower-bound flips, 1 776 upper-bound flips, 27 both
  (1 776 + 1 776 − 27 = 3 525), counted independently by two scripts that agreed.
- **One correction to the cited separating case.** `(0, 0, 38, 2)` at DEFF 1.5 reproduces exactly:
  MOVER-D lower unclamped `-0.9943104520691852`, exact lower unclamped `-1.0112372435695796`,
  printed interval `(-1.0, -0.7728921326614777)`, point `-0.95`, McNemar exact `p = 7.28e-12` — zero
  is excluded, so the verdict is distinguishable. But at that table the shipped tie-break still
  names the **right** arm under both orderings (`exact paired bootstrap`): it is a case where the
  audit is *unanswerable* — the printed `-1.0` equals neither arm's own value — not one where it is
  *mis-answered*. The stronger shape the fact asserts does occur, and the entry could have cited it:
  `(0, 0, 6, 6)` at DEFF 7.0, where both lower bounds fall outside support (`-1.1514326608688510`
  MOVER-D, `-1.1614378277661477` exact), both clamp to `-1.0`, and the `<=` tie-break names MOVER-D
  where the exact arm was the more conservative one.
- **Discarded as already published at the point of use:** the model-bench-specific half.
  `docs/plans/small-model-benchmarking-ml.md` **Rule 4a** (pinned `a707d09`) already carries the
  ruling, the same 28 912 / 173 472 counts, the commute proof and the `support bound` third token;
  `stats.py`'s `_compose` docstring at `d45e5ff` records the pending placement. What that document
  never states — and what was promoted — is the generalisation past a support clamp to *any*
  value-modifying post-transform, rounding and normalisation included.
- **Collateral, already scheduled — no routing needed.** At `d45e5ff`, `envelope_arms` clamps each
  arm and `verdict()` reads `bound_by` off those clamped arms, so the shipped attribution is wrong
  on the 3 525 combinations above. Rule 4a / `P8-1` moves the clamp into `_compose` and takes
  `bound_by` from the unclamped composed value; `_compose`'s own docstring states the edit is
  pending. Not a new defect, and not `cobb`'s to fix.
- **Disposition:** 1 promoted (the generalisation) / 0 discarded outright / 0 kept open — no
  `plan.md` item, since the only unfinished work is already scheduled as `P8-1`. Graph:
  `producedEdges` 1, `mentionEdges` 0 → `otherRemaining` 0 → whole node `DETACH DELETE`d.
  `data-scientist`'s one `MENTIONS`-only edge (`b7e41c92-3d5a-4f18-9c60-2a8e17d34f5b`) is out of
  this unit's scope and untouched.

## 2026-09-08 — `lm-studio-model-notes.md`: `/api/v0/embeddings` carries no measurement fields, and the catalog size is not stable (U22)

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk D (unit U22, entry
  `b7f2c1d4-9e63-4a58-8c21-5f0a7d3e9b16`), made two edits inside the existing `/api/v0/` section —
  no new section, no prompt change.
- **The new clause.** Most of the entry was already published here: the ten catalog keys,
  `/v1/models`' three, `runtime`/`stats`/`model_info` living only on
  `POST /api/v0/chat/completions`, `loaded_context_length`'s absence while not-loaded, and
  `lms.exe`-only reachability from WSL. What was missing is the negative on the third route:
  `POST /api/v0/embeddings` carries **none** of the three. Verified live 2026-09-08 against
  `localhost:1234` with `text-embedding-qwen3-embedding-0.6b` — the response holds exactly
  `data`/`model`/`object`/`usage`, and `usage` itself came back
  `{prompt_tokens: 0, total_tokens: 0}`. That turns "an embeddings-only arm can never populate a
  runtime field" from an inference into a measured constraint on any fingerprint design.
- **The counts were stale, and correcting them strengthened the finding.** The `capabilities` bullet
  was written against a 19-model catalog; today's `curl -s :1234/api/v0/models` returns **16** (10
  `vlm`, 4 `llm`, 2 `embeddings`), so the census reads 12 of 16, not 15 of 19. The bullet's
  *conclusion* is unaffected and now demonstrably robust: the **same four** named entries are still
  the ones without a `capabilities` key, and every entry that has it still holds only
  `["tool_use"]`. The bullet now says to recount rather than quote a denominator.
  `loaded_context_length` also gained its first positive confirmation — 15 of 16 entries were
  `not-loaded` and keyless while the one model a probe had just JIT-loaded carried the key.
- **Plan items:** none.

## 2026-09-08 — `lm-studio-model-notes.md`: `capabilities` is not a tool-calling gate (from `analyst`'s capture, U20)

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk B (unit U20, entry
  `3f7c1a92-5d64-4b0e-9c31-8ae2f0d47b15`), added one bullet inside the existing `/api/v0/` section
  — which already listed `capabilities` among the fields `GET /api/v0/models` returns but said
  nothing about whether to trust it.
- **The re-derivation strengthened the claim.** Re-ran `curl -s :1234/api/v0/models` on this box
  2026-09-08 (19 models, six days after the original observation). The entry says the field is
  unreliable; the measurement says it carries **no discriminating information at all** — every
  entry that has the key holds exactly `["tool_use"]` and none holds anything else, so it never
  says *no*. It reports `["tool_use"]` for the embeddings model
  `text-embedding-qwen3-embedding-0.6b`, and is absent from **four** entries (the entry named two)
  spanning both kinds: `google/gemma-3-4b` and `google/gemma-3-12b` (`vlm`), `gemma-3-4b-vl-it-…`
  (`llm`), and a second embeddings model `text-embedding-nomic-embed-text-v1.5`. Presence does not
  imply a chat model; absence implies nothing. Gate on `type` ∈ {`llm`,`vlm`}. The entry's
  `loaded_context_length` aside reproduced too (absent from all 19 while `state == not-loaded`)
  and went in beside it.
- **Why here rather than in the plan it was about:** the entry was raised against
  `docs/plans/small-model-benchmarking.md` §3.6, which a concurrent session held for the duration
  of this unit. The fact now lives where `data-scientist` reads it; whether §3.6's tool-caller
  refusal changes is that document owner's call.
- **Files:** `claude/data-scientist/lm-studio-model-notes.md`. Source disposition:
  `claude/analyst/kaizen/history.md` (2026-09-08, U20).

## 2026-09-07 — Learnings-graph distillation, chunk C of 3 (U15): 10 entries — 5 promoted, 4 discarded, 1 kept open; the agent is now at zero

- **What:** `cobb` processed the final 10 `data-scientist` `:KaizenEntry` nodes in `kaizen_team`
  (2026-09-03 … 2026-09-07) per `agent-maintenance` §5, closing this agent out. Every claim was
  **re-derived from scratch** — the statistics recomputed in stdlib Python (`decimal` at 60–80
  digits where a fixture was at stake) against the *shipped* module, never confirmed against the
  entry's own cited evidence; the LM Studio timing claims re-measured **live** on a genuinely cold
  box. Seven of the ten are model-bench statistics facts, and a concurrent session held that
  component's source and docs dirty throughout, so **nothing under `model-bench/**` or
  `docs/**small-model-benchmarking*` was written** — the one entry needing such a write is kept
  open below with its targets named.
- **Promoted to the prompt (3):**
  - `c41b8e07` (2026-09-03, a fixture table published at display precision cannot support a tighter
    assertion tolerance) → **prompt**, one sentence on the "Evaluation engineering → Golden sets and
    regression evals" bullet. **Re-derived at 60 digits independently:** the MOVER-D lower bound for
    `(34,6,0,0)` is `0.031762869443060`, so v1.2–v1.4's published `3.1763` pp sat `1.3056e-7` from
    the truth — **130.6x** the 1e-9 proportion tolerance those same subsections mandated — while the
    shipped float implementation's worst error across all five fixtures is `1.640e-16`, about
    `6.1e6` *inside* the mandate. **Two corrections to the entry:** it says "131x" (130.6x, fine) but
    puts the implementation at `1.44e-16`; the worst of the ten bounds is `1.640e-16`, ~14% larger,
    and its "7e6x inside" is correspondingly `6.1e6`. Neither moves the conclusion. The
    project-scoped copy is already published twice — as a boxed rule in
    `docs/plans/small-model-benchmarking-ml.md` §3.2c and as a comment block above
    `REGRESSION_FIXTURES` in `model-bench/tests/test_stats.py` — and **acted on**: the fixtures are
    now published at 10 dp (verified against the 60-digit truth to ≤3.8e-13) and asserted at 1e-9.
    Only the discipline-level rule was promoted, since publishing a fixture table beside a tolerance
    is a recurring shape of *this agent's own* `-ml.md` deliverables, and it is the shape that
    produced the defect.
  - `a9d2f5b3` (2026-09-03, sweep the exact expression the code uses) and `b3d7f1a2`'s transferable
    half → **prompt**, as a new "the arithmetic of a published statistic is part of the claim"
    bullet under "Classical ML & statistics". **Both sweeps reproduced exactly:** `7/40` gives
    `x*1000 == 175.0` but `x/0.001 == 174.99999999999997`; over every `k/n` with `n ≤ 1000` the
    multiply form diverges from exact rational truncation **0** times and the divide form the code
    uses **654** times, hitting `n = 10, 20, 40` — `n = 40` being a published row of the method note.
    Independently, `repr(0.28*25) == '7.000000000000001'`, and over `permille 1..999 × X ≤ 3000` the
    percent spelling `ceil(pct/100*X)` diverges from `-(-num*X//den)` **1626** times, the level-first
    spelling **755**, and the numerator-first spelling **0** — first divergence `(X=25, level=7/25)`
    for both float forms, and **0** divergences at the four levels model-bench actually uses
    (permille 25/500/950/975) up to `X = 2000`, so the guard is defensive there exactly as the note
    says. The project-scoped copies are published in note §11.2.1 and Rule 3a, and at the point of
    use in `model-bench/modelbench/stats.py:646` and `tests/test_stats.py:383` (*"a sweep run against
    the expression the code does not use is how this was missed the first time"*).
  - `5b8e0c62` (2026-09-03), **carrying the merged `c1f2a7d4`** → **prompt**, as a second new bullet
    on the seeded percentile bootstrap. See the REFINES resolution below.
- **Promoted to the knowledge base (2)** — both folded into the **existing** `/api/v0/` section's JIT
  bullet in `lm-studio-model-notes.md`, rewriting it in place rather than stacking a sixth parallel
  section:
  - `c47a9e30` (2026-09-03, `stats.time_to_first_token`/`generation_time` exclude the JIT auto-load,
    so `wall − (ttft + generation_time)` isolates the load and is an in-call reload detector).
    **Re-measured live this session, not read:** LM Studio was up, all 19 models `not-loaded`, so the
    probe got a genuine cold start. `residency()` was `[]` before the call and `['qwen/qwen3-4b-2507']`
    after; the cold call was wall **3 756.3 ms** with `ttft` 33.3 ms and `generation_time` 215.7 ms —
    a gap of **3 507.4 ms**, within 0.6% of the note's independently-measured 3 485.6 ms — against
    **−6.5 … +6.8 ms** over 20 warm calls, so the cold gap is ~**516x** the largest warm gap (the
    note measured 461x). The *detector* framing was the half missing from this file: it previously
    recorded only that `ttft` excludes the load, with a pointer to §11.4. **One correction to the
    entry:** it calls the cold-load spread "an order of magnitude"; the two figures it cites are
    3.625 s and 21.068 s, a **5.8x** ratio, and today's 3.5 s is a third point in the same band. The
    file already stated this correctly as "~6x" and now says so over three observations.
  - `a3f1c8de` (2026-09-06, a withholding rule firing on a covariate is not right-censoring) →
    **knowledge base**, same bullet, because a reader handed the detector above must not read its
    withheld set as a slow tail. **Re-derived by construction rather than by probe:** the threshold
    is on `unexplainedMs`, so a withheld item's *wall clock* is `gap + ttft + generation` — a 1.01 s
    gap around 50 ms of generation is withheld at ~1.06 s while an ordinary long-pole pack turn is
    timed at ~1.3 s, and the ordering inverts with no exotic input. The live warm data supports the
    premise directly: warm gaps are ±7 ms, so on a clean call the wall clock *is* generation. Its
    project-scoped copy is published, and **more carefully than the entry**, in note §11.5 (three
    producers of a withheld timing, only two of which censor the summarised quantity) and §11.5.1
    (v1.12, commit `fc2fcf6`), which turns the entry's "must be computed per render" into a shipped
    per-render predicate rather than a caution. The entry's evidence clause overstates one notch —
    it says the ordering *fails* on the ordinary case where the note says it *cannot be assumed* and
    must be computed; the fact itself is right and is what was folded.
- **The `5b8e0c62` / `c1f2a7d4` REFINES pair — resolved as supersession, promoted once.**
  `5b8e0c62`'s `fact` opens with a curator instruction to merge or supersede `c1f2a7d4`. Both were
  re-derived against the shipped `paired_bootstrap` (clean in the working tree), and **both
  reproduced to the digit**: 12 row-shuffles of the same difference multiset at the fixed seed
  20260902, `(b=8,c=0,n=85)`, `B=10000`, give exactly the two intervals `(3.53, 15.29)` and
  `(3.53, 16.47)` pp that `c1f2a7d4` reports, and 20 seeds at fixed row order give the same two
  values; `(b=5,c=3,n=12)` flips its lower bound between **−25.0** and **−33.3** pp on **107/93** of
  200 seeds — `5b8e0c62`'s figure exactly — and 107/93 of 200 row permutations (the entry says
  102/97; the split is permutation-RNG-dependent, the phenomenon is not); `(b=4,c=0,n=30)` is
  identical across 60 seeds; `(b=4,c=2,n=30)` is stable at `B=10000` and moves on 6 of my 60 seeds
  at `B=2000` (entry: 8/60, same seed-set dependence). The predictor checks out to five places:
  exact `P(K≤13) = 0.97281` and `P(K≤14) = 0.98738` for `Bin(85, 8/85)` against a Monte-Carlo
  standard error of `sqrt(.975×.025/10000) = 0.0016`, putting the 97.5th percentile 0.0022 from an
  atom boundary. **`5b8e0c62` supersedes `c1f2a7d4` outright**: it contains the original's bottom
  line (row order is a real input at a fixed seed, because `random.Random.choice` draws an *index*)
  and corrects its framing from universal to conditional, which is the half that decides whether a
  reader trusts a re-derivation that comes back stable. `c1f2a7d4`'s one unique contribution — the
  closed form — was folded into the promoted bullet rather than dropped. Promoting both would have
  shipped a flat claim beside its own refutation. The project-scoped copy of the merged statement is
  already published in the refined form in three places (note §3.2d/§3.4, review item 3 with the
  same 0.0022-vs-0.0016 numbers, and the shipped `stats.py:245` docstring), so only the
  discipline-level rule was promoted.
- **Discarded (3, plus `c1f2a7d4` as superseded above):**
  - `b1e6c0d2` (2026-09-03, a strict-dominance MDD can be smaller than an observed non-significant
    difference, so a hard-coded *"the observed X pp is below that"* is structurally false for mixed
    discordance) — **mechanism re-derived exactly, and already fixed in code and documented
    verbatim.** Driving the shipped `stats._mdd_clause` at `n=20`, DEFF 1.0, α=0.05 reproduces
    `mdd80 = 36.7 pp` and, at `b=13, c=5` (observed 40.0 pp, `mcnemar_exact = 0.09625`), renders the
    *"is at or above that, but the MDD assumes strict dominance…"* branch — the entry's own example,
    with its p-value, now rendering correctly. The clause is conditional at `stats.py:736`; sweeping
    every `(b,c)` at `n ∈ {12,15,20,30,38,40,48}` prints the false wording **0** times under the
    current code and would print it on 2 422 of 3 820 tables under the unconditional version. The
    finding is published in the shipped `_mdd_clause` docstring (with its numbers), in
    `docs/reviews/small-model-benchmarking-ml.md` §604-613, and in `model-bench/docs/HISTORY.md:219`.
    **One caveat recorded rather than promoted:** the entry's "1580 tables print the clause, 268
    (17%) print it falsely" is **not reproducible without the sweep's exact verdict gating** — my
    nearest reconstruction (verdict-2 tables only, i.e. McNemar p > 0.05, DEFF 1.0, no floor gate)
    gives 1746 and 386 (22.1%). Same order, same conclusion, but the counts are conditional on
    gating the entry does not state, and the shipped docstring carries them as if absolute.
  - `b3d7f1a2` (2026-09-03, float rank/bin arithmetic has produced an off-by-one twice in
    independent formulas; use integer arithmetic) — **both instances re-derived exactly** (numbers
    above, under `a9d2f5b3`) and **already published**: note §11.2.1 carries the `0.28*25` case, the
    integer form `-(-level.numerator * X // level.denominator)`, *and* the explicit framing the entry
    reaches for (*"Same bin-edge class as Rule 3a's `(7/40)/0.001` case, one operation over"*), plus
    the "defensive, not load-bearing" status at the four levels this tool actually uses. Its
    transferable half rides in the `a9d2f5b3` promotion; nothing distinct is left.
  - `6ef71251` (2026-09-07, in `ceil(level*X)` operand order decides exactness — 1626 / 755 / 0
    divergences for the percent, level-first and numerator-first spellings) — **swept independently
    and reproduced to the count**, including the first divergence at `(X=25, level=7/25)` for both
    float forms, and confirmed that `modelbench/stats.py:299` and `results.py:573` both still spell
    it `int(round(pct / 100.0 * (len-1)))`. **Already published, by the very work that produced it:**
    note **v1.18** §11.2.1 carries the three-row table with 1626/755/0 verbatim *and* explicitly
    corrects the v1.17 mis-attribution the entry reports (*"v1.17 printed it under
    `math.ceil(permille * X / 1000)`, which is the numerator-first spelling and diverges **nowhere**
    in that sweep"*), landed in commit `bbbf18e`. The entry arrived from a concurrent session during
    U13's run and describes a state the repo had already left.
- **Kept open (1) — blocked by the model-bench write constraint, target named:**
  - `7f3c1a92` (2026-09-03, the pinned `_Z_95` is not equal to `NormalDist().inv_cdf(0.975)`, so
    `==` fails while `< 1e-12` passes). U14 left this to this pass with two readings, both of which
    hold. **It is not a duplicate of the already-cleared `0f3b6a1e`** — that entry was about prose
    `1.96` versus the pin (provenance), this one about the pin literal versus the computed double
    (exact equality); complementary halves of one rule. And it is **already published in the
    strongest available form**, as the committed executable assertion
    `test_z_95_matches_the_inverse_normal_cdf` (`model-bench/tests/test_stats.py:82`), which asserts
    both `_Z_95 != NormalDist().inv_cdf(0.975)` and `abs(...) < 1e-12` and whose docstring states the
    divergence and forbids tightening it. **But its stated mechanism carries an arithmetic error, and
    so does every published copy.** Re-derived independently: the delta is `4.440892098500626e-16`
    while `math.ulp(1.9599639845400536)` is `2.220446049250313e-16`, and the two doubles are **two**
    representable steps apart — confirmed three ways (delta/ulp = 2.0, IEEE-754 bit distance = 2, and
    `nextafter(inv_cdf, +inf)` applied **twice** reaching the pin exactly). So it is **two ULPs, not
    one**. Also confirmed: the pin really is the 16-significant-digit decimal of that double
    (`'%.16g' % inv_cdf` round-trips to the pin; `%.17g` does not), which is what makes the entry's
    framing right for the wrong count. The bottom line — unequal doubles, `==` fails, `< 1e-12`
    passes — is untouched. **The "one ULP" wording is committed in three places, all outside `cobb`'s
    write remit and all under the concurrent session's active edit:**
    `model-bench/tests/test_stats.py:89` (the docstring above), `model-bench/docs/HISTORY.md:387`,
    and `docs/reviews/small-model-benchmarking-ml.md:247` (finding n-ML-3). Opened as **K-004** in
    `plan.md` with those three paths, for the human to route once that session lands. The entry is
    cleared from the graph; `plan.md` is its durable record.
- **`MENTIONS` edges added: 0.** All ten are statistics or LM-Studio-measurement facts inside this
  agent's own discipline, and every project-scoped half is already published in the component it
  concerns, so tagging would only queue documented content into another agent's pass (chunk A/B
  reasoning, unchanged). `7f3c1a92`'s blocked correction spans a `-ml` review this agent *itself*
  owns under root `AGENTS.md`'s by-kind table and two model-bench files the human is routing, so a
  tag would misdirect it rather than carry it.
- **Verification basis, per the standing requirement.** **Two entries were verified by live probe**
  — `c47a9e30` and (in its premise) `a3f1c8de`: LM Studio answered on `:1234`, all 19 models were
  `not-loaded`, and one cold call plus 20 warm calls against `qwen/qwen3-4b-2507` (Q4_K_M) were
  measured this session. **The other eight were verified by artifact and by recomputation** —
  `decimal` at 60–80 significant digits for the MOVER-D fixtures and the inverse-normal constant,
  exhaustive integer sweeps for the rank/bin claims, and direct calls into the *shipped, clean*
  `modelbench.stats` for the bootstrap and MDD claims. **model-bench's test suite was deliberately
  not run** — its `conftest.py`, `test_fingerprint.py`, `test_results.py` and `fingerprint.py` were
  dirty in the working tree throughout, so a failure could not have been attributed; `stats.py` and
  `test_stats.py` were clean, which is why importing and driving the module directly was sound.
  No shared FalkorDB graph was read or written beyond `kaizen_team`, and no scratch graph key was
  created. Publication checks again grepped **outside** the docs tree — shipped docstrings, inline
  comments, test bodies and `model-bench/docs/HISTORY.md` — which is where four of the six
  already-published findings were actually found.
- **Cleared:** all 10 entries `DETACH DELETE`d after this entry was written, each having exactly one
  `PRODUCED` edge and no `MENTIONS`. `data-scientist` is at **zero** produced entries.
- **Docs touched:** `claude/data-scientist/{data-scientist.md, lm-studio-model-notes.md,
  kaizen/history.md, kaizen/plan.md}`. Nothing under `model-bench/` or `docs/` was written.


## 2026-09-07 — Learnings-graph distillation, chunk B of 3 (U14): 9 entries — 1 promoted, 7 discarded, 1 kept open

- **What:** `cobb` processed the 9 `data-scientist` `:KaizenEntry` nodes dated 2026-08-31…2026-09-02
  in `kaizen_team` per `agent-maintenance` §5 (chunk A was the 10 oldest; the 2026-09-03…09-07
  entries are chunk C and were not touched). Every claim was **re-derived from scratch** — the
  statistics recomputed in stdlib Python from the raw data, the falkor-chat mechanics re-read from
  source — never confirmed against the entry's own cited evidence. Two entries survived
  re-derivation with their *arithmetic* intact but their *framing* falsified; one was falsified
  outright.
- **Promoted (1):**
  - `0e07beeb` (2026-09-02, marginal-Wilson overlap is inert, not conservative, as a difference
    test) → **prompt**, folded into the existing "Classical ML & statistics → Uncertainty" bullet
    rather than added as a new one. The bullet already mandates the Wilson interval as this lab's
    small-n convention and says nothing about the most common misuse of exactly that tool; the
    existing "refusing to bless differences the sample cannot support" clause guards the opposite
    failure (over-claiming), so the addition is genuinely uncovered ground, not a restatement.
    **Every number re-derived independently:** 40/40 Wilson = (0.9124, 1.0000) vs 34/40 Wilson =
    (0.7093, 0.9294) — overlapping although perfectly nested; the same table as McNemar exact
    (b=6, c=0) gives p = 0.03125; Newcombe MOVER-D gives (3.2, 29.1) pp, matching the entry to the
    published precision. The separation sweep confirms the strong form: at n ∈ {20, 30, 38, 40} and
    a baseline of 0.90/0.925/0.95, **no** candidate score separates, up to and including a perfect
    one. Minimum net discordant wins for α = 0.05 at c = 0…4 is 6/8/10/12/13, so the observable
    floor is 6/n; the 80%-power MDD search gives n·δ = 7.33 (n=20) … 7.81 (n=120), so ≈7.7/n. The
    project-scoped copy of this decision is already shipped in `model-bench`'s own code and docs —
    only the discipline-level rule was promoted, since that is the half that reaches a future
    session in any component.
- **Discarded (7):**
  - `a3f1c2e0` (2026-08-31, `ministral-3-3b` has no `temperature` pin, which "alone can explain
    most of" a ~40-point swing in a repro script's precondition rate) — **falsified twice over.**
    The pin exists: `falkor-chat/config/models.json` pins `temperature: 0` for
    `lmstudio/mistralai/ministral-3-3b`, added by commit `9d98aa0` *on the entry's own date*, so
    the stated mechanism was remediated immediately and the entry never described a standing
    condition. The causal claim is separately refuted by the lab's own later measurement:
    `falkor-chat/docs/BACKLOG.md` K-062 §18 ran three arms **under the identical pinned
    temperature, model and script**, differing only in `systemPrompt` text, and measured 53.6% /
    17.9% / 60.7% — baseline vs. lever (a) non-overlapping at Fisher p ≈ 0.011. That entry states
    the conclusion plainly: "**The `temperature: 0` pin did not narrow this swing.**" U13 had
    already promoted the stronger, later form of this lesson into `lm-studio-model-notes.md`
    ("a pin buys comparability, not repeatability").
  - `f3a7b2c4` (2026-08-31, a small addition to a repro script's turn text collapsed the
    held-rejection base rate from ~58% to ~4%) — **already published, in a more careful form than
    the entry.** `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §13.2 states the
    finding with both rates, both Wilson CIs and the non-overlap, and then explicitly declines the
    causal attribution the entry asserts ("this pass did not isolate which factor (or both) is
    responsible"). Its consequence — that the pass was powered on a stratum that then yielded one
    usable rep — is stated there too. The generalized rule is published as a standing live
    constraint in K-062: an occurrence-rate difference between two sessions or two prompt variants
    cannot by itself be read as evidence about the variable under test. Nothing left to promote.
  - `b3f1c9a4` (2026-08-31, a deterministic per-loop customer id in a throwaway probe silently
    reuses cart state across invocations) — **mechanism re-derived and correct, remedy already
    published three times.** `repository.add_to_cart` is `ON MATCH SET item.quantity =
    item.quantity + $qty` and its own docstring says "Add means increment, not replace"; the tool
    schema says the same to the model ("quantities accumulate, they don't replace"). The incident
    is `…-ml.md` §12.2 verbatim, including the increment citation and the "test-harness
    id-collision artifact" diagnosis; the **remedy** is published as a construction rule in §13.2
    (a "uuid4-derived customer id … never a loop index, specifically to foreclose the exact
    id-collision hazard §12.2 caught") and as a standing requirement in
    `docs/test-plans/salesperson-tool-reliability-regression.md` ("each of the 6 conversation reps
    uses a fresh customer id … so no rep's cart or profile state can leak into another's"). The
    entry's own `suggestedHome` of `skills/joern-cpg` is a mis-file — that skill builds Code
    Property Graphs and has no relation to live-harness authoring.
  - `b3f0a1c4` (2026-08-31, an exact-argument dedup guard is defeated by an optional-parameter
    default applied inside the tool wrapper) — **fixed in code and documented at the point of use,
    verbatim.** `executor.py` now computes the key through `_resolve_dedup_arguments`, and a
    ~25-line comment block above it states the entry's fact, its evidence (§15.2, rep-20), the
    exact two colliding argument shapes, and why each write tool gets its own resolver rather than
    a shared one — including that `remove_from_cart` deliberately has no such default. The
    accompanying docstring records that only the *key* is collapsed, never the dispatched call.
  - `f95fc29b` (2026-09-02, `judge_calibration.json`'s raw agreement flatters the judge; relevance
    κ = 0.211 vs. faithfulness κ = 0.833) — **arithmetic re-derived exactly, and already published
    in the very document the entry's own context names.** Recomputed from the committed file:
    relevance p₀ = 0.700, pₑ = 0.620 (from 7/3 gold vs. 8/2 judge marginals), κ = 0.2105;
    faithfulness p₀ = 0.900, pₑ = 0.400, κ = 0.8333; 2 of the 3 gold-irrelevant answers judged
    relevant. `docs/plans/small-model-benchmarking-ml.md` already carries every one of those
    numbers in a table attributed to a recompute from that same file, calls the relevance axis
    "close to worthless" in §6, and **acted on it** — the design drops the relevance axis entirely
    and reuses only faithfulness. One caveat the entry does not raise and that a promotion would
    have shipped unqualified: this repo has an explicit, reasoned standing decision *not* to
    report κ at this n (`docs/test-reports/graphrag-eval-2026-08-16.md`, `golden-set-expansion-ml.md`
    §"not κ — n=15–20 is too small"), and it is right — κ = 0.211 at n = 10 carries a bootstrap
    95% CI of (−0.32, 0.80) and the class-conditional 2/3 has a Wilson CI of (0.21, 0.94). The
    published treatment, which flags the figure for a gate rather than deciding on it, is the
    correct handling; the entry's flat "not usable as a measurement" is stronger than n = 10
    supports.
  - `7c2d84b0` (2026-09-02, two traps in the published paired-statistics tables — the MDD table is
    ceilinged not rounded, and the Kish design effect is the *square* of the CI width ratio) —
    **re-derived exactly and already documented at the point of use.** Independent exact search
    over the McNemar rejection region (strict-dominance model, b ~ Bin(n, δ), c = 0, reject at
    b ≥ 6) reproduces n = 40 → 19.0464 pp, power(0.190) = 0.7980, power(0.191) = 0.8023, and the
    ceiling reproduces the whole published row set — 36.7 / 25.1 / 20.1 / 19.1 / 12.9 / 9.2 / 6.6
    at n = 20 / 30 / 38 / 40 / 60 / 85 / 120 — while round-to-nearest gives 36.6 / 25.1 / 20.0 /
    19.0 / 12.9 / 9.1 / 6.5. **One correction to the entry:** it says rounding "disagrees with
    every published figure"; it disagrees with **five of seven** (n = 30 and n = 60 agree), and
    understates in each of the five. Both traps are already written into the shipped module:
    `model-bench/modelbench/stats.py` documents "MDD₈₀, exact and **rounded up** to the printed
    precision … Ceiling to 19.1 pp gives 0.8023" and, separately, "the design effect is a variance
    ratio — the width ratio squared", with a `width_ratio` helper whose docstring says it is
    "**not** the design effect" and a note that following the earlier wording literally "divides by
    2.6". The Kish identity checks out independently: m = 7, ρ = 1 → DEFF = 7, √7 = 2.646,
    n_eff = 280/7 = 40 = the cluster count.
  - `0f3b6a1e` (2026-09-02, this lab has ONE 95% z-constant and the choice is numerically
    irrelevant) — **already published in four documents and two committed tests, and its framing is
    the half that is wrong.** A from-scratch MOVER-D implementation reproduced all ten published
    fixture bounds to 10 decimal places and put the pin-vs-1.96 divergence at 3.017 × 10⁻⁴ pp,
    matching the entry. But "numerically irrelevant … only matters for exact-equality assertions"
    is false in the regime this lab actually operates in: the method note mandates a 1e-9
    *proportion* tolerance, 3.0 × 10⁻⁴ pp is 3.0 × 10⁻⁶ as a proportion, and the shipped test
    `test_the_pinned_z_constant_is_load_bearing_at_this_tolerance` asserts that `z = 1.96` misses
    **every** bound — about 3 000× outside tolerance. The published wording has this right and the
    entry does not: `1.96` is a typographic rounding rather than a rival convention, which is a
    statement about provenance, **not** about interchangeability. Also already published:
    `small-model-benchmarking-ml.md` §3.2a's "at most 3.0 × 10⁻⁴ pp", the coordination log's "M-1
    is not a numerical defect", a whole both-z fixture table in the review, and
    `model-bench/tests/test_stats.py`'s `assert _Z_95 != 1.96`.
- **Kept open (1):** `e1a6c4d2` (2026-08-31, `ModelGateway.from_env()` resolves every declared
  provider's `{env:}`/`{file:}` substitution eagerly, so pointing a live harness at
  `config/opencode.example.json` raises `ModelConfigError` on an `openai` provider it never calls
  unless `OPENAI_API_KEY` is set to a placeholder first) — **mechanism re-derived and exactly
  correct**, and correct for a reason the entry did not name: it is not `from_env` but the
  constructor it calls, `ModelGateway.__init__` → `_build_providers`, which builds a `ProviderSpec`
  for the union of *every* catalog and overlay provider id and calls `_substitute` on each one's
  `apiKey`, raising `ModelConfigError("{env:NAME} is not set")` — eagerly, not lazily per `.llm()`
  call. Four independent scripts have now hit and worked around it. It is **not** promotable into
  this agent's own files: it is a falkor-chat config-loading mechanic, not an ML-method or
  LM-Studio fact, and `lm-studio-model-notes.md` is the wrong home for it (the same reasoning that
  discarded `b1e3f6a2` in chunk A). Its one genuinely useful home is a clause in
  `falkor-chat/docs/manuals/llm-provider-config.md` §2, which tells an operator that a missing
  `{env:}` variable fails startup but **not** that this applies to providers nothing ever resolves
  to — the surprising half. That file is `tico`-owned and outside `cobb`'s write remit, so the
  entry is logged here, opened as **K-003** in `plan.md`, and tagged `MENTIONS → tico` in the graph
  so it resurfaces in that agent's own distillation pass rather than being lost. Counter-argument
  weighed and recorded: the fact *is* already documented at the point of use, in a full comment in
  `server/tests/eval/test_guard_calibration_live.py` naming `_build_providers` by line — and the
  entry's own author found it by copying that precedent. That is why this is a manual clause and
  not a `falkor-chat/AGENTS.md` line: an always-loaded context file is the wrong price for a fact
  that binds only when someone writes a new harness driver.
- **`MENTIONS` edges added: 1** — `e1a6c4d2` → `tico`, per the kept-open routing above. The other
  eight touch no agent substantively beyond the producer, and every one of them is already
  published in the code or docs of the component it concerns, so tagging would only queue
  already-documented content into another agent's pass (chunk A's reasoning, unchanged).
- **Verification basis:** **no live model probe was needed or used.** Four entries are pure
  statistics and were re-derived by computation in stdlib Python from the committed raw data
  (`judge_calibration.json`, the shipped MOVER-D regression fixtures) — a live probe would have
  added nothing a recomputation cannot settle. Five are falkor-chat mechanics and were re-derived
  by **reading source and committed artifacts**: `repository.py`, `services.py`, `executor.py`,
  `tools.py`, `modelconfig.py`, `config/models.json` and `git log` on the pin commit. No shared
  FalkorDB graph was read or written, no probe script was executed, and no scratch key was created.
  Publication checks deliberately grepped **outside** each component's `docs/` index — scripts,
  tests, source comments, `BACKLOG.md` and shipped module docstrings — which is where five of the
  seven discards were actually found.
- **Cleared:** 8 entries `DETACH DELETE`d after this entry was written; `e1a6c4d2` had only its
  `PRODUCED` edge resolved, leaving the node alive on its `MENTIONS → tico` edge.
- **Docs touched:** `claude/data-scientist/{data-scientist.md, kaizen/history.md, kaizen/plan.md}`.
  `lm-studio-model-notes.md` was read in full and **not** written to — nothing in this chunk was an
  LM Studio or small-model-realism fact, and its existing claims were re-checked against
  `config/models.json` and found current after chunk A's correction.

## 2026-09-07 — Learnings-graph distillation, chunk A of 3 (U13): 10 oldest entries — 2 promoted, 8 discarded

- **What:** `cobb` processed the 10 oldest `data-scientist` `:KaizenEntry` nodes in `kaizen_team`
  (2026-08-26 … 2026-08-30) per `agent-maintenance` §5. Every entry was re-derived independently
  rather than confirmed against its own cited evidence; three re-derivations **falsified** the
  entry outright and one found it fixed in code since it was written. Verification basis is stated
  per entry below — LM Studio *was* reachable this session, so four entries got live probes.
- **Promoted (2):**
  - `b6e2a1f4` (2026-08-29, published tool-calling benchmarks don't test the K-056 failure mode)
    → **prompt**, one clause on the "Model selection" bullet: a benchmark number is evidence only
    for the construct that benchmark actually measures. *Re-derived live* against the primary
    source (`gorilla.cs.berkeley.edu` BFCL V3 multi-turn blog), not the entry's own citation: BFCL
    multi-turn is **pre-scripted with human-labelled ground-truth trajectories** and grades
    dependent-call chaining within one task — it cannot observe spontaneous cessation of tool use
    in open-ended conversation. The falkor-chat-specific half is already published
    (`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §~776 and the xLAM table), so
    only the discipline-level rule was promoted; the project-scoped copy would never reach a
    future model-selection session.
  - `b2f7a2b5` (2026-08-29, `temperature: 0` is not run-to-run deterministic on this stack)
    → **knowledge base**, *folded into* `lm-studio-model-notes.md`'s existing provenance section
    (habit 2), not stacked as a new section — that file had already been written to four times
    this pass. The fold also **corrected a stale claim in place**: the section asserted "the repo
    sets **no** `temperature` key anywhere for any kind", which `falkor-chat/config/models.json`
    now contradicts (`temperature: 0` pinned for `qwen/qwen3-4b-2507` and `ministral-3-3b`). The
    39/40-vs-1/40 measurement itself was **verified by artifact, not by a live re-run** — the
    eval script was never committed, and re-running 40 conversations would touch shared graphs;
    the rate is published with a Wilson CI in
    `falkor-chat/docs/plans/workflow-salesperson-demo-coordination.md` (U40) and independently
    reinforced by K-062 in `falkor-chat/docs/BACKLOG.md`, which measured that the pin did **not**
    narrow a much wider between-session swing. That K-062 statement is the stronger, later form
    of the same lesson, so the promoted text cites it rather than restating the entry.
- **Discarded (8):**
  - `e2a2b1a0` (2026-08-26, `ws:acme` holds only QA fixture data — "~10-12 entities, almost all
    Organization/Other, ~6-8 relationships, **no** Person/Location/Product/Event/Concept")
    — **falsified by a read-only live probe.** `ws:acme` today holds 544 `Entity` nodes across
    **all seven** taxonomy types (Other 146, Product 139, Concept 92, Organization 50, Event 49,
    Location 36, Person 32), 29 `Document`, 87 `Chunk`, 381 `RELATES_TO`. Every one of the five
    types the entry said were absent is present. The corpus grew after the entry was written, so
    its conclusion — that the document-ingestion graph is unusable as a second-schema golden-set
    candidate — no longer follows from its own premise. An entry that asserts a *live graph's
    contents* is the fastest-rotting kind there is.
  - `c4a7d891` (2026-08-28, LM Studio "silently ignores" `tool_choice: "required"`, and adding it
    triggers runaway repetition) — **mechanism falsified by live probe**, and already published.
    A control call (tools offered, no `tool_choice`) reproduced the skip shape exactly: 0
    `tool_calls`, plain prose, `finish_reason=stop`. Re-sending the identical request with
    `tool_choice: "required"` returned `finish_reason=tool_calls`, one clean `post_message` call,
    no text and **no** degenerate loop; sending `tool_choice: "auto"` explicitly did the same.
    So the serving stack **does** implement the field — the observed ignore was specific to one
    degenerate prompt state, not an LM Studio behaviour. The published account
    (`salesperson-tool-reliability-ml.md` §4.2) already hedges correctly ("for this request
    shape", n=1); it is the kaizen entry that over-generalized. The original byte-identical
    failing prompt was never committed, so that exact shape could not be replayed — a synthetic
    six-turn rapport conversation did not reproduce the skip at all (control tool-called).
  - `b7e3f1a2` (2026-08-29, qwen3-4b "systematically" emits numerics as quoted JSON strings,
    silently breaking a Cypher comparison) — **fixed in code, documented at the point of use, and
    not reproducible.** `querygen.py` now coerces a numeric-looking string by the property's
    declared type and raises on one that genuinely doesn't parse ("U29f fix A", with the quirk
    named verbatim in the docstring and the inline comment), and `tools.py`'s
    `_QUERY_REQUEST_INSTRUCTIONS` now says `a bare JSON number … (e.g. 50, never "50")`. Live
    probe: 5/5 bare JSON numbers at `temperature: 0` — three draws under the current hardened
    prompt and two under a loose prompt reconstructing the entry's own stated condition ("a
    string, number, or boolean"). The consequence clause ("no error, just zero matching rows") is
    now false by construction.
  - `c3a8b3c6` (2026-08-29, `gpt-oss-20b` HTTP 400 "peg-native format" on multi-tool-call turns)
    — **already published verbatim**, including the exact error string, 6/8 with a Wilson CI, the
    harmony-format-vs-grammar diagnosis and the repeated-`post_message` behaviour, in
    `salesperson-tool-reliability-ml.md` §~412-425. Not re-probed live: the model is
    `not-loaded` and a cold 20B MXFP4 load buys nothing the published account lacks.
  - `d4b9c4d7` (2026-08-29, a mid-node `ProviderCallError` after a committed tool call leaves
    `WorkflowRun.status=failed` with zero `StepRun`/`TraceEvent`) — **already published in the
    same passage** as `c3a8b3c6`, which states the observability gap in the entry's own terms
    ("left the engine's own `StepRun`/`TraceEvent` audit trail entirely unwritten for that turn
    … a total absence of any record for an action that actually happened"). Mechanism
    independently re-derived from source and confirmed, with one detail the entry missed: because
    `_record` never runs, the `StepRun -[:PRODUCED]-> Message` audit link and the `toolsUsed`
    property are lost too, so the committed `Message` is left **orphaned**, not merely untraced.
    The deferred record→trace lifecycle is already documented in `StepResult`'s own docstring
    (Option B, K-023), so no new prose was warranted.
  - `a1f3c9d2` (2026-08-30, a tool's internal LLM call resolves through the step-kind default,
    not the calling node's `config.model` pin) — **already published at the point of use.**
    `falkor-chat/docs/HISTORY.md` states it as a parenthetical on the K-057 verification line:
    "real `ModelGateway.from_env()` (not `StaticModelGateway` — `query_graph_data` resolves its
    own model independently of the `assistant` step's pin)", covering both the divergence and the
    test-double corollary. Re-derived from source and confirmed exactly:
    `executor.py` passes `requested=config.get("model")`, `tools.py`'s `QueryGraphDataTool.run`
    calls `self._models.llm("step", ws=ctx.ws)` with no `requested=`. One staleness note: the
    entry cites `salesperson@v4`; the shipped version is now `v5`.
  - `b1e3f6a2` (2026-08-28, the run-level `trace` flag persists across `resume_workflow_run`)
    — mechanism **re-derived and correct** (`trigger.py` rule 2 resumes a waiting run rather than
    starting a new one; `_drive_loop` selects the tracer from the persisted `run["trace"]`;
    `app.py` constructs `WorkflowTrigger` with no `trace` kwarg, so the default `False` stands),
    but it follows directly from `docs/DESIGN.md`'s run-model note that `trace` is a field **on**
    `WorkflowRun` gating all trace writes. Discarded as already-derivable from documented design;
    the weakest of the eight discards, and it is a falkor-chat harness mechanic rather than an
    ML-method fact, so it had no home in this agent's knowledge base either.
  - `a1e6f1a4` (2026-08-29, `/v1/models` lists more models than `config/models.json` pins; unloaded
    models auto-load; "no `lms` CLI available in WSL2") — **near-duplicate of content already
    promoted into `lm-studio-model-notes.md` this pass, and its one novel clause is false.** The
    19-model catalog, the JIT auto-load on first request, and the `lms` CLI situation are all
    already in that file; re-confirmed live today (19 models, `ministral-3-3b` and `gpt-oss-20b`
    both listed and both `not-loaded`). The "no `lms` CLI available in WSL2" clause is wrong — the
    Windows binary is reachable from WSL and the file has said so, `Verified: 2026-09-07`, since
    U11.
- **`MENTIONS` edges added: none.** Four entries touch another discipline (`d4b9c4d7` and
  `b1e3f6a2` are engine/observability, `a1f3c9d2` carries a test-double-fidelity corollary), but
  every one of them is already published in `falkor-chat`'s own docs, so tagging would only queue
  already-documented content into another agent's future pass.
- **Verification basis:** live probes for `e2a2b1a0` (read-only Cypher against `ws:acme`),
  `c4a7d891`, `b7e3f1a2`, `a1e6f1a4` (LM Studio, reachable this session) and `b6e2a1f4` (primary
  source over the web); source re-derivation for `b1e3f6a2`, `d4b9c4d7`, `a1f3c9d2`; **artifact
  only** for `b2f7a2b5` and `c3a8b3c6` (the eval scripts were never committed, and re-running
  either would touch shared graphs or cold-load a 20B model). No shared graph was written to and
  no scratch key was created.
- **Cleared:** all 10 entries resolved out of `kaizen_team` after this entry was written.
- **Docs touched:** `claude/data-scientist/{data-scientist.md,lm-studio-model-notes.md,
  kaizen/{history,plan}.md}`.


## 2026-09-07 — `lm-studio-model-notes.md`: a wording-iteration eval caution promoted from `coder`'s kaizen distillation (U11)

- **What:** New section — on `mistralai/ministral-3-3b`, a *second* iteration of targeted prompt /
  tool-description wording cut net task correctness from 16/20 (80%) to 14/20 (70%) while not
  improving the defect it targeted, because it suppressed a multi-call self-correction the model
  had been performing on its own. Two rules follow: score **net** correctness on every wording
  iteration, not just the targeted defect rate; and treat an observed self-correction behaviour as
  part of the baseline you can lose. Model-specific effect size, portable direction.
- **Why:** the numbers and the mechanism exist only in `falkor-chat/docs/HISTORY.md` (2026-08-31,
  K-057) and `docs/reviews/salesperson-tool-reliability-ml.md` §11/§14 — a milestone lookup doc and
  a per-item review, neither of which a `data-scientist` designing a *future* prompt-wording eval
  would read. The generalized caution is exactly what this knowledge base is for. Re-derived by
  `cobb` from both documents (no model was loaded on the shared LM Studio server to verify).
- **Files:** `claude/data-scientist/lm-studio-model-notes.md`. Source disposition:
  `claude/coder/kaizen/history.md` (2026-09-07, U11).

## 2026-09-07 — `lm-studio-model-notes.md`: the measurement-surface section refined (second inbound promotion from `architect`'s kaizen distillation, chunk B)

- **What:** Three clauses folded into the section created earlier the same day (entry below) — **no
  second section**, per chunk A's own hand-off note. (1) Both catalog routes are effectively free
  (19 models in 1.6-6.5 ms over six calls, `/v1/` no faster than `/api/v0/`), so the choice between
  them is about content, never cost. (2) Each `lms.exe` call costs **~0.30 s** (0.30/0.31/0.32 s
  over three `lms ps --json` runs) — the WSL-to-Windows subprocess price, ~100x the HTTP routes.
  (3) A request naming an unloaded model triggers **JIT auto-load**, so `lms ps --json` → `[]`
  means the next call is cold, not that it will refuse.
- **Why:** `cobb` distilling `kaizen_team` entry `6f2b1d94-3c7a-4e51-b0d8-9a2e5c74f118`
  (2026-09-03, produced by `architect`) — a refinement of `443cc4cc`, not a contradiction.
- **Verified:** re-derived live 2026-09-07 against the LM Studio server, **without loading a
  model** — the same constraint chunk A worked under, since the server is shared with live
  sessions. The JIT clause is corroborated by an independent measurement in
  `docs/plans/small-model-benchmarking-ml.md` §11.4.
- **One figure deliberately not banked.** The source entry carried a 21.068 s cold-call
  measurement. §11.4 measures 3 625.0 ms on the same box, and plan v1.10 records that no load-cost
  figure is left anywhere the design is sized against, the two measured cold loads differing by
  ~6x. The KB therefore states the *behavior* plus the stable invariant — LM-Studio-side `ttft`
  **excludes** the JIT load while wall clock includes it — and cites §11.4 for the numbers.
- **Full disposition record:** `claude/architect/kaizen/history.md`, 2026-09-07 chunk-B entry.
- **Plan items:** none.

## 2026-09-07 — `lm-studio-model-notes.md` gains the measurement-surface section (inbound promotion from `architect`'s kaizen distillation)

- **What:** New section in `claude/data-scientist/lm-studio-model-notes.md` — *"The machine-readable
  measurement surface is on `/api/v0/`, not `/v1/` — and `lms` is reachable from WSL only as
  `lms.exe`"*. It records which LM Studio facts can be collected automatically (`/api/v0/models`
  per-model fingerprint fields; `/api/v0/chat/completions`'s `stats`/`model_info`/`runtime`, all
  absent from the `/v1/` OpenAI route; `lms.exe` reachable from WSL at
  `/mnt/c/Users/<user>/.lmstudio/bin/lms.exe` with `server status --json`, `ps --json` and
  `load --estimate-only` working) and the two that cannot — **the LM Studio app version** (`lms
  version` prints only a CLI commit hash) and **the KV-cache setting** (no API field, no `lms load`
  flag), which must be operator-attested.
- **Why:** `kaizen_team` entry `443cc4cc-9ad2-4ea9-b93c-4ae62412bbb1` was produced by `architect`
  (2026-09-02, designing `docs/plans/small-model-benchmarking.md`) but is squarely in
  `data-scientist`'s domain, and this file is the live-verified knowledge base for exactly it.
  Per the U2/U6 precedent, `cobb` dispositioned it here directly rather than `MENTIONS`-tagging a
  second agent onto it. Full disposition record: `claude/architect/kaizen/history.md` 2026-09-07,
  unit U7 of `claude/docs/plans/kaizen-distillation2-coordination.md`.
- **Verification:** re-derived live 2026-09-07, not taken from the entry. `GET /api/v0/models` → 200
  with the ten documented per-model keys; `command -v lms` exits 1 while the `.exe` path answers;
  `lms server status --json` → `{"running":true,"port":1234}`; `lms ps --json` → `[]`; `lms load
  --help` shows `--estimate-only` and `--context-length` and no KV-cache flag; `lms version` prints
  `CLI commit: <sha>` only. The one clause **not** re-derived is the
  `/api/v0/chat/completions` response shape — confirming it live would mean loading a model on the
  shared LM Studio server, which the distillation pass must not do; verified instead against
  LM Studio's own REST reference (`lmstudio.ai/docs/developer/rest/endpoints`), which quotes all
  three field groups. The section carries its verification date.
- **Open, for whoever runs `architect`'s chunk B:** entry `6f2b1d94-3c7a-4e51-b0d8-9a2e5c74f118`
  (2026-09-03) covers the same surface with response timings. It refines this section rather than
  contradicting it — **fold it in, do not add a second section**.
- **Plan items:** none.

## 2026-09-02 — "This lab's terrain" no longer describes the retired salesperson app (U6 of `salesperson-ui`)

- **What:** `data-scientist.md` `:56` listed two graph-backed AI apps, the second being
  *"`salesperson` (LangChain/LangGraph over a FalkorDB knowledge graph, optional local LLM via LM
  Studio)"*. That component is being retired; the salesperson agent now lives inside falkor-chat's
  workflow engine, and the `salesperson/` name is being taken over by a React storefront UI with no
  ML surface at all. The sentence now reads: falkor-chat (FalkorDB as the single store; GraphRAG =
  in-graph vector search + traversal, **and the workflow-engine-backed `salesperson` agent it
  hosts**). Everything after it — read the component's docs and actual prompts before opining — is
  unchanged.
- **Why:** left as written, an ML-method question about "the salesperson agent" would have been
  answered against a LangChain/LangGraph architecture that is no longer the one running. Found by
  `cobb` in the cross-agent sweep of `salesperson-ui` unit U6.

## 2026-08-25 — `kaizen_team` distillation: 3 current-shape entries, all discarded as already documented

- **What:** `cobb` processed all 3 current-shape entries `PRODUCED` by `data-scientist` in the
  shared `kaizen_team` graph (agent-maintenance skill §5, unit U7 of the team-wide distillation
  pass, `claude/docs/plans/kaizen-distillation-coordination.md`). Legacy (`author`-property) read
  returned 0 rows — nothing pre-M8 remained for this agent.
  - **`f3b1a2c4` (2026-08-22) — discarded, already documented.** Fact: falkor-chat's
    `llm.extract_own_line_json_object`'s "whole reply is one JSON object" branch returns before
    `require_key` is ever consulted, so `extraction.py` cannot rely on `require_key="entities"` to
    validate shape. **Re-verified against live code:** `server/falkorchat/llm.py:561-563`
    (`whole = _load_json_object(text); if whole is not None: return whole`) returns before
    `require_key` is referenced at line 578 — still true. **Already captured twice in project
    docs**, independently: `extraction.py`'s own module docstring (lines 16-24, "ML note F1") and
    `docs/plans/document-ingestion-ml.md` §F1/§3.2 — both state the exact same finding and its
    consequence (`extraction.py` does its own mandatory schema validation, doesn't rely on
    `require_key`). No third copy needed.
  - **`b3e1b6f0` (2026-08-24) — discarded, already documented.** Fact: running falkor-chat with
    two `ModelGateway` kinds (chat/extraction + embedding) against the same local LM Studio
    instance under concurrent background load causes model-swap thrashing (HTTP 400 engine-startup
    aborts, sometimes 10+ min to settle). **Re-verified:** this exact finding, with the same root
    cause (two models competing for load slots) and the same workaround (serialize the calls), is
    already written up in `docs/reviews/document-ingestion-ml.md` §1 (the entry's own source
    document) and independently reproduced and recorded again in
    `docs/test-reports/document-ingestion-report.md`'s "Environment notes" section, which
    explicitly cross-references the prior finding as "the same class of instability". Already a
    standing, twice-confirmed project-docs fact — no promotion needed.
  - **`c4f2c7a1` (2026-08-24) — discarded, already documented.** Fact: the shared
    `~/.config/opencode/opencode.json` LM Studio `baseURL` (`192.168.0.69:1234`) was unreachable
    from this WSL2 session while `localhost:1234` answered; `FALKORCHAT_OPENCODE_CONFIG` lets a
    scratch copy override the `baseURL` without touching the shared file. **Re-verified:** the
    stale-LAN-IP fact and the `FALKORCHAT_OPENCODE_CONFIG` workaround are both already recorded in
    `docs/reviews/document-ingestion-ml.md` §1 and repeated in
    `docs/test-reports/document-ingestion-report.md`'s "Environment notes" (which explicitly notes
    "still unfixed as of this pass" — i.e. the shared config file is deliberately never edited).
    Considered `opencode/local-llm.md` (this lab's other LM-Studio-connectivity doc) as an
    alternative home, but that file is scoped to OpenCode agent/provider config for Severino, not
    falkor-chat's `ModelGateway`/environment operations — the existing falkor-chat docs are the
    right home and already have it.
  - **No `MENTIONS` tags added** — all three entries are squarely `data-scientist`'s own domain
    (falkor-chat LLM extraction/ingestion ML work), not about another agent.
  - **No plan items opened** — nothing kept open; every entry resolved to a clean discard with a
    verified, cited existing location.
- **Why:** Routine distillation pass — three-plus-week-old entries from the K-050 M5 document-
  ingestion ML review work, never yet processed.
- **Verified:** re-derived each fact directly (code read for `f3b1a2c4`; doc cross-reference for
  the other two, confirming the *same* fact, not just a citation that still resolves) rather than
  trusting the entries' own citations. All three `PRODUCED` edges resolved with `otherRemaining=0`
  (no `MENTIONS` edges on any), so each node was fully `DETACH DELETE`d after this entry was
  written.
- **Docs touched:** `claude/data-scientist/kaizen/history.md` only — `plan.md` unchanged (nothing
  kept open).

## 2026-08-24 — Prompt-waste compression, Stage C5: two "this lab" attributives cut from the LLM-as-judge bullet
- **What:** Unit C5 of `claude/docs/plans/prompt-waste-reduction.md` (with `tdd-engineer.md` and
  `qa-engineer.md` as one commit). **2,098 → 2,083 w (−15, −0.7%).** Two edits, one pass.
- **The file's citation habit (plan finding 9 — name it before cutting):** the **"this lab"
  attributive parenthetical**. This is the only prompt on the team that cites the lab as the
  *authority* for a promoted rule — a habit it has because its rules come from evaluation work done
  here. `cobb` independently checked the four remaining `this lab` instances against that definition
  and confirmed the habit inventory is **exhausted at two cuts**; none of the four qualifies.
- **Removed (class 6, provenance):** from the "LLM-as-judge with its validity caveats" bullet,
  "— a real, recurring pattern in this lab's guard judges" and "(a real hardware-driven pattern
  here)". Both state where the lesson came from, not how to recognize the situation; both triggers
  survive intact in the bolded lead-ins ("**For a judge deliberately biased toward one verdict**",
  "**When the judge collapses onto the same model as the agent-under-test**"), and the recognition
  aid "(e.g. bias-to-suspend / abstention-favoring by design)" was deliberately kept.
- **Gate (b) — where the provenance now lives, in full:** this file's **2026-08-11** inbox-distillation
  entry (bias-to-suspend judges needing class-conditional gating, promoted into the LLM-as-judge
  bullet) and its **2026-08-21** `kaizen_team` distillation entry (`8f6e20a1` — the judge collapsing
  onto the same model as the agent-under-test). Neither entry spelled out the two attributives'
  specific content, so recording it here completes the trace: the bias-to-suspend shape recurs
  across this lab's **guard judges** specifically, and the judge/agent model collapse is **forced by
  hardware** — one local LM Studio model serving both roles — rather than chosen. The weak *prior*
  the second parenthetical carried ("expect this here") is not orphaned by the cut: the "Model
  selection" bullet two entries up already establishes that this lab runs local models via LM
  Studio, and the trigger is directly observable to the agent designing the eval.
- **Gate (a) inventory — all preserved:** the advisory-scientist identity and the never-implement
  rule; the isolated-subagent contract (`AskUserQuestion` unavailable, return the sharp question);
  all three standing modes and the `architect`/`analyst` altitude splits; every Core-expertise
  bullet across the five sections, including the full `1/n`-vs-CI-width gate rule, the
  report-each-probe-individually rule, the class-conditional-rates gating rule with its κ critique,
  the split-the-self-preference-caveat rule, the `graph-dba` boundary, the Wilson-interval
  convention with its anti-trigger parenthetical, and the perishable-facts rule in **both** its
  places; "This lab's terrain"; all five "How you work" steps; the three deliverable kinds with
  their exact paths, structures and verdict scale; the header-block instruction; all four
  Guardrails including the write-guard scope and the interactive-commit grant with its full
  never-list and delegated-subagent carve-out; the Cypher capture template and call line. Audit
  check-8 tokens verified present after the edits. *(This file carries no `CPG:` line — correct; it
  is not one of the six CPG-adoption agents.)*
- **Considered and rejected — judged keeps, both upheld by the lint** (recorded in `kaizen/plan.md`):
  "(e.g. 2-of-3 probes failing)" and "(still a legitimate rubric-following signal)" — both change
  what the agent does; the perishable-facts rule stated in both "Model selection" and the
  "No fabricated numbers" guardrail; and "You do **not** implement" in the opening paragraph vs. the
  `Write`/`Edit` guardrail.
- **One recorded rationale corrected by the lint.** The C5 inventory justified keeping the
  perishability rule twice on the grounds that *a capability claim isn't a number, so the guardrail's
  first sentence doesn't cover it*. That explains why the guardrail's second sentence exists but not
  why it isn't redundant against the fuller statement in "Model selection". The keep survives on a
  different basis — **two decision points**: model-selection time vs. claim-writing time, and a
  deliverable can carry a capability claim with no model-selection step in sight. Corrected in
  `kaizen/plan.md` so a later unit doesn't cut the clause after finding the stated reason weak.
- **Residual after this unit: 31 w, all class-7 keeps; class-6 = 0 w.** `cobb`'s measurement. The
  file is at its editorial floor.
- **Verified:** gates (a)–(e) green. `./claude/scripts/audit-team.sh` **PASS**. `cobb` §7 lint:
  **0 blockers, 0 majors, 0 minors, 0 findings introduced by the edits** — the cleanest of the three
  files, with dimensions 1–6 clean across the board and enforcement parity independently confirmed
  against the shared guard core. Orphan-phrase grep across the repo clean.
- **Plan items:** none opened. Two judged-and-kept records and one rationale correction added to the
  parking lot.

## 2026-08-23 — Prompt-waste compression, Stage B wave 1 (boilerplate sweep)
- **What:** Applied the pilot-validated boilerplate compressions from
  `claude/docs/plans/prompt-waste-reduction.md` (§3 doctrine, Stage B), same shapes as the
  `architect.md` pilot — two of the three shared blocks; this file has no CPG-freshness clause.
  (1) Interactive-commit-grant passage (§ Guardrails, Bash bullet): dropped the provenance
  sentence "Stakeholder decision, 2026-08-21 — see `kaizen/history.md`." and ", same as before";
  the "(spawned via `Agent`/`Task`)" clarifier moved from the interactive-definition parenthetical
  to the carve-out sentence (was stated in both). (2) Learning capture: intro dropped "directly"
  and "identified by a real `:Agent` node it's `PRODUCED`-linked to," (the Cypher template below
  shows the MERGE + PRODUCED edge); tail dropped the inbox-replacement history sentence and
  "exactly like the old inbox was".
- **Rule inventory (gate a), edited regions — all preserved:** interactive-mode definition,
  explicit-path grant, full never-list, delegated-subagent carve-out, deliverable left for `teco`
  post-verification (block 1); capture trigger + graph + Cypher template, skip-known-facts,
  raw-capture/`cobb` promotes, never edit own definition (block 2). Audit-check-8 tokens
  untouched.
- **Removed class-5/6 material, recorded where:** inbox-replacement history → this file's
  2026-08-21 "kaizen/inbox.md deleted" entry; commit-grant provenance → this file's 2026-08-21
  grant entry + `claude/AGENTS.md` § Hook machinery.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint pass on the result.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** The Bash guardrail's "investigation only" bullet now also grants: when running
  interactively (`claude --agent data-scientist`, a human present turn-by-turn — not a delegated
  subagent), may `git add`/`git commit` its own method note(s)/methodology review(s) from the
  session, by explicit path, never bulk-staged/pushed/reset/rebased/amended; the grant does not
  apply when spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there (including this agent's own distillation, immediately below) has already been distilled and cleared — and this file's own pre-migration content was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: 8 entries — 3 to the prompt, 3 to `lm-studio-model-notes.md`, 1 discarded as an already-resolved one-off, 1 blocked from its original target by archival and redirected

- **What:** `cobb` processed all 8 `author:'data-scientist'` entries in the shared `kaizen_team`
  graph (agent-maintenance skill §5). Read via single-column `substring()` paging (avoids the
  multi-column chat-rendering corruption documented in `cypher-mcp/README.md`).
  - **Promoted (3) → `data-scientist.md` prompt, folded into existing bullets (not new top-level
    bullets — kept cognitive load flat):**
    - `8f6e20a1` — when the judge collapses onto the same model as the agent-under-test, a
      blanket self-preference caveat conflates a fixed-content sub-pass (little risk) with a
      live-output sub-pass (real risk); split it. Folded into the LLM-as-judge bullet.
    - `11475cca` + `f3b90490` (same "Evaluation engineering" bullet, folded together) — before
      blessing a zero-tolerance small-golden-set gate, compare the one-unit delta (`1/n`) to the
      metric's CI width at that `n`; and when a probe set is gated with AND/OR logic, still
      report each probe's individual outcome in prose, not just the bloc boolean — both are
      reusable pre-sign-off habits, not falkor-chat-specific.
    - `e6f3a1c4` — this lab's small-n pass/fail convention is the Wilson score interval, not
      Clopper-Pearson/rule-of-three (independently re-derived in Python, matches the archived
      `m3-guard-calibration.md` §6 figures to rounding). **Redirected from its own suggested
      target**: the entry's context doc, `golden-set-expansion-ml.md`, is now `Status: archived`
      (frozen except header-pointer metadata) — the fact is durable and reusable well beyond that
      one document, so it went into the "Uncertainty" bullet instead of being lost.
    - Folded `9642e9e0` in kind but routed to the knowledge base instead (next bullet) — its
      LM-Studio-specific verification commands are better as a consultable recipe than prompt
      prose the agent pays for every session.
  - **Promoted (3) → new sections in `claude/data-scientist/lm-studio-model-notes.md`:**
    - `e820b9e0` — Mistral/Ministral GGUF templates enforce strict role alternation (HTTP 400 on
      two consecutive same-role messages); Qwen3 tolerates it silently. Live-verified against
      `falkor-chat`'s real `triage@v1` intake prompt.
    - `a1c2ef6d` — two differently-named LM Studio catalog ids can alias one loaded model slot
      (confirmed via `/api/v0/models` state-flipping and byte-identical completions) — verify
      before assuming two entries are two different weight files.
    - `9642e9e0` — a live-run report's provenance can silently diverge from the repo's static
      model config per box (the provider file is machine-local, outside the repo, and
      `ProviderCatalog` validates only the provider id, not the model id); two reusable habits:
      live-check `/api/v0/models` rather than trusting static config, and grep the repo for
      `temperature` before assuming a pinned sampling parameter.
  - **Discarded (1):** `b3f2a8e0` — a golden-set fixture id-numbering footnote (a draft plan
    section had "used" `tn-08`/`tn-09` ahead of the real fixture file) whose only plausible
    target, `golden-set-expansion-ml.md`, is now archived and the numbering it describes is
    already resolved in the delivered `golden_guards.jsonl`; too narrow and already-moot to carry
    forward into a different document.
  - **Verified:** live-re-derived `e6f3a1c4`'s Wilson figures in Python (stdlib `math` only) before
    promoting — reproduced `wilson_upper(0,10)=27.75%`, `wilson_upper(0,30)=11.35%`, matching the
    entry's own citation.
  - **Docs touched:** `claude/data-scientist/{data-scientist.md,lm-studio-model-notes.md,
    kaizen/history.md}`.
- **Why:** User-requested distillation pass ("who's next?" → data-scientist had the oldest pending
  entries, 2026-08-15).
- **Plan items:** none opened — every entry had a direct promotion target or a clear discard
  rationale.

## 2026-08-21 — Persona fix: dropped stale "senior" framing (team certification, §7 fold-in)

- **What:** Opening line "You are a senior **data scientist and AI/ML specialist**..." →
  "You are a **data scientist and AI/ML specialist**...". Dropped the one word.
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in.
  The team dropped "senior" framing collection-wide on 2026-06-20 (overconfidence concern;
  persona-prompting evidence shows role labels are weak-to-neutral for correctness —
  `claude/cobb/kaizen/history.md`, 2026-06-20 entry, "Collection harmonization" — applied
  explicitly to `cobb` itself and stated as bringing "the whole Claude collection" in line).
  `data-scientist.md` had never been swept for it; genuine drift against a dated, explicit
  team decision, not a fresh design call.
- **Verified:** `bash claude/scripts/audit-team.sh` — same 113 PASS / 2 pre-existing FAILs before
  and after (diff, not bare gate).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_data-scientist`), mirroring `graph-dba`; `mcp__cypher__query` granted
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_data-scientist` (FalkorDB, via `mcp__cypher__query`) instead of appending
  to `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 4
  pre-existing entries were parsed out programmatically and imported into the graph verbatim
  (entryId assigned, `author: 'data-scientist'`), preserving every field; its own header explains
  the freeze and gives the live-read query. Frontmatter `tools:` gained `mcp__cypher__query` —
  this agent previously had no MCP tool access at all, needed now for both this capture path and
  any future graph reads. The trailing "Your write guard allows exactly this inbox path" clause
  was dropped — the write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP tool, so
  it no longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-11 — Inbox distillation: 4 entries — 1 prompt addition, 1 new knowledge base, 1 to `python-web-quirks`, 1 discarded as stale

- **What:** `cobb` processed all 4 entries in `data-scientist/kaizen/inbox.md` (§5).
- **Promoted:**
  - Bias-to-suspend judges need class-conditional gating (false-advance/advance-recall), not
    κ/accuracy → new clause in "Core expertise → LLM systems → LLM-as-judge".
  - Ministral-3B vs. Qwen3-4B tool-calling reliability → new on-demand knowledge base,
    `claude/data-scientist/lm-studio-model-notes.md`, pointed to from "Core expertise → Model
    selection".
  - Bare `json.loads` on an LLM judge's output being fence-fragile → `skills/python-web-quirks/
    SKILL.md` (general knowledge; the project-specific instance already has an open tracking item,
    K-027, in `falkor-chat/docs/BACKLOG.md`, so no new backlog action needed).
- **Discarded:** `read_thread`'s `authorType` being a list, not a string — the flagged gap was in a
  since-completed M3 plan doc's prose; the live query itself (`labels(author) AS authorType`) is
  self-documenting (`labels()` obviously returns a list) and is already cross-referenced correctly
  in `falkor-chat/docs/HISTORY.md`.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/data-scientist/{data-scientist.md,lm-studio-model-notes.md,
  kaizen/{history,inbox}.md}` · `skills/python-web-quirks/SKILL.md`.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Method notes and methodology reviews open with the canonical header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line added to *Your deliverables*, after the two document bullets and before the "return the path" line: *"Open the document with the header block from root `AGENTS.md`."* Placed so it covers both written deliverables (`docs/plans/<slug>-ml.md` and `docs/reviews/<slug>-ml.md`) and not the inline-consultation bullet, which produces no document. No frontmatter, hook, `description` or catalog change.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6 makes a three-field header (`Status:` · `Owner:` · `Tracks:`) the repo's lifecycle signal, replacing the milestone filename prefix and the move-to-`archive/` rule; both `-ml` documents are in the closed role set and both are cited by path from an architect plan, so they need the same header as everything they sit beside. The line is a **pointer, not an inlined template** (v1.4 M20) — root `AGENTS.md` is already in every agent's context via the root `CLAUDE.md` `@AGENTS.md` import — and is byte-identical across the six producing prompts because the convention's coverage check greps for it literally. `claude/README.md` row 18 re-checked — it cites both write paths and the hook, not document structure; no edit needed.
- **Plan items:** none. (K-001's first-run shakedown, when it happens, now also exercises the header block.)

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 676 → 606 chars (-10%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (data-scientist↔architect, data-scientist↔analyst, data-scientist↔graph-dba) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`, `teco`, `tico`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `data-scientist`'s `guard-ds-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed methodology-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1472 to 674 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-ds-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/data-scientist/hooks/guard-ds-doc-writes.sh`) to `$HOME/.claude/agents/data-scientist/hooks/guard-ds-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/data-scientist` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Created
- **What:** Initial version of the `data-scientist` agent — the team's AI/ML/data-science specialist, created to work alongside `architect` (supplies the ML/DS method inside a design) and `analyst` (methodology review of plans/code). Advisory-only shape chosen by the user over a hands-on (graph-dba-style) shape: read-only on code, `Write`/`Edit` scoped to method notes (`docs/plans/<slug>-ml.md`) and methodology reviews (`docs/reviews/<slug>-ml.md`), harness-enforced by `hooks/guard-ds-doc-writes.sh` (PreToolUse, matcher `Write|Edit`, same contract as the analyst's guard but allowing both doc homes). Tools match architect/analyst (`Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`); model opus; subagent-aware (questions return as the deliverable).
- **Why:** The team had no ML/DS-methodology depth — model/embedding selection, RAG/GraphRAG evaluation design, metric choice, experiment design, statistical validity all landed on generalists. This lab's two themes (graph-backed AI apps, agent engineering) make the gap recurring.
- **Boundary pairs declared** (added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS`, reciprocal clauses added to partners' descriptions): `architect:data-scientist` (software plan vs. ML method inside it), `analyst:data-scientist` (general static review vs. methodology review), `graph-dba:data-scientist` (in-graph vector mechanics vs. embedding/eval method). teco's routing table + handoff contracts gained a data-scientist row/entry.
