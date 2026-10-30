# Connected code graph — ML & evaluation method note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (`docs/requirements/connected-code-graph.md`) · **Extends:** `docs/plans/connected-code-graph.md` · **Extended by:** `docs/plans/connected-code-graph-p0-ml.md`, `docs/plans/connected-code-graph-eval-protocol-ml.md` · **Version:** 2

_Revision 2026-10-26 (v2): added §8 (stakeholder decisions S1–S38 and the re-checked use cases:
statistical meaning of the new and changed ACs, AC-24 and RCA-baseline designs, untested-code risk
ranking, new stakeholder questions); §1–§7 are v1 and still hold except where §8.1 supersedes them._

## 0. Question, decision at stake, verdict

**Question.** The architect's proposal (`docs/plans/connected-code-graph.md`, v1) keeps every
*fact* deterministic and pushes ML into a few named places: plain-language interpretation, RCA
symptom anchoring and ranking, answer phrasing, and optional model-suggested links (§3.6–§3.7,
§3.13). It leaves the method for those places, and the whole evaluation design (§5.2, R4, R12),
to this note (§6.3). The decision this note serves: **what gets built in P0/P1 for each ML
touchpoint, and what numbers each AC is judged against.** That fixes how much LLM the system
depends on (and so how far a local-only deployment is viable), and how expensive the ground-truth
sets are.

**Constraints that bind.** FR-19 (same facts for the same question and snapshot); FR-8 (over-list,
flag doubt; a miss is worse than a false alarm); FR-9 (local-only must work, so every LLM
touchpoint has to be viable on this lab's small LM Studio models); "correctness over speed".

**Verdict on the proposal's ML/evaluation design: sound with changes.** The load-bearing
decision is right: facts are computed at build time and queried deterministically, and the LLM
sits only at the edges. Five changes are needed before P0 freezes contracts (§5): a dev/held-out
split with a pre-registration rule for every ground-truth set (M1); a two-part noise bound that
actually stops AC-5 from being gamed, including a rule for when a collapsed "any of N" group
counts as a hit (M2); a deterministic RCA ranker, with the LLM kept out of the ranking of record
(M3); deterministic, exact retrieval for symptom anchors, with no approximate vector search
on the fact path (M4); and no model-suggested links in answers, with the LLM's build-time role
recast as suggesting rule packs for a human to accept (M5).

**The recommendation in one line per touchpoint** (rationale in §2–§3):

| Touchpoint | LLM? | Method |
|---|---|---|
| Structured inputs (file:line, symbol, diff, route, topic, stack trace, test name) | **No** | Deterministic parsers; the LLM is never on this path |
| Free-prose question → structured request | **Yes, narrow** | Single-shot, schema-constrained extraction of use case + anchor *spans*; deterministic resolver; echoed and cached |
| Plain-language symptom → RCA anchors | **Yes, narrow** | LLM extracts search terms (echoed); deterministic lexical retrieval over identifiers, literals and log templates produces the anchor set |
| Error/log text → anchors | **No** | Build-time log-template extraction; deterministic tiered matching |
| RCA ranking | **No** (v1) | Transparent weighted feature score, fixed weights, stable tie-break; an LLM re-ranker only if it beats this on held-out bugs |
| Answer phrasing | **Partly** | Facts are template-rendered and never LLM-written; an optional LLM summary paragraph is checked for added entities |
| Proposing *possible* links | **No** (in answers) | Offline, LLM/co-change suggests candidate rule-pack entries for human acceptance; co-change is also a miss detector for evaluation |

## 1. Findings from the real system

What this note rests on, beyond the two documents it extends:

- **Small local models do schema-constrained extraction well, and multi-turn tool use badly.**
  On `model-bench`'s 2026-09-19 sweep (`model-bench/reports/catalog-sweep-2026-09-19-consolidated.md`),
  `nlq-structured-query` (natural language → a structured query over a small schema, exact match
  after canonicalization) had five 3–4B models at 1.000; `qwen/qwen3-4b-2507` scored 34/34
  (Wilson 95% [0.898, 1.000], p95 1.4 s). `tool-caller-shop-assistant` (multi-turn tool calling)
  topped out at 6/12 clean scripts, and the reference model scored 0/12. That pack is too small to
  rank models (its floor is 50 pp), but it gives no support for an agentic local loop. **Construct
  caveat:** the nlq pack's questions are short catalog lookups ("How much does the Wireless
  Charging Pad cost?") over a toy schema. Code questions carry identifiers, partial names, paths
  and pasted traces. The pack is evidence that the *output shape* (fill a small schema in one
  shot) is within reach of a local model. It is not evidence of accuracy on this task; §4.4's
  golden set measures that.
- **Temperature 0 is not determinism on this stack.** 40 identical runs against
  `qwen/qwen3-4b-2507` at a pinned `temperature: 0` diverged in 1/40
  (`claude/data-scientist/lm-studio-model-notes.md`). Any FR-19 argument that relies on "pin the
  temperature" is false here. Determinism has to come from *not re-asking the model* (§3.2).
- **Context and residency are tight.** This LM Studio setup keeps one model resident at a time:
  loading an embedder evicts the chat model (same notes file). `falkor-chat`'s storefront
  deployment found a shared 8,192-token KV pool overflowing at three concurrent turns (commit
  `e4cdba9f`, K-078 RCA DEF-1). An LLM step that reads code bodies for many hypotheses, or
  summarizes hundreds of answer items, does not fit. An LLM step that reads one question plus a
  schema (~1–2k tokens) does.
- **Grounding checks can be deterministic.** `model-bench`'s `chat-responder` pack scores
  grounding by containment against supplied context, with no judge. The top local model reached
  24/30 (0.800, [0.627, 0.905]). Prose generated by a small model adds unsupported content often
  enough that **the fact list itself must never be model-written** (§3.3).
- **The proving repo already holds labeled real failures.** `falkor-chat/docs/reviews/` has six
  `-rca` documents (e.g. `salesperson-cart-product-id-rca.md`, `salesperson-fabricated-prices-rca.md`)
  with known root causes and recorded symptoms. Six is too few to measure ranking, but they are a
  free external-validity check on a synthetic seeded-bug set (§4.3).
- **The proposal itself makes most recall risk a noise problem.** Under §3.3–§3.4, an unresolved
  call degrades to a name-matched *possible* link, and an unknown key degrades to "any provider of
  that kind". Recall is therefore lost only where **no rule fires at all** (an unrecognized
  convention, reflection, a callee reached under a different name). Precision is lost everywhere
  resolution is weak. That asymmetry decides several answers below. In particular, P0's
  call-resolution rate per language drives *noise*, while the recall risk lives in recognizer
  coverage.

## 2. Answers to the architect's §6.3 questions

### Q1 — Where LLMs are used, how each use is validated, and whether local suffices

**Precondition for all of (a)–(e): route structured input around the LLM entirely.** Most of
FR-17's starting points are not natural language: a file + line, a qualified symbol, a diff or
commit range, a URL route, a gRPC method, a topic name, a table/column, a config key, a stack
trace, a test name. Each has a deterministic parser (§3.6 step 1 of the proposal already
describes most of them). The plain-language front should **first try those parsers on the raw
text** (a pasted trace or route inside a sentence is still a trace or route) and call the LLM only
for what remains. That shrinks the non-deterministic surface to free prose, which is also the
only place an LLM adds value.

| Use | Recommended shape | Validation | Local viable? |
|---|---|---|---|
| **(a) Question → structured request** | One-shot, grammar/JSON-schema-constrained call. Output: `use_case` (enum of 5), `anchors[]` as **verbatim spans copied from the question** with a type hint, optional `scope`/`depth`/`direction`. The model never invents an identifier. Spans go to the deterministic anchor resolver. An ambiguous resolution returns *all* candidates as anchors (FR-8 spirit), or a "did you mean" list when the candidate count exceeds a cap | Golden question set (§4.4). Primary metric: **resolved-request exact match** (same use case and same resolved anchor IDs as gold), not string match | **Likely**, to be confirmed by §4.4. The output shape matches `nlq-structured-query`, where 3–4B models scored ~1.0, but the construct differs (§1). Do not build this as a multi-turn tool-calling agent on a local model (§1) |
| **(b) Symptom → RCA anchors** | The LLM extracts **search terms and candidate anchor kinds** (an entity name, a screen, an operation, an error phrase) and echoes them. Deterministic **lexical retrieval** over symbol names, string literals, log/exception templates, route keys and docstrings (BM25-style, identifier-aware tokenization: camelCase/snake_case splitting) produces the anchor set: every hit above a recall-calibrated score floor, *possible* status | Anchor recall on seeded bugs with a blind-written symptom (§4.3): the true cause's code region must be inside backward reach of at least one anchor | **Yes** for term extraction (a short prompt). Embeddings optional and not on the v1 fact path (§3.4) |
| **(c) Answer phrasing** | Facts (every item, every chain) are rendered by **deterministic templates**: grouped text and Markdown report. The LLM may add only a short summary paragraph, written from **aggregates** (counts per component and status, top-k items), never the full list | Deterministic **no-added-entity check**: every symbol, file, component, route or count the summary names must appear in the canonical answer, otherwise the summary is dropped and the answer ships without it. Omission is impossible by construction because the list is template-rendered | **Yes.** The input stays small because it is aggregates |
| **(d) RCA ranking** | **No LLM in v1.** Deterministic scorer (Q2) | Rank metric on the seeded-bug set (§4.3) | n/a |
| **(e) Model-suggested links** | **Not in answers.** Recast as an offline **rule-pack suggester**: a model (or co-change statistics, Q3) proposes candidate recognizer rules (e.g. "`busClient.emit(topic, …)` looks like a publisher"). A human accepts or rejects them. Accepted rules are ordinary deterministic packs | Precision of suggestions at human review; recall gain measured on the held-out set *after* acceptance, like any pack change | Optional. Hosted is fine where policy allows. Local is fine but weaker, and since it is offline and human-gated, a weak model costs reviewer time, not correctness |

**Why (e) changes shape.** A model-suggested link inside answers can only *add* links, never
remove them, so it cannot hurt AC-5. It does inflate noise, though, and it is unauditable
("the model thought so" is not an FR-20 evidence chain). Its real value is finding conventions no
pack recognizes, which is exactly the recall gap of §1's last bullet. Turning a suggestion into a
reviewed pack rule captures that value permanently, deterministically, with evidence, and keeps
FR-19 trivially true.

### Q2 — RCA ranking: method, evaluation set, metric, tolerated order variance

**Method: a transparent, deterministic, weighted feature score.** For a hypothesis *h* reached
from anchors *A*:

`score(h) = Σ_i w_i · f_i(h)`, ties broken by stable symbol ID.

Recommended features, each normalized to [0, 1] and each shown in the answer as its contribution,
which is what "the ranking shows its evidence" means (AC-12):

1. **Anchor agreement:** the fraction of anchors whose backward reach contains *h*. With several
   services' logs this is the strongest single signal.
2. **Chain strength:** status of the weakest hop (confirmed = 1, possible = 0.5, `runtime-built`
   = 0.25), times a decay in path length (e.g. `0.85^hops`).
3. **Recent change:** whether *h*'s enclosing symbol changed in the overlay diff or in the last
   *N* commits before the snapshot (git, deterministic per snapshot).
4. **Evidence lexical match:** the score of the error/log template match that produced the anchor
   (Q5), propagated to hypotheses on that anchor's chains.
5. **Test signal**, when supplied: *h* is covered by a failing test's runtime coverage; *h* is
   covered by passing tests only (a weak negative).
6. **Cross-boundary prior:** a small bonus for hypotheses across a contract hop when the symptom
   sits at a consumer. The proving use case is exactly that class, but the weight must be tuned on
   the dev split only, or it overfits the seeded set's construction.

**Weights:** hand-set priors in P5, then tuned on the **dev** split only (coordinate search or
logistic regression on "is the true cause" with ~6 features, which is still interpretable). Report
on the held-out split. A learned ranker beyond that (gradient boosting, learning-to-rank) is
rejected for v1: tens of seeded bugs cannot support it without overfitting, and it would lose the
per-feature explanation.

**LLM as ranker: rejected for v1, with a reversal trigger.** Reasons: (i) the local context budget
(§1) cannot hold code for dozens of hypotheses; (ii) it makes the order non-deterministic, which
FR-19 tolerates but users do not like; (iii) it cannot show a per-feature explanation, and an LLM
rationale is not evidence. **Trigger to revisit:** an LLM re-ranker applied to the deterministic
top-*k* (say k = 10, code snippets ≤ 1.5k tokens) that beats the deterministic ranker on held-out
hit@3 by McNemar's exact test (paired by bug; ≥ 6 net discordant wins is the observable floor,
`claude/data-scientist/statistical-method-techniques.md`). If adopted, it re-orders only within
the top-*k* and never removes a hypothesis, so the set stays fixed.

**Evaluation set and metric.** §4.3 gives the protocol and sizing. Metrics, in priority order:

- **Set recall** (the gate, per FR-12 and AC-12): the true cause is in the hypothesis tree, for
  every bug × input form. Reported with a Wilson bound over bugs.
- **hit@k** with k = 3 and 5 (primary ranking metric, because a person evaluates the top of the
  list), per input form.
- **MRR** (secondary; sensitive to the long tail).
- **Baseline:** the same candidate set ordered by path length alone (BFS order). The ranker must
  beat it on hit@5, paired by bug. A ranker that cannot beat BFS order is not earning its
  complexity.

**Order variance FR-19 tolerates in practice:** with the deterministic ranker, **zero**. The same
request on the same snapshot gives the same order. So the FR-19 allowance is only needed if a
model re-ranker is ever adopted. If it is, specify the tolerance as: top-1 identical in ≥ 95% of
repeated runs, and Kendall τ ≥ 0.8 over the top 10, measured over 5 repeats per held-out bug.
These numbers are proposals; the stakeholder sets them (§6, RQ3).

### Q3 — Statistical or embedding methods for inferring *possible* links

**Recommendation: none in answers in v1. Use two signals as *miss detectors* that feed
evaluation and pack authoring.**

- **Co-change history** (logical coupling: files in different units that change together in the
  same commits beyond chance, e.g. lift or a hypergeometric p-value over the commit window). This
  is a good detector of an unrecognized dependency. As an answer link it is poor: shared
  refactors, formatting sweeps and doc+code commits produce strong coupling with no runtime
  dependency, and new code has no history at all. Use: (i) a list of **unlinked cross-unit
  co-change pairs** each snapshot, triaged like runtime-only misses (fix a pack, or confirm no
  link); (ii) the RCA "recent change" feature (Q2).
- **Name/path/embedding similarity across services** (e.g. a consumer's `"/orders"` vs a provider's
  `OrderController`). Already covered better by the deterministic matchers plus the unknown-key
  fallback. Embedding similarity would add mostly noise, plus a dependency on a pinned embedding
  model in every build.

**If either is later promoted into answers**, it must (a) be computed at build time and stored
with the model or parameters fingerprint, so it is deterministic per snapshot; (b) carry its own
reason code (`co-change`, `model-suggested`) and the `possible` status; and (c) pass an
experiment on the held-out set: it raises recall on items the deterministic linker missed, at a
possible-precision for its own links no worse than the noise floor (§4.2). Promoting it without
that experiment is not justified.

### Q4 — Measuring AC-5 "never miss", with confidence, plus the noise bound

Full design in §4.1–§4.2. The short answers:

- **Construction protocol:** two kinds of ground truth, both **frozen before the run** and
  written by someone who has seen neither the system's output nor its packs: (1) a **census** of
  the proving repo's naturally occurring cross-boundary dependencies, written by blind source
  reading, stratified by link kind; (2) **planted items** in the multi-repo fixture system covering
  the hard shapes (wrapper, template literal, runtime-built URL, dynamic import, config-bound base
  URL, prompt mention). Each set is split into **dev** (pack authors may look at it) and
  **held-out** (only the gate runs it). A held-out item that is debugged becomes dev.
- **Recall with confidence:** 100% observed recall is a point estimate. Report the **Wilson 95%
  lower bound**. With zero misses that bound is n/(n + 3.84): **n = 35 gives ≥ 0.90; n = 73 gives
  ≥ 0.95.** Recommended held-out size: **≥ 75 seeded impacts pooled across kinds** for the
  headline claim, with per-kind counts reported individually (per-kind bounds will be wide, and
  that should be visible, not hidden).
- **Noise bound (R4):** two parts. (i) **Confirmed precision ≥ 0.95** observed, measured by
  labeling every confirmed item on the held-out answers (a false *confirmed* is a correctness
  defect, not noise). (ii) **Possible-item precision and burden**, measured on a labeled random
  sample of possible items, with a floor the stakeholder sets after P1 measures it (§4.2 gives the
  proposal and why the number cannot responsibly be fixed before P1).
- **Runtime-oracle misses** form a separate, human-independent recall estimate: the fraction of
  distinct runtime-observed cross-boundary edges present as links. Reported next to the seeded
  recall, never pooled with it, because it is biased toward exercised paths. Each oracle miss is
  triaged, and each confirmed miss becomes a new **held-out** item for the *next* gate (§4.1).

### Q5 — Log/error-text matching for failure-evidence anchors

**Build time — template extraction.** For every logging, exception-construction and
error-response call site (recognized by packs: `logging.*`, `logger.*`, `raise X(...)`,
`console.error`, slf4j, `Log.*`, `fmt.Errorf`, …), reconstruct the message as a **template**:
constant segments plus typed holes, using the argument value sets the bundle already carries
(f-strings, `%`-format, `.format`, template literals, `String.format`, slf4j `{}`). Store a
compiled regex, the constant tokens, and an **information score**: the summed inverse document
frequency (IDF) of the constant tokens across all templates in the system. `"error: %s"` scores
near zero; `"cart line references unknown product id %s"` scores high.

**Query time — tiered, deterministic matching** of each input line, after masking volatile
tokens (numbers, UUIDs, hex, timestamps, paths, quoted strings, Drain-style):

1. **Exact template match** (the regex matches the whole line or a whole segment) → anchor,
   status confirmed-for-matching.
2. **Constant-token containment:** all constant tokens of the template appear in order → anchor,
   possible.
3. **Weighted token overlap:** IDF-weighted Jaccard between the line's masked tokens and the
   template's constant tokens ≥ θ → anchor, possible, with the score as RCA feature 4.

**Keeping "never miss" while controlling explosion.** Do not cap the candidate count; FR-8 forbids
silent truncation. Instead: (i) **θ is chosen on the dev set** as the largest value that keeps
100% recall of the true emitting site, and is then frozen; (ii) templates with information below a
floor are matched but **collapsed** in the answer ("also matches 37 low-information templates such
as `error: %s`: expand"), and get a near-zero ranking weight; (iii) stack-trace frames, when
present, dominate: a frame is a confirmed anchor regardless of text matching. Explosion then shows
up as a long collapsed tail, not as a flooded tree.

**Validation (cheap and label-free).** Run the proving repo's suites and the AC-7 codebase with
**caller location enabled in log output** (Python `logging` `%(pathname)s:%(lineno)d`, logback
caller data, and so on). Strip the location, feed the text to the matcher, and check that the true
site is among the candidates. The analysis unit is the **distinct template**, not the log line
(one chatty template must not dominate). Metrics: recall@all (gate, Wilson-bounded, target
100%), median and p90 candidate count after collapse, and the true site's rank. Sizing is cheap:
hundreds of distinct templates fall out of one suite run.

### Q6 — Evaluating AC-15 (security) and reachability claims

**Seeded set per finding kind, with matched negatives.** AC-15 as drafted (one example per kind)
is a smoke test. For a measured claim, the fixture system gets, **per kind, ≥ 10 true positives and
≥ 10 near-miss negatives**, at least a third of each spanning a boundary:

| Kind | Positives | Near-miss negatives (must *not* be reported as definite) |
|---|---|---|
| Cross-boundary taint | HTTP/gRPC/queue param → another unit's SQL/shell/file/eval sink | Same flow through a recognized sanitizer; parameterized query; flow into a non-sink |
| Missing authorization | Entry → sensitive sink with no `auth-check` on any chain | Same with an auth decorator, middleware, or a check in the *calling* service |
| Secrets | Real-format keys in code, config and Compose; secret flowing into a log or response | Test fixtures with fake-format values, high-entropy hashes, public keys |
| Vulnerable dependency | See reachability table below | — |

Report **per-kind recall** (Wilson-bounded) and **per-kind false-positive rate on the negatives**.
For calibration rather than as a gate, also run established external suites where a language
applies: OWASP Benchmark (Java) for taint, NIST Juliet for C/C++. Verify the current versions and
licences before adopting either; they are single-program suites, so they say nothing about
cross-boundary behavior.

**Reachability claims: judge soundness asymmetrically.** Build labeled (dependency, advisory)
pairs in the fixture whose truth is known by construction:

| Case | Truth | Acceptable report |
|---|---|---|
| Imported; vulnerable symbol called on a path from an entry point | reachable | function-level reachable (advisory names symbols) / package-level reachable |
| Imported and package API called; vulnerable symbol **not** called | not reachable at function level | function-level "not reachable" **only** if the advisory names symbols; otherwise package-level reachable, labeled "function level unknown" |
| Declared in manifest, never imported | not reachable | "declared, not imported" |
| Transitive, called through a direct dependency | reachable | reachable via chain, or possible |

The gate is the **unsound "not reachable" rate**: the system must never state "not reachable" for
a reachable case (one counterexample fails; a never-miss rule for security). Over-claiming
"reachable" counts as noise and is reported. Where an ecosystem has an independent reachability
oracle, compare against it. Go's `govulncheck` is the known one (verify its current behavior
before relying on it). A disagreement is a triage item, not an automatic failure.

## 3. Method notes

### 3.1 The boundary rule: an LLM may *interpret* and *phrase*, never *assert*

Every fact the system reports (an impact, a link, a hypothesis, a gap, a finding) must have a
deterministic evidence chain (FR-20). An LLM output is not evidence. So the rule for any future
LLM use is: **it may change which question is asked (visibly, echoed), and how the answer is
worded (checked), but never which facts an asked question yields.** The touchpoints in §0's table
all follow it. A new proposal that does not (an LLM deciding whether a link exists at query time,
an LLM pruning a candidate set) breaks FR-19 and FR-20 and should be rejected on that basis, not
debated on accuracy.

### 3.2 Keeping FR-19 honest when the model is non-deterministic

The proposal's §3.6 argument ("facts are a pure function of snapshot, verdicts, access scope and
structured request") is correct. It holds for the plain-language client only if the
**interpretation** is made stable and visible. Four mechanisms, all cheap:

1. **Echo:** every answer carries the structured request it ran (already proposed, R2).
2. **Interpretation cache:** key = (normalized question text, snapshot ID, interpreter model
   fingerprint, prompt hash). The same question asked again against the same snapshot reuses the
   cached interpretation instead of re-sampling the model. That makes FR-19 hold *literally* for
   repeat asks, which pinning temperature cannot do on this stack (§1). The cache is invalidated
   when the snapshot or the model changes. Both are stated in the answer.
3. **Provenance in the canonical answer:** interpreter model id, quantization, endpoint
   (local/hosted), prompt hash, temperature and cache hit/miss. Without it a reader cannot tell a
   different *interpretation* from a different *fact*. (On this lab's stack the live model can
   diverge from static config per box, so read the fingerprint from the endpoint, `GET
   /api/v0/models`, not from config; see `claude/data-scientist/lm-studio-model-notes.md`.)
4. **Pin and re-run:** the echoed request is a valid structured request, so any client can re-run
   it exactly. AC-16's plain-language leg then tests two separable things (§4.4, §6 RQ8):
   interpretation accuracy (probabilistic, measured) and facts given the interpretation
   (deterministic, asserted).

**Determinism traps beyond the LLM, in the fact path itself:**

- **Approximate nearest-neighbor retrieval.** If any anchor resolution uses a vector index (HNSW),
  the returned set is approximate (it can omit a true neighbor) and is not guaranteed stable
  across index rebuilds of the same snapshot. Both break FR-19 and threaten never-miss. Rule:
  **no approximate vector search on the fact path.** Use lexical retrieval, or, if embeddings are
  used at all, exact cosine scoring over a lexically pre-filtered candidate set (§3.4). (Index
  mechanics are `graph-dba`'s; the rule is a method constraint.)
- **Depth bounds.** A bounded reach is deterministic, but the bound is part of the request. Echo
  it (the proposal already states bounds in the answer), and measure recall *at* the shipped
  default bound (§4.1).
- **Unordered result sets.** Cypher row order is not guaranteed. The proposal's stable sort (§3.6
  step 5) must apply before any truncation, paging or "top-k" step, including the RCA ranker's
  tie-break.

### 3.3 Answer phrasing: the list is rendered, not generated

The proposal says the LLM front end "phrases the facts without adding or removing any". That
cannot be enforced on free text from a 3–4B model: `model-bench`'s best local grounding rate was
0.80 (§1). It can be enforced structurally: **templates render every item and every chain**,
grouped per FR-10. The LLM writes at most a short lead paragraph from aggregates, and that
paragraph passes a deterministic **no-added-entity check** before display (every named symbol,
file, component or route must appear in the canonical facts; every stated count must equal the
canonical count). On failure the paragraph is dropped, not repaired. This makes "never drops a
fact" true by construction and "never adds a fact" mechanically checked. Measure the drop rate
during P4 as a quality signal for the chosen model, not as a gate.

### 3.4 Retrieval for symptom anchors: lexical first, embeddings deferred

Code-search terms in symptoms are mostly domain nouns and error phrases that also appear in
identifiers, literals and log templates. Identifier-aware lexical retrieval (split camelCase,
snake_case and paths; light stemming; IDF weighting) is deterministic, explainable, cheap, and
fits FalkorDB full-text indexing. Embedding retrieval earns its complexity only when symptoms and
code share no vocabulary ("the page hangs" vs a timeout in a retry loop). **Experiment to justify
adding it** (P5): on the seeded-bug held-out set, plain-language form only, compare lexical vs
lexical ∪ embedding-top-k (exact cosine over symbols with docstrings, a pinned local embedder such
as the Qwen3-Embedding models `model-bench` already measured) on **anchor recall** (the true cause
is within backward reach of an anchor) and on anchor-set size. Adopt the union only if it recovers
held-out bugs the lexical path misses (paired by bug) at an acceptable growth in anchor count.
Note the residency constraint (§1): on this box, an embedding call evicts the chat model, so a
symptom question that uses both pays a reload. "Correctness over speed" tolerates that, but
`teco`-flow latency (R14) will feel it.

### 3.5 Local vs hosted (FR-9)

| Touchpoint | Local (LM Studio, 3–4B class, ~8k usable context on this box) | Hosted |
|---|---|---|
| (a) Interpretation | **Expected sufficient** with schema-constrained single-shot output and deterministic parsers in front. Must pass §4.4's gate per model | Better on messy prose and long pasted context; same gate |
| (b) Symptom term extraction | Sufficient (short input, short output) | Marginal gain |
| (c) Summary paragraph | Sufficient from aggregates; expect a non-trivial drop rate | Lower drop rate |
| (d) Ranking | Not used (deterministic) | Not used (deterministic) |
| (e) Rule suggestions (offline) | Usable, more reviewer time | Better; only where policy allows |

**Consequences.** (i) Local-only is viable for every v1 touchpoint because none of them carries
facts. A weaker model degrades *interpretation accuracy* (visible, echoed, correctable by
re-asking or pinning) and summary quality, never fact correctness. That property is worth stating
to the stakeholder: it is what makes FR-9 safe. (ii) **The golden sets are the gate for any model
swap**, local or hosted: §4.4 runs per configured interpreter model, and a codebase's default
model is the one that passed. (iii) Model capabilities and catalog contents are perishable (the
LM Studio catalog on this box went from 19 to 16 models within days, same notes file), so the
model choice is made at P4 against the golden set, not now.

### 3.6 A method note on the P0 "precise indexer" bar (proposal §3.2)

The proposal adopts a per-language precise indexer "where P0 measures Joern's call resolution
below a bar that `data-scientist`'s recall target implies". Per §1's asymmetry, **call-resolution
rate does not drive recall; it drives noise**, because unresolved calls fall back to name-matched
*possible*. Recall is lost only when the true callee is not reachable by name-match either
(dynamic dispatch through a differently named interface method, callbacks, reflection, DI). So the
P0 spike should measure two numbers per language, on a sample of call sites whose true callee is
known (from runtime traces of the sample program, or a typed indexer where one exists):

1. **Fallback recall:** the fraction of true call edges found as resolved *or* name-matched. **This
   is the gate.** If it is below ~0.98 on the sample (lower bound reported), the language needs the
   indexer, or a recognizer for the missing dispatch idiom, regardless of noise.
2. **Name-match fan-out:** the median and p90 number of name-matched candidates per unresolved
   call site. This is the noise driver. If p90 fan-out is high (say > 20 candidates), the indexer
   is justified on noise grounds, as a cost decision rather than a correctness one.

The ~0.98 figure is a proposal: it leaves room for the per-link-kind recall target once errors
compound along multi-hop chains (a 3-hop chain at 0.98 per hop keeps ~0.94). Revisit it after P0
measures typical chain lengths.

## 4. Evaluation design

Conventions throughout: intervals are **Wilson 95%** (this lab's convention for small-n
pass/fail bounds); paired comparisons use **McNemar's exact test**; every gate reports each
stratum individually as well as the pooled figure; every ground-truth set has a **dev** part
(anyone may look and debug) and a **held-out** part (frozen, run only at a gate).

### 4.1 Recall — AC-5 "never miss"

**What is measured.** An *item* is a pair (anchor, expected impacted element), for example
(`falkor-chat` route `GET /shop/api/health`, `salesperson/src/api/endpoints.ts` caller symbol).
The engine is asked the impact question for the anchor at the **shipped default depth bound**. The
item is a **hit** if the expected element appears as confirmed, or as possible **and individually
enumerated with its own evidence chain**. A collapsed group ("may call any of N endpoints of X")
counts as a hit only if the element is a listed member of that group. Each member then also counts
toward the burden metric (§4.2), so collapsing cannot hide noise.

**Ground-truth sources (two kinds, both blind):**

1. **Census (realism).** A complete, per-kind enumeration of the proving repo's cross-boundary
   dependencies: storefront → server HTTP routes, MCP tool providers ↔ every config and prompt
   that names them, FalkorDB labels ↔ every reader/writer, `cypher-mcp` ↔ hooks and scripts,
   Compose/`.mcp.json` wiring. **Who:** an agent instance (or person) with repo read access only:
   no access to the system's output, its packs, or its tests, and not a pack author. **How:**
   source reading guided by the provider side (list every route, tool, label first, then hunt its
   consumers with grep and reading). Each item records file:line at both ends and the "why".
   **Second reader:** a second blind author enumerates one stratum (say MCP) independently. The
   union is the ground truth, and the disagreement rate between the two is reported as a
   ground-truth quality figure. If the readers miss each other's items often, the census is
   undercounting and recall claims are optimistic.
2. **Planted (difficulty).** In the multi-repo fixture system (proposal §5.1), planted links of
   every kind, stratified by **difficulty shape**: exact literal; constant across files; template
   literal; client wrapper (role lifting); config-bound base URL; runtime-built key (must be
   *possible*); dynamic import; DI or reflection; prompt mention; cross-repository library call
   with version skew. The planter writes the expected status and reason code too. That gives
   AC-6 its ground truth (§4.6).

**Mechanically derived truth, where a contract is declarative.** On the AC-7 codebase, gRPC
`.proto` files enumerate every service/method, and generated stubs pin both sides, so
the gRPC stratum can be derived exhaustively by a script independent of the system (§4.5). Prefer
this over hand enumeration wherever it exists. It is cheaper and complete.

**Leakage control (the most likely way this evaluation fails).** The proposal's §5.2 triage loop
("each miss is triaged into a recognizer gap (fix the pack) …") is good engineering and bad
measurement if run on the gating set: fixing packs until the held-out list passes turns it into a
training set, and "100% recall" then measures the debugging effort. Rules:

- Pack authors work against **dev** items only.
- The **held-out** set is frozen (content-hashed, committed) **before** the first gated run, and
  is run only at gates.
- A held-out item that fails and is then fixed **moves to dev**. The next gate needs replacement
  held-out items: fresh census strata, newly planted items, or runtime-oracle misses (below).
- Report every gate's result **as first run**, never as the post-fix rerun.

**Sizing.** With zero misses, the Wilson lower bound is n/(n + 3.84):

| Held-out n (pooled) | Lower bound, 0 misses | Lower bound, 1 miss |
|---|---|---|
| 20 | 0.839 | 0.764 |
| 35 | 0.901 | 0.855 |
| 50 | 0.929 | 0.895 |
| 75 | 0.951 | 0.928 |
| 100 | 0.963 | 0.946 |

**Recommendation:** proving repo held-out **≥ 75 items pooled** (census plus planted; the census
alone may yield fewer, so planting fills the gap), with **≥ 8 per required link kind**, so that
each stratum is at least present and individually reported. At 8 items a perfect stratum's bound
is only ~0.68. That is honest: per-kind claims at this size are smoke tests, and the document
should say so. The same ≥ 75 applies to the AC-7 codebase (§4.5). The analysis unit is the
**item**. Items sharing an anchor are correlated (one missing recognizer misses all of them), so
also report recall over **distinct anchors** and treat that as the conservative figure.

**Runtime oracle (independent of humans).** Distinct cross-boundary edges observed at runtime
(distributed traces on the AC-7 codebase; on the proving repo, server access logs or HTTP-client
instrumentation during the suites and a scripted storefront session) define **oracle recall** =
observed edges present as links ÷ observed edges. It is reported next to seeded recall, never
pooled, because it only sees exercised paths and over-represents common ones. Each oracle miss is
triaged (pack gap, extractor gap, genuinely dynamic), and each confirmed miss is added to the
**next** gate's held-out set, which is the sustainable source of fresh held-out items.

**Two further miss detectors (no gate, triage only):** unlinked cross-unit co-change pairs (Q3),
and FR-14's `runtime-only` coverage disagreements (proposal §3.4).

### 4.2 Noise — the false-alarm bound that makes AC-5 meaningful

Without a noise bound, listing everything as *possible* passes AC-5 (R4). Two parts:

1. **Confirmed precision.** Every confirmed item in the held-out answers is labeled true/false by a
   labeler who is not a pack author, against a written rubric ("a real dependency exists in at
   least one deployable configuration"). **Gate: ≥ 0.95 observed, and every false confirmed item
   is a defect to fix.** A false *confirmed* misleads; it is not the tolerated kind of error.
2. **Possible precision and burden.** Draw a **seeded random sample of ≥ 100 possible items**
   (expanded group members included) across held-out answers, stratified by reason code. Label
   them the same way. Report: possible precision per reason code (Wilson); **burden** = expanded
   possible items per answer (median, p90); and **reduction vs a naive over-lister** = burden of
   the baseline that lists every name-match and every provider of the kind ÷ the system's burden,
   at equal recall. The last metric makes the degenerate strategy visible: a system that equals
   its baseline has learned nothing.

**Thresholds.** A responsible possible-precision floor cannot be fixed before P1 measures the
actual distribution. Proposed starting values for the stakeholder to ratify after P1 (estimates,
not measurements): **possible precision ≥ 0.20 pooled** (at most four false alarms per real one
among *possible* items, consistent with FR-8's "a miss is worse"), **p90 burden ≤ 100 expanded
items per answer**, and **reduction ≥ 5×** over the naive over-lister. Each reason code whose
precision falls under ~0.05 should be reviewed as a candidate for collapsing or a better rule
rather than gated. Labeling cost: ~100 items × a few minutes each per gate, affordable for an
agent with repo access, spot-checked by a person.

**Why the labeler can see system output here but not in §4.1.** Precision is a judgment *about*
emitted items, so seeing them is unavoidable and harmless. Recall ground truth written after
seeing the output is biased toward what the system already finds. That is the difference between
the two protocols.

### 4.3 RCA — AC-12 seeded cross-component bugs and ranking quality

**Generation (cheap, and blind where it matters).** Hand-writing 40+ realistic cross-component
bugs is expensive. Use **contract-boundary mutation** on the fixture system and the proving repo:

1. A mutation catalogue applied at the **provider or data side** of a boundary, with the symptom
   surfacing at a consumer: rename a response field; change a field's type or units; drop a
   column or rename a graph label in the writer only; change a route path or verb; change a topic
   name on one side; change an enum value; return null or empty in an edge case; alter an auth
   check; swap two arguments. Each mutation records the **true cause location** (the mutated
   symbol).
2. Run the suites and a scripted end-to-end session. Keep mutants that produce a **visible
   symptom in a different component** from the cause (AC-12's condition). Capture the four input
   forms mechanically: the error message, the stack trace, the log excerpt (several services
   where applicable).
3. The **plain-language symptom** is written by someone who sees only the user-visible behavior
   or the captured output, never the diff, which keeps the fourth form blind.
4. **External validity:** the six real `-rca` documents in `falkor-chat/docs/reviews/` (§1) are
   replayed as a separate, small "real bugs" stratum. They are too few to gate on. They are reported
   individually as a check that mutants are not systematically easier than real bugs.

**Metrics.** Set recall (gate; per bug × form), hit@3 and hit@5 per form (primary), MRR
(secondary), each against the BFS-order baseline (Q2). The analysis unit is the **bug**. The four
forms are repeated measures of one bug: report each form's rate over bugs, and never pool the
bug × form cells as if they were independent n.

**Sizing.** Ranking claims need paired power against the baseline. McNemar's exact test cannot
reach α = 0.05 with fewer than 6 net discordant bugs, so a held-out set under ~30 cannot
demonstrate an improvement over BFS order unless the gap is very large. **Recommendation: ≥ 40
held-out bugs plus ≥ 20 dev bugs** (dev tunes weights and θ). For orientation: at n = 40, an
observed hit@5 of 32/40 has a Wilson interval of [0.65, 0.90]; 28/40 gives [0.55, 0.82].
**Proposed gate** (estimate, to be ratified once P5 measures a first dev run): set recall 100% for
every form; hit@5 ≥ 0.70 per structured form (error, trace, log), ≥ 0.50 for the plain-language
form; and a significant paired win over the baseline on hit@5 for at least the pooled structured
forms.

### 4.4 Plain-language interpretation — the golden question set

No drafted AC measures this, though FR-10, FR-16 and AC-16's plain-language leg all depend on it
(§6 RQ9).

**Set construction.** Cells: **5 use cases × 4 starting-point groups** (FR-17). Per cell ~5
questions → **~100 held-out** plus **~40 dev** (prompt iteration happens only on dev). Plus a
**15-item ambiguity stratum** (a symbol name that resolves to several symbols, an unknown name, a
two-use-case question, a question that names no anchor), whose correct behavior is "return the
candidates or ask", never a silent guess. Authors are people or agents who have **not seen the
interpreter prompt**, writing the way a developer actually asks: partial names, wrong case,
pasted fragments, Portuguese as well as English if the team asks that way (the stakeholder writes
both). Gold labels: the structured request **after resolution** (use case plus the resolved anchor
IDs against a pinned snapshot), so the metric is immune to harmless span differences.

**Metrics.** Resolved-request exact match (primary); per-field accuracy (use case, anchor set,
scope) as diagnostics; ambiguity-handling rate on the ambiguity stratum; **stability**: each item
run 5 times with the cache disabled, reporting the item-level flip rate (expected non-zero on this
stack, §1; the cache in §3.2 is what makes users not see it).

**Gate (proposal):** exact match **≥ 0.92 observed on n = 100 held-out**, which keeps the Wilson
lower bound ≥ ~0.85 ([0.850, 0.959]), with every use case individually ≥ 0.80 reported, and
ambiguity handling ≥ 13/15. Run per candidate interpreter model, local and hosted; the codebase's
default model is one that passed.

### 4.5 AC-7 — ground truth on the second codebase (OpenTelemetry demo)

**The choice is good, with conditions.** The OpenTelemetry demo covers the required shape (several
FR-1 languages, gRPC, Kafka, PostgreSQL, a cache), and, decisively for evaluation, it is
**already instrumented with OpenTelemetry**, so its traces are a free runtime oracle (§4.1).
Service mix varies by release (the proposal says so), so pin a tag and verify it.

**Ground truth, by stratum, from independent sources:**

| Stratum | Source of truth | Independence |
|---|---|---|
| gRPC | `.proto` service/method definitions × services that register a server for them × services holding a client stub for them, extracted by a small standalone script | Mechanical, complete |
| HTTP | Frontend/proxy route definitions plus Compose/env wiring, by blind reading | Human, blind |
| Kafka | Topic names at producers and consumers (checkout → accounting, fraud-detection), by blind reading | Human, blind |
| SQL / cache | The accounting service's tables/queries; cache key patterns | Human, blind |
| Service-level sanity | The maintainers' published architecture diagram | External, but **service-level only**: it validates that each service pair has ≥ 1 link, not code-level items |
| Runtime | Traces from a load-generator run | Oracle recall, reported separately |

**Conditions:**

1. **Pre-registration.** Freeze the pack set (version-hash it) **before the first build on the
   demo**. The first-contact result is the AC-7 result. Pack fixes made afterwards are legitimate,
   but their re-run is reported as "after adaptation". Otherwise "unfamiliar" stops being true by
   the second run.
2. **Contamination:** the demo is a famous public repository, so any LLM in the loop may have
   memorized it. That does not affect the deterministic linker. It does inflate LLM-assisted parts
   (interpretation, rule suggestions) evaluated on it, so do **not** use the demo for §4.4.
3. **One codebase is n = 1 for generality.** AC-7 proves the system works on *a* second codebase.
   For a workplace-generality claim, hold a **third** codebase back, untouched until a later gate.
   It must be a different shop's style, with a messaging stack and an in-house-looking wrapper.
   Google's Online Boutique (`microservices-demo`) is a plausible candidate (multi-language,
   gRPC), but verify that it has messaging and SQL, since I have not confirmed it does.
4. **AC-19 split:** the proposal splits the demo into three repositories. The split must be done
   **before** ground truth is written, and the census must include the cross-repository items the
   split creates.

### 4.6 Remaining ACs with an ML/statistical angle

- **AC-6 (doubt flagged).** Ground truth is the planted difficulty shapes (§4.1). Each planted
  unprovable link has an expected reason category. **Pass:** every planted unprovable link is
  listed as possible with a reason in the expected category, and **none appears as confirmed**
  (the second clause is the real test; an unprovable link shown as confirmed is the dangerous
  error).
- **AC-15 (security):** per Q6.
- **AC-16 (same facts across clients):** two parts. (i) Deterministic: the same structured request
  through MCP, CLI and the published raw query gives byte-identical canonical facts. One
  counterexample fails, no statistics needed. Also rebuild the snapshot twice from the same
  commits and diff it (proposal §5.1). (ii) Probabilistic, plain-language leg: covered by §4.4.
  AC-16 itself asserts only that when the echoed interpretation equals the structured request, the
  facts are identical.
- **Log/error matching (feeds AC-12, AC-17):** per Q5.

### 4.7 Gate summary

| Gate | Data | Metric | Proposed threshold | Phase |
|---|---|---|---|---|
| AC-5 recall (proving repo) | ≥ 75 held-out items, ≥ 8 per kind | Recall, Wilson LB; also over distinct anchors | 100% observed (LB ≥ 0.95) | P1 (HTTP/MCP/scripts strata), P3 (data) |
| AC-5 recall (AC-7) | ≥ 75 held-out, gRPC derived mechanically | Same; first-contact result | 100% observed | P2 |
| Noise | All confirmed items; ≥ 100 sampled possible | Confirmed precision; possible precision; burden; reduction | ≥ 0.95; ≥ 0.20; p90 ≤ 100; ≥ 5× (ratify after P1) | P1 onward |
| Oracle recall | Runtime traces/logs | Observed edges linked ÷ observed | Reported, misses triaged | P1 onward |
| Fallback call recall | P0 sample with known callees | Resolved ∪ name-matched recall; fan-out | ≥ ~0.98; p90 fan-out reported | P0 |
| AC-12 RCA | ≥ 40 held-out mutant bugs + 6 real | Set recall; hit@3/5; MRR; vs BFS | 100%; hit@5 ≥ 0.70 structured, ≥ 0.50 prose; paired win | P5 |
| Log matching | Distinct templates from suite runs | Recall@all; candidates p90; rank | 100%; reported | P5 |
| Interpretation | 100 held-out + 15 ambiguity | Resolved exact match; ambiguity handling; flip rate | ≥ 0.92 (LB ≈ 0.85); ≥ 13/15 | P4 |
| AC-15 security | ≥ 10 pos + ≥ 10 neg per kind | Per-kind recall, FP rate; unsound "not reachable" | Reported; **0** unsound | P5 |
| AC-16 | Structured request × 4 clients; rebuild twice | Canonical diff | Identical | P1 / P4 |

## 5. Findings on the proposal

Severity scale: blocker / major / minor / nit. No blockers: nothing here invalidates the
architecture. The majors change contracts P0 is meant to freeze, so they should be folded in before
P0 starts.

**Major**

- **M1 — §5.2: the triage loop runs on the gating set, so "100% recall" would measure debugging
  effort.** §5.2 writes the seeded list blind (good), then triages each miss into "fix the pack"
  with no dev/held-out separation. *Change:* adopt §4.1's leakage rules (dev vs frozen held-out,
  first-run reporting, fixed held-out items move to dev, oracle misses replenish the next gate).
  Add pre-registration of the pack set before first contact with the AC-7 codebase (§4.5).
- **M2 — R4 / §3.4: the noise bound is named but has no shape, and the collapsed "any of N"
  rendering can game both recall and noise.** §3.4's collapsed group would satisfy AC-5 for every
  member while showing one line of noise. *Change:* adopt §4.1's hit rule (a hit needs the element
  individually enumerated, or listed inside an expandable group with its own chain) and §4.2's
  two-part bound (confirmed precision ≥ 0.95 gated; possible precision, expanded burden and
  reduction vs a naive over-lister, measured from P1 and gated once ratified). State in §3.4 that
  burden is always counted on expanded groups.
- **M3 — §3.7 RCA: ranking left open, and the "likelihood" framing invites an LLM ranker.** *Change:*
  specify the deterministic weighted-feature ranker of Q2 (features shown per hypothesis, weights
  tuned on dev only, stable tie-break). That makes the order deterministic, which is stronger than
  FR-19 requires and lets AC-12's "ranking shows its evidence" mean per-feature contributions. Keep
  an LLM re-ranker as a gated later experiment over the top-k only. Add the BFS-order baseline to
  §5.
- **M4 — §3.6 step 1: failure-evidence and symptom anchoring is underspecified at exactly the
  points that decide never-miss and FR-19.** "Token similarity as *possible*" has no threshold or
  method, symptom → anchors is not described beyond "the LLM front end", and nothing rules out an
  approximate vector index. *Change:* adopt Q5 (build-time log templates with IDF information
  scores; tiered exact → containment → weighted overlap; θ frozen from dev at 100% recall;
  low-information templates collapsed, never truncated) and §3.4 (lexical retrieval for symptom
  anchors, embeddings only after a paired experiment). Add the rule **no approximate
  nearest-neighbor search on the fact path** (§3.2) and route it to `graph-dba` as an index
  constraint.
- **M5 — §3.13 / §6.3(e): `model-suggested` links stored as *possible*.** Rejecting query-time
  LLM links is right, but build-time model-suggested links still add unauditable noise (no FR-20
  chain) for a recall gain better captured another way. *Change:* drop `model-suggested` from the
  reason-code list for v1. Recast the LLM (and co-change statistics) as an offline rule-pack
  suggester with human acceptance (Q1e, Q3), which reuses the FR-27 machinery and keeps every
  answer deterministic. Keep the reason code reserved for a future, experiment-gated promotion
  (Q3).

**Minor**

- **m1 — §3.6, the LLM front end "phrases the facts without adding or removing any": unenforceable
  as written.** *Change:* template-render every item; allow only an aggregate-based LLM summary
  that passes a deterministic no-added-entity check (§3.3).
- **m2 — §3.6, the plain-language client should try deterministic parsers first.** Pasted traces,
  routes, file:line and diffs inside a sentence should never go through the model (Q1). This
  shrinks the non-deterministic surface and the golden set's job.
- **m3 — §3.6 step 5 canonical answer: add the interpretation provenance** (model id, quantization,
  local/hosted, prompt hash, cache hit) and an **interpretation cache** keyed by question, snapshot
  and model (§3.2). Without the cache, FR-19 holds only "given the echoed interpretation". With it,
  it holds literally for repeat asks.
- **m4 — §3.2 / P0: the precise-indexer bar is framed as call-resolution rate.** Under the
  proposal's own fallback, resolution rate drives noise and fallback recall drives misses. *Change:*
  P0 measures **fallback recall** (gate) and **name-match fan-out** (cost) on a sample with known
  callees (§3.6 of this note).
- **m5 — §3.7 RCA and impact depth bounds vs never-miss.** A bound can cut off the true item. The
  bound is stated in answers (good). *Change:* §5 measures recall at the shipped default bound, and
  the answer flags when the frontier was hit ("reach stopped at depth d with k unexplored edges"),
  so truncation by bound is visible rather than silent.
- **m6 — §5.1 AC-16 test conflates interpretation with determinism.** *Change:* split it into the
  deterministic structured-client diff and §4.4's interpretation accuracy (§4.6).
- **m7 — §5.2 runtime oracle: state its bias.** It only sees exercised paths. Report oracle recall
  separately, never pooled with seeded recall (§4.1).
- **m8 — §2.6 / P2: the OpenTelemetry demo's "published service map" is service-level.** It cannot
  serve as the code-level seeded list on its own. *Change:* derive the gRPC stratum mechanically
  from `.proto` files, write the other strata blind, use the map only as a service-level sanity
  check, and keep the demo out of any LLM evaluation because of contamination (§4.5).

**Nit**

- **n1 — §5.2 "target 100%":** phrase it as "100% observed, with the Wilson lower bound reported";
  a 100% target on a finite set is an observation, not a population claim.
- **n2 — §3.7 RCA feature list:** "overlap with recent change (git)" should name the window (N
  commits before the snapshot, or the overlay diff) so the feature is deterministic per snapshot.

**Not changed, and endorsed:** facts at build time with deterministic queries (§3.1); possible as
a first-class status inside programs too (§1, §3.1 rule 5); the published query for AC-16 (R1);
the echoed interpretation (R2); R12's "keep AC-12 as smoke plus a measured set".

## 6. Requirement-level findings for the stakeholder

None of these edits the requirements document. Each is a suggestion for `tico` to put to the
stakeholder at the AC review. Thresholds marked *proposal* are estimates to be ratified after the
first measurements (P1 for noise, P4 for interpretation, P5 for RCA).

- **RQ1 — AC-5 is gameable as written** (agrees with the proposal's R4, and goes further on
  wording). *Suggested rewording:* "Given a held-out list of known cross-boundary impacts, frozen
  before the run and written by someone who has not seen the system's output, every one appears in
  the answer, as confirmed or as possible and individually listed with its evidence. On the same
  answers, at least 95% of confirmed items are correct, and possible items meet the agreed
  false-alarm bound." Add the bound as its own criterion (*proposal:* possible-item precision ≥
  0.20 and p90 ≤ 100 possible items per answer, ratified after P1).
- **RQ2 — AC-12 measures appearance, not ranking** (agrees with R12). *Suggested rewording:* keep
  the current text as the smoke test, and add: "Over a set of at least 40 seeded cross-component
  bugs, held out from tuning, the true cause appears in the tree for every bug and every input
  form; it ranks in the top 5 for at least 70% of bugs given an error, trace or log (50% given a
  plain-language symptom); and the ranking beats ordering by path length alone." The numbers are
  *proposals*.
- **RQ3 — FR-19's RCA order allowance is probably unnecessary.** A deterministic ranker (§2 Q2)
  gives the same order every time. *Suggested decision:* tighten FR-19 to "same facts **and same
  RCA order**", and keep the allowance in reserve only if a model-based re-ranker is ever adopted.
  In that case the allowed variation should be bounded (*proposal:* top hypothesis identical in ≥ 95%
  of repeats). Also agree to R2's scoping: FR-19 applies to the same **interpreted** question, and
  the interpretation is shown with every answer.
- **RQ4 — FR-12 "ranked by likelihood" implies calibrated probabilities**, which tens of seeded bugs
  cannot support. *Suggested rewording:* "ranked by a stated score, with the evidence contributing
  to each rank". If the stakeholder does want probabilities ("70% likely"), that is a much larger
  data commitment (hundreds of labeled incidents for calibration) and should be decided
  explicitly.
- **RQ5 — AC-6 is hard to test as "a link the system cannot prove".** *Suggested rewording:* "Given
  planted links of each unprovable kind (runtime-built address, dynamic import, reflection or
  dependency injection), each is shown as possible with a reason naming its kind, and **none is
  shown as confirmed**."
- **RQ6 — AC-15 with one example per kind is a smoke test, and it says nothing about false
  alarms or false reassurance.** *Suggested addition:* "Per finding kind, a seeded set with at least
  10 real cases and 10 look-alike safe cases; recall and false-alarm rate are reported per kind;
  and the system never states that a vulnerable dependency is unreachable when it is reachable."
  The last clause is the security counterpart of "never miss".
- **RQ7 — AC-7 is one codebase, and "unfamiliar" erodes with every adaptation.** *Suggested
  additions:* "the system's configuration is frozen before it first sees the codebase, and the
  first-contact result is the one reported"; and, for the workplace-generality goal, a third
  held-back codebase at a later milestone.
- **RQ8 — AC-16's plain-language leg depends on interpretation, which is probabilistic.**
  *Suggested rewording:* "…returns the same facts; for the plain-language client, whenever its
  shown interpretation matches the structured question". Interpretation accuracy is then judged by
  RQ9.
- **RQ9 — Gap: no AC measures whether plain-language questions are understood**, though FR-10 and
  FR-16 rest on it. *Suggested new AC:* "On a held-out set of about 100 plain-language questions
  covering all five use cases and all four starting-point kinds, the system's interpretation
  matches the intended question in at least 92% of cases; for ambiguous questions it lists the
  candidates or asks, rather than guessing." The 92% is a *proposal*, and it applies to whichever
  model, local or hosted, a codebase is configured to use.
- **RQ10 — FR-8 needs an exchange rate to be operational.** "Missing a real impact is worse than a
  false alarm" is right but unbounded; RQ1's false-alarm bound *is* the exchange rate. Worth
  saying to the stakeholder in those terms: how many false alarms per real impact is acceptable
  before people stop reading the list? The *proposal* of four is a starting point to revisit with
  real answers in hand.
- **RQ11 — FR-9: say what a weaker local model costs.** Under this design a weaker local model
  lowers *interpretation accuracy and summary quality*, never fact correctness (§3.5). A one-line
  addition to FR-9's rationale would set the right expectation: "local-only may mean more
  questions need rephrasing; it never changes the facts".

## 7. Risks & open questions

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| E1 | **Census undercounts** (blind readers miss real links too) | Recall overstated | Two blind readers on one stratum, disagreement rate reported; runtime oracle and co-change as independent miss detectors (§4.1) |
| E2 | **Mutant bugs easier than real ones** | RCA hit rates overstated | Real-bug stratum from `falkor-chat/docs/reviews/*-rca.md`, reported individually; mutation catalogue weighted toward contract-semantic changes (units, nullability) rather than crash-on-rename |
| E3 | **Ground-truth cost** (~75 + 75 recall items, ~60 bugs, ~140 questions, ~80 security cases, plus labeling ≥ 100 possible items per gate) | Evaluation becomes the bottleneck | Mechanical derivation where contracts are declarative (gRPC, Compose); mutation for bugs; caller-location logs for template ground truth; strata built per phase, not all in P0 |
| E4 | **Thresholds set before data** | Gates either trivial or unpassable | Every threshold above is labeled *proposal* with the phase whose measurement ratifies it; P0 fixes the *protocol and sizes*, not the numbers |
| E5 | **Local interpreter accuracy unknown on code questions** | Plain-language client weaker locally | §4.4 per-model gate; deterministic parsers in front; echo and re-ask; hosted where policy allows |
| E6 | **Held-out exhaustion on the small proving repo** | No fresh held-out after a few gates | Oracle misses and new planted items replenish it; the AC-7 and third codebases carry later generality claims |

**Open questions** (none blocks this note; each is decided by routine evidence later):

1. Does the stakeholder ask in Portuguese as well? If so, §4.4's set needs both languages per cell
   (assumed yes for some items; cheap to confirm at the AC review).
2. Is an "unknown key → every provider of the kind" fallback acceptable in a large workplace
   system, or does it produce burden beyond any bound? P1 measures it on this repo, but the
   workplace's dynamic-wiring rate (K2) is unknown. Capturing one anonymized workplace service's
   statistics (counts only, no code) would answer it.
3. Which roles produce ground truth in this lab? Recommendation: census and question sets by an
   agent instance with repo-read-only access and no view of the system (for example a
   `qa-engineer` brief that withholds packs and output), spot-checked by the stakeholder.

## 8. Pass 2 — stakeholder decisions and the re-checked use cases (2026-10-26)

Inputs: plan v2 (`docs/plans/connected-code-graph.md`, which adopted M1–M5 and m1–m8, §2.7);
`analyst`'s review (`docs/reviews/connected-code-graph.md`); and the requirements as of
`c98c3578`, which carry decisions S1–S38 plus the re-checked use cases (FR-11–FR-15, AC-12–AC-15,
new AC-24 and AC-25).

### 8.1 What the decisions settle or supersede in §1–§7

- **Settled as recommended:** RQ1→S8, RQ2→S9, RQ3→S3 (fixed order), RQ4→S10, RQ5→S11, RQ6→S12,
  RQ7→S14, RQ8→S2, RQ9→S15/AC-24, RQ11→S16. **RQ10's exchange rate is 4:1**, provisional; its
  denominator still needs pinning (§8.5 SQ2).
- **§7 open questions closed.** (1) Portuguese: **yes** (S37), so §4.4's set becomes bilingual
  (§8.3). (2) Workplace statistics: **declined** (S38), so the "any provider of the kind" burden
  is checked only on this repo and the OpenTelemetry demo, and the risk is accepted. (3) Ground-
  truth producer: **an isolated read-only agent, spot-checked by the stakeholder** (S36). §8.2
  adds how to check that agent's *completeness*, which a spot-check cannot.
- **Plain language moves into P1 (S35).** Part of §4.4's set must exist at P1, but only the impact
  use case is built then. AC-24 can be gated in full only after P5. Until then, the impact cells
  are gated and the rest are reported as "not yet built".
- **Per-test runtime coverage is required (S29).** That makes runtime coverage a usable RCA
  feature (Q2 feature 5) and gives the AC-14 "tests a change reaches" question a runtime ground
  truth (§8.2).
- **`analyst` findings on this note:**
  - m3 (θ vs "no score cut-off"): **accepted.** θ and the symptom-anchor score floor are a
    deliberate, frozen, recall-calibrated exception. They are applied by the engine over the
    *complete* full-text result set, never as an index `LIMIT`.
  - m1 (resolved-call precision): **accepted.** The P0 spike measures **resolved-call precision**
    next to fallback recall (§3.6), because a wrong resolution is both a silent miss and a false
    *confirmed*. Gate it with AC-5's confirmed-precision bar (≥ 0.95 observed) on the sample.
  - B5.4 (fixture and ground-truth strata in no phase): **agreed.** The held-out strata that P1
    gates on are P0 deliverables.
  - B5.1: AC-7 pre-registration is **per stratum**: each stratum's first contact is the first build
    that includes support for it. That is the one coherent form of "first contact" given that SQL
    arrives in P3.
- **New: impact through data flow and control dependencies (FR-11, AC-25).** Control-dependency
  fan-out is the largest new noise source in the design. AC-5's noise labeling needs a rubric for
  indirect items. *Proposal:* an indirect item is **real** if some execution exists in which a
  change at the anchor changes the item's inputs or whether it runs. The labeler records distance,
  so possible-precision can be reported by distance band. The provisional 4:1 cap should be
  checked per band too. A pooled ratio that passes only because direct items dilute a noisy
  distance-3 band misleads.

### 8.2 (a) Are the new and changed criteria statistically meaningful at their sizes?

Intervals are Wilson 95%. "P(pass)" is the binomial probability that a system with the stated
*true* rate meets the gate on one first-run attempt. It is the number that tells the stakeholder
how much luck a gate contains.

| Criterion | Size | Meaningful? | What I'd change |
|---|---|---|---|
| AC-5 recall: every item of ≥ 75 | 75 | **Yes, and very strict.** 0 misses gives a lower bound of 0.951. But a zero-miss gate on 75 first-run items passes a system with true recall 0.99 only **47%** of the time (0.995 → 69%; 0.999 → 93%; 0.98 → 22%) | Keep it, since it is the stakeholder's stated priority, but write down the **fail procedure** now: each miss gets a root cause and a fix; the failed items move to dev; the gate reruns on **fresh** held-out items. Otherwise the first failure invites quiet re-runs on the same list (§8.5 SQ1) |
| AC-5 key completeness | — | **Not measured by any criterion.** The stakeholder's spot-check verifies that the key's items are *correct*. It cannot show the key is *complete*, and an incomplete key overstates recall | **Two isolated key-writing agents** with different strategies (provider-first vs consumer-first), different instances, and if possible different models. Estimate the true total with **capture–recapture**: N̂ = n₁·n₂ / m, where m is the overlap. Report key completeness ≈ \|union\| / N̂. Both agents share LLM blind spots, so N̂ is itself an underestimate; label it a lower bound on incompleteness. The runtime oracle (§4.1) stays the independent check |
| AC-5 confirmed ≥ 95% real | All confirmed items in the held-out answers (likely hundreds) | **Yes.** Every confirmed item is labeled (a census of that set, not a sample) | State that it is an observed rate over all confirmed items, with the interval reported |
| AC-5 false alarms ≤ 4 per real impact | ≥ 100 sampled possible items | **Yes as a provisional ratio, but its denominator is ambiguous.** "Per real impact" (all real items, confirmed included) is much looser than "per real *possible* item" (§4.2's 0.20 precision floor) when most real items are confirmed | Pin the denominator (§8.5 SQ2). Estimate it as a ratio of sums pooled over held-out answers, with a bootstrap-over-answers interval (resample answers, not items; items within an answer are correlated). Also report per distance band (§8.1) and p90 per-answer burden |
| AC-6 planted unprovable kinds | ≥ 1 per kind as written | Deterministic property. **One instance per kind is a coincidence-level test** | ≥ 3 planted instances per named kind, in different syntactic shapes |
| AC-7 first contact | Per stratum | Meaningful only if pre-registration is per stratum (B5.1, §8.1) | — |
| AC-12 top-5 ≥ 70% (structured) | 40 bugs | **Coarse.** 28/40 has interval [0.55, 0.82]. A system with true top-5 of 0.70 passes 58% of the time; 0.80 passes 96%; 0.60 passes 13% | Acceptable for a provisional bar. Read it as "true rate is probably ≥ ~0.65". Do not compare models or versions on this gate alone; use paired tests (§8.3). Define the per-bug unit for the three structured forms (§8.3) |
| AC-12 top-5 ≥ 50% (symptom) | 40 bugs | **Coarse.** 20/40 is [0.35, 0.65]. True 0.50 passes 56%; 0.60 passes 93% | Same reading; the symptom form is also where the blind-writer protocol (§4.3 step 3) matters most |
| AC-12 beats baseline | 40 bugs, paired | **Yes if the test is defined.** As written, "beats" is undefined, and BFS order has heavy ties that make a naive baseline arbitrary | §8.3 defines the baseline (exact tie expectation) and the test (paired sign test over bugs) |
| AC-12 single planted cases (config, schema, wiring, data-flow-only, recent change, outside branch, no-match, ruled-out re-rank) | 1 each | Functional checks, deterministic; fine as such. **They say nothing about ranking those cause kinds** | Make config, schema-mismatch and wiring causes **strata of the 40** (≥ 5 each), so their top-5 rates are at least reported. Add ≥ 5 infrastructure-caused failures where the correct top hypothesis is the "outside the mapped code" branch, and ≥ 20 negative-control inputs for "no match" (§8.3) |
| AC-12 recent-change feature | — | **Leak risk.** In a mutation-generated set, *every* true cause is the most recent change. The recent-change feature then scores the answer for free, and weight tuning learns "recent = cause" | Place the mutation **outside** the recency window in ~half the bugs (commit it behind realistic distractor commits, or back-date it relative to the snapshot's window). Report held-out hit@5 **with and without** the feature (ablation). Tune weights with the same mix. The six real RCA cases give the realistic check |
| AC-12 "trace + log reinforce each other" | Bugs that yield both | **Not sound as a universal "at least as high as either alone".** See below | Two-part version (§8.5 SQ3) |
| AC-14 risk ranking | — | No ground truth for "risk". It can be checked for being stated and deterministic, but not for being *right* | §8.4: stated score; optional retrospective check |
| AC-14 "exactly the tests" a change reaches | Per planted change | **Conflicts with FR-8.** Static reach over-lists by design (possible links), so "exactly" fails on any possible hop | Reword (§8.5 SQ5): every exercising test listed (zero misses); per-test runtime data, where supplied, separates confirmed from possible |
| AC-15 10 real + 10 look-alike per kind, 5 kinds | 10 / 10 | **Smoke-level per kind.** 10/10 detected gives a recall lower bound of 0.72; 0/10 false alarms gives an upper bound of 0.28. Pooled over 5 kinds (50/50) gives 0.93 and 0.07 | Fine as provisional. Report per kind *and* pooled, and never claim per-kind precision from 10 negatives. The **never-unreachable** clause is the real gate (zero tolerance, by design) |
| AC-15 secret values never appear | — | Deterministic, but needs a test design | **Canary secrets**: plant unique high-entropy canaries, then search every stored view, answer, report and log for each canary and for every substring of ≥ 8 characters. Zero hits. Also: the "fingerprint" must be a **keyed** hash (HMAC with a local key), not a plain hash. A plain SHA of a short or low-entropy secret can be brute-forced, which amounts to storing the value (route to `security-expert`) |
| AC-24 ≥ 92% of ~100, EN + PT | ~100 | **Pooled yes; per language marginal.** 92/100 has interval [0.85, 0.96]. 46/50 per language is [0.81, 0.97] | Gate pooled; report per language with a **per-language floor** (proposal: ≥ 0.85 observed), so a collapse in one language cannot hide in the pool. Details in §8.3 |
| AC-24 "every ambiguous one" | ~15 | Zero tolerance on ~15: 15/15 gives a lower bound of 0.80 | Fine, but it must be paired with an **over-asking** check, or "always answer with options" passes it (§8.3) |
| AC-25 data-flow + control effect | 1 each | Functional | ≥ 3 each; noise by distance band (§8.1) |

**"Reinforce each other" (FR-12, AC-12): judgment.** The draft criterion, that a trace and a log
together rank the true cause at least as high as either alone, read as rank(combined) ≤
min(rank(trace), rank(log)) for every bug, is **not sound as a universal rule**:

- Evidence can legitimately conflict. A log line from another service adds anchors whose backward
  reach brings in strong competitors. Under any additive score (Q2), the true cause can drop below
  its trace-only rank while the ranking still behaves correctly. A universal monotonicity demand
  would push the design toward hacks, such as forcing the trace-only order to dominate, that make
  combination pointless.
- It also says nothing about the case the stakeholder actually means by "reinforce": two weak
  pieces of evidence that *each* rank the cause around 8th should rank it near the top
  *together*. The draft criterion is satisfied by a ranker that ignores the second piece
  entirely.

**Recommended testable version** (for the stakeholder; §8.5 SQ3), over the bugs that yield both a
trace and a log (a stratum of ≥ 20 of the 40):
1. **Helps on aggregate:** combined top-5 rate ≥ the better single form's top-5 rate (observed),
   and combined MRR ≥ the better single form's MRR.
2. **Never degenerates:** in every bug, rank(combined) ≤ max(rank(trace), rank(log)). The combined
   evidence is never worse than the *worse* single piece. This is a deterministic guard that an
   agreement-weighted score can reasonably meet, and a failure of it signals a real defect.
3. **Actually combines:** in at least some bugs, rank(combined) < min(rank(trace), rank(log)).
   This is reported as a count. If it is zero across the stratum, the system is not combining
   evidence, whatever (1) says.
The same three apply to symptom + trace.

### 8.3 (b) Evaluation designs: AC-24 and the RCA baseline (plus RCA feedback, "no match", outside branch)

#### AC-24 — understanding plain-language questions (supersedes §4.4 where they differ)

**Set.** The grid is still 5 use cases × 4 starting-point groups = 20 cells. **Held-out ≈ 100 =
50 English + 50 Portuguese, 2–3 per cell per language**, plus the ambiguity stratum (≥ 15, split
across both languages) and a dev set of ~40 (20 + 20) for prompt work.
- **Native, not translated.** Each language's questions are written natively by the key-writing
  agent, prompted to write as a developer asks: partial names, wrong case, pasted fragments,
  code-switching (Portuguese prose around English identifiers is the realistic Portuguese case).
  Translations would understate the difficulty.
- **One small paired probe, reported only:** 10 dev questions written in both languages, to see
  whether a language gap exists (McNemar on 10 pairs cannot reach significance, so it is a
  descriptive signal, not a test).
- **Gold** is the resolved structured request against a pinned snapshot: use case, the resolved
  anchor IDs, scope and direction (the latter now includes the reverse "what does this depend
  on?", FR-11). **Ambiguity is labeled by two independent annotators.** An item is in the
  ambiguity stratum only if both mark it ambiguous; disagreements are dropped from the gate and
  reported. The stakeholder, a native Portuguese speaker, spot-checks the Portuguese gold in
  particular.
- **Phasing (S35):** at P1 only the impact column is buildable. Gate on the impact cells (~25
  questions, 12–13 per language), with the bar reported but treated as provisional. With n ≈ 25,
  23/25 = 0.92 has interval [0.75, 0.98], so it is a smoke gate. The full AC-24 gate runs when all
  five use cases exist.

**Scoring rule (deterministic, against gold):**

| System output | Unambiguous item | Ambiguous item |
|---|---|---|
| Single interpretation = gold | understood | **fail** (a silent guess, even if lucky) |
| Single interpretation ≠ gold | not understood | fail |
| Options, gold among them | **not understood** (over-asking) | understood |
| Options, gold not among them | not understood | fail |

The "options on an unambiguous item" cell is what stops "always answer with options" from passing
both clauses. Report the **over-asking rate** separately, because a model that hedges a lot is a
different failure from one that misreads.

**Gate (provisional):** pooled understood ≥ 0.92 over the ~100 unambiguous held-out items; each
language ≥ 0.85 observed; every ambiguity-stratum item answered with options containing the gold.
Report the item flip rate over 5 uncached repeats (§4.4). Run per configured interpreter model;
§8.5 SQ7 asks which models the 92% binds.

**Leakage:** held-out questions never appear as few-shot examples. Only dev questions may. The
OpenTelemetry demo is not a source of questions (contamination, §4.5).

#### RCA baseline — "closest to the error first"

**Definition.** Same candidate set as the system (so the comparison isolates ranking, not
retrieval). Order by **hop distance from the nearest anchor** (over all anchors, when several
pieces of evidence are given).

**Ties: score the baseline's expectation exactly, never break ties arbitrarily.** Hop distance
produces large tie blocks. Breaking them by symbol ID would make the baseline's hit@5 depend on
alphabetical accident, in either direction. For a bug whose true cause sits in a tie block of size
b with a candidates strictly closer: expected hit@5 = min(1, max(0, 5 − a) / b), and expected
rank = a + (b + 1)/2. Both are exact; no random draw, no seed.

**A second, stronger reference (reported, not gated):** "closest first, recently changed first"
(sort by in-window change, then distance). It is roughly what a developer does with `git log`. A
ranker that beats the agreed baseline but not this one is not yet earning its complexity, and the
stakeholder should know.

**The test.**
- **Analysis unit is the bug.** For the three structured forms, per-bug score s = number of forms
  (0–3) with the true cause in the top 5; the baseline's s is the sum of its exact expectations.
- **Paired sign test** over bugs on sign(s_system − s_baseline), ties dropped, one-sided
  α = 0.05. The symptom form is tested the same way, separately.
- "Beats the baseline" = the structured-form test is significant. The symptom-form result is
  reported, plus Holm across the two tests if both are to be claimed.
- Resolving power: an exact sign test needs **≥ 5 net wins with no losses** (one-sided) to reach
  α = 0.05, and more as losses appear. With 40 bugs, a ranker that wins about 12 bugs and loses
  about 2 is detectable; a few-point average gain is not. Report MRR difference with a paired
  bootstrap-over-bugs interval as the effect size.

#### RCA feedback (ruled out / confirmed), "no match", and the outside branch

- **Feedback re-ranking.** A *simulated investigator*: starting from the tree, repeatedly mark the
  top hypothesis ruled out if it is not the true cause. **Metric: rule-outs needed to reach the
  true cause.** With feedback that only deletes, this equals rank − 1; a feedback model that also
  down-weights hypotheses sharing the ruled-out chain should do **better**. Report the median over
  held-out bugs against rank − 1 (paired). Marks are part of the echoed request, so FR-19 holds
  per session.
- **"No match" vs never-miss.** Q5's recall-calibrated tier-3 θ means an unrelated error string
  will often weakly match *something*, so a literal "matches nothing" rarely fires. Proposed
  definition (§8.5 SQ4): **no match = no stack frame, no exact or full-template match**. Weak
  tier-3 matches are still listed under "weak matches only". Evaluation: ≥ 20 **negative
  controls** (error strings from the OpenTelemetry demo or other code fed to the proving repo, and
  vice versa) must yield "no match". Every held-out bug × form must **not** yield it. A false "no
  match" is a miss.
- **Outside the mapped code.** The branch needs a score like any hypothesis, or it is decorative.
  Feature: an **infrastructure-signature pack** (connection refused, DNS failure, timeout, OOM
  kill, TLS errors, third-party exception classes) matched against the evidence. Evaluation:
  ≥ 5 recorded infrastructure failures (stop the database container, kill the broker, block
  a network route). A wrong port in a mapped config file is *not* one of these: FR-12 makes it a
  wiring hypothesis traced to its entry. The outside branch should rank first; report its rank.

### 8.4 (c) Risk ranking for untested code (FR-14, AC-14)

**Recommendation: a stated, additive points score per untested function, deterministic per
snapshot, with each factor's points shown as the reason.** Same philosophy as the RCA ranker:
transparent and repeatable, with weights that are a policy statement rather than a fitted model.
There is no labeled "risk" data to fit against.

| Factor (FR-14 names the first three) | Measure | Points (proposal) |
|---|---|---|
| **Reach** | Number of distinct **entry points** (routes, RPC methods, topic consumers, MCP tools, CLI mains) whose forward reach contains the function, over confirmed links; cross-boundary entry points counted separately | 0 / 1 / 2 / 3 at 0, 1–2, 3–9, ≥ 10 entry points; +1 if any reaching entry point is in another component |
| **Security relevance** | On a source→sink chain, on a path to a sensitive entity, an `auth-check` itself, or carrying a security finding | 0 or 3 (3 = any of these) |
| **Recent change** | Changed within the codebase's recent-change window (the same per-codebase setting as RCA, FR-12) | 0 / 1 / 2 by number of in-window changes (0, 1, ≥ 2) |
| **Complexity** (suggested addition, §8.5 SQ6) | Cyclomatic complexity from Joern's control-flow graph | 0 / 1 / 2 at < 5, 5–14, ≥ 15 |

**Ordering:** total points descending; ties broken by reach, then by stable symbol ID. Why counts
of **entry points** rather than raw callers: raw caller counts reward utility functions called
from many places within one path, while entry-point reach measures how much user-visible
behavior depends on the function. Possible-only reach is shown but **not scored**, so FR-8
over-listing does not inflate risk. Code reachable only through possible links is listed in its
own tier, "reachability uncertain".

**Statically "reached but runtime-unexecuted"** functions (FR-14's disagreements) are a distinct
class: listed alongside, flagged "statically reached, never executed by any test". They are a
coverage gap even though static analysis counts them reached.

**Evaluation: what can and cannot be claimed.**
- **Checkable now (the AC-14 gate):** the score is stated per item; the order is deterministic
  (same snapshot, same order); and planted functions with known factor values land in the
  expected order (a fixture with, say, a high-reach security-relevant untested function, which
  must outrank an unreachable helper).
- **Predictive validity: not claimable without data, and not required by AC-14.** If wanted
  later, a **retrospective check**: for each bug-fix commit in history (identified by the
  `-rca`/`HISTORY.md` defect trail in this repo, or "fix" commits in the OpenTelemetry demo),
  compute the ranking on the snapshot *before* the fix and record whether the fixed function was
  in the top decile of untested code. Report the **lift** over random (top-decile hit rate ÷
  0.10) with its interval. With the handful of fixes in this repo's history this is anecdote,
  not evidence. Say so in any report, and do **not** re-weight the factors on it (that fits the
  score to this repo's few bugs).
- **Test-kind labels** (unit / integration / end-to-end, AC-14): deterministic rules (framework
  markers, directory conventions, plus whether the test's reach crosses a boundary or drives an
  external interface). Check accuracy on ~50 tests labeled by the blind key agent and report the
  confusion matrix. A misclassified kind is cosmetic for the map but matters for "which tests
  does this change reach?" filters.

### 8.5 (d) Requirement questions for the stakeholder

Each is one decision, followed by the recommended answer. None blocks P0 planning. SQ2–SQ5 change
how an AC is scored, so they should be settled before P0 freezes the evaluation protocol.

- **SQ1 — AC-5: what happens when the zero-miss gate fails?** At 75 items, a system that truly
  finds 99% of impacts still fails about half the time on a first run. *Recommend:* keep zero
  misses as the bar, and add "each miss is root-caused and fixed, and the gate is re-run on
  **new** blind items, never on the same list". Alternative: allow at most one miss with a
  recorded root cause. That is less strict, and I recommend against it given "missing a real
  impact is worse than a false alarm".
- **SQ2 — AC-5's 4:1 cap: four wrong *possible* items per what?** (a) Per real impact listed
  (confirmed included), pooled over all test answers; or (b) per real *possible* item. (a) is the
  user's view ("for each real thing I see, at most four false alarms") and is looser. (b) is
  stricter. *Recommend (a)*, as the AC's wording reads, pooled over answers, with the 90th-
  percentile answer's count and the per-distance figures also reported (indirect, data-flow and
  control-dependency items are the noisiest).
- **SQ3 — FR-12/AC-12 "reinforce each other": confirm the testable form.** The current reading
  ("together ranks the true cause at least as high as either alone, every time") can fail for a
  correctly behaving system, and is passed by one that ignores the second piece of evidence.
  *Recommend:* over bugs with both a trace and a log, (1) together finds the cause in the top 5 at
  least as often as the better single piece; (2) together is never worse than the *worse* single
  piece for any bug; (3) at least some bugs rank higher together than with either alone (§8.2).
- **SQ4 — "No match" vs never miss.** To avoid misses, weak text matches are kept, so a strict
  "matches nothing" would almost never be said. *Recommend:* "no match" means no stack frame and
  no exact or full-template match; weak matches are still shown, labelled "weak matches only",
  together with what was tried and what to provide next.
- **SQ5 — AC-14 "exactly the tests a change reaches" conflicts with over-listing (FR-8).**
  *Recommend:* "lists every test that exercises the changed functions, with none missing; tests
  reached only through unproven links are marked possible; where per-test runtime data is
  supplied, it confirms which tests actually ran the changed code".
- **SQ6 — AC-14 risk ranking: "ranked by a stated risk score", like RCA?** Nobody can show yet
  that the ranking predicts where bugs will be; there is no data for it. *Recommend:* yes, word it
  as a stated score with its reasons, and add **complexity** (how branchy the function is) as a
  fourth factor next to callers, security relevance and recent change.
- **SQ7 — AC-24: which model must reach 92%?** FR-9 accepts that a weaker local model "may need
  more rephrasing". *Recommend:* the 92% binds **each codebase's configured default model**, so a
  local-only codebase must have a local model that passes. Any other configured model has its
  score shown, not gated. If no local model reaches 92%, local-only codebases get a stated lower
  local target rather than a silent pass. Decide that number after P1 measures it.
- **SQ8 — AC-12: ≥ 40 planted bugs, of which ≥ 20 yield both a stack trace and a log, ≥ 5 each
  are config, schema-mismatch and wiring causes, and ≥ 5 infrastructure failures test the
  "outside the mapped code" branch.** *Recommend yes.* This is scope (more planted bugs of specific
  kinds), not a new bar; without it, the RCA additions are checked once each and never ranked.
