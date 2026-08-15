# GraphRAG retrieval + generation evaluation harness — plan gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-026 (M2.5-quality)

## Scope & verdict

Static plan-gate review of `docs/plans/graphrag-eval.md` (the `architect` implementation plan,
just written) against `docs/BACKLOG.md`'s K-026 entry and `docs/plans/graphrag-eval-ml.md` (the
`data-scientist` method note, ✅ 2026-07-10, treated as design authority — its metric/threshold/
split choices are not re-litigated here). I read all three documents in full, then verified every
codebase claim the plan makes against the real code it cites: `repository.py`/`services.py`
(`hybrid_search`), `modelconfig.py` (`KINDS`, `ResolvedModel`, `ModelGateway.from_env`),
`test_modelconfig.py`'s FR-4 AST test, `repository.py`'s `create_channel`/`create_thread`/
`post_first_message`/`post_subsequent_message`/`thread_has_head`, `test_workflow_live.py` in full,
`bootstrap_schema.sh`'s `EMBEDDING_DIM` default, `responder.py`'s `AgentResponder`/`_SYSTEM_PROMPT`,
`guards.py`'s `GuardVerdict`/`Judge`, `llm.py`'s `extract_own_line_json_object`, and — because the
plan's Unit 2/3 tests live under `server/tests/eval/` and therefore inherit `server/tests/conftest.py`
— that root conftest's autouse fixtures. I did not run the eval harness (it doesn't exist yet); this
is a design-time gate, not an execution verification.

**Verdict: needs changes.** One Critical/blocker-grade grounding gap (an autouse pytest fixture the
plan never accounts for silently defeats Unit 2's core dimension-guard and threatens Unit 3's
production-fidelity), one factual error in a code citation the plan claims to have verified, one
concrete dispatch-sizing breach, and one internal self-contradiction in Unit 2's test design. None
of these require rethinking the plan's overall shape — the two-layer split, D1–D6 design calls, and
corpus design are sound — but each is a concrete implementation landmine that should be resolved in
the plan text before Unit 1/2 are dispatched.

---

## Findings

### Critical

**C-1. The autouse `_model_config_env` pytest fixture (`server/tests/conftest.py:100-123`) is never
accounted for, and it silently breaks the two safety mechanisms the plan is proudest of (D3's
embedding-cache invalidation, Unit 2's dimension guard).**

`server/tests/conftest.py:100-123` defines:

```python
@pytest.fixture(autouse=True)
def _model_config_env(monkeypatch):
    ...
    monkeypatch.setenv("FALKORCHAT_OPENCODE_CONFIG", opencode_path)
    monkeypatch.setenv("FALKORCHAT_MODEL_CONFIG", model_config_path)
    monkeypatch.setattr(config, "OPENCODE_CONFIG_PATH", opencode_path)
    monkeypatch.setattr(config, "MODEL_CONFIG_PATH", model_config_path)
```

`autouse=True` on a fixture in the package-root `conftest.py` applies to **every** test collected
under `testpaths = ["tests"]` (`server/pyproject.toml:26`) — including everything the plan adds
under `server/tests/eval/`, regardless of marker. `ModelGateway.from_env()` reads exactly the two
patched attributes (`modelconfig.py:637,644`: `_config.OPENCODE_CONFIG_PATH` /
`_config.MODEL_CONFIG_PATH`), so **any pytest test in this suite that calls
`ModelGateway.from_env()` resolves against `server/tests/data/models.json`, never the real
`config/models.json`.** I read both fixture files:

- `server/tests/data/models.json`'s embedding entry: `"lmstudio/text-embedding-qwen3-embedding-0.6b": {"dim": 4}`.
- The real `config/models.json`: `"lmstudio/text-embedding-qwen3-embedding-0.6b": {"dim": 1024}`.

The model **refs** happen to match today (coincidentally — the fixture was authored to mirror
production refs at `ws:test`'s `TEST_EMBEDDING_DIM = 4`), but the **dimension does not**, and that's
exactly the value the plan's own safety net depends on:

- `docs/plans/graphrag-eval.md` §5 Unit 2, `conftest.py`'s `ws_eval` fixture: "`repo.read_index_dimension(EVAL_WS, label="Message")` is not `None` and **matches the configured
  embedding dim**." If this is implemented the natural way — `ModelGateway.from_env().resolve("embedding").primary.dim` — it will read **4** inside every pytest run, while `ws:eval`'s real
  vector index is genuinely built at **1024** (Unit 1 bootstraps it via the resolved dim). The
  fixture would then treat a *correctly seeded* `ws:eval` as dimension-mismatched on every single
  default `pytest -q` run, and skip — silently defeating Unit 2's entire purpose. The plan's own
  "Done when" claim ("`pytest tests/eval -q` ... passes with `ws:eval` seeded") would not hold.
- D3 (§3): "the deterministic retrieval-eval test loads this cache and asserts every golden query's
  cached `model` field equals `config/models.json`'s current `defaults.embedding`." Read the same
  way (via `ModelGateway`/`config.MODEL_CONFIG_PATH`), this checks against the **test fixture's**
  `defaults.embedding`, not the real file — today harmless because the ref string happens to match,
  but fragile by construction: the two files can (and per K-042's own design, are allowed to) diverge,
  at which point this "invalidation guard" silently validates against the wrong source of truth with
  no error.
- Unit 3's generation sub-pass ("call the **agent-under-test** LLM's `.complete(...)`") doesn't
  specify how that LLM client is constructed. If it goes through `AgentResponder`/`ModelGateway`
  (the natural reading of "agent-under-test," and the only path that would track a *future* change
  to `defaults.agent`), it inherits the same redirection — the "agent under test" would silently be
  whatever `tests/data/models.json`'s `defaults.agent` names, not the real M2 stack model, undermining
  the entire point of measuring production behavior.

The plan's own §2 claims to have "verified against the current codebase, not assumed from the note,"
and it explicitly cites `test_workflow_live.py` as "the shape to mirror verbatim for every live test
this plan adds." That citation is correct as far as it goes, but the plan doesn't draw the
connection: **`test_workflow_live.py` constructs `OpenAICompatibleLLM`/`OpenAICompatibleEmbedder`
directly from env-var literals precisely because going through `ModelGateway` inside a pytest test
hits this exact autouse redirection** — its own docstring (`test_workflow_live.py:14-19`) says as
much ("this module constructs ... directly ... rather than going through `ModelGateway.from_env()`
— a live run needs no config file"). Unit 1's corpus-seed script is unaffected (it's a standalone
script, not a pytest test, so no conftest fixture ever touches it) — but Unit 2's `conftest.py`/
`test_retrieval_eval.py` and Unit 3's `test_judge_live.py` are pytest tests under `server/tests/`,
and the plan gives no instruction to avoid `ModelGateway` inside them.

**Suggested fix:** state explicitly in §3 (next to D3) and §5 Unit 2/3 that any code under
`server/tests/eval/` needing the *real* production config (embedding dim for the `ws_eval` readiness
check, D3's model-ref comparison, the agent-under-test LLM in Unit 3) must **not** resolve it via
`ModelGateway.from_env()`/`config.MODEL_CONFIG_PATH` inside a pytest context — it must read
`config/models.json` directly (e.g. via the raw-path constant `modelconfig.DEFAULT_MODEL_CONFIG_PATH`,
which is a module-level `Path`, untouched by the conftest monkeypatch) or construct clients from
env-var literals exactly as `test_workflow_live.py` does. Unit 1's own script already does the right
thing (`ModelGateway.from_env()` from a bare script) for the opposite reason — worth naming that
asymmetry in the plan so the implementer doesn't "fix" Unit 1 to match Unit 2/3 or vice versa.

---

### Major

**M-1. `AgentResponder._SYSTEM_PROMPT` doesn't exist — `_SYSTEM_PROMPT` is a module-level constant,
not a class attribute.**

`server/falkorchat/responder.py:36-40` defines `_SYSTEM_PROMPT` at module scope, *before* the class
statement at line 43. `AgentResponder` never assigns it as a class attribute. The plan states twice
that it reuses `AgentResponder._SYSTEM_PROMPT`:

- §2: "the judge layer (§5 Unit 3) therefore reuses `AgentResponder._SYSTEM_PROMPT` and
  `AgentResponder._build_prompt(...)` directly."
- §5 Unit 3: "build the prompt via `AgentResponder._build_prompt` + `_SYSTEM_PROMPT` (reused per §2's
  finding, not duplicated)."

`AgentResponder._SYSTEM_PROMPT` raises `AttributeError` as written — Python does not expose a
module-level name as a class attribute just because the class lives in the same file. The correct
reuse is `from falkorchat.responder import _SYSTEM_PROMPT` (or `responder._SYSTEM_PROMPT` on the
imported module) alongside `AgentResponder._build_prompt` (which *is* a real instance method, and —
since it never reads `self` — can be called unbound or via a throwaway instance). This is exactly the
kind of citation §2 says was verified rather than assumed; it wasn't, here. Trivial to fix, but worth
correcting in the plan text so the implementer doesn't inherit the wrong shape.

**M-2. Unit 2 breaches this project's own dispatch-sizing convention (~5 files is a decomposition
signal) — the plan's "each of its 3 units is independently dispatchable and bounded" claim doesn't
hold for Unit 2 as scoped.**

Counting §5 Unit 2's own file list: `golden_retrieval.jsonl`, `embed_golden_queries.py`,
`golden_retrieval.embeddings.json`, `metrics.py`, `conftest.py`, `test_golden_set_integrity.py`,
`test_retrieval_eval.py`, `retrieval_baseline.json` — **8 files**, 6 authored plus 2 generated
artifacts the unit is still responsible for getting right. That's well past the ~5-file signal (Unit
1 = 2 files, Unit 3 = 4 files — both comfortably within bound). Recommend splitting Unit 2 before
dispatch, e.g. **2a** (golden-set authoring: `golden_retrieval.jsonl`, `embed_golden_queries.py`,
`golden_retrieval.embeddings.json`, `test_golden_set_integrity.py`) and **2b** (metrics + baseline
gate: `metrics.py`, `conftest.py`'s `ws_eval` fixture, `test_retrieval_eval.py`,
`retrieval_baseline.json`) — 2b depends on 2a's fixture existing but not on its content being final,
so the split doesn't reintroduce a false Unit-1-style ordering dependency.

**M-3. `test_golden_set_integrity.py`'s dependency on `ws:eval` is self-contradictory, which also
undercuts the "Unit 2's code can be written before Unit 1 runs" claim.**

§5 Unit 2 introduces the test as "network-free, runs against the committed fixtures only (does
**not** need `ws_eval`, **or** needs it read-only just to confirm ids exist)" — then the very next
bullet describes exactly that: "every `relevant_msgId` referenced actually exists in `ws:eval` (a
light `WHERE m.msgId IN $ids` lookup)." A live Cypher lookup against `ws:eval` needs FalkorDB
reachable and the graph to exist; that contradicts (a) the "network-free... zero network calls in
either case" framing in both the intro and the §6 test-strategy table (`Network? none`), and (b) the
opening §5 paragraph's claim that "Unit 2's code can be written/reviewed before [Unit 1] runs, since
its fixture skips cleanly on an absent corpus" — this particular check has no stated skip behavior at
all when `ws:eval` doesn't exist yet. Left as written, a naive implementation would raise (the "empty
key" `ResponseError` pattern `repository.py:740-743` shows other code has to guard against
explicitly) rather than skip, the first time this test runs against an unseeded `ws:eval`. Resolve
the "or" — pick one: either this test genuinely needs a seeded `ws:eval` and should reuse the same
`ws_eval` skip fixture Unit 2's other live test uses (in which case it isn't "before Unit 1 runs"
material and the opening claim needs qualifying), or the msgId-existence check is deferred/optional
until `ws:eval` exists and the test should say so explicitly with its own skip guard.

**M-4 (opinion, per the brief's ask). D6 — the baseline file as a live regression gate from day one —
is a defensible reading of the method note mechanically, but the plan doesn't close the process gap
around it.**

D6's mechanics (auto-write on first run without failing; compare-and-fail on every subsequent run
using the method note's own `recall@10 ≥ baseline ∧ MRR not down >5%` rule) are a faithful, literal
implementation of the method note's own acceptance-threshold table — the note's own text already
describes exactly this two-phase behavior, so D6 isn't inventing gating the note didn't ask for. What
the plan doesn't address: nothing requires anyone to look at the *specific numbers* the first run
produces and affirmatively decide they're an acceptable floor before those numbers become load-
bearing for every subsequent CI run. The corpus behind them is `analyst`-reviewed, not human-verified
(§4, explicitly flagged as an open question — see O-1 below), and the method note itself calls corpus
representativeness the harness's single **highest** risk. An unrepresentative corpus that happens to
produce a mediocre `recall@10` on its first green run would, under D6, immediately start blocking any
future retrieval change that can't beat that number — including changes that are actually
improvements over a *better* baseline nobody ever got to see. **Recommendation:** require the first
`retrieval_baseline.json` to be reviewed (`data-scientist`, since this is a methodology call on
whether the number is a reasonable floor, not a code-correctness question) before it's committed as
gating, rather than treating "the test passed" as sufficient sign-off.

**M-5 (opinion, per the brief's ask). An `analyst` review pass is a reasonable substitute for the
method note's "human verification" on the golden-retrieval set's structural properties, but not for
the ~10-example judge-calibration set specifically.**

§7 item 2 correctly surfaces this as an open question rather than resolving it, and I'll give the
opinion it asks for. For the 30–50 golden-retrieval pairs, `analyst` review is well-matched to what
the method note actually needs checked structurally — topical spread, near-miss pairing, no
answer-key leakage (§4's own framing of what the review gate does) — an independent static reviewer
can verify those properties about as well as a human skimming the same corpus. The judge-calibration
set is different in kind, not just degree: its entire output is a number the method note and the
backlog's own done-condition call **"judge–human agreement."** If the ~10 labels backing that number
are `analyst`-authored rather than human-authored, the harness still *reports* a number under that
name, but it no longer measures what the name promises — it measures judge-vs-analyst agreement,
which is a materially weaker claim (an LLM-based judge and an LLM-based reviewer share failure modes
a human wouldn't). Given the set is only ~10 examples — the cheapest of the three "human verification"
touchpoints to actually do with a real person — I'd recommend a real human spot-check specifically for
the calibration labels, even if the larger golden-retrieval set stays on the `analyst`-review path.

---

### Minor

**N-1. Swapping `config/models.json`'s `defaults.embedding` — one of the harness's own named
motivating use cases — will turn the *default*, network-free `pytest -q` run red until someone
separately re-runs `embed_golden_queries.py`, and nothing connects those two facts.** D3's "mismatch
fails loudly" is the right behavior in isolation, but once `ws:eval`/`golden_retrieval.embeddings.json`
exist, any embedding-model change made *elsewhere* in the codebase (not touching `tests/eval/` at
all) will red the offline baseline suite until the unrelated-looking manual step is remembered. Worth
a callout in the plan (and ideally a cross-reference from `AGENTS.md`'s model-config section) that an
embedding-model swap is now a two-location change.

**N-2. The backlog's done-condition text has no explicit "or marked not-run" carve-out for the judge
layer, unlike K-025's precedent language ("recorded model-gated, structurally demonstrated").** The
plan's `generate_report.py` design (an explicit "not run (live LLM unreachable...)" marker satisfying
"reported") is a reasonable, precedented interpretation given this exact repo's K-025/K-027 pattern
for model-gated acceptance — but it would be cleaner to amend K-026's backlog done-condition text to
say so explicitly, rather than leaving the interpretation to the plan's report generator.

**N-3. Unit 1's seed script (§5, step 5) reimplements `Services.post_message`'s dispatch logic "in
~10 lines" without naming `Repository.thread_has_head` (`repository.py:243`), the one existing
primitive that does the first/subsequent-write decision for the real dispatch loop
(`services.py:704`, `_dispatch_write`).** Worth naming explicitly so the implementer reuses it instead
of rolling an equivalent ad hoc query.

**N-4. Minor internal inconsistency on whether Unit 3 needs `metrics.py`.** §5's opening paragraph
says "Unit 3 depends on Unit 2's `golden_retrieval.jsonl` + `metrics.py` existing (it reuses both)";
§7 item 7 walks this back to "it only needs ... `metrics.py`-adjacent conventions to exist" (Unit 3's
own file list and described logic never actually calls `recall_at_k`/`mrr`). Harmless, but pick one
phrasing.

---

## What's solid

- The two-layer split (deterministic retrieval metrics, zero network dependency, vs. a live-marker-
  gated judge layer) is the right shape and matches the method note's own design faithfully — I found
  no place where the plan quietly reopened a method-note choice.
- §2's codebase-grounding is mostly accurate and genuinely re-derived, not copied from the note: the
  `hybrid_search` call-surface/return-shape description, the `KINDS` frozenset claim, the FR-4 AST
  test's directory scope, `create_channel`/`create_thread`'s non-idempotency, `bootstrap_schema.sh`'s
  1536 default, and the `test_workflow_live.py` skip-never-fail shape all check out exactly as
  described against the live code — the one miss (M-1) is narrow and easily fixed.
- D1 (no fifth `judge` kind added to `ModelGateway`) is well-argued and correctly grounded in
  `modelconfig.py:85-87`'s own "not a change to this set casually" comment; D2 (persistent `ws:eval`,
  never rebuilt per session) is the right call and clearly reasoned against the repo's usual
  throwaway-workspace convention.
- The corpus design (§4 — near-miss pairs reproducing the method note's own 0.786 finding, plus
  orthogonal "easy" topics) is a thoughtful, concrete translation of the method note's abstract
  representativeness ask into an actually-buildable spec.
- Edge cases in §6 (multi-relevant golden pairs, ANN returning fewer than `k`, unparseable judge
  output, empty retrieved context) cover what the method note calls out and nothing obvious is
  missing beyond what's flagged above.

## Open questions

- C-1 needs a decision, not just acknowledgment: should Unit 2/3's "real config" reads go through a
  new *test-only* escape hatch (e.g., a fixture that restores the real `config/models.json` path for
  just `tests/eval/`), or should they bypass `ModelGateway` entirely and read the file/construct
  clients directly? Either resolves the bug; they have different maintenance costs the plan should
  pick between rather than leave to the implementer.
- M-4 and M-5 are opinions this review was explicitly asked to give, not defects — the downstream
  coordinator/user still owns the actual call on both (whether to gate `data-scientist` sign-off
  before committing the first baseline; whether to insist on a human spot-check for the calibration
  set).

---

## Pass 2 (2026-08-15) — re-review of v2

Re-read `docs/plans/graphrag-eval.md` v2 in full (the revision note + every changed section: §2's
new conftest-autouse citation, §3 D6's sign-off addendum and new D7, §4's cross-reference, all of
§5's Unit 1/2a/2b/3, §6's updated table, §7 items 1/3/9/10, and the closing "Ready to implement").
Re-verified every code claim the revision makes, rather than trusting the plan's own "resolved"
framing: `modelconfig.Overlay`/`ProviderCatalog`/`_read_json` (`modelconfig.py:353-459`),
`modelconfig.DEFAULT_MODEL_CONFIG_PATH` (`:118`) against `conftest.py`'s `_model_config_env`
(`:100-123`) to confirm the former really is untouched by the latter's monkeypatch,
`AgentResponder.__init__`/`StaticModelGateway.__init__` for I/O-freedom, and `config.AGENT_ID`
(`config.py:86`). I also re-counted Unit 2a/2b/3's file lists against the ~5-file dispatch signal
and re-read Unit 2a's `test_golden_set_integrity.py` bullet against `golden_retrieval.jsonl`'s own
schema to check the M-3 fix holds end to end, not just for the one check it names.

### C-1 — genuinely resolved

D7 mechanism 1 checks out exactly as claimed. `modelconfig.Overlay.load(path)` (`:435`) calls the
module-level `_read_json(path, ...)` (`:353`), which does `Path(path).expanduser().read_text(...)` —
it uses the `path` argument directly and never consults `os.environ` or any `config.py` attribute
(`var_name` is used only in the error message). `Overlay.__init__` (`:414-432`) parses only the
passed-in `doc`; `model_settings(ref)` (`:451`) is a plain `self.models.get(ref)` lookup — no
dependency on `ProviderCatalog`/the OpenCode file at all, confirming "works standalone with no
other config file present." And `modelconfig.DEFAULT_MODEL_CONFIG_PATH` (`:118`,
`_REPO_ROOT / "config" / "models.json"`) is a distinct module-level constant from
`config.MODEL_CONFIG_PATH` — `conftest.py:100-123`'s autouse fixture monkeypatches `config.
OPENCODE_CONFIG_PATH`/`config.MODEL_CONFIG_PATH` only, never touching `modelconfig.
DEFAULT_MODEL_CONFIG_PATH`. So `Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))
.model_settings("lmstudio/text-embedding-qwen3-embedding-0.6b").get("dim")` really does read
**1024** from the real `config/models.json`, inside a pytest context, unaffected by the redirect.
D7 mechanism 2 (env-var-literal client construction, mirroring `test_workflow_live.py`) needed no
re-verification beyond Pass 1's read of that file — it's the same pattern, just applied
consistently now. **C-1 is closed.**

### M-1 — genuinely resolved

`AgentResponder.__init__` (`responder.py:46-63`) does pure attribute assignment plus
`StaticModelGateway(llm=llm, embedder=embedder)`, and `StaticModelGateway.__init__`
(`modelconfig.py:834-837`) is likewise pure attribute assignment — no I/O in either constructor, so
`AgentResponder(services=None, agent_id=config.AGENT_ID)` is safe to instantiate standalone.
`config.AGENT_ID` exists (`config.py:86`, `os.environ.get("FALKORCHAT_AGENT_ID", "assistant")`) and
is already used exactly this way at three call sites in `app.py` (`:290,304,312`). `_build_prompt`
(`responder.py:73-83`) reads only its `question`/`seeds` arguments and the module-level
`_SYSTEM_PROMPT`, never `self` — calling it on a throwaway, unconfigured instance is safe and
correctly avoids the nonexistent-class-attribute bug from v1. **M-1 is closed.**

### M-2 — genuinely resolved

Recounted against the plan's own v2 file lists: Unit 1 = 2 files (unchanged), **Unit 2a = 4 files**
(`golden_retrieval.jsonl`, `embed_golden_queries.py`, `golden_retrieval.embeddings.json`,
`test_golden_set_integrity.py`), **Unit 2b = 4 files** (`metrics.py`, `conftest.py`,
`test_retrieval_eval.py`, `retrieval_baseline.json`), Unit 3 = 4 files (unchanged). All four units
are now at or under the ~5-file signal. **M-2 is closed.**

### M-3 — only partially resolved: the fix moved one `ws:eval`-dependent check out of Unit 2a, but
left a second one behind under the same "genuinely network/DB-free" label

The msgId-existence check did move to Unit 2b as described, and that half of M-3 is fixed. But
Unit 2a's `test_golden_set_integrity.py` (§5 Unit 2a) still lists, as one of its two remaining
checks:

> "no `query` is a case-insensitive substring of, or a superstring containing, its target
> message's `text` (self-retrieval-inflation guard, method note finding 5/risk 2)"

`golden_retrieval.jsonl`'s schema (repeated in this same section, unchanged from v1) is `{id, query,
relevant_msgIds, topic, rationale}` — it carries no field holding the target message's actual
`text`. The only place that text exists is the `Message.text` property in `ws:eval` itself, keyed by
the `msgId`s in `relevant_msgIds`. So this check, as specified, needs exactly the same thing the
msgId-existence check needed (a live read of `ws:eval`'s `Message.text`) — it just wasn't named in
the M-3 fix, and the surrounding paragraph now claims the opposite: "**genuinely network/DB-free**,
needs neither FalkorDB nor `ws:eval`," repeated in §6's test-strategy table ("Network? none —
genuinely, no `ws:eval` needed either") and in Unit 2a's own "Done when" ("passes with zero
network/FalkorDB dependency, regardless of whether `ws:eval` exists yet — provable by running it
before Unit 1 has ever been run"). As specified, that "Done when" claim does not hold: an
implementation of the self-retrieval check needs either a live graph read (reintroducing the exact
`ws:eval` dependency the fix was meant to remove from this file) or a source of the target text this
schema doesn't carry. This also weakens §7 item 10's sequencing claim that "Unit 2a's own code
(fixtures + integrity test) can be written and run before Unit 1 exists at all" — true for the
embedding-cache-match check, not true for the self-retrieval check as written.

Two independent fixes would close this, either is fine — the plan should pick one, not carry the
ambiguity into implementation:

1. **Add a field to `golden_retrieval.jsonl`'s schema** (e.g. `target_text`) carrying a copy of the
   relevant message's text, authored alongside the pair — keeps Unit 2a genuinely DB-free and the
   check runs as a pure string comparison against the fixture, matching the file's own framing.
2. **Move this check to Unit 2b too**, next to `test_golden_msgids_exist_in_corpus`, reusing the same
   `ws_eval` skip fixture — and drop the "genuinely network/DB-free" claim for the whole file,
   scoping it instead to just the embedding-cache-match check (the only one left that's actually
   DB-free).

I'd lean toward (1): it's a one-field schema addition, keeps the self-retrieval guard runnable in
CI with zero FalkorDB dependency (arguably *more* valuable for this specific check than for the
msgId-existence one, since it's the leakage-inflation guard the method note calls out by name), and
preserves the clean "Unit 2a is written/reviewed before Unit 1's corpus exists" story the split was
built to enable. But the plan's authors should make this call explicitly rather than leave the
current text's internal contradiction for the implementer to discover.

### M-4 / M-5 — clearly and actionably stated

Both coordinator-accepted opinions are folded in with enough specificity for an implementer/QA pass
to act on without re-deriving anything: D6's `data-scientist` sign-off is written into Unit 2b's own
"Done when" as part of the done-condition ("committed only after that review, per D6"), not left as
a vague aspiration; §7 item 3's human-spot-check recommendation for `golden_judge_calibration.jsonl`
is scoped precisely (that set only, not the larger golden-retrieval set) and correctly marked as a
recommendation the coordinator/user still owns, not a hard gate the plan enforces. No further
action needed on either — as instructed, not re-litigating the opinions themselves.

### Updated verdict

**Needs changes — but narrowly now.** Three of the four Pass 1 blocking-grade findings (C-1, M-1,
M-2) are genuinely, verifiably closed — I checked each against the actual code rather than taking
the revision note's word for it, and all three hold. M-3 is only half-fixed: the self-retrieval-
inflation check in Unit 2a's `test_golden_set_integrity.py` still implicitly depends on `ws:eval`
(via a `Message.text` this schema doesn't carry), contradicting that file's own "genuinely
network/DB-free" claim in three places (its intro parenthetical, §6's test-strategy table, and its
"Done when"). This is a small, well-scoped gap — either of the two fixes above closes it — and
doesn't call the unit split or any other design decision into question. Unit 1 can be dispatched
now; **Unit 2a should not be dispatched until this is resolved**, since its own done-condition is
currently unsatisfiable as written. Units 2b and 3 are unaffected and can proceed in parallel with
that fix.

No new findings beyond the M-3 residual — M-4/M-5's framing is fine as-is, D7's two mechanisms both
check out, and I found nothing newly broken by the split itself (Unit 2b's fixture reuse, Unit 3's
dependency on Unit 2a's file existing rather than its content, and the file/step counts across all
four units all hold on inspection).

---

## Pass 3 (2026-08-15) — narrow re-check of the v3 M-3 fix

Targeted re-read of `docs/plans/graphrag-eval.md` v3's diff from v2: the revision note, §5 Unit 2a's
schema/file description (`:396-448`), and §7 item 10 (`:640-650`) plus its echo in the closing
"Ready to implement" section. Per the request, I did not re-verify C-1/M-1/M-2 (Pass 2 already
confirmed those closed against the code, and the revision note states §2/§3/§4/Unit 1/Unit 2b/Unit 3
are unchanged from v2 — I spot-checked that claim by diffing the sections I'd already read in Pass 2
against this file and found no changes outside §5 Unit 2a, the v3 revision note, and §7 item 10/the
closing section, consistent with "narrow follow-up fix").

**The fix, verified:**

- `golden_retrieval.jsonl`'s schema (`:399-405`) now carries a `target_text` field, defined
  (`:406-417`) as "a verbatim copy of the first-listed `relevant_msgIds` entry's `Message.text`,
  authored alongside the pair at golden-set-authoring time." This is exactly the fix I'd leaned
  toward in Pass 2 (option 1: add a field to the schema rather than move the check to Unit 2b).
- The self-retrieval-inflation check (`:433-435`) now reads "no `query` is a ... substring of, or a
  superstring containing, its **own pair's `target_text`**" — a pure fixture-to-fixture string
  comparison, no `msgId` lookup, no graph involved. The embedding-cache-match check (`:436-438`)
  was already DB-free via D7 mechanism 1 and is unchanged.
- With both checks now genuinely free of any `ws:eval`/FalkorDB dependency, the file-level claims
  that were contradictory in v2 are now accurate: the "genuinely network/DB-free... for **either**
  of its two checks" framing (`:427-432`), the `§6` test-strategy table row (`:561`, "none —
  genuinely, for both its checks"), and the "Done when" clause (`:446-448`, "provable by running it
  before Unit 1 has ever been run") all hold together now — I checked each against the corrected
  check description rather than the plan's own summary of itself, and none of them overstate what
  the two checks in `test_golden_set_integrity.py` actually need.
- §7 item 10 (`:640-650`) was reworded to match: "Unit 2a's own code (fixtures + integrity test) can
  be written and run before Unit 1 exists at all, **in full** — both of its checks are genuinely
  network/DB-free ... not just the embedding-cache-match one." This is now a true statement of what
  the file does, not an aspiration — the sequencing claim holds for the whole file.
- The multi-`relevant_msgIds` case is handled sensibly: `target_text` scopes to the first/primary id
  only, and the check is described as "scoped accordingly" (`:412-413`) — this doesn't weaken the
  leakage guard (checking the primary target is sufficient to catch the verbatim-copy failure mode
  the method note is worried about) and doesn't conflict with Unit 2b's retrieval-metrics test, which
  still uses the full `relevant_msgIds` set for recall/MRR — a separate concern `target_text` was
  never meant to touch.

**Nothing newly broken.** I looked for a fresh gap the one-field addition might have introduced —
in particular, whether `target_text` could go stale against the corpus the way the method note
already worries golden-set content can drift after a corpus edit — and concluded this is already
covered by the plan's existing, general caveat (§3 D3: a corpus edit requires the golden set to be
manually re-verified against the new corpus; `target_text` is just one more field subject to that
same already-documented manual-re-verification obligation, not a new category of risk). Nothing
else in the narrow diff — the schema example, the file list, the "Ready to implement" summary —
introduces a new inconsistency.

### Updated verdict

**Approve.** All four Pass 1/Pass 2 findings that gated dispatch — C-1, M-1, M-2, and now M-3 — are
confirmed closed against the actual plan text and, where re-checked, against the actual code. The
two coordinator-accepted opinions (M-4/M-5) are folded in as actionable done-condition items, not
loose recommendations. I found no new issues in this pass. **The plan is ready to dispatch to
implementers as written: Unit 1 → Unit 2a (parallel-writable now) → Unit 2b/Unit 3 per the
dependencies in §5's opening and §7 item 10.**

---

## Corpus content review (Unit 1)

**Scope:** a content review of the *authored corpus itself*, not the plan/design — this stands in
for the method note's "human-verified" requirement for the corpus specifically (plan §4/§7 item 3),
distinct from Pass 1–3's design gates above. Reviewed `scripts/seed_eval_corpus.py`'s full `_CORPUS`
list (all 12 threads, 121 messages verbatim), `scripts/seed_eval_corpus.sh`, and
`server/tests/eval/corpus_provenance.json`. I did not run a live `hybrid_search` query — the report's
distance numbers (0.34–0.39) are accepted as reported, and judged only for plausibility against the
actual message text, per the brief.

**Structural counts, verified directly against the file (not taken on the report's word):**
message/thread totals, channel count, and the near-miss/orthogonal split all check out. 12 threads ×
10 messages + 1 thread × 11 messages (`notification-service-database-choice` has a rare
back-to-back `user`/`user` turn — messages 9–10) = **121**, matching
`corpus_provenance.json`'s `message_count: 121` exactly. 7 distinct `channel_id`s
(`incidents`, `architecture`, `security`, `people-ops`, `planning`, `logistics`, `infra`), matching
the report's claim. Every thread is within the plan's 8–12-message band. The plan's minimums (≥3
near-miss pairs, ≥2–3 orthogonal topics) are both exceeded (4 and 4 respectively).

### 1. Near-miss pairs — real, not just labeled

All four claimed pairs hold up on a close read of the actual text, each for a different, genuine
reason:

- **`payment-timeout-incident` / `search-latency-incident`** — near-identical incident-report
  template (symptom → confirm timing → what changed → rollback confirms hypothesis → "Confirmed
  root cause: ..." → action item → tracked follow-up), same shared vocabulary (`deploy`, `rolled
  back`, `root cause`, `p99`/rate figures), different specifics (payment-service v3.9 outbound
  timeout vs. search-service v2.7 Redis lookup). A paraphrase like "what caused the timeout spike
  after last week's deploy" is genuinely ambiguous between them.
- **`notification-service-database-choice` / `notification-service-retry-strategy`** — the
  *strongest* pair: both literally name `notification-service`, both end in an explicit "Decision:
  ... / Recorded." beat. A query mentioning "notification-service" alone, without the DB-vs-retry
  specifics, would plausibly retrieve either.
- **`oauth-token-refresh-bug` / `session-timeout-policy`** — both are "users getting logged out"
  threads sharing `session`/`logout`/timeout-duration vocabulary, but one is a race-condition bug
  and the other a deliberate policy change — a genuinely confusable pair for a query like "why do
  users keep getting logged out."
- **`customer-support-escalation-process` / `on-call-rotation-schedule`** — both center on
  `on-call`/paging/rotation vocabulary; slightly looser than the other three (one is about SLA
  process, the other about calendar fairness) but still plausibly confusable.

The reported 0.34–0.39 top-3 distances are consistent with what this text would plausibly produce
(shared domain vocabulary + near-identical narrative shape is exactly what pulls embeddings close);
I did not independently reproduce the numbers.

**Worth flagging for whoever drafts Unit 2a's golden queries (not a corpus defect):** the
"report → clarify → diagnose → confirm → follow-up" template isn't confined to the four labeled
pairs. `oauth-token-refresh-bug` ("Root cause: concurrent refresh calls racing...", "deployed it
Monday", a quantified before/after metric) shares real structural overlap with the two incident
threads too, not just with its labeled pair-mate `session-timeout-policy`. This likely means the
*effective* distractor set for an incident-thread query is wider than "its one designated pair,"
which is fine (arguably a more realistic, harder test) but the golden-set author should not assume
a near-miss-pair query's only close competitor is its labeled partner.

### 2. Orthogonal topics — genuinely orthogonal, one soft exception worth a later spot-check

`hr-onboarding-checklist`, `q3-product-roadmap-planning`, and `office-relocation-logistics` read as
cleanly orthogonal to everything else — distinct vocabulary registers (people-ops/HR, product/sales,
facilities) with only incidental, non-distinguishing word overlap (e.g. `office-relocation`'s
one-line mention of a "server closet" doesn't pull it toward the infra threads; the content is
almost entirely about desks/lease dates/conference rooms).

**`database-backup-policy`** is the one soft case: it shares the general infra/ops register and the
literal word "database" with `notification-service-database-choice`, though the actual vocabulary
diverges sharply once you're past the topic word (`snapshot`/`retention`/`compliance`/`restore-drill`
vs. `writes/minute`/`partition`/`notificationId`). I'd expect this pair to sit further apart than the
four labeled near-miss pairs, but it's the one place in the corpus I wouldn't be surprised by a
higher-than-baseline (if still not top-3) similarity. Worth a cheap spot-check once Unit 2a drafts
queries against either thread — not worth reworking the corpus for.

### 3. No answer-key leakage

All 121 messages are conversational (a report/question, a clarifying reply, a decision, a follow-up)
— none are phrased as a literal question-and-answer block a future golden query could pattern-match
against without real retrieval. No message states a "query: ... answer: ..." shape. Clean.

### 4. Specificity for unambiguous targeting

Every thread carries concrete, distinguishing facts an implementer can paraphrase a query against
without ambiguity — version numbers (v3.9/v2.7/v2.6), timestamps (14:20 UTC, "Monday, August 3rd"),
percentages and latencies (3%→0.1%, 180ms→2.1s), headcounts and dates (68 desks, Nov 22–23, 4
engineers starting Sept 2nd), dollar figures ($180,000 ARR), and named services/providers (APNs/FCM,
DynamoDB/PostgreSQL). I did not find a single thread that reads as generic or interchangeable with
another — this check passes cleanly across all 12.

### 5. Realism — the one genuine minor finding

Eleven of the twelve threads are **exactly 10 messages, strictly alternating `user`/`assistant`**,
and all twelve (including the one 11-message thread) follow close to the same five-beat arc: report
→ clarifying question → detail/diagnosis → decision or fix → confirmation/follow-up. That's a
noticeably uniform shape for "reads like a real chat" — a real corpus would more likely show uneven
turn counts, occasional short exchanges, a thread that trails off or changes direction, or two people
talking past each other briefly. This doesn't undermine the eval's functional validity (specificity
and near-miss/orthogonal structure — the properties that actually drive recall@k — are both solid),
but it's a legitimate, visible tell that the corpus is authored-to-spec rather than organic, worth
naming plainly since the brief asked for it. Not a reason to hold Unit 1, but worth keeping in mind
if this corpus is ever extended: vary turn count and arc shape rather than replicating the same
five-beat template a thirteenth time.

### Verdict: approve with suggestions

The corpus meets every structural requirement in plan §4 (message/thread counts, near-miss pair
count, orthogonal-topic count, specificity, provenance consistency) and I found no leakage and no
mislabeled pair — all four near-miss pairs are real on inspection, not just asserted, and the four
orthogonal topics are genuinely orthogonal with one soft, non-blocking exception. Two minor,
non-blocking observations for downstream awareness: the corpus's narrative uniformity (finding 5)
and the wider-than-labeled distractor bleed around the incident-report template (finding 1's
addendum) — both are worth the Unit 2a golden-set author's awareness, neither is a defect that
should hold Unit 1 or require re-authoring the corpus. **Approved as the load-bearing corpus for
Unit 2a's golden-set drafting.**

---

## Golden-set content review (Unit 2a)

**Scope:** a content review of `server/tests/eval/golden_retrieval.jsonl` (38 pairs, delivered by
`tdd-engineer`) against `scripts/seed_eval_corpus.py`'s `_CORPUS` — the same "human verification"
stand-in as the Unit 1 corpus review above (plan §4/§7 item 3), not a code/design review of
`test_golden_set_integrity.py` or `embed_golden_queries.py`. I did not re-run the integrity suite
myself; teco's report of 117 passed is accepted as reported.

**Mechanical sanity checks, verified directly:** `golden_retrieval.embeddings.json` has exactly 38
entries, all keyed `gr-01`..`gr-38`, all with `model: "lmstudio/text-embedding-qwen3-embedding-0.6b"`
and a 1024-length vector — matching `golden_retrieval.jsonl`'s 38 lines and the real embedding
model/dimension. No orphaned or missing entries either direction.

**Every single `target_text` value — all 38 — was cross-checked byte-for-byte against the actual
corpus message it claims to quote** (reconstructing each `msgId` from its `topic`/index and reading
the corresponding message in `_CORPUS`), not spot-checked. All 38 match exactly, with no truncation,
paraphrasing, or drift. The two multi-`relevant_msgIds` pairs (`gr-15`, `gr-34`) both correctly carry
the *first-listed* id's text as `target_text`, per the schema's own rule. I found **zero fabricated
or mismatched `target_text` values** — this is a clean, careful piece of authoring.

**Topical coverage:** all 12 corpus threads are represented (2–4 pairs each; `oauth-token-refresh-bug`
is the thinnest at 2, everything else is 3–4) — full coverage, reasonably balanced, no thread
skipped. `relevant_msgIds` values span a good mix of "diagnosis"/"decision"/"numbers" message types
within each thread rather than clustering on one message per thread.

**No leakage beyond what the integrity test's substring check catches — with one exception worth
fixing (Minor).** Every query I checked reads as a genuine paraphrase (question form vs. the
target's declarative/decision form, different word order, only natural domain-vocabulary overlap)
— **except `gr-31`**:

> query: *"How far back does finance need point-in-time recovery for audit purposes?"*
> target_text: *"Finance needs point-in-time recovery for any date within the last 90 days, for
> audit purposes."*

This passes the integrity test's literal substring check (the query isn't a contiguous substring of
the target, because "for any date within the last 90 days," sits between the two halves it borrows),
but it isn't a real paraphrase either: strip the query's "How far back does" prefix and "?" suffix
and what's left — *"finance need point-in-time recovery ... for audit purposes"* — is the target
sentence's own opening and closing clauses, reused nearly verbatim, with only the middle numeric
clause excised and turned into the question word. This is the weakest pair in the set by a clear
margin; every other query I checked constructs its own sentence rather than deleting a clause from
the target. **Suggested fix:** reword to something that doesn't reuse "point-in-time recovery"/"for
audit purposes" as a contiguous phrase — e.g. "What's the required data-retention window for
financial audits?" or "How many days back does the finance team need to be able to restore data?"
— preserving the same target message without the near-verbatim clause reuse.

**One positive pattern worth naming, not a defect:** `gr-36`'s query ("How many engineers are needed
per on-call shift given the current page volume?") is itself a close paraphrase of a *different*,
non-target corpus message (`on-call-rotation-schedule`'s message 007, the user's own question in the
thread — "How many engineers do we need per shift, given current page volume?"), while its actual
answer/target is message 008 (the assistant's reply immediately after). This is realistic (a golden
query phrased the way a real user actually phrased that exact question in the corpus) and makes for
a legitimately harder retrieval test (the corpus contains a near-duplicate of the query itself that
isn't the right answer) rather than an easier one — no action needed, flagged only because it's the
kind of thing worth knowing about when interpreting a lower-than-expected recall on this specific
pair.

### Verdict: approve with suggestions

The golden set is well-constructed and, on the specific dimension this review exists to check
(does it faithfully and honestly represent the corpus it's drawn from), essentially clean: all 38
`target_text` values verified exact, full topical coverage across all 12 threads, no fabrication,
correct handling of the multi-relevant case. One concrete, fixable finding: **`gr-31`'s query is a
near-verbatim clause reuse of its target text** and should be reworded before this golden set is
treated as load-bearing for the frozen baseline — it's the one pair in 38 that would inflate recall
for a reason other than genuine retrieval quality. Not a blocker for Unit 2b to proceed (37/38 pairs
are clean, and even `gr-31` isn't a leakage *bug* — the integrity test's literal check is working as
designed, this is a step beyond what that mechanical check can catch), but worth a one-line edit
before `retrieval_baseline.json`'s first run is treated as the sign-off-gated baseline (D6).

---

## Pass 4 (2026-08-15) — D1-scoped re-gate (v3 → v4)

**Scope, per the request:** a narrow gate on v4's single change — D1's judge-model default
collapsing onto `qwen/qwen3-4b-2507` (the agent-under-test's own model), a stakeholder-directed
hardware-constraint decision relayed by the coordinator, not a reopened Pass 1–3 finding. Not a
re-review of the plan's substance (Pass 3 already approved that). I read the v4 revision note and
all of §3 D1 in full, then traced every downstream citation the request named (§5 Unit 3, §6, §7
items 2/8, and the closing "Ready to implement" section) against the new D1 text, and — since this
file isn't under git version control (`git status` shows it untracked, so no mechanical diff was
available) — cross-checked D2–D7, §4, and Units 1/2a/2b's text against my own verbatim quotes from
Pass 1–3 to confirm they're genuinely untouched rather than trusting the revision note's claim.

**D1's core change is internally consistent.** The new default, the rationale ("registered" ≠
"usable" on the actual hardware), the explicit acknowledgment that this drops the method note's
self-preference-bias safeguard, and the mitigation (a mandatory caveat + `data-scientist` sign-off
gate mirroring D6) all fit together coherently — there's no place where the plan asserts the
trade-off is costless or silently elides it.

**All four named downstream citations are genuinely updated, not just gestured at:**
- **§5 Unit 3** (`:555-608`): the env-var default, the fixture-reachability note (correctly
  observing this isn't a *new* dependency — the judge role now rides on the same model already
  being loaded for the agent-under-test), the generation sub-pass's client-construction note
  (correctly clarifying two separate client *instances* are still constructed even though they name
  the same model — avoids an implementer conflating "same ref" with "reuse one client"), and the
  new `judge_calibration.json` field (`sameModelAsAgentUnderTest`, computed from the *resolved*
  refs at runtime, not a static default-value comparison — correctly handles an operator overriding
  only one of the two env vars) are all present and consistent with each other.
- **§6** (`:624`): the test-strategy table row for `test_judge_live.py` is updated with both the
  "one loaded model, two roles" framing and the caveat/sign-off sentence.
- **§7 item 2** (`:648-655`): correctly scoped to "the config-wiring mechanism is unchanged, only
  the default value changed," cross-referencing item 8 rather than duplicating it.
- **§7 item 8** (`:696-710`): fully rewritten, not patched — states the registered-vs-usable
  distinction explicitly, names the same four alternative models as the revision note, and states
  both direct consequences (the sign-off/caveat requirement, and the unaffected skip-on-unreachable
  behavior) rather than only the first.
- The closing "Ready to implement" section (`:731,745-763`) also picks up the change in both the
  per-unit list and the "key design calls" paragraph.

No stale references to the old default survived the edit — I specifically checked for this, since
v1–v3's Unit 3 text had a sentence tying `openai/gpt-oss-20b`'s *availability* to the old default
("environment note confirms it's available alongside the 4B in the current LM Studio load"); that
sentence is gone from v4's Unit 3 paragraph, not left dangling next to the new default.

**The limitation note says what it needs to say.** "Self-preference-bias limitation (added in v4)
— accepted, named, and sign-off-gated, not an oversight" (`:227`) is explicit, and the "produced,
but not yet load-bearing" framing (`:242`) unambiguously states the numbers aren't trustworthy until
sign-off — matching the request's exact bar.

**D2–D7 (excluding D1), §4, and Units 1/2a/2b are genuinely untouched.** Cross-referencing this
read against my own quotes in the Pass 1/2/3 sections above (the only diff mechanism available,
since the file isn't committed to git) — D2, D4, D5, D6, D7's text, the full §4 corpus-design
section, and Unit 1/2a/2b's file lists and "Done when" clauses all read identically to what I
already verified word-for-word in earlier passes. Nothing outside D1's blast radius moved.

**One genuine, narrow gap found — not in D1 itself, but in how it wires into `generate_report.py`
(Minor, not blocking).** The Unit 3 file-list bullet for `generate_report.py` (`:604-608`) asks the
report to include, when applicable, "an explicit note on whether the required `data-scientist`
sign-off (D1) has happened yet — mirroring how the report already carries D6's baseline sign-off
status, not a new report mechanism." **That premise doesn't hold.** D6's sign-off is enforced by
*not committing* `retrieval_baseline.json` until the sign-off happens (`:324-326`: "the first
commit of `retrieval_baseline.json` therefore gets an explicit `data-scientist` methodology
sign-off... before it's treated as gating") — the file's mere presence in the committed repo *is*
the sign-off signal, there's no separate status field the report reads. `judge_calibration.json`
doesn't have an equivalent gate: per its own spec (`:590-595`), it's written on **every** completed
live run unconditionally, sign-off or not — so, unlike the baseline file, its presence can't be read
as "sign-off happened." Nothing in v4 defines what the report's "has sign-off happened yet" note
would actually read to know the answer — there's no described sign-off-marker file, no field a
`data-scientist` is asked to set on `judge_calibration.json` after review, nothing. The mandatory
self-preference-bias *caveat* itself is fine (it's unconditionally shown whenever
`sameModelAsAgentUnderTest` is true, which is the safety-critical part and needs no external state);
it's specifically the report's proposed "sign-off status" sub-note that has no data source. Two
independent fixes, either closes it: (a) drop that sub-clause and let the sign-off tracking live
outside the file the way D6's actually does (e.g., in the coordination doc, mirroring the process
rather than the artifact), or (b) define a lightweight sign-off marker (e.g., a
`judge_signoff.json` sidecar the `data-scientist` writes after reviewing) that `generate_report.py`
can actually check for — which would make the "mirrors D6" claim literally true instead of only
true in spirit.

### Updated verdict

**Approve with suggestions.** D1's core change — the default, the rationale, the safety framing —
is sound and correctly propagated to every downstream citation the request asked me to check; D2–D7
and everything else are confirmed untouched. The one finding (the report's unspecified sign-off-
status data source) is narrow, doesn't undermine the change's actual safety property (the caveat
itself is unconditional and correctly triggered regardless of sign-off state), and is a one-line
scope-or-mechanism fix, not a redesign. **v4 is dispatchable as written** — Unit 3 can proceed with
the current text; the sign-off-status gap is worth a follow-up line in the plan before
`generate_report.py` is actually implemented, not a reason to hold anything now.
