# `ItemScorer` extension (`prime`/`embed_text`/`deterministic_arm`) — seam-readiness review for S4

> **Status:** active · **Owner:** `analyst` · **Tracks:** U123 (S3); informs S4 planning

## Scope & verdict

Reviewed, statically: the `ItemScorer` Protocol extension shipped in U123
(`modelbench/runner.py`'s `ItemScorer` class, `_load_item_scorer`, and
`_drive_single_call_items`'s two `getattr`-guarded call sites; `modelbench/cli.py`'s
`_cmd_run` deterministic-arm hook; `modelbench/scoring/retrieval.py` as the one concrete
implementation), judged specifically as an extension point for S4's two upcoming concrete
scorers (`guard-judge`, `nlq-generator`) — **not** a re-review of `retrieval.py`'s already-accepted
arithmetic (U123/U125). Baseline: `docs/plans/small-model-benchmarking-s3-spec.md` §2.2/§3.2/§3.3/§9,
`docs/plans/small-model-benchmarking.md` §3.8.2/§3.8.3, `docs/plans/small-model-benchmarking-coordination.md`
rows U120/U121/U123, and the falkor-chat source S4's two packs transcribe their real mechanism from
(`falkor-chat/server/falkorchat/app.py`'s `_LlmGuardJudge`). This review directly answers the s3-spec's
own §9 open item: *"an independent review pass... before S4's `guard-judge`/`nlq-generator` scorers are
built on top of the same seam, since a second role's needs may reveal the extension is shaped wrong
for anything beyond the embedder."* Confirmed live: the full suite reproduces `1250 passed, 1 failed
(the ruled S5 tripwire), 3 deselected`, matching U123/U125's ledger rows exactly.

**Verdict: needs changes.** Not against the already-accepted U123 delivery — that unit is sound and
stays accepted, and nothing here reopens it. Against the seam's readiness for S4: the extension is
correctly shaped for the embedder's own two gaps (corpus priming, embeddings-branch text), but a
second, concrete look at what `guard-judge`/`nlq-generator` actually need (both real, single-call
**chat**-surface roles per the plan, not embeddings) surfaces two gaps that were not
built and are not yet even named as open items anywhere in the tree — one of them a pre-existing,
untested correctness defect that this seam's own consumption path will trigger the moment either
pack is driven live. Both should be resolved, or explicitly scoped as S4 Step 0/1 work, before an
implementer starts on `guard-judge`'s scorer.

**CPG:** considered, not relevant — `model-bench` is a Python component with no `cpg_model-bench`
graph loaded on this FalkorDB instance; this is a code-level task, so "considered, not relevant"
applies rather than "not applicable."

## Findings

### MAJOR — The chat branch has no scorer-side hook for per-item prompt construction; `embed_text`'s twin doesn't exist yet, and guard-judge needs one

`_drive_single_call_items`'s embeddings branch calls `scorer.embed_text(item_input, pack=pack)`
(`runner.py:359-361`) instead of the hardcoded JSON dump — exactly the fix S3 built for gap 2
(module docstring gap 1, s3-spec §2.2 point 1). The **chat** branch got no equivalent: it still calls
`_item_chat_messages(pack, item_input)` (`runner.py:300-314, 347-354`) unconditionally, which builds
the user message as `json.dumps(item_input, sort_keys=True)` with no scorer input at all — the
function's own docstring calls this "deliberately role-agnostic... only what lets the runner issue a
real call for a stub or fixture item today" and defers "each role's real prompt template" to "a
scorer concern this document does not invent" (`runner.py:304-308`). But no such concern-point
exists on `ItemScorer` — `prime`/`embed_text`/`deterministic_arm` (`runner.py:177-198`) touch priming,
the embeddings branch, and the CLI-only BM25 hook; none of them let a scorer shape the chat branch's
messages.

This is not hypothetical for S4: the plan states `guard-judge`'s pack "carries the judge prompt as
text (`prompts/judge.md`), transcribed from `falkorchat/app.py::_LlmGuardJudge`'s prompt
construction" (`docs/plans/small-model-benchmarking.md:2079-2081`), and that source's real user
message is a labelled, multi-block render with its own truncation rule —
`_render_judge_user(condition, understanding, recent_turns)` builds `"CONDITION: ..."` /
`"CURRENT STATE:\n<json>"` blocks and caps at `JUDGE_USER_MAX_CHARS`
(`falkor-chat/server/falkorchat/app.py:619-634`), never a flat `json.dumps` of the golden item.
`nlq-generator` is worse: its prompt needs the catalog/schema/table context threaded in per §3.8.3,
which the generic item row alone cannot carry. A generic `json.dumps(item_input)` cannot satisfy
either role.

`ItemScorer`'s own docstring lists these two roles as being "carried" by this Protocol shape
(`"""One item-level role's scorer (embedder, guard-judge, nlq-generator, chat-responder)."""`,
`runner.py:158`) — accurate for the two required methods, but the three optional ones are, in
practice, embedder-only; nothing flags that a fourth optional method is still owed before a chat-
surface role can build a real prompt. The s3-spec's own §9 names this exact risk in the abstract
("the extension is shaped wrong for anything beyond the embedder," `s3-spec.md:907-916`) but does
not identify the concrete gap; this review does.

**Suggested fix:** extend `ItemScorer` with a fourth optional method (e.g.
`build_messages(item_input, *, pack) -> list[ChatMessage]`), `getattr`-guarded in
`_drive_single_call_items`'s chat branch exactly the way `embed_text` is guarded on the embeddings
branch — called instead of `_item_chat_messages` when the scorer defines it, falling back to the
existing stub otherwise so no shipped test needs to change. Do this as an explicit, named step in
S4's own plan (mirroring how s3-spec §2.2 named its two gaps as their own implementation step),
not discovered mid-implementation the way U120's gaps were.

### MAJOR — `prompt.systemPrompt`/`toolSchemas` are never resolved from paths to content, and every consumer already treats them as if they were

`Pack.prompt_config()`'s own docstring states `systemPrompt`/`toolSchemas` are "carried through as
the manifest's own declared values — paths, not resolved content... no such caller exists in this
tree yet" to resolve a `prompt.*` path against the pack root the way `Pack.data_path` resolves a
`data.*` one (`packs.py:366-370`, `395-396`). `convo.PromptConfig`'s docstring, written earlier,
states the opposite contract: `systemPrompt`/`toolSchemas` **are** "the resolved content"
(`convo.py:148-153`). Both are honest about the gap, but every real consumer already reads the field
as if resolution had happened: `runner._item_chat_messages` puts `cfg.systemPrompt` straight into a
chat message's `content` (`runner.py:311-312`); `run_pack`'s two `warm_up` calls pass it as
`system_prompt=` verbatim (`runner.py:797`, `813`); `convo.assemble` does the identical thing
(`convo.py:376-378`). The one real pack fixture that declares a `prompt.systemPrompt` value at all,
`tests/fixtures/packs/valid/pack.json`, declares it as a **path** (`"prompts/system.md"`) — but
every test that actually drives `_item_chat_messages`/`convo.assemble` with a real value uses literal
inline text instead (`tests/test_convo.py:63, 561`; `tests/test_runner.py:277`), so this path never
gets exercised. The first chat-surface pack driven for real — `guard-judge`, per S3's own file-layout
convention (`prompts/judge.md`) — will send the literal string `"prompts/judge.md"` to the model as
its system prompt, not the file's text, unless this is fixed first.

This predates and is independent of U123's extension (S3 never touched the chat branch), so it is
not a defect in the delivered unit — but it sits directly on the path S4's coupling to this seam
must cross, is currently untested against a realistic fixture, and both docstrings that describe it
already disagree with each other about the intended contract. Left alone, it produces a silent,
loud-nowhere wrong prompt on `guard-judge`'s first live run.

**Suggested fix:** add a `Pack` method resolving a `prompt.*` path against `pack.root` (mirroring
`data_path`'s existing pattern), have `prompt_config()` (or its callers) use it for `systemPrompt`/
`toolSchemas`, and reconcile the two docstrings once resolution exists. Pair with a test that drives
`_item_chat_messages`/`convo.assemble` against a `systemPrompt` value that is a real path pointing at
a real fixture file, asserting the *file's contents*, not the path string, land in the sent message —
today's suite has no such assertion (`grep -n 'systemPrompt' tests/*.py` shows only literal-text or
`None` cases, confirmed above).

### Minor — `ItemScorer`'s docstring role list slightly overstates what today's optional methods cover

`runner.py:158`'s `"""One item-level role's scorer (embedder, guard-judge, nlq-generator,
chat-responder)."""` is true for `score_item`/`aggregate` but not for `prime`/`embed_text`/
`deterministic_arm`, which are embedder-specific in practice (confirmed: neither `guard-judge` nor
`nlq-generator` needs a corpus-priming hook or an embeddings-branch text builder per
`docs/plans/small-model-benchmarking.md:5699-5713`, and neither pack's plan section mentions a
deterministic reference arm the way retrieval's BM25 arm exists). A skim of this docstring alone
could lead an S4 implementer to assume the extension already covers their role. **Suggested fix:**
one added sentence noting which optional methods are embedder-only today, or fold the note into
whatever docstring change the fix for the first finding above adds.

## What's solid

- The structural-typing discipline is intact end to end: `_load_item_scorer` resolves
  `pack.manifest["scorer"]` via `importlib.import_module` and returns the module itself
  (`runner.py:214-231`); every call site (`_drive_single_call_items`, `cli.py:425-430`) reaches
  scorer methods via `getattr`/plain attribute access, never `isinstance`, and neither `ItemScorer`
  nor `ConversationScorer` is `@runtime_checkable`. Nothing in the extension introduces a class-
  instance assumption.
- `prime`/`embed_text`/`deterministic_arm`'s own signatures are generic enough to be reused as-is
  by a future role that genuinely needs pre-loop live setup, embeddings-branch text, or a
  no-live-call reference arm — none of the three needs renaming now, and none is a breaking change
  to defer; S4 simply won't call them.
- `getattr`-guarding is applied consistently and each call site is commented with its originating
  spec section (`runner.py:332-340`, `356-364`), so what's already built is discoverable; the gap is
  what's *not* built, not how the existing pieces are documented.
- `retrieval.py`'s own docstring is explicit that it is "a module, not a class" satisfying the
  Protocol structurally (`retrieval.py:8-13`), a clear worked example for whoever writes
  `classification.py`/`extraction.py` next.
- The s3-spec's own §9 already flagged this extension as unreviewed and asked for exactly this pass
  before S4 — the process worked as designed.

## Open questions

- Should the fourth `ItemScorer` method (message construction for the chat branch) be **required**
  for any role whose `environment.requires` declares `lmstudio-chat` at the item level, the way
  `embed_text` is treated as "required in practice" for the embeddings branch — or should it stay
  optional with `_item_chat_messages`'s JSON-dump as a legitimate fallback for a future item-level
  chat role simpler than `guard-judge`/`nlq-generator`? This is an S4 design call, not this review's
  to make.
- Is the `prompt.systemPrompt`/`toolSchemas` path-resolution fix (finding 2) sized as its own small
  precursor unit ahead of S4, or folded into S4 Step 0/1 alongside the new Protocol method? Both
  land in the same area of `packs.py`/`runner.py`/`convo.py` and a single implementer touching all
  three at once may be the more efficient sequencing — a scoping call for whoever plans S4.
