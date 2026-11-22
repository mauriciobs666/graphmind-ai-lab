# Connected code graph — P1-35 interpreter design-fault diagnosis

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1-35) · **Extends:** `docs/plans/connected-code-graph-p1-interpreter-model-ml.md` · **Reviews:** `docs/reviews/connected-code-graph-p1-interpreter-design-fault-ml.md`

States the diagnosis the design-fault trigger of that note's §3 asked for, the prompt-edit unit that follows from it, and its effect on the dev set and the
`ccg-prereg-ac24-v1` record. The tries, stop rules, fresh-set controls and power figures are in `docs/plans/connected-code-graph-p1-interpreter-model2-ml.md`
and are cited, not restated. No held-out item, custodian directory or dev gold beyond `cpg/eval/dev/ac24.jsonl` was read.

## 1. Result

**The resolver has no defect against §4.2 as built.** Replaying the saved drafts through `ccg.front.resolve.resolve_draft` reproduces the logged verdict kind and
option count for every missed qid, and an ideal draft for each resolves, or asks, as gold says. The misses are (a) a prompt silent on four constructs these questions
exercise, (b) span-discipline failures of 3-4B models, and (c) one pre-registered, gold-blind resolver policy (the unfound-name ask) that turns a harmless extra span into an
over-ask. No gold label is wrong. "Prompt silence" is a **hypothesis under test**: only the fresh dev set (model2 note §4) shows whether a better prompt fixes a 3-4B model, so the
edit is reported as an intervention, not a cure. Decision for the stakeholder: §6; recommendation A (fix, fresh dev set, pre-stated stop).

## 2. Method and its limits

- Replay: `cpg/.venv` python, `CypherReader` over the live `ccg_graphmind-ai-lab` graph (FalkorDB `127.0.0.1:6390`, read-only), `to_draft(item.repeats[0].draft)` into
  `resolve_draft(reader, draft, normalize_question(text), ("graphmind-ai-lab",))`; scripts `dfault.py`, `dfault2.py` in the session scratchpad. The store is not the run's clone (it also holds
  `falkor-chat/docs/reviews/.../mut/` copies), so QW-19 and QW-27 show extra candidates there; they passed in the runs. The four qids below reproduce the logged `ambiguityKindEmitted` and `nOptions` exactly.
- Try 0 (`qwen/qwen3-4b-2507`) stored no drafts; its rows rest on the verdict (QW-7 and QW-13 match the other two models' drafts; QW-11 and QW-47 are consistent with a direction misread, unconfirmed).
- **Trigger reading.** "Same way on >= 3 runs" holds literally for QW-13 and QW-7. QW-11 and QW-47 miss differently per model and are treated as cue questions because they share the cause in §3.

## 3. Diagnosis per qid

| qid | What the drafts did (replayed) | Class | Resolver on an ideal draft |
|---|---|---|---|
| **QW-7** | ministral: file span of two joined words, verbatim so it passes the guard, then fails the path lookup, `unresolved`. qwen2.5: a file span not in the question (guard drops it), the symbol alone has two real candidates, `several-symbols`. qwen3: same verdict. | Capability (joined or fabricated span) plus prompt silence on span form. | symbol + bare file name resolves to `salesperson/src/api/endpoints.ts::postMessage`, `impact`; symbol alone is a true 2-way ambiguity. |
| **QW-13** | All three add a second group for the renamed parameter `re` (symbol, with a `server` or a fabricated file span); ministral also hints the server name as `tool`, which `_group` checks before `symbol`. `re` finds nothing, so the resolved tool rides as an `unknown-name` ask. | Prompt silence ("one group per thing the question points at"; nothing says a parameter is not an anchor) plus the pre-registered unfound-name ask; hint confusion is capability. | `send_message` as `tool`, with or without the server, resolves; adding `re` as `symbol` gives the logged ask, as `other` resolves. |
| **QW-11** | ministral: right route, direction flipped to `depends-on`. qwen2.5: dropped the verb, five contract points match, `several-symbols`/5. qwen3: `wrong-interpretation`. | Prompt silence (the `impact` cue is only implied by the subject wording of the `depends-on` cue; nothing says a route span keeps its verb) plus capability. Gold sound. | `POST /threads/{}/messages` + `impact` resolves; a verb-less route is a genuine 5-way ambiguity. |
| **QW-47** (gold `two-use-cases`) | ministral: symbol + `tool` span `MCP`, `depends-on`; the `tool` span wins in `_group`, finds nothing, `unknown-name`/0. qwen2.5 and qwen3: silent single `depends-on`. | Prompt cue gap: the only `null` cues are "an overview of" and "o panorama de", and "never answer null when the question implies a direction" pushes a small model to pick. Hint confusion is capability. Gold sound. | `direction: null` gives `two-use-cases` with both readings, matching gold. |

Same family, outside the trigger: qwen2.5 QW-3 rewrote `... line 564` as `...:564` (guard drops it, `no-anchor`); qwen2.5 QW-39 and QW-42 dropped the verb.

**Not covered by any rule (expected residuals, no sixth rule):** qwen2.5 QW-19 came back with use case `root-cause` (`not-built`) and ministral QW-30 read a component name as a repository `scope`
(`not-built`). Both are misreadings of fields the prompt already states and neither is a resolver defect. They cost slack: a model that repairs every other miss sits at the threshold. They appear in the report if the search fails (model2 note §5 S5, §7).

## 4. Fix: the prompt-edit unit (`tdd-engineer`; gate `analyst` on the diff, then `data-scientist` re-reads the wording, model2 note §3)

**No resolver or grammar change** (`RESOLVER_VERSION` `1.0`, `resolverRules`, `grammar/interpret.json`, `SAMPLING` stay). A resolver edit would be written against a dev item, and the unfound-name ask
is the documented policy. **Parked for P2** (`architect`/`data-scientist`): report an unfound name beside a resolved anchor as a notice rather than an ambiguity; it needs an answer-schema field `answer-v1.schema.json` lacks.

**Files.** Edit `cpg/ccg/front/prompts/interpret.txt` and `cpg/tests/test_front_prompt.py` (module docstring "at most 4" becomes "at most 6"). Nothing hard-codes `promptHash` (grep of `cpg/` finds `494fb949…` only in result files), so the hash, every
`cacheKey` (`ccg.front.cache.key_of` takes `prompt_hash()`) and `coreTreeHash` change by themselves; `grammarHash`, `samplingHash`, `packSetHash` do not. Add one pin in `cpg/tests/test_front_interpret.py` if absent: two prompt texts give two cache keys.
The canary digest is not a repo file: it is re-recorded by `ac24_smoke.py canary`/the dev run (`CANARY_QUESTIONS` unchanged). Keep one system plus one user message. Report prompt length in tokens, old and new, in the commit (not a gate).

**Wording** (add to the `Fields` block; every name is made up, none is a dev question's word; if a standing pin below goes red on a word, reword that word, never loosen the pin):

| Rule | Text to add or change in `interpret.txt` |
|---|---|
| R1 route verb | to the `span` field: `A route span keeps its HTTP verb exactly as written: in "PUT /fx-items/{id} is slow" the span is "PUT /fx-items/{id}".` |
| R2 span form | to the `span` field: `A span is one unbroken stretch of the question. Never join words and never add a "/" or ":" that is not there. A component name written before a file name is not part of the file span: copy only the file name.` |
| R3 parameter | to the `anchors` field: `A parameter, field or option that the question only says will be changed is not an anchor: anchor the tool, route or function that owns it.` |
| R4a direction | in the `direction` field: `impact` clause becomes `when the question asks what is affected, who or what calls, uses, consumes or reaches the anchor, what would fail`; the `depends-on` clause keeps `what the anchor itself calls or uses` unchanged. |
| R4b null (class, not a token) | in the `direction` field: `null when the question asks how the anchor connects to the rest of the system, names both directions, or names neither. Cues such as "an overview of", "o panorama de", "relations of", "relações de", "everything about", "how it fits in" are examples only.` (six cues; "relations of" and "relações de" are two of them, never the only ones.) |
| R5 hint | to the `kindHint` clause: `tool is the tool's own name and server is the server's. A protocol or product label in brackets after a tool name, such as "(FXQ)", is neither: give it other or leave it out.` |

**Two new examples** (ceiling six, F4; both `fx-*` only, sentence frames unlike any dev question, one Portuguese; the existing four stay; the pins below must pass on them):
- `Q: PUT /fx-items/{id}: renaming the fx_limit field, which callers are hit?` / `A: {"useCase":"impact","direction":"impact","anchors":[{"spans":[{"text":"PUT /fx-items/{id}","kindHint":"route"}]}],"scope":null,"readings":[]}` (shows R1, R3, R4a).
- `Q: quais as conexões do fx_ledger_close em fx-books/close.py?` / `A: {"useCase":"impact","direction":null,"anchors":[{"spans":[{"text":"fx_ledger_close","kindHint":"symbol"},{"text":"fx-books/close.py","kindHint":"file"}]}],"scope":null,"readings":[]}` (shows R4b in Portuguese).

**Pins** (write each red first, then green; each names its planted mutant, model2 note §3; mutants are run, not described):
1. `test_there_are_at_most_four_examples...`: rename to six, `1 <= len(EXAMPLES) <= 6`, plus `len(EXAMPLES) == 6` so the examples are present. Mutant: delete one new example.
2. R1: assert the field text contains the verb sentence and a route example whose span starts with an upper-case HTTP verb. Mutant: delete the R1 sentence.
3. R2: assert the field text contains `unbroken stretch`, `never join words` and the component-before-file sentence. Mutant: delete the R2 sentence.
4. R3: assert the `anchors` text contains the parameter sentence. Mutant: delete it.
5. R4a (rewrites `test_the_depends_on_cue_is_worded_by_subject...`; do not just delete it): split the `direction` line at `"depends-on"`. Assert `who or what calls, uses, consumes or reaches the anchor` is in the `impact` part, `DEPENDS_ON_CUE` is in the `depends-on` part, and every `\bcalls\b` in the `depends-on` part is the one inside `DEPENDS_ON_CUE`; keep the `Never answer null` assertion. Mutant: swap the two cues.
6. R4b (rewrites `test_the_neutral_null_direction_cue_appears_in_no_dev_question...` and drops its `"relations" not in words(PROMPT)` line, which kept the QW-47 cue out and is retired now dev1 is spent): assert the null clause contains the class sentence (`how the anchor connects to the rest of the system, names both directions, or names neither`) and at least three of the six cues; keep `NEUTRAL_CUES` in the prompt with `"an overview of"` counted twice and `"o panorama de"` once. Mutant: remove the class sentence keeping every cue word.
7. R5: assert the hint sentence uses a `fx`-style label such as `(FXQ)` and says `other` or `leave it out`; the existing "names every kind hint" pin stays. Mutant: replace the sentence by the word `MCP`.
8. Standing pins, kept as written: no 3-word phrase shared with any dev question (`ngrams(PROMPT,3)`), no 2-word phrase between example questions and dev questions, no dev question in the prompt, example-validity and example-axes pins, `fx-` in every example span.
9. Extend `test_no_dev_question_appears_in_the_prompt_and_no_dev_anchor_name_either` from tokens with `/ _ .` to bare identifiers and acronyms: collect from `ac24.jsonl` question text and every gold `keyNorm`/`qname` last segment the tokens that are all upper-case (3+ letters), contain `_`, or have an inner capital; exclude the HTTP verbs `GET POST PUT PATCH DELETE HEAD OPTIONS` (example 3 and R1 legitimately use them). Assert none is in `PROMPT` (word-boundary). Planted: insert `MCP` into the prompt, the pin must fail (a second planted mutant: `postMessage`).

## 5. What this does to the dev set and the record

- **The 14 dev items are spent for entry-check purposes** (earlier note §3); they keep their unit-test uses. EP §10.2 allows prompt work on dev questions, so the prompt edit itself is not a violation; the entry check's validity is what is spent.
- **Held-out stays unread and unspent**; `heldout-manifest.json` is not touched.
- **Order** (model2 note §4.1): prompt commit, `analyst` gate on the diff, `data-scientist` wording re-read; only then the fresh set `cpg/eval/dev/ac24-r2.jsonl` is authored (writer, annotators, custodian, controls, filters, seed and shape as model2 note §4); nobody who edited or gated the prompt reads it. Tries and stop rules: model2 note §5. `teco` confirms the writer sandbox, pinned binary, custodian root and `EVAL_COMMIT` checkout from `cpg/eval/headless/README.md` first (unverified here); fresh-set work can start inside P1 only after that check.
- **The record is not appended until a dev run on the final prompt passes.** `prereg-record` builds from the final run's `treeStamp` and refuses a dirty tree or a differing stamp. `devAttempts` lists the four old tries with `promptHash` `494fb949e16ca53914dd929ae6d0e3a60d99908b587819abcccbfa5d43bb5e15` and every new try with its own: add an optional `promptHash` key to `validate_dev_attempts` in `cpg/eval/tools/ac24_smoke.py` (a `harnessHash` change; `tdd-engineer`, red pin in `cpg/tests/test_ac24_smoke.py` first). **Disclosure in the record:** the prompt was revised after the first dev check failed, the first dev set is spent, and the dev1 prompt hash is stated.
- **Harness text fix (contradicts EP §10.2).** `ac24_smoke.py` line 552 prints `"evaluate a larger model; never tune the prompt, grammar or resolver on the dev items"`, which EP §10.2 contradicts (only dev items may be used in prompt work) and which would print again on dev2. One-line fix: `"entry check failed: do not edit the prompt, grammar or resolver after dev2 exists; held-out items are never used for tuning"`, with the pin on the string in `cpg/tests/test_ac24_smoke.py`. Owner `tdd-engineer`, same commit as the `promptHash` change (one `harnessHash` change); the string is a doc-level fix, not a scoring change.

## 6. Decision for the stakeholder

| | What | Cost | Risk |
|---|---|---|---|
| **A (recommended)** | §4 edit, fresh dev set and tries per the model2 note, then the record and the held-out run once. Stop rule S5: all three fail, report **"AC-24 not met at P1, carried to P2"** with all dev counts, the prompt hashes and missed qids by class. | One writer and two annotator headless sessions, one custodian step, a small prompt/test change, up to three dev runs. | A new failure is then capability, reported as such; the writer infrastructure may need re-pinning. |
| **B** | Stop now: "not met at P1, carried to P2". | Nothing. | AC-24 stays unmeasured at P1 and a fixable prompt gap ships to P2. |
| **C** | A larger local model on the unchanged prompt. | Three dev runs plus a new model list. | Does not repair QW-13/QW-7 (not model-level); a pass hides a prompt gap the held-out run exposes. |

AC-24's text: "if none does, a lower local target is set once the first phase has measured it". Recommendation A: the prompt is still free to change (the record is unwritten), held-out is untouched, the cost is bounded by S5. Spending the writer sessions is the stakeholder's call.

## 7. Steps if A is chosen

1. `tdd-engineer`: §4 unit and the §5 harness changes (promptHash key, string fix); suite green with `cpg/` as the working directory; commit; `analyst` gates the diff.
2. `data-scientist`: wording re-read of the committed prompt against the model2 note §3 table.
3. `teco`: infrastructure check, then the fresh set per the model2 note §4; commit file and sha256.
4. `qa-engineer`: canary, then the tries of the model2 note §5 with `--dev-items cpg/eval/dev/ac24-r2.jsonl`, drafts stored.
5. Pass: `prereg-record`, `teco` appends with the disclosure, held-out once. S5 stop: record the outcome and this diagnosis in `docs/HISTORY.md` and the P2 backlog.

## 8. Risks

- The added rules lengthen a prompt for a 3-4B model and can lose items the old one passed (QW-3, QW-30, QW-32); the fresh set and the model2 note's S6 diagnostic measure that.
- The replay store differs from the run's (§2); the four qids reproduce, and a changed store could change an option count, never a verdict class.
- The qwen3 reading rests on verdicts; every new try stores drafts.
- dev2 measures transfer to fresh items from the same writer population, not to a new writer (model2 note §4); a pass is a smoke test (power: model2 note §6).
