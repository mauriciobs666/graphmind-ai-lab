# Connected code graph — P1-35 interpreter design-fault diagnosis

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1-35) · **Extends:** `docs/plans/connected-code-graph-p1-interpreter-model-ml.md`

States the diagnosis the design-fault trigger of that note's §3 asked for, the smallest fix per defect, and what
it does to the dev set and to the `ccg-prereg-ac24-v1` record. No held-out item, custodian directory or dev gold
beyond `cpg/eval/dev/ac24.jsonl` was read.

## 1. Result in one paragraph

**The resolver has no defect against §4.2 as built.** Replaying the saved drafts of tries 1 and 2 through
`ccg.front.resolve.resolve_draft` reproduces the logged verdict kind and option count for every missed qid, and an
ideal draft for each of the four resolves or asks exactly as gold says. The misses are (a) a **prompt that is silent on
four span/cue constructs** these questions exercise (fixable, changes `promptHash`), (b) **span-discipline
failures of 3-4B models** (capability, not fixable by us), and (c) one **pre-registered, gold-blind resolver policy**
(the unfound-name ask) that turns a harmless extra span into an over-ask. No gold label is wrong. The fix is a prompt
edit; it spends the 14 dev items, so a re-check needs a fresh dev set (§5). Recommendation: do the fix and a fresh dev set inside P1,
with a pre-stated stop; if that also fails, report "AC-24 not met at P1, carried to P2" (§6).

## 2. Method and its limits

- Replay: `cpg/.venv` python, `CypherReader` over the live `ccg_graphmind-ai-lab` graph (FalkorDB `127.0.0.1:6390`,
  read-only queries), `to_draft(item.repeats[0].draft)` into `resolve_draft(reader, draft, normalize_question(text), ("graphmind-ai-lab",))`.
  Scripts: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/dfault.py` and `dfault2.py`.
- The replay store is **not** the run's throwaway clone: it also holds `falkor-chat/docs/reviews/.../mut/` mutation copies, so QW-19 and QW-27
  show extra candidates there that the run did not see. Those two items passed in the runs and are out of scope; the four qids below
  reproduce the logged `ambiguityKindEmitted` and `nOptions` exactly (ministral QW-7 `unresolved`, lookup `salesperson endpoints.ts`;
  qwen2.5 QW-7 2 options, QW-13 1 option `unknown-name`, QW-11 5 options, QW-39 and QW-42 2 options, QW-3 `no-anchor`; ministral QW-13 and QW-47 `unknown-name`, 0 options).
- Try 0 (`qwen/qwen3-4b-2507`) stored no drafts, so its rows below rest on the verdict only (QW-7 `several-symbols`/2 and QW-13 `unknown-name`/0 match
  the other two models' drafts; QW-11 `wrong-interpretation`, QW-47 `silent-single-interpretation` are consistent with a direction misread, not confirmed).
- **Trigger reading.** "Same way on >= 3 runs" holds literally for QW-13 and QW-7 (an over-ask on all three). QW-11 and QW-47
  miss differently per model; they are treated as cue questions because they share the cause in §3.

## 3. Diagnosis per qid

| qid | What the drafts did (replayed) | Class | Resolver on an ideal draft |
|---|---|---|---|
| **QW-7** `salesperson endpoints.ts postMessage() — who uses it in the UI?` | ministral: file span `salesperson endpoints.ts` (two words joined; verbatim, so it passes the guard, then fails the path lookup) gives `unresolved`. qwen2.5: file span `salesperson/endpoints.ts` is not in the question, the guard drops it, the symbol alone has two real candidates (`falkor-chat/web/app.js::postMessage`, `salesperson/src/api/endpoints.ts::postMessage`) gives `several-symbols`. qwen3: same verdict as qwen2.5. | **Capability** (joined or fabricated span) **plus prompt silence** (nothing says a file span is one path token as written). | `postMessage()` + `endpoints.ts` resolves to `salesperson/src/api/endpoints.ts::postMessage`, `impact`. Symbol alone is a true 2-candidate ambiguity: the resolver is right. |
| **QW-13** ``send_message tool on falkor-chat mcp: if i rename the `re` param who breaks`` | All three emit a **second group for `re`** (the parameter being renamed): symbol `re` (all), plus `server mcp` (ministral) or file `falkor-chat/mcp` (qwen2.5, fabricated, dropped by the guard). Ministral also hints `falkor-chat` as `tool`, which `_group` checks before `symbol`. `re` finds nothing, so the resolved `send_message` rides as an `unknown-name` ask (1 option for qwen2.5, 0 for ministral). | **Prompt silence + resolver policy.** The prompt says "one group per thing the question points at" and never says a parameter is not an anchor. The unfound-name ask (`resolve.py` module docstring, `-p1-ml.md` §4.3) is **by design and pre-registered gold-blind**; it is not a defect against the spec. Hint confusion (`tool`/`server`) is capability. | `send_message` tool alone, or with `falkor-chat` as `server`, resolves `impact`. Adding `re` as `symbol` gives the logged `unknown-name`; as `other` resolves (replayed). |
| **QW-11** `POST /threads/{thread_id}/messages — what calls this route` | ministral: right route, direction `depends-on` (flipped; the prompt's `depends-on` cue contains the word "calls"). qwen2.5: **dropped the verb** (`/threads/{thread_id}/messages`), which matches five contract points (`*`, `GET`, `POST`, `{}` and `{?}` forms) so `several-symbols`/5. qwen3: `wrong-interpretation` (resolved, not gold). | **Prompt silence** (the `impact` cue is only implied by the subject wording of the `depends-on` cue, `test_front_prompt.py:DEPENDS_ON_CUE`; nothing says a route span keeps its verb) **plus capability**. Gold is sound: callers of a route are `impact`, as the record's cue text says. | `POST /threads/{}/messages` + `impact` resolves; the flipped direction resolves too, to the wrong gold. A verb-less route is a genuine 5-way ambiguity. |
| **QW-47** `relações da tool ingest_documents (MCP)` (gold `two-use-cases`) | ministral: `symbol ingest_documents` + `tool MCP`, direction `depends-on`; the `tool` span `MCP` wins in `_group`, finds nothing, `unknown-name`/0 options. qwen2.5: right `tool` + `server MCP`, direction `depends-on`, so a silent single reading. qwen3: silent single. | **Prompt silence**: the only `null` direction cues are "an overview of" and "o panorama de", while "Never answer null when the question implies a direction" pushes a 3-4B model to pick. All three pick `depends-on`. Gold is arguable only in the sense that any ambiguous item is; "relações" naming both directions is a sound label. Hint confusion (`MCP` as `tool`) is capability. | `direction: null` gives `two-use-cases` with both `ingest_documents` readings, matching gold. |

Other dev misses seen in the saved drafts, not part of the trigger, same family: qwen2.5 QW-3 normalised `cypher-mcp/server.py line 564` to
`cypher-mcp/server.py:564` (not verbatim, guard drops it, `no-anchor`); qwen2.5 QW-39/QW-42 dropped the verb; ministral QW-30 read the
component name `decision-models` as a repository `scope` (the resolver refuses a scope: not built). These show span discipline, not a resolver fault.

## 4. Smallest fix per defect

**No resolver change.** A fix in `resolve.py` would be written against a dev item: dropping an unfound `symbol` beside a resolved contract
silently discards a name the asker gave (the module docstring's reason for the ask), and re-ordering `tool` over `symbol` in `_group` has no
rule behind it. `RESOLVER_VERSION` stays `1.0`; `resolverRules` text stays.
**Parked, not P1:** report an unfound name beside a resolved anchor as a notice instead of an ambiguity. It needs an answer-schema field
(`answer-v1.schema.json` has none), so it is a P2 backlog item for `architect`/`data-scientist`, not a defect fix.

**One prompt edit** to `cpg/ccg/front/prompts/interpret.txt` (grammar `grammar/interpret.json` and `SAMPLING` untouched). Five additions, each stated
as a general rule with made-up `fx-*` names, never the dev item's own words (the file's convention):
1. *Route spans keep the HTTP verb exactly as written* (`POST /fx/{id}` is one span; a path without its verb is a different span). Cures qwen2.5 on QW-11/39/42.
2. *A span is one contiguous stretch of the question: never join words, never add `/` or `:` that is not there; a component name beside a file name is not part of the file span.* Cures QW-7 and qwen2.5 QW-3.
3. *A parameter, field or option named only as the thing being changed is not an anchor; anchor the tool, route or function that owns it.* Cures QW-13.
4. *Directions:* add "who or what calls, uses or consumes the anchor" to the `impact` cue, and "relations of / relações de" beside the `null` examples. Cures QW-11 (ministral) and QW-47.
5. *kindHint:* `tool` is the tool's own name, `server` the server's; "MCP" is neither. Cures ministral QW-47/QW-13.
Add at most two `fx-*` examples (one route with a verb and a changed parameter; one relations question with `direction: null`); keep one system plus one user message (the ministral template rejects more).

**Hashes.** Changes: `promptHash` (was `494fb949e16ca53914dd929ae6d0e3a60d99908b587819abcccbfa5d43bb5e15`), every `cacheKey`, `coreTreeHash` (prompt lives under `cpg/ccg/`),
the canary digest (re-recorded at the next dev run; `CANARY_QUESTIONS` unchanged). Unchanged: `grammarHash` `2f7a1edd…`, `samplingHash`, `RESOLVER_VERSION`, `harnessHash` (unless step 4 below), `packSetHash`.
Tests to update: `cpg/tests/test_front_prompt.py` (hash/cue pins; add a pin per new rule), `test_front_interpret.py`, `test_ac24_smoke.py` where they hard-code the prompt hash.
The implementer is `tdd-engineer` (pins first); gate `data-scientist` on cue wording, `analyst` on the diff.

## 5. What this does to the dev set and the record

- **The 14 dev items are spent for entry-check purposes.** The prompt edit is derived from their misses, so a pass on them would show fit, not
  that the interpreter works (`-interpreter-model-ml.md` §3, design-fault trigger). They keep their other uses (the unit tests citing QW-11, fixtures).
  Note `docs/plans/connected-code-graph-eval-protocol-ml.md` §10.2 allows prompt work on dev questions; it is the entry check's validity, not that rule, that spends them.
- **Held-out stays unspent and unread.** The prompt change happens before the record, as the protocol requires ("committed after the last prompt change and before the held-out run", `-p1-ml.md` §4.3).
- **The record `ccg-prereg-ac24-v1` is not appended until a dev run on the final prompt passes.** `prereg-record` builds from the final dev result's `treeStamp` and refuses a dirty tree or a stamp that differs from it, so the order is: prompt commit, then dev run on the new tree, then record.
- **`devAttempts`** carries all four old tries (model, counts, `resultSha256`) and every new one. Add the prompt hash to each entry (`validate_dev_attempts` in `cpg/eval/tools/ac24_smoke.py` takes `{model, counts, unusable, resultSha256}`; one optional `promptHash` key, a `harnessHash` change, hence tdd-engineer, gate analyst). The record states the disclosure: prompt revised after the first dev check failed; first dev set spent.
- **The tries list resets.** `-interpreter-model-ml.md` §3 ("no re-run of a model") bound the old prompt. A new prompt hash is a new configuration; `data-scientist` writes the successor note `docs/plans/connected-code-graph-p1-interpreter-model2-ml.md` (Extends) pre-stating the new tries and stop rule before any run. Suggested: `qwen/qwen3-4b-2507` first (the closest miss), then the unspent `nvidia/nemotron-3-nano-4b`, at most three runs.

**Fresh dev set: who and how** (EP §10, `docs/plans/connected-code-graph-p0-2.md` §3 S52-S53, `cpg/eval/headless/README.md`):
- **Writer:** the headless `QW` role (agent `ccg-question-writer`, pinned Claude build via `cpg/eval/headless/run-writer.sh prepare-bin|prepare|start`), launched by `teco`, under the same §6 writer protocol and the S03 id forms at `EVAL_COMMIT`. It must **not** be shown this document, the saved drafts, the miss list, or the old dev items' failures, so the new items are not shaped to the fix; neither the prompt-edit author nor `architect` writes items.
- **Labels:** two isolated annotators `AN1`/`AN2` label ambiguity independently (EP §10.3); disagreements resolve by the S47 rule.
- **Custodian (GC):** a fresh `qa-engineer` copies the items, assigns `qid`s and writes **a new file `cpg/eval/dev/ac24-r2.jsonl`**. It must not re-run `s47_scripted_steps.py`/`split_manifest.py` (those rewrite `ac24.jsonl` and the held-out split; `heldout-manifest.json` must stay byte-identical). `qa-engineer` checks counts-only that no new item equals or duplicates a held-out/reserve/old-dev `itemId` or question text (EP §12.1's "an item already in dev/held-out/reserve is discarded"), through the custodian's probe-hash check, without the builders reading held-out.
- **Shape:** 12 unambiguous (3 per group x language cell of `code-location`/`contract-point` x `en`/`pt`; at least 2 `depends-on`) plus at least 2 ambiguous covering `two-use-cases` and one other `ambiguityKind`, so the thresholds stay 11/12 and 2/2 with no recomputation (otherwise `data-scientist` recomputes with the `ceil` rule). Run the harness with `--dev-items cpg/eval/dev/ac24-r2.jsonl` (a custodian path is refused by design).
- **Can it be done inside P1?** Yes in method (one QW session, two AN sessions, one GC step, a dev run per try). **Unverified:** that the pinned writer binary (`PIN`, K-P0-12), the custodian root and the `EVAL_COMMIT` checkout still work; `teco` must run `status`/`prepare` first. Gold ids must be valid at `EVAL_COMMIT`, not HEAD.

## 6. Decision for the stakeholder

AC-24's own text: "if none does, a lower local target is set once the first phase has measured it". Options:

| | What | Cost | Risk |
|---|---|---|---|
| **A (recommended)** | Prompt edit (§4), fresh dev set (§5), successor tries note, re-check, then record and held-out run once. Pre-stated stop: if the fresh dev check fails on all (at most three) tries, report **"AC-24 not met at P1, carried to P2"** with all dev counts and this diagnosis. | One writer + two annotator headless sessions, one GC step, a small prompt/test change, up to three LM Studio dev runs. | Another failure is then capability, and is reported as such; the writer infrastructure may need re-pinning. |
| **B** | Stop now: report "not met at P1, carried to P2"; P2 measures once on held-out with the best model and sets the lower local target. | Nothing. | AC-24 stays **unmeasured** at P1 (the 14-item check is a smoke test, not the measurement the requirement names); a known-fixable prompt gap ships to P2. |
| **C** | Re-try with a larger local model on the **unchanged** prompt (no fix, no new dev set; a model swap does not tune on dev). | Three dev runs plus a new `data-scientist` list (the old one was <= 4B on latency grounds). | Does not repair QW-13/QW-7, which the trigger says are not model-level; a pass would hide a prompt gap the held-out run then exposes. |

Recommendation: **A**. The prompt is still free to change (the record is unwritten), the held-out set is untouched, and the extra cost is
bounded by the pre-stated stop. C can run in parallel at no protocol cost but is not a substitute. Whether to spend the writer sessions is the stakeholder's call; I did not assume it.

## 7. Steps if A is chosen

1. `data-scientist`: successor note `-interpreter-model2-ml.md` (new tries, stop rule, fresh-set shape); gate on the cue wording in §4.
2. `tdd-engineer`: red pins in `cpg/tests/test_front_prompt.py` for the five rules, then edit `interpret.txt`; optional `promptHash` in `devAttempts` (`ac24_smoke.py`, `test_ac24_smoke.py`). Suite green with `cpg/` as the working directory. Commit; `analyst` gates.
3. `teco`: author the fresh set per §5 (writer, annotators, GC), commit `cpg/eval/dev/ac24-r2.jsonl` and a counts-only report; confirm `heldout-manifest.json` is unchanged.
4. `qa-engineer`: canary, then the dev runs per the note, `--dev-items cpg/eval/dev/ac24-r2.jsonl`, drafts stored; first pass stops the search.
5. On a pass: `prereg-record` (never hand-built), `teco` appends `ccg-prereg-ac24-v1` with the disclosure, then the held-out run once. On the pre-stated stop: record the "not met" outcome and the diagnosis in `docs/HISTORY.md` and the P2 backlog.

## 8. Risks

- A pass on a 12-item set is a smoke test (the earlier note's binomial estimate stands: about a third of a 0.85-true-rate model passes per try); report it so.
- The five prompt rules lengthen a prompt for a 3-4B model and could regress items that passed (QW-3, QW-30, QW-32); the fresh set measures that, the old dev set cannot.
- Replay used a different store than the run (§2); the four qids reproduce, but a changed store could change an option count, never a verdict class.
- The qwen3 diagnosis rests on verdicts, not drafts; step 4 stores drafts for every new try.
