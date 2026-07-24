# Review — K-027 slice A: parse-layer robustness (implementation gate)

> Reviewer: `analyst` · Date: 2026-07-24 · Type: static code review + executed behavioural probing
> **Baseline reviewed:** the uncommitted working-tree diff vs `HEAD` (`2ee6eba`), restricted to
> `git diff -- falkor-chat/server falkor-chat/docs/BACKLOG.md falkor-chat/docs/HISTORY.md` —
> `server/falkorchat/llm.py`, `server/falkorchat/app.py`, `server/tests/test_llm.py`,
> `server/tests/test_app.py`, `docs/BACKLOG.md`, `docs/HISTORY.md`. Read in full, plus the surrounding
> consumers (`executor._run_agent_node`/`_handle_tool_call`, `guards._coerce_verdict`, `tools.py`
> registry, `responder.py`, the shipped `seed_workflows.sh` prompts).
> **Specs judged against:** `docs/BACKLOG.md` **K-027 item 1** + the "Addendum from the K-025 QA pass
> (2026-07-21)" (pre-change text via `git show HEAD:falkor-chat/docs/BACKLOG.md`),
> `docs/archive/test-reports/m3-workflow-engine-report.md` §3.9 / AC-2a / **DEF-K027-B**, `AGENTS.md`.
> **Out of scope, not faulted:** the parallel unit's paths (`docs/plans/workflow-def-structure-read.md`,
> `docs/reviews/workflow-def-structure-read.md`, `docs/plans/m3-followups-coordination.md`,
> `claude/*/kaizen/inbox.md`) and the concurrent K-031/K-033/K-034 churn — the `K-034` block at
> `BACKLOG.md:644+` is another agent's filing, not this change. K-027 items 2–5 are out of scope.

## Verdict: **needs changes**

**Finding counts: 1 blocker · 2 major · 7 minor · 3 nit.**

The slice is the right move at the right time, cleanly layered, and its scope discipline is exact — I
verified every claim rather than taking it on trust. The implementer's own account is accurate on every
point I could test: precedence is genuinely untouched, the removed `"non-object judge output"` branch is
genuinely unreachable, `_scan_parenthesized`'s string/escape handling is correct, and the red/green test
arithmetic (11 red / 8 green before) is exactly right.

What fails the gate is **the direction the slice says it closed**. The documented safety property — *"a
call this layer invents is worse than one it misses"*, *"tolerance runs in the safe direction only"* —
is asserted in three places (`app.py:351-353`, `HISTORY.md`, `BACKLOG.md` K-027 item 1) and is **false
in two concrete, reachable ways I reproduced**. Both are narrow, and both have a surgical fix that keeps
all 19 new tests green. The blocker is the judge one, because it makes the served guard advance where
`HEAD` correctly suspended — in exactly the metric K-027 item 3 exists to gate.

### Verification I performed in this pass (evidence, not inference)

| Claim | Result |
|---|---|
| Test collection | `pytest --collect-only -q` → **552/553 collected, 1 deselected** — matches the claimed total exactly |
| Touched test files pass | `pytest tests/test_llm.py tests/test_app.py -q` → **45 passed** |
| `ruff check .` | `All checks passed!` ✅ |
| `_drive_loop` SHA-lock | Re-ran the `docs/archive/plans/m3-process-flow.md` §3.1 awk command → **`71055f756280`** ✅ |
| No Cypher/DDL/script change | `git status` + `git diff --stat`: only the six declared files; `executor.py`, `guards.py`, `repository.py`, `scripts/`, `docs/QUERIES.md` **untouched** ✅ |
| Precedence unchanged | `git diff` on `_parse_chat_message` is **docstring-only**; `llm.py:142-152` is byte-identical to `HEAD` ✅ |
| `"non-object judge output"` unreachable | `extract_json_object` can only return `dict \| None` — every non-`None` return is behind `isinstance(parsed, dict)` (`llm.py:364-365`). Branch is dead ✅ |
| Rationale string has no consumer | `_coerce_verdict` (`guards.py:453-471`) reads `decision` and passes `rationale` through opaquely; `_rationale_contradicts` only runs when `decision is True`, which this string never accompanies; `executor._trace_step` (`:844-845`) writes it to a debug payload. Grepped `falkorchat/`, `tests/`, `web/`, `scripts/` — **no literal match on either string** ✅ |
| Behavioural envelope | ~50 hand-built inputs driven through `_parse_content_tool_calls`, `_parse_chat_message`, `extract_json_object` and `_build_llm_judge`+`_coerce_verdict`. All quoted results below are verbatim probe output |

**"11 red / 8 green before the fix"** — verified by construction, not by execution: all 8 `test_llm`
recovery pins exercise `_parse_bare_call_syntax`, which does not exist at `HEAD`; the 3 `test_app`
recovery pins assert `decision is True` on replies `HEAD`'s bare `json.loads` rejects (fenced ×2,
prose-wrapped ×1) ⇒ exactly 11 red. The other 6 + 2 take an identical path at `HEAD`. Self-consistent.

### ⚠️ Environment state the coordinator needs to know — I did **not** run the full suite

A plain `pytest` wipes the global `reference` graph via the `wf_repo` fixture, and the documented
re-seed is *create-only*: if a workflow test leaves a def behind under a production `key@version`,
`seed_workflows.sh` prints `already present — no-op` and the production def is **not** restored
(`AGENTS.md`). For a change that is 100 % offline string parsing, that shared-state risk buys nothing,
so I ran only the two touched files. **Consequence: the "552 passed" claim is corroborated by the
collection count and by the two touched files passing, but is not verified by a full green run.**

Separately, and **unrelated to this diff** — I checked the live instance while assessing risk:

```
GRAPH.QUERY reference "MATCH (n) RETURN labels(n)[0], count(*)"   → 0 rows (graph is EMPTY)
GRAPH.QUERY ws:acme  "MATCH (s:WorkflowDefSnapshot) RETURN …"     → triage@v1, access-request@v1
```

`reference` currently holds **no** `WorkflowDef` at all, while `ws:acme` still holds both snapshots —
the documented post-wipe state, so `@mention`-to-start will silently no-op until someone re-seeds. I
did **not** re-seed: another agent is concurrently working on publish/materialize semantics (K-031 /
K-034) and a create-only re-publish into that state is a shared-state write with a known split-brain
hazard, not a routine cleanup. **Route `./scripts/seed_workflows.sh acme` to whoever owns the estate,
after the concurrent unit lands.**

---

## Blocker

### B-1 · The tolerant judge parse reads a **quoted/hypothetical** JSON verdict out of prose — a new false-advance path, in the direction the docs promise is closed

**Evidence:** `app.py:362-365` + `llm.py:354-357` (the "first `{` … last `}`" candidate).

Reproduced end-to-end through the wired judge and `guards._coerce_verdict`:

```
reply: 'If the user had named the service I would answer {"decision": true, "rationale": "named"}
        but they did not, so I answer false.'

HEAD  → {"decision": False, "rationale": "unparseable judge output"}          (safe)
NEW   → {"decision": True,  "rationale": "named"}
        _coerce_verdict → GuardVerdict(decision=True, rationale='named')      ← ADVANCES
```

A second shape, also a strict regression: `'[{"decision": true, "rationale": "…"}]'` — at `HEAD` a
non-dict ⇒ `decision=False`; now the inner object is lifted out of the array and **advances**.

The `_coerce_verdict` backstop does **not** catch either. The *quoted* rationale ("named") is clean, so
`_rationale_contradicts` (`guards.py:474-489`) finds no cue; the judge's actual conclusion ("I answer
false") is discarded before `_coerce_verdict` ever sees it. When the lifted fragment carries no
`rationale` at all, the run advances with an **empty** rationale and the `guard_judgment` trace records
`… -> True: ` — a false advance that is also undiagnosable after the fact. The prose-extraction path
strips exactly the evidence the safety backstop needs.

**Why it matters.** This is a strict regression against `HEAD` in the safety-critical direction, on the
**served** guard, for exactly the metric that gates K-027 item 3 (D9: false-advance ≤ 10 %). A small
model narrating a counterexample or restating the schema is a common shape. I confirmed the parse is
*order-blind, not conservative*: with two objects present the `{`…`}` span is invalid JSON and the
whole verdict is lost (`unparseable`), and a schema-echo-then-verdict reply also degrades to
`unparseable` — so whichever object the span happens to capture wins, and the model's **last** (i.e.
concluding) object is not preferred.

Compounding it, `app.py:351-353` states the opposite as a design guarantee:

> *"Tolerance runs in the safe direction only: a reply with no JSON object at all still resolves to
> `decision=False`, so prose is never read as a verdict."*

True only for the case it names. `HISTORY.md` and `BACKLOG.md` K-027 item 1 repeat it. A future owner
calibrating the judge (item 3) will read that as a proven invariant.

**Suggested improvement (both halves):**
1. **Behaviour** — make the judge path conservative rather than best-effort. Cheapest correct form:
   accept the prose-embedded candidate **only when it contains a `decision` key and is the last
   balanced JSON object in the reply**; or, stricter and sufficient for the D13 Ministral shape that
   motivated item 1, accept only a reply that is *entirely* one JSON object after fence-stripping.
   The D13 evidence was about **fences**, not prose — the prose fallback is unforced here. All three
   new judge pins (`test_app.py:249,260,266`) survive the stricter rule: the prose-wrapped pin has
   exactly one object and it is last.
2. **Docs** — correct the claim at `app.py:351-353`, in `HISTORY.md` and in `BACKLOG.md` K-027 item 1
   to what is true: *prose alone* is never a verdict, but *prose containing a JSON object* is. **This
   half is non-negotiable even if (1) is deferred to item 3** — an asserted-but-false safety invariant
   is worse than a known gap.
3. Add both reproductions above as negative pins. No existing test covers a JSON object embedded in
   *judge* prose that the judge does not endorse, nor the array-wrapped verdict — the two inputs whose
   behaviour actually changed are precisely the two that are unpinned (see "test quality", below).

---

## Major

### M-1 · Bare-call recovery fires on an **illustrative or explicitly disclaimed** call inside a multi-line message, and silently discards the model's real answer

**Evidence:** `llm.py:233-235` (the stated guard), `llm.py:261-277` (the loop), `llm.py:150`
(`text=None` on recovery).

The comment at `llm.py:233-235` states the safety rationale:

> *"MULTILINE `^` is the false-positive guard — prose about a tool ("I will call post_message(...)")
> never starts its line with the identifier, so it is never mistaken for a call."*

That holds only for **single-line** prose. Models routinely put the example on its own line. Verbatim
probe output against the delivered code:

| Content | Result |
|---|---|
| `` The API is simple.\n``post_message({"text": "x"})``\nThat is all you need to know. `` | **dispatches** `post_message` |
| `` You should not do this:\n```python\npost_message({"text": "hi"})\n```\nInstead, ask first. `` | **dispatches** `post_message` |
| `` I will NOT do this:\npost_message({"text": "hi"})\nbecause it is premature. `` | **dispatches** `post_message` |
| `` You wrote:\n\n> here is my code\n\npost_message({"text": "hello world"})\n\nI cannot run that. `` | **dispatches** `post_message` |
| A tool catalogue listing three calls, one per line | **dispatches all three**, incl. `human_handoff` |
| `` Example:\n\n    post_message({"text": "hi"}) `` (4-space markdown code block) | **dispatches** `post_message` |

Note the code fence is irrelevant to the guard: `MULTILINE ^` matches inside a fence of any language
tag, so `_strip_code_fence` is not doing the containment work a reader might assume.

**Two harms compound.** (a) The illustrative payload is **dispatched** — for `post_message` that is a
real, irreversible write to a real thread: the exact "invented call" this slice claims to have closed.
(b) `_parse_chat_message` sets `text=None` on a recovered call (`llm.py:150`), so the model's actual
prose is thrown away: it never reaches `last_text` in `executor._run_agent_node` (`executor.py:586-587`),
`_assistant_turn` feeds empty content back, and the node iterates instead of terminating. An `answer`
node that explains its own tooling **loses its answer and posts the example**. That is a behavioural
*regression* on a path that previously worked, not merely a missed recovery.

**The test estate does not cover this.** All 6 negative pins (`test_llm.py:312,323,332,340,354,362`)
are **single-line** cases. There is no multi-line negative pin at all, so the gap is invisible to the
suite — while `test_chat_recovers_a_bare_call_on_its_own_line_after_prose` (`:263`) pins the *opposite*
for the prose-before case, a widening the DEF-K027-B report gives no evidence for (the observed output
was the call and nothing else).

**Why major and not a blocker** — verified mitigations, all real:
- `executor._handle_tool_call` rejects any name outside the node's granted set (`executor.py:626`) and
  any call missing a required argument (`executor.py:630`), so an invented call can only reach a tool
  the node was *already* allowed to call.
- `LMStudioLLM.chat` is called from exactly one place, `executor.py:584`. The M2 `@mention` responder
  uses `complete()` and is untouched by this diff.
- None of the three shipped `triage@v1` prompts (`scripts/seed_workflows.sh:167,182,207`) contains an
  example call on its own line, so the shipped def does not actively provoke the shape.

**Suggested improvement — verified against the full corpus.** Require the **last accepted call to be
the final non-whitespace content** of the fence-stripped message (reject the whole recovery if any
non-blank text follows it). Run against all 8 positive pins and the false positives above:

```
T1 observed          CALL → CALL      FP-A quoted fenced block   CALL → text
T2 json fence        CALL → CALL      FP-A2 example block        CALL → text
T3 plain fence       CALL → CALL      FP-B  call in middle       CALL → text
T4 prose then call   CALL → CALL      FP-C  disclaimed intent    CALL → text
T5 multiline json    CALL → CALL
T6 no args           CALL → CALL
T7 empty obj         CALL → CALL
T8 two calls         CALL → CALL
```

**All 8 positive pins survive; every false positive closes.** It is a ~4-line change in
`_parse_bare_call_syntax` and it preserves `test_chat_recovers_a_bare_call_on_its_own_line_after_prose`,
because there the call *is* last. Residual after the fix: a message whose final line is an illustrative
call still fires — bounded, and worth one sentence in the docstring. If the owner prefers to keep the
current width instead, then `llm.py:233-235` **must** be rewritten to state the real rule and the six
rows above **must** be pinned as accepted behaviour, so the next widening is a deliberate act.

### M-2 · The shadowing follow-up is correctly diagnosed but under-weighted as "small"

**Evidence:** `llm.py:188-197` (JSON probe first), `BACKLOG.md` "Discovered during slice A" bullet 1.

Confirmed exactly as the implementer describes:

```
create_user({"name": "bob"})       → ToolCall(name='bob',    arguments={})
run_tool({"action": "delete"})     → ToolCall(name='delete', arguments={})
x({"tool": "y", "args": {"a": 1}}) → ToolCall(name='y',      arguments={"a": 1})
```

I verified the "harmless today" premise: the three registered tools take `text`/`mentions`
(`tools.py:193-199`), `query` (`tools.py:278-284`) and `reason` (`tools.py:338-341`) — no
`name`/`action`/`tool`. **The premise holds, and deferring the fix is the right call for a slice whose
whole point is not to widen precedence.** The problem is the *shape* of the failure, not its
likelihood: it does not fail loudly, it manufactures a call named after a **user-supplied value** and
silently **drops the arguments**. Add a tool `create_user(name)` and the AC-6 check (`executor.py:626`)
rejects `'bob'` as ungranted, burns an iteration, and tells the model its own argument is not a tool —
an almost undebuggable trace. A bullet filed under "candidate follow-ups, **small**" will not surface at
the moment someone adds that tool.

Two corrections to the note itself: it omits the argument-dropping, and it justifies the deferral as
*"means reordering the content fallback; out of slice-A scope"*. Reordering is in fact safe and ~2
lines — a JSON envelope (`{"name": …}`) never starts a line with `identifier(`, so running the bare
probe first returns `[]` on every existing JSON case and falls through unchanged. The deferral is still
right; the *reason* should be "no evidence it bites today", not "expensive".

**Suggested improvement:** keep the deferral, but (a) raise it from a bullet to a named, numbered
K-item (or an explicit K-029 sub-item) so it is schedulable, and (b) add a one-line comment at the
probe-order site `llm.py:188` — *"argument keys `name`/`action`/`tool` shadow the call name here; see
K-0xx before registering a tool with such a parameter"* — because that is where the next author looks.

---

## Minor

### m-1 · Undisclosed behaviour change: `{"tool_calls": [], …}` envelopes now fall through

`llm.py:190-193` — the new `if calls:` guard replaced an unconditional `return`, so an empty or
all-nameless envelope now falls through to `_normalize_tool_call` and then to the bare probe:

```
'{"tool_calls": [], "name": "graphrag_retrieve", "arguments": {"query": "q"}}'
  HEAD → []            NEW → ToolCall(graphrag_retrieve, {"query": "q"})
```

Arguably an improvement, but it is a **third** behaviour change and `HISTORY.md`'s "Behaviour change a
caller could notice" section lists only two; the docstring at `llm.py:181` calls the JSON branch
"(unchanged)", which it is not. `{"tool_calls": []}` alone still → text (confirmed).
**Suggested:** drop the word "unchanged", add the case to the HISTORY list, and pin it either way.

### m-2 · `consumed` does not advance on a **rejected** scan, so a nested identifier can be recovered from inside a discarded expression

`llm.py:276` — `consumed = after` runs only on the accept path:

```
'outer(text="\nfoo({"a": 1})\n")'  → ToolCall(foo, {"a": 1})
```

Contrived, but the cursor's stated job ("inside an already-recovered call's arguments", `llm.py:263`)
is narrower than the job it should do. **Suggested:** set `consumed = after` immediately after a
successful `_scan_parenthesized`, before the accept/reject decision — a one-line move that makes the
comment true. (The current placement is *deliberately* safe in the other direction — a failed scan must
not swallow a later real call, which I verified it doesn't — so keep that property when moving it.)

### m-3 · Two of the four "discovered, deliberately not fixed" backlog bullets are inaccurate

`BACKLOG.md` "Discovered during slice A", bullets 2–3:

- *"Fences with a non-`json` language tag (```` ```tool_code ````) are not stripped"* — literally true
  of the tag text, but the stated consequence does not hold. Both consumers recover such content
  anyway: ```` ```tool_code\npost_message({"text":"hi"})\n``` ```` → **recovered as a call**, and
  ```` ```tool_code\n{"decision": true,…}\n``` ```` → **recovered as a verdict**. The backticks *are*
  stripped; the leftover tag line is harmless to the line-anchored regex and to the first-`{`/last-`}`
  span. As written, the bullet sends the next author to fix a non-issue.
- *"Namespaced call names (`functions.post_message({…})`) are not recognised"* — **confirmed accurate**.

**Suggested:** delete or rewrite bullet 2; keep bullets 1, 3 and 4.

### m-4 · "This would have converted the observed run" is asserted, not established

`HISTORY.md` §1 and `BACKLOG.md` addendum (b), vs `m3-workflow-engine-report.md:260-273`. The report's
DEF-K027-B quotation is **line-wrapped and tail-elided**; the fixture (`test_llm.py:224-229`)
reconstructs it as one line. Probed with the wrap kept as a real newline:

```
'post_message({"text": "…a broken deploy on the\nbilling service. Could you please…"})'  →  []
```

— not recovered, because a raw newline inside a JSON string fails `json.loads`. The same diff records
this as a known non-fix (bullet 4), ~40 lines from the claim that the fix *would have converted the
run*. Exactly one of those is true, and which one depends on whether the wrap at report line 265 is the
model's newline or the report author's.

**Suggested:** (a) relabel the fixture *reconstructed from the report, de-wrapped* — the word
"**Verbatim**" at `test_llm.py:221` is what makes this invisible; (b) close it cheaply instead of
arguing it: `json.loads(inner, strict=False)` in `_parse_call_arguments` (`llm.py:321`) accepts control
characters inside strings and recovers the wrapped shape with **no** widening of *which* text counts as
a call — then pin the wrapped string; (c) until then, soften the claim in both docs. This matters
because item 5's re-probe is being unblocked on the strength of it.

### m-5 · Duplicate identical bare calls are both dispatched

`llm.py:273-276`; `test_llm.py:296` pins multi-call recovery as desired.

```
'post_message({"text": "a"})\npost_message({"text": "a"})' → two ToolCalls, both dispatched
```

Multi-call recovery is a deliberate, defensible choice; the *identical-repeat* sub-case is not, and
self-repetition is a known 4B failure mode. Result: two identical messages in the thread. `HEAD`'s
content path could only ever yield one call, so this is new. **Suggested:** de-duplicate on
`(name, arguments)` inside `_parse_bare_call_syntax`, or cap the bare path at one call — and pin it.

### m-6 · The `AGENTS.md` skip is **correct on the letter**, but the change creates the kind of invariant that file exists to carry

I verified the implementer's premise: `AGENTS.md` and `docs/DESIGN.md` contain **no** mention of
`_parse_content_tool_calls`, `extract_json_object`, `_build_llm_judge` or the parse layer (the only
`llm.py` hit is the unrelated ruff note at `AGENTS.md:197`). Nothing there is now false, and I would not
have asked for an edit on those grounds alone. **But** the change promotes `_extract_json_object` to
public `extract_json_object` and makes it a **shared seam across two modules** (`llm.py:337` ←
`app.py:30`) with a stated tolerance contract both consumers now depend on — and it introduces a
heuristic that can **dispatch a tool from model prose**. That is the same class as the invariants
`AGENTS.md` already carries ("`role` is derived, never trusted", "the empty-`UNWIND` guard is
load-bearing"), and K-027 item 5 plans to re-measure model behaviour *against this layer*.
**Suggested (after B-1 is settled, so the sentence is true when written):** one bullet in the M1-server
/ K-014 live-agent-loop list stating the tolerance contract of `llm.extract_json_object`, that both the
tool-call and judge paths ride it, and that the false-positive direction is the binding constraint.

### m-7 · K-027 is still marked `🔵 proposed` after a delivered slice

`BACKLOG.md:315`. The legend at `BACKLOG.md:6` offers `🟡 in-progress`, which is what an item with one
delivered slice and four open items is. Cosmetic, but the milestone map is read by agents deciding what
is safe to pick up.

---

## Nits

- **n-1 ·** `llm.py:249` — *"Arguments must be an empty list or a single JSON object"*: "empty list"
  reads as *a JSON array* `[]`, which is in fact **rejected** (`post_message([{"text": "hi"}])` → text,
  confirmed). Say "an empty argument list".
- **n-2 ·** `llm.py:247` / `:234-235` — "the identifier starts a line" understates the regex: `^[ \t]*`
  allows leading whitespace, so an *indented* call is recovered and `post_message ({…})` (space before
  the paren) is too. Probably desirable; just say so. The neighbouring rejections are worth a sentence
  as well, because they are not predictable from the stated rule: markdown bullets `- name(…)`,
  numbered `1. name(…)`, bold `**name(…)**` and a trailing `;` all correctly stay text — I checked each.
- **n-3 ·** `app.py:313` — the second function-local `import json as _json` (in `_render_judge_user`)
  survives; `HISTORY.md` correctly acknowledges this as "half of n-1". Fine to leave, but it is a
  two-line cleanup now that its sibling is gone.
- *(Not a finding, noted for the owner:)* `_scan_parenthesized` is O(n) per candidate line and returns
  `None` only after scanning to end-of-text, so dense content is O(n²) — measured **0.64 s for 3 000
  line-start `a(` in ~6 KB**. Model output is token-bounded (~16 KB worst case ⇒ low seconds), so not
  worth code today; a `if len(calls) >= N: break` would remove the tail if it ever matters.

---

## What's solid

- **Precedence is genuinely untouched.** `_parse_chat_message` (`llm.py:135-152`) differs from `HEAD`
  by docstring only; native-over-content is preserved and pinned by `test_llm.py:362`.
- **The removed `"non-object judge output"` branch is safely dead**, and the rationale-string rename
  cannot break a consumer — verified three ways (type-level, `_coerce_verdict` read, whole-component
  grep). Correctly declared in `HISTORY.md`.
- **`_scan_parenthesized` survived everything I threw at it**: parens inside JSON strings, `\"` escapes,
  `\\`-then-quote, unicode escapes, nested objects/arrays, duplicate keys, tabs, CRLF, a stray
  unbalanced quote before a real call, and a failed first match that must not advance `consumed` (it
  doesn't — the right design). No crash, no swallow, no mis-parse. Escape state is properly scoped to
  `in_string`; unbalanced input yields `None` and stays text (pinned, `test_llm.py:354`). Single-quoted
  strings are not string-aware, but that failure lands on the *safe* side — verified.
- **The narrow-recognition rules that are claimed do hold**, individually: inline mid-sentence calls,
  trailing prose on the call's line, `name(text="hi")`, `name("hi")`, `name(...)`, JSON *array*
  arguments and a trailing `;` all stay text. The negative pins are real pins, not tautologies — they
  assert `result.text == prose` (identity), not merely `not is_tool_call`.
- **Tests pin behaviour, not implementation.** Every new test drives the public `llm.chat(...)` /
  `_build_llm_judge(...)` seam through an injected transport/stub — none reaches into
  `_parse_bare_call_syntax` or `_scan_parenthesized`, so they would survive a scanner rewrite and still
  catch a recovery-contract regression. The DEF-K027-B fixture is a named module constant with its
  provenance documented in place. The gap is **coverage direction**, not test craft: B-1 and M-1 are
  both cases where the negative pins guard the half that never changed.
- **Scope discipline is exact.** No Cypher, no DDL, no script, no schema; `executor.py` and `guards.py`
  byte-identical; `_drive_loop` SHA re-verified `71055f756280`; K-027 items 2–5 and the carried
  `guards.py` m-1/m-2/m-3 findings explicitly left open; K-027 **not** closed.
- **The `BACKLOG.md` edit is well-targeted**: item 1 ✅ and addendum **(b)** ✅, with (a)'s engine half
  named as "the *only* remaining part", item 5's precondition marked unblocked, and four deliberate
  non-fixes recorded rather than left to be rediscovered. That disclosure is why m-2/m-3/m-4 are
  findings about *wording*, not about concealment. `HISTORY.md` is detailed, dated, honest about the
  behaviour changes it knew about, and correctly reasons rule 6 (RAM) to a no-op.

---

## Open questions (for the coordinator, not fixes)

1. **B-1 scope.** Does the behaviour fix land in *this* slice, or does the doc correction land now and
   the behaviour roll into K-027 item 3 (calibration), which owns the false-advance metric anyway?
   Either is defensible; shipping the *claim* uncorrected is not. My recommendation: do both here — it
   is a smaller change than the calibration harness will be, and item 3's measurements are only
   meaningful against a settled parse.
2. **M-1's trade-off is a product call.** Tightening to "the call must be last" costs the ability to
   recover a call the model buries mid-message. My read: the executor's re-prompt loop makes a *missed*
   call cheap (the node iterates) and an *invented* `post_message` expensive (an irreversible thread
   write), so tightening is right — but the owner of the triage flow should confirm.
3. **Was the line break at `m3-workflow-engine-report.md:265` the model's or the report author's?** It
   decides whether slice A converts the observed DEF-K027-B run (m-4). Only the QA engineer who
   captured it, or a surviving raw `StepRun.output`, can answer.
4. **Should the bare path be gated on the granted tool set?** The layering note at `llm.py:124-126`
   says name validation belongs to the agent loop, and that separation is worth keeping. But the
   bare-call probe is *heuristic recognition*, not wire-shape parsing — passing the granted names down
   purely as a recognition filter would close M-2 and most of M-1 in one stroke. A real layering
   decision, flagged rather than asserted.

## Routing

- **B-1, M-1, m-1, m-2, m-5, n-1, n-2, n-3** → `tdd-engineer` (each finding names its input strings;
  the pins are the deliverable).
- **B-1 (1) and open questions 2 & 4** → `architect` if the narrowings are contested — they change
  judge and recovery behaviour, which K-027 item 3 will calibrate against.
- **M-2, m-3, m-4, m-6, m-7** → docs/backlog, whoever lands the next K-027 slice.
- **Re-seed `reference`** (see the environment note above) → estate owner, after the concurrent
  K-031/K-034 unit lands. Not a finding against this diff.

---
---

# Re-gate — K-027 slice A fix pass (round 2)

> Reviewer: `analyst` · Date: 2026-07-24 · Type: static re-review + executed behavioural probing
> **Baseline re-reviewed:** the uncommitted working tree, restricted to the K-027 files
> (`server/falkorchat/llm.py`, `server/falkorchat/app.py`, `server/tests/test_llm.py`,
> `server/tests/test_app.py`, `AGENTS.md`, `docs/BACKLOG.md`, `docs/HISTORY.md`) against the 13
> dispositions the implementer claims. **Explicitly out of scope and not faulted:** the concurrent
> K-031 unit (`api.py`, `schemas.py`, `services.py`, `repository.py`, `test_api.py`,
> `test_repository.py`, `test_services.py`, `scripts/verify_workflows.sh`, `DESIGN.md`,
> `QUERIES.md`, the K-031/K-033/K-034 backlog + HISTORY blocks). K-027 items 2–5 remain out of scope.
> **Method note:** the pre-fix working tree is not in git, so "unchanged" claims about round-1 code
> are judged by *line-number invariance against the line numbers this review quoted in round 1*
> plus body inspection — stated as such below, never as a diff.

## Re-gate verdict: **needs changes** (one major; the blocker is closed)

**Round-2 counts: 0 blocker · 1 major · 4 minor · 2 nit.** Twelve of the thirteen dispositions land,
most of them well; **B-1 is genuinely closed** and I reproduced it. What re-opens the gate is a
single finding, **N-1**: the M-1 rule anchors only the *last* recovered call, so every earlier
own-line call-shaped expression in the same message still dispatches — and three documents now
assert the opposite as a closed safety property. The behaviour half is a verified ~3-line change;
the **doc half is mandatory** under the same rule round 1 applied to B-1 ("an asserted-but-false
safety invariant is worse than a known gap"). Nothing else here blocks.

### Verification I performed in this pass

| Check | Result |
|---|---|
| Touched test files | `pytest tests/test_llm.py tests/test_app.py -q` → **56 passed** (45 → 56 = the claimed +11) ✅ |
| `ruff check` on the four owned files | `All checks passed!` ✅ |
| Full suite | **Not run, by instruction** — a plain `pytest` wipes `reference`, and a concurrent agent is live against the same FalkorDB. The "595 passed" claim is **not** corroborated here. |
| `executor.py` / `guards.py` / `scripts/` byte-untouched | `git status`: neither file appears as modified; the only new script is K-031's `verify_workflows.sh` ✅ |
| B-1 reproduction | Re-ran the exact round-1 input through `_build_llm_judge` → `guards._coerce_verdict` ✅ (below) |
| M-1 corpus | All six round-1 dispatching shapes + 14 new boundary shapes driven through `_parse_chat_message` ✅ |
| Positive-pin survival | Round-1 line numbers 221/224–229/**263**/**296**/312/323/332/340/354/362 (`test_llm.py`) and 249/260/266 (`test_app.py`) all still land on the same construct; the new blocks are pure insertions at `test_llm.py:362` and `:426` and `test_app.py:290`. Bodies inspected. ✅ |
| Proposed N-1 fix | Prototyped read-only in a scratchpad and run over all 8 positive pins + the FP corpus (source untouched) |

---

## Per-finding disposition

| # | Round-1 finding | Status | Evidence |
|---|---|---|---|
| **B-1** | quoted/hypothetical judge verdict advances | **CLOSED** ✅ | Both named shapes now suspend; seam split at `llm.py:416` / `app.py:370`; 4 new pins (`test_app.py:290,303,312,325`) |
| **M-1** | illustrative call dispatches | **PARTIAL — re-opened as N-1** ⚠️ | 5 of 6 shapes close *in isolation*; all 6 re-open when a genuine call ends the message (`llm.py:310`) |
| **M-2** | shadowing under-weighted | **CLOSED** ✅ | `K-035` at `BACKLOG.md:855-899` (owner, three ranked remedies, test strategy, both round-1 corrections folded in); tripwire at `llm.py:188-195` — the exact site asked for |
| **m-1** | `{"tool_calls": []}` fall-through undisclosed | **CLOSED in substance, one word left** ⚠️ | HISTORY behaviour list now has 3 items; both directions pinned (`test_llm.py:429,440`). But `llm.py:181` **still says "(unchanged)"** — the specific word the finding asked to drop → **N-5** |
| **m-2** | `consumed` cursor | **CLOSED** ✅ | `llm.py:295` advances after a *successful* scan; a failed scan still does not advance (`:290-291`), so the safe property round 1 asked to preserve is preserved; pinned `test_llm.py:407` |
| **m-3** | two inaccurate backlog bullets | **CLOSED** ✅ | Bullet struck through with the reason (`BACKLOG.md:458-463`); I re-verified both consumers recover ```` ```tool_code ```` content (call **and** verdict) under the *delivered* code |
| **m-4** | "would have converted the observed run" | **CLOSED in substance, wording residue** ⚠️ | Claim softened + evidence of the search recorded (`BACKLOG.md:409-414`); HISTORY no longer asserts conversion. Residue: "as recorded" is literally false → **N-4**; the fixture comment still leads with "Verbatim" (`test_llm.py:221`) though it now discloses the de-wrap in the next two lines |
| **m-5** | duplicate calls both dispatched | **CLOSED** ✅ | `llm.py:302-304` fingerprints `(name, arguments)`; pinned `test_llm.py:417`; verified ×2 and ×3 repeats collapse, distinct calls preserved |
| **m-6** | AGENTS.md tolerance contract | **CLOSED, with one false sentence** ⚠️ | `AGENTS.md:225-237` is the right bullet in the right list; its last sentence (`:236`) is the N-1 over-claim |
| **m-7** | K-027 still `🔵 proposed` | **NOT ADDRESSED** ❌ | `BACKLOG.md:315` still reads `🔵 proposed` while item 1 and addendum (b) are ✅. Absent from the claimed dispositions entirely → **N-3** |
| **n-1** | "empty list" wording | **CLOSED** ✅ | `llm.py:263` — "Its argument is empty or a single JSON **object**" |
| **n-2** | `^[ \t]*` understated | **CLOSED** ✅ (one shape still unstated) | `llm.py:242-246` names the indentation allowance and the list-marker rejections. `post_message ({…})` (space before paren) is still recovered and still unstated → **N-6** |
| **n-3** | second function-local `import json` | **CLOSED** ✅ | Both copies gone; module-level `import json` at `app.py:18` |
| *(O(n²) note)* | rejected | **Accepted as rejected** ✅ | Round 1 filed it as "not a finding"; token-bounded input is the round-1 reasoning verbatim |

---

## New findings

### N-1 · Major — the "final non-whitespace content" rule anchors **only the last call**; every earlier own-line call-shaped expression is still dispatched, and three docs say otherwise

**Evidence:** `llm.py:286-312` — the loop appends *every* accepted candidate (`:306-308`); the rule-3
guard at `:310` tests `text[end_of_last_call:]`, i.e. only what follows the **last** one. Nothing
constrains what sits *between* accepted calls.

Verbatim probe output against the delivered code (driven through `_parse_chat_message`):

| Content | Result |
|---|---|
| `` I considered handing off:\n```\nhuman_handoff()\n```\nBut instead I will ask:\npost_message({"text": "Which service?"}) `` | **dispatches both** `human_handoff` **and** `post_message` |
| `` I will NOT do this:\nhuman_handoff()\nInstead:\npost_message({"text": "q"}) `` | **dispatches both** |
| `` You wrote:\n\n    post_message({"text": "hello"})\n\nI will do it:\npost_message({"text": "done"}) `` | **two `post_message` dispatches** — the echoed user text is written to the thread |
| `` Available tools:\npost_message({…})\ngraphrag_retrieve({…})\nhuman_handoff() `` (the round-1 catalogue, minus its trailing sentence) | **dispatches all three**, incl. `human_handoff` |

`executor._run_agent_node` dispatches **every** element of `result.tool_calls` (`executor.py:595`),
so each of these is a real, irreversible write for `post_message`. This is precisely the harm M-1
named, reachable through a shape at least as natural as the ones that were closed: "here is what I
considered / here is an example / **here is what I actually do**" ends with a genuine call, and the
recovery exists exactly to serve that trailing call.

**Why it matters more than its width.** Three documents state the closed property in a form that is
**false as written**, and the suite cannot catch it:

- `AGENTS.md:236` — *"a recovered `name({json})` must be the **final non-whitespace content** of the
  message"*. False for every call but the last. This is the invariants file; a reader takes away
  "an illustrative call cannot be dispatched".
- `HISTORY.md:137-139` — *"Deliberately rejected as text: … trailing prose on the call's line **or on
  any later line**"*. Falsified by rows 1–3 above.
- `BACKLOG.md:402-408` — same rule, same omission; the "Residual, accepted" sentence names only the
  final-line case.
- `llm.py:266-274` — rule 3 is *literally* true ("The **last** accepted call is…"), but the sentence
  that follows ("Rule 3 is what makes 'the expression owns its lines' true rather than aspirational")
  and the "Residual, by design" paragraph both omit this family.
- All three new M-1 pins (`test_llm.py:371,383,395`) end with prose. **Not one negative pin ends with
  a genuine call**, so the family is invisible to the suite — the same coverage-direction failure
  round 1 diagnosed one level down, and which `BACKLOG.md:471-474` records as the slice's own lesson.

**Suggested improvement — verified, not proposed blind.** Require that, from the **first** accepted
call onward, only calls and whitespace appear: reject the whole recovery when
`text[end_of_last_call:match.start()].strip()` is non-empty at the second and later accepted calls
(≈3 lines beside `llm.py:301`). Prototyped in a scratchpad and run over the corpus:

```
8 positive pins            all unchanged (T1…T8, incl. two-calls-on-separate-lines)
m-5 identical repeat       unchanged (one dispatch)
considered-then-did        CALL,CALL → text      echoed snippet then real   CALL,CALL → text
disclaimed then real       CALL,CALL → text      FP1/FP2 (already closed)   unchanged
```

**Residual even then, and honest about it:** a bare *catalogue* of contiguous calls (row 4, with no
prose between them) still fires — it is structurally identical to an intended multi-call. Closing
that needs either a cap of one call on the bare path (which costs
`test_chat_recovers_two_bare_calls_on_separate_lines`) or open question 4's granted-name recognition
filter. **If the owner prefers to keep the current width**, then the doc half is not optional: the
three sentences above must state the real rule ("the *last* recovered call must end the message;
earlier own-line calls ride along and are dispatched"), and rows 1–4 must be pinned as **accepted**
behaviour so the next reader is not misled and the next narrowing is a deliberate act.

**Also verified, and correctly disclosed:** the round-1 4-space-markdown shape
(`Example:\n\n    post_message({…})`) still dispatches. That one *is* the documented residual
(`llm.py:272-274`) and I am not re-filing it — but note the docs' "illustrative call on the final
line" phrasing is the *only* residual named, which is what makes N-1's omission read as closure.

### N-2 · Minor — the judge's own-line residual is real, one-sided in the *advance* direction, and pinned only in its inline form

Verbatim, against the delivered judge → `guards._coerce_verdict`:

```
'The reply shape is:\n{"decision": true, "rationale": "..."}\nIn this case the condition is not met.'
   → {'decision': True, 'rationale': '...'}   → GuardVerdict(decision=True)   ← ADVANCES
'If they had named it I would answer:\n{"decision": true, …}\nBut they did not, so I answer false.'
   → {'decision': True, 'rationale': 'named'} → GuardVerdict(decision=True)   ← ADVANCES
```

This is the residual the implementer declares (`llm.py:437-439`, `BACKLOG.md:348`, HISTORY), and it
is **within the range round 1 itself sanctioned** — the review's own option-1 rule ("last balanced
object carrying `decision`") accepts these too. So it is not a re-open. Two gaps worth closing
cheaply: (a) `test_app.py:303` pins the **inline** schema echo only — the own-line twin, which is the
one that advances, is unpinned in either direction; (b) `app.py:352-360` lists what resolves to
`decision=False` but never names this residual, so the docstring a future judge-calibrator reads is
one-sided while `llm.py`'s is complete. **Suggested:** add the own-line echo as a labelled
characterisation pin, and one sentence in `app.py`'s docstring. The stricter alternative round 1
offered (accept only a reply that is *entirely* one object after fence-stripping) stays available
for item 3 if calibration shows this shape in the false-advance count.

### N-3 · Minor — m-7 was dropped silently

`BACKLOG.md:315` still reads `### K-027 — … (🔵 proposed …)` after a delivered, twice-gated slice
that ✅-marks item 1 and addendum (b) inside the same item. The legend at `BACKLOG.md:6` offers
`🟡 in-progress`. The finding is absent from the claimed dispositions — not rejected with a reason,
just missing, which is the failure mode that makes a "every finding adopted" claim expensive to
trust. **Suggested:** flip to `🟡 in-progress`, or state the rejection.

### N-4 · Minor — `BACKLOG.md:409-410`'s softened claim is still false when read literally

> *"This **would have converted the observed shape as recorded**"*

The shape **as recorded** in `m3-workflow-engine-report.md:265` is line-wrapped. I drove the wrapped
form through the *delivered* code: still `text`, not a call (a raw newline inside the JSON string
fails `json.loads` in `_parse_call_arguments`, `llm.py:352-358`) — bullet 4 twenty lines below says
so. What was converted is the **de-wrapped reconstruction** (`test_llm.py:224-228`). **Suggested:**
"…converted the observed shape **as reconstructed (de-wrapped)**", and relabel the fixture comment's
leading word "Verbatim" (`test_llm.py:221`) — the two lines under it already disclose the
reconstruction honestly, which is why this is minor rather than a repeat of m-4.

### N-5 · Minor — `llm.py:181` still calls the JSON branch "(unchanged)"

m-1's whole point. The branch's `if calls:` / `if call:` guards changed its behaviour for
`{"tool_calls": [], "name": …, "arguments": …}` (now a call, was text — pinned at
`test_llm.py:429`). HISTORY discloses it; the docstring still tells the next reader the opposite.
**Suggested:** drop the word, or say "JSON first (probe order unchanged; the empty-envelope
fall-through is new — see HISTORY 2026-07-24)".

### N-6 · Nit — two small parse behaviours still unstated, one new

- `post_message ({"text": "hi"})` (space or tab before the paren) is recovered — round 1's n-2 named
  it; `llm.py:242-246` documents the *indentation* half only.
- A **multi-line array-wrapped** verdict now advances: `'[\n{"decision": true, …}\n]'` →
  `decision=True`, while the single-line `'[{"decision": true, …}]'` round 1 named correctly
  suspends. It follows from the documented own-line rule and the object *is* asserted, so I read it
  as acceptable; it is simply unpinned in either direction. Worth one pin next to
  `test_app.py:276`.

---

## Direct answers to the two questions the coordinator asked

**1 · Is B-1 closed? Yes — reproduced.** The exact round-1 input now suspends:

```
reply: 'If the user had named the service I would answer {"decision": true, "rationale": "named"}
        but they did not, so I answer false.'
NEW → {"decision": False, "rationale": "unparseable judge output"}
      guards._coerce_verdict → GuardVerdict(decision=False)          ← SUSPENDS (was: ADVANCES)
```

The second named shape, `'[{"decision": true, …}]'`, also suspends. The ambiguity I was asked to
probe — *a real verdict on its own line whose quoted twin is also on its own line* — is handled
correctly by the "exactly one qualifying object" rule:

```
'I would have answered:\n{"decision": true, …}\nbut in fact:\n{"decision": false, …}'  → False
'{"decision": false}\n{"decision": true}'                                              → False
```

Two disagreeing candidates ⇒ suspend, in both orders. `require_key="decision"` correctly ignores a
non-verdict second object (`'Answer:\n{"decision": true, "rationale": "a"}\n{"note": "x"}'` → True),
which is the right asymmetry. All eleven positive judge shapes I tried (bare, fenced ```json,
unlabelled fence, ```` ```tool_code ````, indented, prose-before-and-after, trailing whitespace)
still parse. The one residual is **N-2**, disclosed by the implementer and inside the range round 1
sanctioned. **B-1 closed.**

**2 · Is M-1's rule watertight? No — it moved the boundary rather than sealing it.** Of the six
round-1 shapes, five stay text *as written*, and the sixth (4-space markdown block) is the declared
residual. But the rule's anchor is positional and applies to the **last** call only, so **all six
re-open verbatim the moment the model ends the message with a genuine call** — see N-1's table,
where a disclaimed `human_handoff()` and an echoed user `post_message` are both dispatched. The new
boundary itself behaves as advertised and errs safe: a trailing newline / trailing whitespace / CRLF
after the call still recovers; a closing ```` ``` ```` after the call recovers **only** when the
whole message was fenced (`_strip_code_fence` is anchored at the start, `llm.py:364-372`) and
otherwise falls back to text; a call followed by a bare `.` — same line or next — falls back to
text. All of those misses are in the safe direction and cost one executor re-prompt.

**3 · Did the 8 positive pins survive?** Yes, on the strongest evidence available without the
pre-fix tree: every line number round 1 quoted still lands on the same construct (fixture 224-229,
prose-then-call 263, two-calls 296, the six negatives 312→362), the two new `test_llm.py` blocks are
pure insertions at 362 and 426, and `test_app.py`'s three judge-recovery pins are untouched at
249/260/266 (the `_ReplyLLM`/`_judge_verdict` helpers at 231-245 predate this pass, or those numbers
would have shifted). Bodies read: none was weakened — the positives still assert `name`,
`arguments` and `id` equality, the negatives still assert `result.text == prose` identity. 56 passed.

**4 · The `text=None` discard.** Unchanged at `llm.py:150`, and correctly so — it is the
pre-existing embedded-JSON contract, pinned deliberately at `test_llm.py:271-272`. Blast radius is
**materially reduced**: recovery now requires the call to end the message, so the discarded prose is
only the *lead-in* to a call the model did make, not (as in round 1) the model's entire real answer
while an example fired. Residual behaviour is acceptable: an `answer` node that ends in a genuine
`post_message` has already put its answer in the tool argument. The one case where it still bites is
N-1's family, and closing N-1 closes that too.

**5 · m-4's softening.** Defensible in substance — the search for the real `StepRun.output` is
recorded, the stronger claim is explicitly disowned, and item 5 is told not to rely on it. Only the
two words "as recorded" still assert something I falsified (N-4).

**6 · K-035.** Accurate and rot-resistant: the three reproduction rows are verbatim, the
not-currently-reachable premise is stated *as the risk* rather than as reassurance, the remedies are
ranked with the layering decision flagged, and the test strategy pins the negative direction too.
The tripwire is at `llm.py:188-195` — the probe-order site round 1 asked for, and the right one for
the *parser* author. The reader it most needs to reach (someone registering a tool) is in
`tools.py`; a one-line pointer beside the tool registry would close that, and is a suggestion, not a
finding.

**7 · Did the fix pass break anything?** No behavioural regression found. `_scan_balanced` is a
correct generalisation of `_scan_parenthesized` (I re-threw the round-1 corpus at it: parens and
braces inside strings, `\"`, `\\"`, unbalanced input, CRLF — all unchanged), `extract_json_object` is
behaviourally identical after the `_strip_code_fence` extraction, the JSON→bare probe order is
preserved, and native-over-content precedence is still pinned. The contradictions this pass
introduced are documentary, not behavioural: N-1 (three docs), N-5 (one docstring word).

---

## What's solid in the fix pass

- **The seam split is the right shape, not just a passing test.** Two named functions with opposite
  biases, each carrying an explicit tolerance contract, the permissive one justified by the
  downstream re-validation that actually exists (`executor._handle_tool_call`), and the conservative
  one refusing to guess between two candidates. The `require_key` parameter keeps the
  conservatism from over-firing on an unrelated second object. This is a better answer than either
  option round 1 proposed.
- **The docs do the hard half.** `HISTORY.md` states plainly that the previous claim *was false*,
  names the blocker, and reproduces the input — the opposite of quiet correction. `BACKLOG.md` item
  1 carries the accurate property statement, and the m-3 bullet is struck through *with its reason*
  rather than deleted.
- **Honest counting.** 8 red / 3 characterisation, with the characterisation pins labelled in place
  and the reason given ("the old span parse already lost it to invalid JSON rather than by design").
  I verified the arithmetic reaches 56.
- **Scope discipline held again.** `executor.py` and `guards.py` do not appear as modified; no
  Cypher, DDL, script or schema; the K-035 filing is a backlog entry plus a comment, not a
  precedence change smuggled in under a deferral.
- **m-2's fix keeps the property round 1 asked it to keep** — the cursor advances on a *successful*
  scan and still does not advance on a failed one, so a rejected expression can neither hide a
  nested identifier nor swallow a later real call. Both directions are now pinned.

## Open questions (coordinator, not fixes)

1. **N-1's behaviour half — this slice or item 3?** The doc half must land now either way. My
   recommendation is both here: it is 3 lines, I verified it against the full corpus, and item 3's
   false-advance/precision measurements are only meaningful against a settled recogniser.
2. **The catalogue residual** (contiguous own-line calls, no prose between) is not closable by
   position alone. It is the second argument for open question 4 of round 1 — passing the granted
   names down as a recognition filter — which would also close K-035. Still a real layering
   decision, still flagged rather than asserted.
3. **Unverified in this pass:** the "595 passed / ruff clean repo-wide" claim. I ran only the two
   touched files and ruff on the four owned files, by instruction. Whoever integrates both units
   should run the full suite once, and re-seed `reference` afterwards per `AGENTS.md`.

## Routing (round 2)

- **N-1 (doc half, mandatory), N-3, N-4, N-5** → `tdd-engineer` / docs owner; each names its file
  and line.
- **N-1 (behaviour half), N-2 pins, N-6 pins** → `tdd-engineer`; the prototype rule and the exact
  input strings are in N-1.
- **N-1's width trade-off and the catalogue residual** → `architect`, if the current width is
  preferred over the tightening.

---

# Re-gate — K-027 slice A fix pass 2 (round 3)

> Reviewer: `analyst` · Date: 2026-07-24 · **Narrow re-gate only** — N-1…N-6 closure, positive-pin
> survival, and the doc half. Not a re-review of the change; rounds 1–2 stand. K-031's files and the
> `BACKLOG.md:745` staleness (coordinator-owned) are out of scope. Method: read the delivered
> `llm.py` / `app.py` / both test files / the three docs, plus a 32-shape behavioural probe driven
> through `_parse_bare_call_syntax` (read-only script in the scratchpad; source untouched).

## Round-3 verdict: **approve**

0 blocker · 0 major · 0 minor · 0 new findings. All six round-2 findings close, on evidence.

## Per-finding disposition

| # | Status | Evidence |
|---|---|---|
| **N-1 (behaviour)** | **CLOSED** ✅ | `llm.py:314-318` — the between-calls guard is the prototype verbatim, placed *before* `end_of_last_call` advances and *before* the dedup `continue`, so it also fires on a repeat-with-prose-between. All four N-1 table rows now return **TEXT**; so do `Example:…/Now the real one:…`, a bare ```` ``` ```` between two calls, and a numeric marker between two calls. |
| **N-1 (docs)** | **CLOSED** ✅ | `AGENTS.md:235-242`, `HISTORY.md:135-146` + `:199-219`, `BACKLOG.md:404-419` all now state the *delivered* rule ("from the first accepted call onward … nothing but calls and whitespace") and name **both** residuals. Repo-wide grep for the old "final non-whitespace content" wording returns only historical/narrative uses (`HISTORY.md:201`, `test_llm.py:427`, the coordination log) — no live assertion survives. |
| **N-2** | **CLOSED** ✅ | Own-line judge echo pinned as labelled characterisation (`test_app.py:312-325`); `app.py:365-371` carries the matching "**Residual, declared**" paragraph, so the two docstrings are no longer one-sided. |
| **N-3** | **CLOSED** ✅ | `BACKLOG.md:315` — `K-027 … (🟡 in-progress …)`. |
| **N-4** | **CLOSED** ✅ | `BACKLOG.md:417` "as reconstructed (de-wrapped)"; the fixture comment (`test_llm.py:221`) now leads with "**Reconstructed (de-wrapped)**". |
| **N-5** | **CLOSED** ✅ | `llm.py:181-182` — "probe order unchanged; the empty-envelope fall-through below is new". |
| **N-6** | **CLOSED** ✅ | Space-before-paren stated (`llm.py:246`) and pinned (`test_llm.py:300`); multi-line array-wrapped verdict pinned (`test_app.py:337`). |

## Boundary probe — the new rule, verified

- **Legitimate multi-call turns still work**, all shapes: two calls separated by `\n`, a blank line,
  `\r\n`, a tab-only line, an indented second call, three calls, multi-line JSON args, a
  whole-message fence, a trailing newline, and dup-then-distinct (`m-5` dedup intact).
- **Rejected candidates fail safe.** A non-JSON-arg call, an unbalanced paren, or a same-line-prose
  call sitting *between* two genuine calls discards the whole recovery (→ TEXT), because the tail
  check at `llm.py:328` still sees the rejected span. So the docs' rule 3 is true as written, not
  only for accepted calls.
- **Declared residuals behave exactly as documented, not worse:** the contiguous own-line catalogue
  still fires (3 calls), and an illustrative call on the final line still fires. Nothing widened.
- Prose (or a *rejected* call-shaped expression) *before* the first accepted call is still allowed —
  that is the `recovers_a_bare_call_on_its_own_line_after_prose` positive, and the docs say so
  ("prose on lines *before the first* call are still recovered").

## Pins

All **9** positive bare-call pins (`test_llm.py:242,253,260,267,278,285,293,300,311`) and all **4**
judge positives (`test_app.py:204,249,260,266`) read by name **and** body: assertions are unchanged
in strength — names, full `arguments` dicts, call `id` ordering, `text is None`, and rationale
substrings. Nothing relaxed to an `is not None` or a length check. The `+8` on the round-2 count of
56 is entirely insertions (5 in `test_llm.py`: the space-before-paren positive, three N-1 negatives,
the catalogue characterisation; 3 in `test_app.py`). Re-ran the two files myself: `test_llm.py` →
**37 passed** (0.03s, fully offline), `test_app.py -k judge` → **18 passed** — consistent with the
coordinator's 64.

## Notes, not findings

- `test_chat_still_recovers_a_contiguous_catalogue_of_calls` exercises the prose-*header* allowance
  as well as the catalogue residual (its fixture opens with `Available tools:`). Both are documented;
  the pin simply covers two allowances at once.
- The rule now rejects a *legitimately* multi-call turn that narrates between its calls
  ("First, retrieve:\n…\nThen answer:\n…") — a miss costing one executor re-prompt, disclosed in
  HISTORY's rejected list. Correct side of the trade; recorded so the next reader is not surprised.
