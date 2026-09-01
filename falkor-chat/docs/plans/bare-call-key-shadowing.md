# Bare-call argument-key shadowing — closing K-035 (finding M-2)

> **Status:** active · **Owner:** `architect` · **Tracks:** K-035

## 0. Summary of the decision (read this first)

K-035 offers three candidate remedies, cheapest first. This plan picks **candidate 1** — in
`_parse_content_tool_calls`, suppress the loose `name`/`action`/`tool` envelope match when the
message content is itself shaped like a bare call expression (matches `_BARE_CALL_OPEN`) — over
candidate 2 (reorder the probes) and candidate 3 (thread the granted tool names down as a
recognition filter). §3 gives the full rationale; in short: candidate 1 closes all three repro
rows with a change scoped to exactly the shadowing shape, leaves the `tool_calls`-list branch's
precedence untouched for every other input (candidate 2 cannot make that guarantee without a full
regression pass — see §3), and does not reopen the layering question the K-027 review flagged as
"a real layering decision" for candidate 3.

**The diff is one added `if` around one existing call, in one function, in one file**
(`server/falkorchat/llm.py:274-302`, `_parse_content_tool_calls`) — plus a doc/comment correction
at the same site and the routine backlog/history housekeeping for a delivered item. No other
function's signature changes; `_normalize_tool_call` (`llm.py:305-317`) is untouched.

**Risks/RAM: confirmed none**, per the backlog's own framing — this stays a pure parse-layer
change inside one already-private function. No Cypher, no schema, no graph/index surface, no
public interface or data-shape change (`ToolCall`, `ChatResult`, and `LLM.chat`'s signature are
all unchanged).

**CPG:** considered, not relevant — this is a single already-open file, two named functions, with
exactly one call site of the touched function (`_normalize_tool_call`, confirmed by
`grep -rn "_normalize_tool_call" server/falkorchat/*.py server/tests/*.py`, one hit inside
`_parse_content_tool_calls` itself) and one call site of the touched function's own caller
(`_parse_chat_message`, `llm.py:244`). Direct reading answered every impact-analysis question this
task raised; no cross-module call-graph or data-flow question came up that `cpg_falkorchat` would
have answered faster or more reliably.

---

## 1. Goal & scope

**Goal.** Fix `llm._parse_content_tool_calls` so that a bare call whose *argument object* carries
a `name`, `action`, or `tool` key is recognized as the bare call it is — `ToolCall(name=<the
call's own identifier>, arguments=<the argument object, verbatim>)` — instead of being
mis-recognized as a loose JSON envelope named after the shadowing value, silently losing the real
call name and the rest of the arguments.

**In scope:**
1. The guard itself in `_parse_content_tool_calls` (`llm.py:274-302`).
2. The stale docstring/`NOTE (K-035)` comment at the same site (`llm.py:274-292`), corrected to
   describe the fix and the one residual it deliberately does not close (§3, "Residual, by
   design").
3. Moving the `_BARE_CALL_OPEN` compiled regex (`llm.py:345`) to before `_parse_content_tool_calls`
   (`llm.py:274`) — it is now referenced by two functions defined at that point, not one; the move
   is a straight relocation, not a behavior change.
4. Six new tests in `server/tests/test_llm.py`, all driving the public `llm.chat(...)` seam (see
   §5) — the backlog's own three repro-table pins, a stronger multi-key envelope regression pin, a
   negative pin for the M-1/K-035 interaction, plus one optional pin documenting an accepted
   residual.
5. `docs/BACKLOG.md`/`docs/HISTORY.md` housekeeping for a delivered item (§6 step 5) — this is a
   module-documentation-convention obligation (root `AGENTS.md`, "Module documentation
   convention"), not optional cleanup.

**Out of scope (explicitly):**
- Candidates 2 and 3 (not chosen — §3).
- M-1's own residuals (the contiguous-catalogue and trailing-call cases the K-027 review's N-1
  reopened) — those are routed to `tdd-engineer` directly off the review, not part of K-035.
- Any change to `executor.py`'s AC-6 check or `_handle_tool_call` — confirmed below (§3) that this
  fix does not touch or need to touch that layer.
- The residual named in §3 ("Residual, by design") — accepted, not fixed, by design; test 6 in §5
  pins it as a documented characterization, not a defect.

---

## 2. Context & findings

**The bug, exactly as filed.** `_parse_content_tool_calls` (`llm.py:274-302`) runs the JSON probe
before the bare-call probe. For content shaped like `create_user({"name": "bob"})`:
`extract_json_object` (`llm.py:488-521`) lifts the *argument* object `{"name": "bob"}` out of the
call expression (its first-`{`-to-last-`}` extraction is call-expression-blind — it does not know
`create_user(` precedes what it found). `_normalize_tool_call` (`llm.py:305-317`) then maps that
object's own `name` key onto the `ToolCall.name` it returns, so the call comes back as
`ToolCall(name='bob', arguments={})` — the real call name (`create_user`) is never seen, and the
real arguments (`{"name": "bob"}`) are dropped. `_parse_bare_call_syntax` (`llm.py:348-426`, the
function that *would* have parsed this correctly) never runs, because the JSON probe already
returned.

**Exact repro (K-027 review, `docs/reviews/k027-parse-robustness.md` finding M-2, verbatim):**

| model emits | parsed as (today, buggy) |
|---|---|
| `create_user({"name": "bob"})` | `ToolCall(name='bob', arguments={})` |
| `run_tool({"action": "delete"})` | `ToolCall(name='delete', arguments={})` |
| `x({"tool": "y", "args": {"a": 1}})` | `ToolCall(name='y', arguments={"a": 1})` |

**Why it is silent and misleading, not loud** (backlog's own framing, confirmed): the manufactured
`ToolCall.name` is a user-supplied value, not a typo an obvious error would catch. If that name
happened to collide with an actually-granted tool, it would dispatch wrong; today it is always
*ungranted* (no registered tool takes a `name`/`action`/`tool` parameter — verified against the
current source, not the review's now-stale line numbers: `PostMessageTool`'s `text`/`mentions`
schema at `tools.py:244-254`, `GraphragRetrieveTool`'s `query` schema at `tools.py:343-348`,
`HumanHandoffTool`'s `reason` schema at `tools.py:1041`), so `executor._handle_tool_call`'s AC-6
check (`executor.py:1015-1017`)
rejects it, burns a re-prompt iteration, and tells the model its own argument is not a granted
tool — an almost-undebuggable trace from the model's point of view.

**The tripwire.** `llm.py:285-292` carries a `NOTE (K-035)` comment naming this exact item, at the
probe-order site, for the next author registering a tool with such a parameter. This plan's fix
makes the *bare-call* shape of the shadow (the repro table above) safe; the comment must be
rewritten, not just left in place, once the fix lands (§4 step 2) — leaving stale "harmless only
while…" language after the fix ships would mislead the very reader it is there to protect.

**`_normalize_tool_call` has exactly one call site** (`llm.py:299`, inside
`_parse_content_tool_calls`) — confirmed by grep, no other function or test calls it directly. This
is what makes the fix safe to land as a guard at that one call site rather than a signature change:
no other caller needs to reason about the new condition.

**The layering question (open question 4 of the K-027 review), resolved for this plan's scope.**
`OpenAICompatibleLLM.chat`'s own docstring (`llm.py:145-161`) states the design intent plainly:
"Name-against-granted-set and arg-schema validation live in the agent loop (U8), not here — `chat`
only parses the wire shape into `ChatResult`." `executor._handle_tool_call` (`executor.py:959-1108`)
is where that validation actually happens — `call.name not in granted_set` is AC-6's rejection
(`executor.py:1015-1017`), and it is the *only* place granted-tool-name authorization occurs. The
chosen remedy (candidate 1) does not read the granted set at all, so it cannot violate or blur that
layering — it is a pure recognition-order fix entirely inside the parse layer. (Candidate 3 *would*
have touched this line — see §3 for why it is not chosen here.)

**Existing test suite shape** (`server/tests/test_llm.py`, 761 lines, all `chat(...)`-seam tests,
no test drives the private probes directly — confirmed by grep, zero hits for
`_parse_content_tool_calls`/`_normalize_tool_call` outside `llm.py` itself). The file already has a
labeled section for K-027's bare-call recovery (`# --- chat(): bare function-call syntax in
content` at line 282) and its negative-direction sibling (`# --- the negative direction: prose must
stay prose` at line 388), plus a labeled gate-m-1 section for the `tool_calls`-envelope fallthrough
(`# --- the JSON probe's tool_calls envelope (gate m-1)` at line 578). None of those existing tests'
content strings start a line with `identifier(` while also carrying a `name`/`action`/`tool` key in
the extracted JSON object — traced individually in §5's "no regression" note — so none of them
change behavior under this fix.

---

## 3. Design & rationale

**Chosen fix, precisely.** In `_parse_content_tool_calls`, change:

```python
        call = _normalize_tool_call(obj)
        if call:
            return [call]
```

to:

```python
        # K-035: an argument object's own name/action/tool key must not be mistaken
        # for the call envelope when the content is itself a bare `name(...)` call
        # expression — the bare-call probe below is what recovers the call's real
        # identifier in that shape.
        if not (isinstance(content, str) and _BARE_CALL_OPEN.search(content)):
            call = _normalize_tool_call(obj)
            if call:
                return [call]
```

`_BARE_CALL_OPEN` (currently defined at `llm.py:345`, just above `_parse_bare_call_syntax`) moves
to just above `_parse_content_tool_calls` (before `llm.py:274`) — pure relocation (module-level
`re.compile`, no state), needed only because it is now referenced by two functions and the first
reference should not read as a forward-reference to a constant defined 70 lines further down.
Nothing else in `_parse_content_tool_calls` changes; the `tool_calls`-list branch above the guard
(`llm.py:295-298`) is untouched and unconditional, exactly as today.

**Why this closes all three repro rows.** For each row, `content` is exactly the bare call
expression, so `_BARE_CALL_OPEN.search(content)` matches (the identifier opens the first line) and
the guard suppresses the loose envelope match. Control falls through to
`_parse_bare_call_syntax(content)` (`llm.py:348-426`, unchanged), which was always the function
built to parse this exact shape and does so correctly: `create_user({"name": "bob"})` →
`ToolCall(name='create_user', arguments={"name": "bob"})`; `run_tool({"action": "delete"})` →
`ToolCall(name='run_tool', arguments={"action": "delete"})`; `x({"tool": "y", "args": {"a": 1}})`
→ `ToolCall(name='x', arguments={"tool": "y", "args": {"a": 1}})`. Verified by hand-tracing every
branch involved (`extract_json_object`, `_scan_balanced`, `_parse_call_arguments`) against the
current source — no live run was needed to confirm this, and none is required before
`tdd-engineer` writes the tests in §5 red-first.

**Why this does not regress a genuine envelope.** The guard is keyed on `_BARE_CALL_OPEN`, which
requires a line that *opens* with `identifier(` (`llm.py:345`'s own regex,
`^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(`, `MULTILINE`). A JSON envelope's content always opens
with `{` (bare, fenced, or embedded in prose — `extract_json_object`'s own docstring), never with a
bare identifier followed directly by `(`. So the guard is false for every existing JSON-shape test
in `test_llm.py` (traced individually in §5) and stays false for the new "genuine three-key
envelope" regression pin (test 4, §5).

**Alternatives considered and rejected:**

- **Candidate 2 — reorder the probes (bare-call first).** The K-027 review itself notes reordering
  is "safe and ~2 lines" *for the cases it checked*, but flags that it "needs its own regression
  pass over every JSON shape it must not break." Investigating that pass here surfaces the actual
  hazard: `_parse_content_tool_calls`'s `tool_calls`-list branch (`llm.py:295-298`, native
  multi-call envelopes) is reached only via the JSON probe. A content string shaped like
  `x({"tool_calls": [...]})` is *simultaneously* a bare call (`x(...)`) and, if the JSON probe ran
  first, a `tool_calls`-list envelope. Under the current probe order it resolves via the
  `tool_calls`-list branch; under an unconditional reorder it would resolve via the bare-call
  probe instead — `ToolCall(name='x', arguments={"tool_calls": [...]})`, a different `ToolCall`
  entirely. No test in `test_llm.py` exercises this specific collision today, which is exactly the
  review's point: closing it needs a deliberate regression pass, not an assumption that "JSON never
  matches the bare regex" is the whole story. Candidate 1 never routes any input away from the
  `tool_calls`-list branch — it only ever suppresses the *single-object* loose match, and only when
  the content is bare-call-shaped — so it carries none of this risk. This is the concrete version
  of "cheapest, available today" the backlog names candidate 1 as.
- **Candidate 3 — pass granted tool names down as a recognition filter.** The K-027 review's own
  open question 4 calls this "a real layering decision," not a mechanical one, and the backlog
  entry repeats that framing. §2 confirms `chat`'s docstring and `_handle_tool_call`'s AC-6 check
  are the load-bearing statement of "name validation lives in the agent loop, not the parser" — and
  confirms this plan's chosen fix never touches that boundary because it never reads the granted
  set. Candidate 3 would cross it deliberately (using granted names for *recognition*, not
  *authorization*, is a defensible position — but it is a *position*, requiring sign-off, not a
  fact this plan can establish by reading code). It is also wider in effect than K-035's own scope:
  the backlog says candidate 3 "also closes most of M-1's residual" — meaning it would change
  `_parse_bare_call_syntax`'s own matching behavior (the contiguous-catalogue and trailing-call
  cases the K-027 re-gate reopened as N-1), which is a different, already-separately-routed
  finding, not this item's. Folding it in here would make this fix's blast radius match a decision
  nobody has been asked to make yet. Deferred, not rejected — worth revisiting if a *future* K-item
  takes on M-1's residuals holistically and wants one filter that closes both at once.

**Residual, by design.** The guard is content-wide (`_BARE_CALL_OPEN.search` anywhere in
`content`), not span-correlated with the specific JSON object `extract_json_object` found. A
message that combines a genuine envelope with an *unrelated* bare-call-shaped line elsewhere —
e.g. `'{"name": "graphrag_retrieve", "arguments": {"query": "billing"}}\nfoo(bar)'` — trips the
guard even though the envelope itself is not the shadowing shape; `_parse_bare_call_syntax` then
finds no valid call anywhere in the content and the whole thing falls through to plain text,
discarding a genuine tool call. This is bounded (both a genuine-shaped JSON envelope *and* an
unrelated trailing bare-call-shaped line in the same message is not an observed real-model shape)
and on the safe side of the trade (text over a wrongly-dispatched call), matching this file's own
stated convention for accepted residuals (`_parse_bare_call_syntax`'s docstring, "Residuals, by
design"). Test 6 in §5 pins it as a documented characterization. Closing it precisely would need
span-correlating the guard to the specific extracted object — a real but separable enhancement, not
needed to close K-035's three repro rows, and not worth the extra complexity for a shape with no
observed occurrence.

---

## 4. Step-by-step implementation

All work is in `falkor-chat/server/`.

1. **`server/falkorchat/llm.py`** — move `_BARE_CALL_OPEN`'s definition (currently `llm.py:345`,
   including its docstring-style comment block) to immediately above `_parse_content_tool_calls`
   (currently `llm.py:274`). Pure relocation — same regex, same flags, same name.
2. **`server/falkorchat/llm.py`** — in `_parse_content_tool_calls`, apply the guard exactly as
   shown in §3. Rewrite the function's docstring and the `NOTE (K-035)` comment block
   (`llm.py:274-292` today) to describe the *fixed* behavior: the JSON probe still runs first for
   content that is not bare-call-shaped; when content is bare-call-shaped, the loose
   `name`/`action`/`tool` match is suppressed in favor of the bare-call probe, which owns
   recovering the call's real identifier. Include one line naming the residual from §3
   ("Residual, by design") so a future reader has the same context this plan does, and drop the
   now-stale "harmless only while no granted tool declares such a parameter" framing — it described
   the *deferred* state, not the fixed one.
3. **`server/tests/test_llm.py`** — add the five tests from §5, in a new labeled section placed
   after `test_chat_keeps_a_bare_empty_tool_calls_envelope_as_text` (ends line 597 today) and
   before `test_chat_prefers_native_tool_calls_over_a_bare_call_in_content` (line 600 today):
   `# --- chat(): K-035 — an argument key must not shadow the bare call's own name ---`.
4. **Run the full offline suite** (`cd server && .venv/bin/python -m pytest -q`, FalkorDB up per
   this session's constraints) and confirm zero regressions — in particular the existing
   content-embedded-JSON tests named in §5's "no regression" note, and the full K-027 bare-call
   section (lines 282-576 today), which exercises `_BARE_CALL_OPEN` far more than any other part of
   the suite.
5. **`docs/BACKLOG.md`** — remove the K-035 entry (`BACKLOG.md:383-427` today) once delivered; per
   root `AGENTS.md`'s module-documentation convention, a delivered item is not kept in `BACKLOG.md`,
   not even as an index row.
6. **`docs/HISTORY.md`** — add one dated entry (most-recent-first position) recording: the fix
   (candidate 1, guard in `_parse_content_tool_calls`), the finding it closes (K-035 / K-027 review
   M-2), and the test additions. Follow the existing entry style (see the 2026-08-31 K-061 entry for
   the level of detail/structure expected).

No other file changes. `executor.py` is read, not written (§2/§3 confirm the fix does not need it).

---

## 5. Test strategy

All tests are offline pins in `server/tests/test_llm.py`, driving `llm.chat(...)` via the existing
`_chat_content(content)` helper (`test_llm.py:303-307`) — never the private probes, per the
backlog's own instruction and this file's existing convention. `_TOOLS`/`_chat_transport` (already
defined, `test_llm.py:131-155`) need no changes.

**1-3 — the backlog's three repro rows, each pinned to the call's own identifier** (this is the
floor the backlog names explicitly):

```python
def test_chat_recovers_a_bare_call_whose_argument_key_would_shadow_via_name():
    result = _chat_content('create_user({"name": "bob"})')

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "create_user"
    assert call.arguments == {"name": "bob"}


def test_chat_recovers_a_bare_call_whose_argument_key_would_shadow_via_action():
    result = _chat_content('run_tool({"action": "delete"})')

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "run_tool"
    assert call.arguments == {"action": "delete"}


def test_chat_recovers_a_bare_call_whose_argument_key_would_shadow_via_tool():
    result = _chat_content('x({"tool": "y", "args": {"a": 1}})')

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "x"
    assert call.arguments == {"tool": "y", "args": {"a": 1}}
```

**4 — a genuine envelope carrying all three shadow-candidate keys still resolves via the existing
`name` > `action` > `tool` precedence** (stronger than the backlog's own minimal two-key example;
the plain two-key case — `{"name": …, "arguments": …}` with no surrounding call expression — is
already pinned by the existing `test_chat_parses_content_embedded_json_fallback`,
`test_llm.py:200-217`, and needs no new test, only re-verification per step 4 of §4):

```python
def test_chat_still_parses_a_genuine_envelope_carrying_all_three_shadow_keys():
    # Not bare-call-shaped (opens with `{`, not `identifier(`) — the K-035 guard
    # must not suppress this recognized envelope, and existing name > action > tool
    # precedence (_normalize_tool_call) must still pick `name`.
    payload = json.dumps(
        {
            "name": "graphrag_retrieve",
            "action": "ignored_action",
            "tool": "ignored_tool",
            "arguments": {"query": "billing"},
        }
    )
    message = {"role": "assistant", "content": payload}
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "billing"}
```

**5 — the negative direction: an argument-embedded `name` must not resurrect a call the M-1
final-content rule already rejects** (the backlog's third required pin — this is the interaction
between the K-035 guard and `_parse_bare_call_syntax`'s own rule 3, `llm.py:410-414`, "prose after
the last call discards the whole recovery"):

```python
def test_chat_keeps_a_shadowed_bare_call_with_trailing_prose_as_text():
    # The K-035 guard correctly routes this to the bare-call probe (the call opens
    # its line), but the bare-call probe's own M-1 rule then rejects it — trailing
    # prose after the call means it was being narrated, not made. The argument's
    # `name` key must not let the JSON envelope path resurrect it instead.
    prose = 'create_user({"name": "bob"})\nI did this because the user asked.'
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose
```

**6 — optional, documents the accepted residual from §3** (not required to close K-035; include it
because it makes an otherwise-silent trade-off explicit and testable, matching this file's existing
convention for pinned residuals, e.g. `test_chat_still_recovers_a_contiguous_catalogue_of_calls`,
`test_llm.py:540-556`):

```python
def test_chat_drops_a_genuine_envelope_when_unrelated_trailing_text_looks_bare_call_shaped():
    # Residual, by design (K-035 plan §3): the guard is content-wide, not
    # span-correlated with the specific object `extract_json_object` found. An
    # unrelated bare-call-shaped line elsewhere in the same message suppresses a
    # genuine envelope too. Not an observed real-model shape; documented rather
    # than closed.
    prose = (
        '{"name": "graphrag_retrieve", "arguments": {"query": "billing"}}\nfoo(bar)'
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose
```

**No regression, traced individually** (verify during step 4 of §4, not just by inspection):
`test_chat_parses_content_embedded_json_fallback` (`:200`), `test_chat_parses_structured_output_
action_shape_in_fenced_content` (`:220`), `test_chat_treats_non_tool_json_content_as_text` (`:251`),
`test_chat_falls_through_an_empty_tool_calls_envelope_to_the_sibling_call` (`:581`),
`test_chat_keeps_a_bare_empty_tool_calls_envelope_as_text` (`:592`), and
`test_chat_does_not_recover_an_identifier_nested_in_a_rejected_expression` (`:559`, gate m-2) — none
of their content strings open a line with `identifier(`, so `_BARE_CALL_OPEN.search` is false for
all of them and the guard never engages; §2/§3 trace each by hand. The full K-027 bare-call section
(`:282-576`) is likewise unaffected — those contents either have no JSON-object-shaped substring
with a `name`/`action`/`tool` key at all, or (for the ones that do carry `text`/`query` keys as
arguments) `_normalize_tool_call` already returns `None` on today's code before the new guard would
even matter, since those keys aren't `name`/`action`/`tool`.

**Edge case explicitly not needed:** a bare call whose argument object shadows via a *combination*
of two of the three keys (e.g. both `name` and `action`) is already covered by tests 1-3's
individual coverage of the guard itself — the guard's condition does not distinguish which key
would have matched, only whether the content is bare-call-shaped, so no additional combination
needs its own test.

---

## 6. Risks & open questions

- **Risk: none identified beyond the accepted residual (§3).** No public interface, data shape, or
  cross-layer contract changes. `ToolCall`, `ChatResult`, `LLM.chat`'s signature, and
  `executor._handle_tool_call`'s AC-6 check are all untouched and unread by the fix.
- **Risk/RAM (rule 6): confirmed none**, matching the backlog's own framing — parse layer only, no
  node type, index, property, or vector dimension; no Cypher, no schema, no script.
- **Open question: none requiring a stop-and-ask.** The one genuine design fork (which of the three
  candidates) is resolved in §3 with the backlog's own reasoning plus the additional collision
  hazard (`tool_calls`-list branch) found by tracing candidate 2, and the layering question
  (candidate 3) is resolved by direct evidence that the chosen fix never reads the granted set.
- **For the implementer:** step 4 of §4 (running the full offline suite) is the actual verification
  that the "no regression, traced individually" claims in §5 hold — the hand-tracing in this plan
  is the design-time justification for choosing this approach, not a substitute for running the
  suite.
