# `_assemble_messages` role-alternation fix — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** K-048

## 1. Goal & scope

Fix `WorkflowExecutor._assemble_messages` (`server/falkorchat/executor.py:912-934`) so it never
emits two consecutive same-role (`user`/`user` or `assistant`/`assistant`) messages in the opening
message list it builds for an `agent`-typed workflow node — the shape that hard-crashes a
strict-alternation chat template (live-confirmed: LM Studio's Ministral-3B, HTTP 400) and is
structurally guaranteed on `intake`'s first call and every `research`→`answer` handoff.

**In scope:** `_assemble_messages` itself, plus one new small module-level helper it calls.
**Out of scope:** `_drive_loop` (SHA-locked, re-verified below — untouched), the tool-call loop
body in `_run_agent_node` (its own appends are already alternation-safe — see §3), any schema/
Cypher/index change (none of this touches the graph), and any change to what tools are offered or
how the agent loop iterates.

**CPG:** used `cpg_falkorchat` — confirmed via `MATCH (call:CALL) WHERE call.METHOD_FULL_NAME
CONTAINS '_assemble_messages' RETURN call.METHOD_FULL_NAME, call.CODE` that
`_assemble_messages` has exactly one call site in the whole graph
(`WorkflowExecutor._run_agent_node`, the same one found by grep), so the impact surface of
changing its output shape is confirmed closed, not merely assumed from a text search.

## 2. Context & findings

- **The function today** (`server/falkorchat/executor.py:912-934`):
  ```python
  @staticmethod
  def _assemble_messages(
      config: dict[str, Any], run_ctx: dict[str, Any],
      thread_msgs: list[dict[str, Any]],
  ) -> list[dict[str, Any]]:
      messages: list[dict[str, Any]] = []
      system = config.get("systemPrompt", "")
      if system:
          messages.append({"role": "system", "content": system})
      for m in thread_msgs:
          role = "assistant" if m.get("role") == "assistant" else "user"
          speaker = m.get("displayName") or m.get("authorId") or "member"
          messages.append(
              {"role": role, "content": f"{speaker}: {m.get('text', '')}"}
          )
      context = json.dumps(run_ctx, separators=(",", ":"), sort_keys=True)
      messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
      return messages
  ```
  It is called from exactly one place, `_run_agent_node` (`executor.py:703`):
  `messages = self._assemble_messages(config, run_ctx, thread_msgs)`. `thread_msgs` is the
  return of `_read_thread_context` (`executor.py:894-910`), which is either `[]` (no `threadId`,
  or `self._services is None` — the offline stub path) or `services.read_thread(...)[-20:]`.
- **Role is a strict binary, never a third value.** `Services._validate_and_derive_role`
  (`server/falkorchat/services.py:822`): `role = "user" if actor_kind == "User" else "assistant"`.
  Every message written to a thread — human or any agent, including an agent *other than* the
  one running this node — carries only `"user"` or `"assistant"`. `_assemble_messages`'s own
  `else "user"` fallback (line 927) is therefore dead-defensive, not a real third case.
- **This means a same-role run can occur *inside* `thread_msgs` itself, not only at the
  CONTEXT tail.** Nothing in the write path enforces that consecutive thread messages
  alternate authorship — two human turns with no agent reply between them (a person sends a
  follow-up before the agent responds), or two different agents each posting once, both produce
  two consecutive `"user"`/`"assistant"` reads from `read_thread`. This is a sibling of the
  confirmed CONTEXT-tail defect, currently unconfirmed-but-plausible rather than live-verified
  (the backlog's live-verified repro is specifically the CONTEXT-tail shape), and it drove the
  choice in §3.
- **Confirmed defect shape, from the backlog (verbatim facts I re-derived, not just copied):**
  `intake`'s first call always sees `thread_msgs` ending in the `user`-role trigger message that
  started the run (already posted to the thread as the triggering `@mention` before the run
  begins); `research` is granted only `graphrag_retrieve`, never `post_message`, so it never posts
  a thread-visible turn, meaning `answer`'s first call after a `research` hop also sees
  `thread_msgs` ending in a `user` turn. Both land the always-`user` `CONTEXT` block right after
  another `user` message.
- **`_run_agent_node`'s only assumption about the returned list is its length for a trace string**
  (`executor.py:713-716`, `f"...: {len(messages)} msgs, ..."`) — no index-based access, no
  assumption about how many turns came from the thread vs. the tail. Confirmed by grep
  (`messages\[` matches nothing in `executor.py`) and by reading every use of `messages` in
  `_run_agent_node`: it is only ever passed whole to `llm.chat(messages, offered)` (line 717) or
  appended to (lines 767, 773). No existing test asserts an exact message count either
  (`grep -n "len(.*messages" server/tests/test_executor_agent.py` — no hits). A change that
  reduces the count when messages coalesce is safe.
- **`_drive_loop` re-verified untouched.** Reproduced the backlog's own hash recipe
  (`docs/DESIGN.md` §9) on the current tree:
  ```
  awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py \
    | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12
  ```
  → `71055f756280`, unchanged, confirming the awk-delimited span the lock covers is
  `_drive_loop` through the line right before the `# ── seams` marker (`executor.py:451` through
  the line before `executor.py:512`). `_assemble_messages` (`:912`) and its caller
  `_run_agent_node` (`:618`) are a different method entirely, physically hundreds of lines below
  the locked span — this fix touches neither the locked text nor anything the lock's hash
  function reads.

## 3. Design & rationale

**Chosen remedy: a general adjacent-same-role coalescing helper, applied at every append site in
`_assemble_messages`** (both the thread-turn loop and the trailing `CONTEXT` append) — not a
tail-only special case.

Add one new module-level function, next to the existing `_assistant_turn` helper
(`executor.py:218`, same file, same "small message-shape helper" convention):

```python
def _append_turn(messages: list[dict[str, Any]], role: str, content: str) -> None:
    """Append a `user`/`assistant` turn, coalescing into the previous message instead of
    appending a new one when that would produce two consecutive same-role turns (K-048): a
    strict-alternation chat template (live-confirmed: LM Studio's Ministral-3B) hard-rejects
    that shape with an HTTP 400 ("conversation roles must alternate user and assistant
    roles"); a tolerant template (Qwen) accepts it today, which is exactly why this can't be
    "obviously" broken in production. Only ever merges when the previous message's role
    equals this turn's role — it never merges into a leading `system` message (a different
    role, and always singular), so it is a no-op whenever the sequence is already
    alternating. Covers both known trigger shapes with one primitive: two consecutive thread
    turns from the same side (no reply in between — plausible, not yet live-observed) and the
    always-`user` trailing CONTEXT block landing after a `user`-authored last turn
    (`intake`'s first call; every `research`→`answer` handoff, since `research` never posts a
    thread-visible turn)."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
    else:
        messages.append({"role": role, "content": content})
```

`_assemble_messages` changes to route its `user`/`assistant` appends through it (the leading
`system` append is untouched — still a direct `messages.append(...)`, since it can never collide
with anything before it):

```python
@staticmethod
def _assemble_messages(
    config: dict[str, Any], run_ctx: dict[str, Any],
    thread_msgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = config.get("systemPrompt", "")
    if system:
        messages.append({"role": "system", "content": system})
    for m in thread_msgs:
        role = "assistant" if m.get("role") == "assistant" else "user"
        speaker = m.get("displayName") or m.get("authorId") or "member"
        _append_turn(messages, role, f"{speaker}: {m.get('text', '')}")
    context = json.dumps(run_ctx, separators=(",", ":"), sort_keys=True)
    _append_turn(messages, "user", f"CONTEXT:\n{context}")
    return messages
```

Signature is **unchanged**: still `(config, run_ctx, thread_msgs) -> list[dict[str, Any]]`, still
a `@staticmethod`. `_append_turn` needs no `self`/class access, so it is a plain module function
(matching `_assistant_turn`'s existing precedent at line 218), not a new method on the class.

The merge separator is `"\n\n"` between the previous message's `content` and the new turn's
`content` — both are always plain strings here (`f"{speaker}: {text}"` or the `CONTEXT:\n{...}"`
block), so string concatenation is safe with no type juggling. This is a specific, load-bearing
choice the implementer should not improvise a different one for: it keeps each coalesced turn
readable as its own paragraph, consistent with how a single turn already prefixes the speaker
name.

### Alternatives considered

1. **Candidate 1 from the backlog — merge only the CONTEXT block into the prior turn, and only
   when that turn is `user`-role.** Fixes exactly the live-verified defect and nothing else.
   Rejected as the sole fix: it leaves the sibling shape (two consecutive same-role thread turns,
   independent of the CONTEXT block — see §2) exposed to the identical crash on the identical
   strict-alternation template, for no implementation savings — a tail-only special case is not
   simpler than a two-line coalescing helper reused at every append site, and choosing it means
   accepting a known-plausible latent 400 rather than closing it at the same cost.
2. **Candidate 2 from the backlog — unconditionally fold every thread turn and the CONTEXT block
   into one trailing `user` message, regardless of role sequence.** Rejected: strictly larger
   blast radius than the defect requires. Today's already-alternating case (thread ending in
   `assistant`) is not broken — collapsing it anyway would discard the turn-by-turn
   `user`/`assistant` distinction the model currently sees (which turn was the model's own prior
   answer vs. what a human said), for every single agent-node call, not just the ones that are
   actually mis-shaped. That is a real behavioral/quality risk for the *tolerant* model (Qwen)
   with zero corresponding benefit, since Qwen was never crashing.
3. **Chosen — general adjacent-same-role coalescing, applied everywhere `_assemble_messages`
   appends a turn.** Strictly dominates candidate 1 (closes the sibling shape at the same
   implementation cost) and is far smaller-blast-radius than candidate 2 (only touches a
   sequence when it is *actually* about to violate alternation — an already-alternating sequence
   is provably untouched, since the merge condition `messages[-1]["role"] == role` can only be
   true when it wasn't going to alternate anyway). This is the "detect alternation breaks
   generally" option the task raised as candidate 3, worked out concretely.

### Regression risk — exact before/after shapes

**Thread ending in `assistant` (today's already-correct, tolerant-and-strict-safe path) — must
stay byte-for-byte identical.** Example: system prompt set, `thread_msgs` = `[user-turn,
assistant-turn]`.

- Before: `[{system}, {user: "Alice: hi"}, {assistant: "Bot: reply"}, {user: "CONTEXT:\n{...}"}]`
  — 4 messages, roles `system, user, assistant, user`. Already alternating (system doesn't count
  against the user/assistant rule); no crash on either template today.
- After: identical. Trace through `_append_turn`: `system` appended directly (unaffected).
  `user` turn → `messages[-1]` is `system` (role differs) → appended as new message. `assistant`
  turn → `messages[-1]` is `user` (role differs) → appended as new message. `CONTEXT` (`user`) →
  `messages[-1]` is `assistant` (role differs) → appended as new message. **Zero merges fire**,
  so the output is exactly the 4-message list above, same objects' shape, same content strings.
  This is the regression guarantee: the helper is a strict no-op whenever the input was already
  alternating.

**Thread ending in `user` (today's broken, crash-on-strict-templates path) — must change.**
Example: same system prompt, `thread_msgs` = `[user-turn]` only (the `intake`-first-call /
`research`→`answer` shape).

- Before: `[{system}, {user: "Alice: hi"}, {user: "CONTEXT:\n{...}"}]` — 3 messages, roles
  `system, user, user`. This is the confirmed-crashing shape.
- After: `system` appended directly. `user` turn → `messages[-1]` is `system` (differs) →
  appended. `CONTEXT` (`user`) → `messages[-1]` is now the just-appended `user` message (role
  **matches**) → **merged**: that message's `content` becomes
  `"Alice: hi\n\nCONTEXT:\n{...}"`, no new message appended. Result: `[{system}, {user: "Alice:
  hi\n\nCONTEXT:\n{...}"}]` — 2 messages, roles `system, user`. Strictly alternating; the
  confirmed crash shape can no longer occur.

**The sibling shape this fix additionally closes (not live-verified as a production crash today,
but the same reasoning applies): two consecutive same-role thread turns before `CONTEXT` is even
appended** — e.g. `thread_msgs` = `[user-turn-A, user-turn-B]` (two human messages with no agent
reply between them). Before: `[{system}, {user: A}, {user: B}, {user: "CONTEXT:..."}]` — three
consecutive `user` messages. After: `_append_turn` merges `B` into `A`'s message on the second
call (same role, `messages[-1]` is the just-appended `A`), then merges `CONTEXT` into that same
message on the third call → `[{system}, {user: "A\n\nB\n\nCONTEXT:..."}]`. Alternating.

## 4. Step-by-step implementation (for `tdd-engineer`, strict TDD — failing test first)

All work is in two files: `server/falkorchat/executor.py` (production change) and
`server/tests/test_executor_agent.py` (new tests — same file already imports `WorkflowExecutor`
and exercises this class; no new test file needed).

1. **Red — pin today's already-correct shape first** (so the fix can't silently regress it,
   per the task's own requirement). Add a test that calls `WorkflowExecutor._assemble_messages`
   directly (it's a `@staticmethod`; no executor instance needed) with a `thread_msgs` list
   ending in an `assistant`-role row, and asserts the exact 4-message shape from §3
   ("thread ending in `assistant`"): roles `["system", "user", "assistant", "user"]`, and that
   the last message's content is exactly `"CONTEXT:\n{...}"` (not merged with anything). This
   test should **pass against the current, unmodified code** — it is a characterization test,
   not a red one; run it first to confirm today's behavior matches what §3 claims before
   touching production code.
2. **Red — the crash-shape test.** Add a test with `thread_msgs` ending in a `user`-role row
   (reuse the existing `_thread_rows(n)` fixture at `test_executor_agent.py:454-460`, which
   already produces `role: "user"` rows, or a single hand-built row) and assert: exactly 2
   messages, roles `["system", "user"]`, and the merged content contains both the original
   turn's text and `"CONTEXT:\n"`. This test **fails** against current code (today it produces 3
   messages with roles `["system", "user", "user"]`).
3. **Green.** Implement `_append_turn` and the `_assemble_messages` edit exactly as specified in
   §3. Run both new tests plus the full `test_executor_agent.py` file — every existing test in
   that file drives through `_run_agent_node`, not `_assemble_messages` directly, and none
   asserts an exact message count (confirmed in §2), so none should need updating. If any
   existing test does need a touch, that is a signal the "no other assumption" finding in §2 was
   wrong — stop and re-check before patching the test.
4. **Recommended, not mandated by the backlog: add the sibling-shape test** (§3's third example
   — two consecutive `user` thread turns, no `assistant` between them) to lock in the reason a
   general fix was chosen over candidate 1. Cheap to add alongside step 2's fixture.
5. **Full offline suite.** `.venv/bin/python -m pytest -q` (server/) — must stay green. This is
   network-free per `falkor-chat/AGENTS.md`; no FalkorDB write path is touched by this change, so
   no re-seed obligation applies (nothing here runs a default pytest against `reference`'s data
   in a way this change could break, and `test_executor_agent.py` uses only in-process stubs).
6. **Live regression pass — reuse the existing test, do not write a new one.**
   `server/tests/test_workflow_live.py::test_triage_flow_runs_end_to_end_against_live_llm`
   (`pytestmark = pytest.mark.live`, deselected by default) already drives the real `triage@v1`
   def end-to-end against the live LLM (today, Qwen) through exactly `intake`→`research`→
   `answer`, asserting the run reaches `done` and that `answer` posts a real reply
   (AC-1…AC-4). This *is* the "regression pass against the existing live triage flow" the
   backlog's test strategy calls for — both structural trigger shapes this fix changes
   (`intake`'s first call, the `research`→`answer` handoff) are exercised by this one test.
   Run `.venv/bin/python -m pytest -m live -s server/tests/test_workflow_live.py` once, after
   step 3 lands, as this fix's live verification. **Do not add a new standing live test or a new
   live Ministral fixture** — the backlog explicitly says the fix must not require a live
   Ministral instance to be part of the standing suite, and the existing offline tests (steps
   1-2) already pin the exact shape a strict-alternation template needs; the live pass here is
   about model *behavior* (does triage still clarify/research/answer sensibly with the merged
   message), not about the alternation crash itself, which is fully covered offline.

## 5. Test strategy (summary — see §4 for the concrete sequence)

| Level | What | Why |
|---|---|---|
| Offline unit (new) | `_assemble_messages` with thread ending `assistant` → pinned 4-message, byte-exact shape | Proves the fix is a no-op on the already-correct path (the regression the backlog explicitly worries about) |
| Offline unit (new) | `_assemble_messages` with thread ending `user` → pinned 2-message, merged shape, no two consecutive same-role entries | Proves the confirmed crash shape is gone |
| Offline unit (new, recommended) | `_assemble_messages` with two consecutive `user` thread turns → pinned merged shape | Validates the rationale for choosing a general fix over backlog candidate 1 |
| Offline suite (existing) | Full `test_executor_agent.py` + `pytest -q` | No behavioral assumption elsewhere in the codebase breaks (§2 confirms none should) |
| Live (existing, one-time run) | `pytest -m live -s server/tests/test_workflow_live.py::test_triage_flow_runs_end_to_end_against_live_llm` | The backlog's own "regression pass against the existing live triage flow," reusing the test that already covers both trigger shapes against the tolerant model — no new live infra |

Edge cases covered: empty `thread_msgs` with no system prompt (`[user: CONTEXT]`, unaffected —
no prior message to merge into); empty `thread_msgs` with a system prompt (`[system, user:
CONTEXT]`, unaffected, same reasoning); a thread ending in `assistant` (no-op, pinned); a thread
ending in `user` (merged, pinned); two-plus consecutive same-role thread turns before `CONTEXT`
(merged, recommended test).

## 6. Risks & open questions

- **Merge-separator choice (`"\n\n"`) is a judgment call, not a verified requirement.** No test
  in the existing suite asserts on exact `CONTEXT` block formatting, and no live evidence bears
  on whether `"\n\n"` vs. some other separator changes model output quality. This is a low-risk,
  easily-revisited choice — flagging it so a reviewer doesn't mistake it for something more
  deeply load-bearing than it is.
- **The sibling shape (§2/§3) is inferred, not live-verified**, the way the CONTEXT-tail shape
  was. If `analyst` or a later reviewer wants live confirmation before treating it as closed, that
  would need a live repro symmetrical to the backlog's own Ministral repro (two consecutive
  human messages, no CONTEXT block even needed) — out of scope for this plan to run, but cheap
  for `tdd-engineer` or a follow-up to do if requested.
- **No rollback concern.** This is a pure function with one call site, no persisted state, no
  schema/migration involved; reverting the diff fully reverts behavior.
- **No re-lock ceremony needed** (confirmed in §2) — `_drive_loop`'s hash is unaffected because
  this change is physically and semantically outside that method.
