# Kaizen — Learnings Inbox: tdd-engineer

> Append-only capture of durable, non-obvious environment facts the `tdd-engineer` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-15 — A reachability-`skip` does NOT gate a live test when the live dep is normally up; only marker deselection does

- **Evidence:** `falkor-chat/server/tests/conftest.py` gates its FalkorDB integration tests with
  `if not _falkordb_reachable(): pytest.skip(...)`. Copying that pattern for an LM-Studio-backed
  test is a trap: LM Studio was reachable (`curl localhost:1234/v1/models` → 200), so the skip
  never fires and the "network-free" default `pytest` silently starts making LLM calls (the U14
  test takes 20–40s+ vs. the suite's 6s). Registering `markers = ["live: ..."]` **plus**
  `addopts = '-ra -m "not live"'` in `pyproject.toml` deselects instead: default run =
  `312 passed, 1 deselected` (verified), and a command-line `-m live` overrides the addopts `-m`
  (verified: `pytest -m live --collect-only` → `1/313 tests collected (312 deselected)`).
  The reachability-skip is still worth keeping *inside* the live test — but as the "don't fail for
  env reasons" net, not as the gate.
- **Context:** K-022 U14 — writing the first marker-gated live LLM e2e in `falkor-chat`, where the
  hard constraint was that the network-free baseline stay green and fast with LM Studio running.
- **Suggested home:** prompt (a general test-gating rule: gate on a marker when the dep is usually
  present; reachability-skip only guards against env absence, it does not opt tests out)

## 2026-07-15 — FalkorDB's empty-`UNWIND` row collapse silently turns "no rows" into an `IndexError` at the *caller*, not a graceful empty write

- **Evidence:** `falkor-chat/server/falkorchat/repository.py:1397` (`materialize_snapshot`) does
  `row = res.result_set[0]` after a `MERGE`-and-`UNWIND $transitions` query. Calling it with
  `transitions=[]` (a legitimate single-terminal-step def) raised
  `IndexError: list index out of range` — the bare `UNWIND []` collapsed the whole row stream
  before the `RETURN`, so the query wrote nothing AND returned nothing. `AGENTS.md` documents this
  quirk only for the §4 mention write-block (which is defended by an
  `UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END)` guard); `materialize_snapshot`
  has no such guard. The failure surfaces as an unrelated-looking Python `IndexError`, which is
  easy to misread as a repository bug rather than the known engine quirk.
- **Context:** K-022 Defect B — authoring a drive-level reproduction test needed a minimal workflow
  def; the natural shape (one terminal step, zero transitions) crashed on setup and briefly looked
  like a RED for the wrong reason. Worked around by using a 2-step/1-transition def.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` — generalize the empty-`UNWIND` note
  from "the mentions write-block" to "any `UNWIND $list` whose caller indexes `result_set[0]`";
  `materialize_snapshot` is a second, unguarded instance)

## 2026-07-24 — A first `docker ps` in a session can hang past a 120s tool timeout; later calls are ~0.3s — do not conclude "the sandbox blocks docker"

- **Evidence:** opening environment probe `docker ps --format … ; redis-cli -p 6379 PING` (sandboxed)
  exceeded the 120s Bash timeout, was backgrounded, and eventually completed **exit 0 with empty
  output**. Re-running `docker ps` with `dangerouslyDisableSandbox` returned instantly, which
  *looked* like a sandbox restriction. After `./scripts/start_falkordb.sh -d`, the identical
  **sandboxed** command ran in **0.348s** (`docker ps` → `falkordb-dev`) and sandboxed
  `redis-cli PING` in **0.020s**. So the sandbox was never the cause. Not attributed further —
  candidates are a cold Docker/WSL2 daemon on first use and `redis-cli` against a dead port; both
  are consistent with the timings, neither was isolated.
- **Context:** `falkor-chat` K-027 slice A — the run's first act was checking whether FalkorDB was
  up, and the hang nearly bought a wrong diagnosis ("docker is unavailable here") before the
  baseline was even established.
- **Suggested home:** prompt (when an environment probe hangs, re-probe after starting the service
  before concluding the harness blocks the tool — and probe one tool per command so a hang is
  attributable)

## 2026-07-24 — A negative-pin corpus that is uniformly single-line cannot test a *line-anchored* recognition rule — the shipped rule was wider than its docstring and no pin could see it

- **Evidence:** `falkor-chat` K-027 slice A shipped `llm._parse_bare_call_syntax` with a docstring
  claiming recovery "only when the expression owns its lines", backed by 6 negative pins
  (`server/tests/test_llm.py`, pre-fix lines 312/323/332/340/354/362). **All 6 were single-line
  inputs.** The regex `^[ \t]*(ident)[ \t]*\(` with `re.MULTILINE` actually enforces "*some* line
  looks like a call", so three multi-line shapes fired and dispatched a real thread write from an
  *illustrative* call — e.g. ``'You should not do this:\n```python\npost_message({"text": "hi"})\n```\nInstead, ask first.'``
  → `ToolCall(post_message)` with `text=None` (the model's real answer discarded). Found by
  `analyst` at the review gate, not by the suite. Fix = require the last accepted call to be the
  final non-whitespace content; all 8 positive pins survived, all 3 false positives closed.
- **Context:** K-027 slice A analyst-gate fix pass — the gap was invisible to a green 552-test suite
  precisely because every pin exercised the one shape the rule handles correctly.
- **Suggested home:** prompt (under "Cover the edges": when the rule under test is *positional* —
  anchored to line start/end, string boundaries, file start/EOF — the negative corpus must vary the
  dimension the rule is anchored on. A pin corpus uniform in that dimension proves nothing about it.)

## 2026-07-24 — Two extractors with the same name-shape but opposite safety postures need the *consumer's* blast radius, not the parse difficulty, to pick between them

- **Evidence:** same slice, blocker B-1. `llm.extract_json_object` (permissive, first-`{`…last-`}`)
  was reused for both the tool-call path and the guard judge because "a fenced verdict is the same
  defect class as a fenced tool call". It is not: the tool-call consumer re-validates name + schema
  in `executor._handle_tool_call`, so a quoted object is caught; the judge **acts on the object
  directly**, so a *quoted* verdict — `'…I would answer {"decision": true, "rationale": "named"} but
  they did not, so I answer false.'` — **advanced** a guard that the previous bare `json.loads`
  correctly suspended. `guards._coerce_verdict`'s bias-to-suspend cannot catch it: the quoted
  rationale is clean, so `_rationale_contradicts` finds no cue. Split into a conservative
  `extract_own_line_json_object(…, require_key=…)` for the judge.
- **Context:** the same fix pass; the shared helper was introduced as a de-duplication win and was
  the slice's one genuine safety regression against `HEAD`.
- **Suggested home:** prompt or knowledge base (when factoring two callers onto one tolerant parser,
  compare what each caller *does* with the result — a validating consumer and an acting consumer do
  not share a tolerance contract, however identical their inputs look)

## 2026-07-24 — A positional accept-rule anchored on ONE element of a collection is not a rule about the collection, and a pin corpus that ends the way the anchor expects can never show it

- **Evidence:** same K-027 slice, gate finding N-1. `llm._parse_bare_call_syntax` enforced "the
  **last** accepted call must be the final non-whitespace content of the message" and the docs
  (three of them) restated that as "an illustrated call can never be dispatched". But the loop
  appends *every* accepted candidate and `executor._run_agent_node` **dispatches every element** of
  `result.tool_calls` — nothing constrained what sat *between* accepted calls. So
  `'I considered:\n```\nhuman_handoff()\n```\nBut instead:\npost_message({…})'` dispatched **both**,
  and the entire previously-closed false-positive family re-opened the moment the model ended its
  turn with a genuine call — the *common* case, not an edge case. The fix was ~3 lines (reject when
  `text[end_of_last_call:match.start()].strip()` is non-empty), all 8 positive pins unchanged.
  The suite could not see it: all negative pins ended in **prose**, so not one exercised a message
  ending in a genuine call.
- **Context:** second consecutive gate on the same function where the *code* rule was narrower than
  the *documented* rule, and the test corpus was uniform in exactly the dimension that mattered.
- **Suggested home:** prompt (under "Cover the edges" / "One reason to fail per test"): when an
  accept-rule is positional **and** the parser returns a *list*, pin the rule at each position it
  can occupy — first, middle, last — and check what the *consumer* does with the whole list. An
  anchor on the last element says nothing about the others; if the consumer iterates, a rule that
  binds one element is not a safety property, and any doc asserting it as one is false.

## 2026-07-24 — A green pytest exit code is not evidence an integration suite actually ran — read the skip count

- **Evidence:** `falkor-chat/server/tests/conftest.py:54` (`_falkordb_reachable()`) turns the entire
  graph-backed half of the suite into `pytest.skip(...)` when FalkorDB is unreachable, rather than
  failing. `pytest -q` still exits 0 with roughly half the tests silently skipped — the exit code
  alone cannot distinguish "everything passed" from "half the suite never ran."
- **Context:** falkor-chat AGENTS.md doc-restructure audit — the project's own AGENTS.md had a
  standing warning to "always report/read `N passed, M skipped`, never just the absence of
  failures," which is exactly the kind of durable verification habit this inbox exists for.
- **Suggested home:** prompt ("Verify honestly" step: check skip counts, not just exit status,
  whenever a suite has an environment-reachability gate)

## 2026-07-24 — A fixture that wipes shared/global state at *setup* rather than *teardown* leaves that state dirty after the last test in the session

- **Evidence:** `falkor-chat/server/tests/conftest.py`'s `wf_repo` fixture runs
  `MATCH (n) DETACH DELETE n` on the **global** `reference` graph at fixture **setup**, once per
  workflow test (isolating that test from prior ones). Because the wipe never runs at teardown, the
  *last* workflow test to execute in a session leaves its own published defs sitting in `reference`
  afterward — so a subsequent "re-seed" script can print a false "already present — no-op",
  reporting a test's publish as if it were a real seed.
- **Context:** same audit — falkor-chat's docs had to carry a standing warning about this because
  the asymmetry (isolate-via-setup-wipe, no teardown-wipe) is invisible from a passing test run.
- **Suggested home:** prompt (a general fixture-design check: a fixture that wipes/mutates
  **shared/global** state to isolate itself from prior tests should also tear that state down,
  or the last test's leftovers get mistaken for real seeded state by whatever runs next)

## 2026-07-31 — A "have you already produced X" idempotency guard can often reuse an existing output-accumulator list instead of a new flag

- **Evidence:** `falkor-chat/server/falkorchat/executor.py`'s `_run_agent_node` buffers posted
  msgIds onto a per-node-loop `emissions: list[str]` (via `_buffer_emission`, only appended on a
  successful `post_message` dispatch). Implementing K-039's "implicit post_message fallback when
  the model ends on plain text without calling its granted tool" naively as
  `if "post_message" in granted_set and result.text:` double-posted: a node that explicitly called
  `post_message` successfully on an earlier turn and then ended a *later* turn with a plain "done"
  narration (a normal, common shape — confirmed by the pre-existing
  `test_agent_node_captures_posted_msg_ids_as_emissions` regressing) got a *second*,
  implicit post of "done". The fix was `and not emissions` — `emissions` already means "has this
  node posted a message yet in this loop," so no new flag/counter was needed; reusing the
  accumulator that already exists for a different purpose (post-record `PRODUCED` linking) gave
  the idempotency check for free. Running the pre-existing suite after adding new code (not just
  the new tests) is what caught this — the new tests alone were all green.
- **Context:** K-039 immediate mitigation (`docs/reviews/mention-reply-delivery-rca.md`
  suggestion 1) — a bug-fix TDD unit in `falkor-chat/server`.
- **Suggested home:** prompt (a general TDD-loop reminder: after a fix, run the *whole* existing
  suite, not just the new reproduction tests — an idempotency/double-fire bug in a fallback path
  is exactly the kind of thing only an *adjacent, unrelated-looking* pre-existing test catches).
