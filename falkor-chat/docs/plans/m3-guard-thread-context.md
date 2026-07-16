# M3 Defect A — guard thread-context seam (restore the DS RECENT-TURNS fallback)

> **Status:** design/plan, ready to implement. **Author:** architect · **Date:** 2026-07-15
> **Owner (implementer):** tdd-engineer · **Fixes:** Defect A (`m3-executor-coordination.md`,
> 2026-07-15 U14 entry) — the intake→research fuzzy guard can never fire.
> **Restores:** `docs/plans/m3-executor-ml.md` §Q1 (extract-then-judge + **RECENT TURNS fallback,
> N=6**) — a specified design lost in implementation, not a new invention. The DS note's own
> **risk #4** predicted this failure.
> **Unblocks:** AC-2b / AC-3 / AC-4 live; **U15** (qa-engineer acceptance).
> **Graph impact: ZERO** — no DDL, no new query, no `QUERIES.md`/`DESIGN.md`/`bootstrap_schema.sh`
> change. The guard is parsed app-side only (AGENTS.md rule 8). **No graph-dba gate is re-opened.**

---

## 1. Goal & scope

**Goal.** Make the `{"kind":"llm"}` transition guard able to see the conversation it is judging:
wire the executor's already-fetched thread turns through `guards.evaluate_guard` into the injected
judge, and implement the DS note's **RECENT-TURNS (N=6) fallback** alongside the existing
`understanding` extraction, with the DS precedence between them. The currently-RED
`server/tests/test_workflow_live.py` then goes green **because the judge finally has evidence**, not
because a prompt was tuned to say yes.

**In scope**
- `server/falkorchat/guards.py` — recent-turns extraction, precedence, judge kwarg.
- `server/falkorchat/executor.py` — two touch points only: `StepResult` gains a `thread` field;
  `_select_transition` passes `thread=result.thread` instead of the hardcoded `None` (line 566).
- `server/falkorchat/app.py` — `_build_llm_judge` builds the real DS §Q1 prompt (RECENT TURNS block).
- `server/tests/{test_guards.py,test_executor_agent.py,test_app.py}` — the offline pins.
- **Optionally (recommended, §7/OQ-2):** `scripts/seed_workflows.sh` intake `systemPrompt` +
  `m3-executor.md` §8 — the *complement*, not the fix.

**Explicitly out of scope**
- **The §2.1 A/B/C drive loop and `record_step_and_advance` (§12.2 / M4) — untouched.** `_drive_loop`
  is **not edited at all** by this design (see §3, S2 — that is the reason for the chosen seam).
- **m-2 (executor never writes `ctx` back)** — not fixed here; not needed here (§6.1).
- **m-C (unbounded `read_thread` per agent node)** — not fixed here; **not made worse** (§6.2).
- **Defect B** (`_handle_tool_call` / `tools.py` mention validation) — parallel work, stay out (§6.4).
- **Guard-judge calibration** (`golden_guards.jsonl`, κ ≥ 0.6 / false-advance ≤ 10%) — OQ-4.
- **The research→answer transition stays unconditional (D5).** Nothing in this design touches it.

---

## 2. Root cause — verified in source (re-verified by me, 2026-07-15)

The judge is asked to rule on an **empty state**, every single turn, and correctly biases to suspend.
Four links, all verified by reading the files:

1. **`server/falkorchat/executor.py:566`** — `_select_transition` calls
   `evaluate_guard(guard, ctx=run_ctx, run=run, step_output=result.output, thread=None, judge=self._guard_judge)`.
   The `thread` argument is the literal `None`. *(verified: `grep -n "thread=None"` → 566)*
2. **`server/falkorchat/guards.py:74`** — `evaluate_guard` declares `thread: Any` and the parameter
   is **never read** anywhere in the module body. *(verified by reading the whole 159-line file)*
3. **`server/falkorchat/guards.py:135-148`** — `_extract_understanding(step_output, ctx)` reads only
   (a) `step_output` parsed as a JSON object (unwrapping an `understanding` envelope), then
   (b) `ctx["understanding"]`, then returns `{}`. There is no third source.
4. **`server/falkorchat/app.py:268-276`** — the production judge sends only
   `{"CONDITION": …, "UNDERSTANDING": …}`. **No RECENT TURNS block exists.** The system prompt
   (`app.py:244-251`) likewise names only CONDITION + UNDERSTANDING.

**Why both `understanding` sources are guaranteed empty in the shipped flow:**
- The intake `systemPrompt` seeded by `scripts/seed_workflows.sh:96-99` ("Ask the user clarifying
  questions until you can state their request precisely; ask one question at a time.") — seeded
  verbatim from `m3-executor.md` §8 — **never asks the model for an `understanding` object**. The
  node's final text is prose ⇒ `_load_obj(step_output)` → `{}` (guards.py:151-159).
- The executor **never writes `ctx` back** (known **m-2**, `docs/reviews/m3-executor-impl.md`), and
  the trigger seeds `ctx` with run bookkeeping only ⇒ `ctx.get("understanding")` → `None`.

So `_extract_understanding` returns `{}` on every turn, and the judge's entire user message is
`{"CONDITION":"the user has provided enough information to research their request","UNDERSTANDING":{}}`.
`guards._coerce_verdict` then does exactly what it is specified to do (bias-to-suspend). **The engine
is correct; the evidence channel is missing.** The live fingerprint in the U14 record — the judge
saying *"The user has not provided any information to research their request"* on the very turn the
intake node said *"Thank you for providing all the details, Alice"* — is precisely the signature of
an empty `UNDERSTANDING`, not of a bad calibration on a rich transcript.

> **This is a broken contract across three files, not a prompt problem.** The `thread` parameter is a
> *declared seam that lies*: every caller passes it, no callee honors it, and a future def author has
> no way to discover that. That is why option 1 (def-prompt only) was rejected, and the rejection is
> right (§7).

---

## 3. The design

The fix restores the DS §Q1 data flow end to end. **The turns the judge needs have already been read**
— `_run_agent_node` fetches them at `executor.py:422` (`thread_msgs = self._read_thread_context(...)`,
capped at `THREAD_CONTEXT_WINDOW = 20`) and then drops them on the floor. The design **rides that
existing read out on the StepResult** rather than issuing a second one.

```
_run_agent_node ──thread_msgs──> StepResult.thread
                                      │
_drive_loop (UNCHANGED, byte-for-byte)│
   result = self._execute_step(...)   │
   decision = self._select_transition(outgoing, run, run_ctx, result)   ← same call, same args
                                      │
_select_transition ──thread=result.thread──> guards.evaluate_guard
                                                 │
                    understanding = _extract_understanding(step_output, ctx)   (primary, unchanged)
                    recent_turns  = [] if understanding else _recent_turns(thread, n=6)   (DS omit rule)
                                                 │
                    judge(condition, understanding=…, recent_turns=…, ctx=…, step_output=…)
                                                 │
app._build_llm_judge ──> DS §Q1 prompt: CONDITION / CURRENT STATE / RECENT TURNS (context only)
```

### Why this seam (the decisive trade-off)

| Alternative | Verdict |
|---|---|
| **Ride the thread out on `StepResult` (chosen)** | `_drive_loop` and `_select_transition`'s **call site** are untouched → constraint #1 (§2.1 loop locked byte-for-byte) is satisfied *by construction*, not by inspection. Zero extra graph reads → m-C neutral. Additive dataclass field. |
| `_select_transition` fetches the thread itself | Needs `ctx: CallContext` at `_select_transition`, which means **editing the `_select_transition(...)` call line inside `_drive_loop`** → violates the byte-for-byte lock. Also a **second** `read_thread` per step → makes **m-C strictly worse**. Rejected. |
| Pass thread via `run_ctx` | Requires the m-2 ctx-write (a new §12 query → **re-opens the graph-dba gate**) for no benefit. Rejected. |

### S1 — `executor.py`: `StepResult` carries the node's turns

```python
@dataclass(frozen=True)
class StepResult:
    output: str = ""
    on: str = "done"
    trace: list[tuple[str, str]] = field(default_factory=list)
    emissions: list[str] = field(default_factory=list)
    thread: list[dict[str, Any]] = field(default_factory=list)   # NEW
```
Docstring addition (the contract): *"`thread` is the recent thread window the node already read
(`_read_thread_context`) — carried out so the transition guard can judge against the live
conversation (DS §Q1 RECENT-TURNS fallback) **without a second read** (m-C). Empty for the offline
stub path and for non-agent steps; `evaluate_guard` degrades to the `understanding`-only path."*

In `_run_agent_node`, pass `thread=thread_msgs` on **both** return paths — the final-text return
(`executor.py:439-440`) and the graceful `maxIterations`-exhaustion return (`executor.py:456`). The
stub path in `_execute_step` (`executor.py:389`) stays `StepResult(output="", on="done")` — default
`[]`. **No other executor behavior changes.**

### S2 — `executor.py:566`: honor the seam

```python
verdict = evaluate_guard(
    guard, ctx=run_ctx, run=run, step_output=result.output,
    thread=result.thread, judge=self._guard_judge,          # was: thread=None
)
```
One line. `_drive_loop` (lines 330-369) is **not edited**.

### S3 — `guards.py`: recent-turns extraction + the DS precedence

Add module constants and one helper; change `evaluate_guard`'s `llm` branch only.

```python
RECENT_TURNS_N = 6        # DS note §Q1: "RECENT TURNS (context only) … N = 6, newest last"
TURN_TEXT_MAX = 400       # per-turn char cap (rule 6 — a long turn cannot dominate the prompt)
```

```python
def _recent_turns(thread: Any, n: int = RECENT_TURNS_N) -> list[dict[str, str]]:
    """The 'fallback' half of extract-then-judge (DS §Q1): the last `n` thread turns,
    newest last, normalized to a compact {speaker, role, text}. Tolerant by design —
    `None` / non-list / malformed rows → []; the judge must never crash a drive."""
```
- Input rows are `repository.read_thread` shape (verified `repository.py:590-596`):
  `{msgId, text, role, createdAt, authorId, displayName, authorType}` — already **chronological**
  (`ORDER BY m.createdAt`), so `thread[-n:]` **is** "newest last".
- Emit `{"speaker": displayName or authorId or "member", "role": role or "user",
  "text": text[:TURN_TEXT_MAX]}`. Skip rows with no text. Never raise.

In `evaluate_guard`'s `llm` branch (guards.py:96-101):

```python
understanding = _extract_understanding(step_output, ctx)
# DS §Q1: RECENT TURNS is the *fallback* — "omit if understanding is present". Truthiness,
# not presence: an emitted-but-empty {} understanding still falls back to the turns.
recent_turns = [] if understanding else _recent_turns(thread)
raw = judge(
    parsed.get("text", ""),
    understanding=understanding, recent_turns=recent_turns,
    ctx=ctx, step_output=step_output,
)
```

**The precedence lives in `guards.py`, not in the judge** — deliberate: it is a *method* decision
(DS §Q1), it is then unit-testable offline with a stub judge (T1/T2 below), and every judge
implementation inherits it instead of re-deriving it. Judges stay dumb renderers. *(Reversible in one
place if a future judge wants both signals — noted, not built.)*

Update the module docstring (it currently says the judge receives the understanding "*not the raw
transcript*") to state the two-tier reality: **understanding primary; last-6 turns as the DS-specified
degraded fallback when no understanding was emitted.** Update the `Judge` type comment and
`evaluate_guard`'s `thread:` param doc ("the recent thread window carried out of the node on
`StepResult.thread`; `None`/`[]` → the understanding-only path").

`WorkflowExecutor.__init__`'s `guard_judge` docstring (`executor.py:210-212`) names the judge
signature — update it to include `recent_turns`.

### S4 — `app.py`: the real DS §Q1 judge prompt

`_build_llm_judge`'s inner `judge` gains the `recent_turns` kwarg and builds the note's prompt
verbatim in intent:

```
CONDITION: {condition}

CURRENT STATE:
{understanding_json}                 # omitted when empty

RECENT TURNS (context only):         # omitted when empty (DS: "omit if understanding is present")
Alice: The checkout service was returning 502 errors…
Assistant: Which version was deployed?
…                                    # newest last
```
- **System prompt** (`app.py:244-251`) extended: it may be given CURRENT STATE and/or RECENT TURNS;
  judge the CONDITION against whatever it is given; when in doubt answer false; **and (see R-1) the
  `rationale` must state only the evidence supporting the decision** — when advancing, say what the
  user *provided*, not what is absent.
- **Cap:** the assembled user message is capped at `JUDGE_USER_MAX_CHARS = 6000` (≈1500 tokens, the
  DS cap) by dropping **oldest** turns first, then hard-truncating. 6 turns × 400 chars ≈ 2.4k, so the
  cap is a backstop, not a routine path.
- Parse/malformed handling unchanged (`{"decision": False, …}`), and `guards._coerce_verdict` still
  applies bias-to-suspend downstream. **Both layers stay.**

### S5 — (recommended complement, stakeholder call — OQ-2) the intake def prompt

`scripts/seed_workflows.sh:96-99`, intake `systemPrompt` → additionally require the structured
emission the DS *primary* path depends on:

> "Ask the user clarifying questions until you can state their request precisely; ask one question at
> a time. Post questions to the user with `post_message`. When you are done with your turn, reply
> with ONLY this JSON object (it is internal state, not shown to the user):
> `{"understanding": {"request": "<one sentence>", "known": ["<fact>", …], "missing": ["<what you still need>", …]}}`"

This is **not a substitute** for S1–S4 and does not stand alone (§7). Mirror the wording into
`m3-executor.md` §8's intake row in the same change.

---

## 4. Files to change

| File | Change | Done looks like |
|---|---|---|
| `server/falkorchat/guards.py` | `RECENT_TURNS_N`/`TURN_TEXT_MAX`, `_recent_turns()`, precedence + `recent_turns=` in the `llm` branch, docstrings | `thread` is read; a prose `step_output` + a live thread yields a non-empty `recent_turns` to the judge |
| `server/falkorchat/executor.py` | `StepResult.thread` field (+docstring); `thread=thread_msgs` on both `_run_agent_node` returns; **line 566** `thread=result.thread`; `guard_judge` docstring | `_drive_loop` unchanged; `git diff` shows no edit inside lines 330-369 |
| `server/falkorchat/app.py` | `_JUDGE_SYSTEM_PROMPT` + `_build_llm_judge(recent_turns=…)`, prompt assembly, `JUDGE_USER_MAX_CHARS` | the judge's user message contains a RECENT TURNS block when no understanding was emitted |
| `server/tests/test_guards.py` | `StubJudge.__call__` gains `recent_turns`; T1–T6 | offline, network-free |
| `server/tests/test_executor_agent.py` | T7–T8 (the seam pin + the one-read pin) | offline, network-free |
| `server/tests/test_app.py` | T9–T11 (prompt shape/cap, stub LLM) | offline, network-free |
| `scripts/seed_workflows.sh` *(S5, if approved)* | intake `systemPrompt` | re-run is still a clean `already present — no-op`… **see R-4** |
| `docs/plans/m3-executor.md` | §2.5 (+§8 if S5) | §6 doc impact |
| `docs/plans/m3-executor-ml.md` | one-line "implemented at …" pointer | §6 doc impact |

**Not changed:** `repository.py`, `services.py`, `tools.py`, `trigger.py`, `config.py`,
`QUERIES.md`, `DESIGN.md`, `bootstrap_schema.sh`, `AGENTS.md`, `test_workflow_live.py` (§5).

---

## 5. Test strategy

TDD, in this order; the **312-test network-free baseline must be green after every step** (`pytest`
→ `N passed, 1 deselected`; N grows only — no test is lost, no marker is loosened).

### Step 1 — `guards.py` unit pins (RED first, `server/tests/test_guards.py`)

Update `StubJudge.__call__(self, condition, *, understanding, recent_turns, ctx, step_output)` and
record `recent_turns`. Existing calls keep `thread=None` where the thread is irrelevant.

| # | Behavior | Assert |
|---|---|---|
| **T1** | understanding present ⇒ **turns omitted** (DS omit rule) | `step_output='{"understanding":{"request":"r","known":["k"],"missing":[]}}'`, `thread=[10 rows]` → `judge.calls[0]["understanding"] == {...}` **and** `recent_turns == []` |
| **T2** | understanding empty ⇒ **last 6 turns, newest last** | prose `step_output`, `ctx={}`, `thread=[10 rows]` → `len(recent_turns) == 6`, `recent_turns[-1]["text"]` is the newest row's text, `recent_turns[0]` is row #5 |
| **T3** | shape normalization | rows carry `displayName`/`authorId`/`role`/`text` → each turn is exactly `{"speaker","role","text"}`; `displayName=None` falls back to `authorId`, then `"member"` |
| **T4** | per-turn truncation | a 5000-char turn → `len(text) == TURN_TEXT_MAX` |
| **T5** | tolerance (the offline stub path) | `thread=None` / `[]` / `[{}]` / `"garbage"` + empty understanding → `recent_turns == []`, **no raise**, verdict still produced |
| **T6** | emitted-but-empty understanding falls back | `step_output='{"understanding":{}}'`, `thread=[3 rows]` → `recent_turns` has 3 (truthiness, not presence) |
| **T6b** | empty/`None` guard still never calls the judge, whatever the thread | `_boom_judge` + `thread=[rows]` → `decision is True` |

**T-cues (R-1 — load-bearing, do not skip).** `_rationale_contradicts` (guards.py:130-132) substring-
matches `_NEGATION_CUES` against the rationale. With a *transcript-fed* judge the rationales get
wordier, and an **affirmative** rationale can trip a cue and force a false suspend — permanently
re-breaking the flow in a way that looks identical to Defect A. Pin the contract:

| Rationale (judge said `decision:true`) | Expected |
|---|---|
| `"The user provided the service, version and symptom; no more info is needed."` | **advances** (`decision is True`) — today this trips `"more info"` |
| `"Everything is known; nothing is unclear."` | **advances** — today this trips `"unclear"` |
| `"The request is clear and no relevant details are missing."` | **advances** — today this trips `"no relevant"` |
| `"Not enough information yet."` | suspends (unchanged — the real contradiction case) |
| `"The user still needs to provide the version."` | suspends (unchanged) |

Implementer: make these pass with the **minimal** change to the cue matching (the intent of the check
is a backstop against an *internally inconsistent* verdict, not a prose grep). The prompt-side rule in
S4 (rationale states only supporting evidence) is the first line of defense; the cue tightening is the
second. **Do not delete the check** — it is the DS bias-to-suspend policy.

### Step 2 — `executor.py` seam pins (`server/tests/test_executor_agent.py`)

The existing `StubServices` (`test_executor_agent.py:83-91`) already records `read_thread` calls and
returns a scripted transcript — reuse it.

| # | Behavior | Assert |
|---|---|---|
| **T7** | **the Defect-A regression pin.** A `type:'agent'` step with a stub LLM returning **prose**, a scripted 8-turn thread, and an `{"kind":"llm"}` outgoing guard with a recording stub judge | the judge received `recent_turns` of length 6, non-empty text. *(This test fails on today's `thread=None` — it is the pin.)* |
| **T8** | **m-C non-regression:** the guard costs **zero** extra reads | after the drive, `StubServices.calls == 1` for that step (the node's read), **not** 2 |
| **T9** | stub path unaffected | a non-agent step / no wired LLM → `StepResult.thread == []`, guard falls back, no crash |

### Step 3 — `app.py` judge-prompt pins (`server/tests/test_app.py`, stub LLM — no network)

| # | Behavior | Assert |
|---|---|---|
| **T10** | turns rendered when present | user message contains `RECENT TURNS`, the speaker names, newest last |
| **T11** | omitted when understanding present | no `RECENT TURNS` substring |
| **T12** | cap | 50 huge turns → `len(user_message) <= JUDGE_USER_MAX_CHARS`, oldest dropped first, newest turn survives |
| **T13** | malformed reply → `{"decision": False, …}` | existing behavior preserved |

### Step 4 — the live proof (`server/tests/test_workflow_live.py` — **do not weaken**)

`cd server && .venv/bin/python -m pytest -m live -s` (needs FalkorDB + LM Studio).

**Green for the right reason** means all of:
1. `status == "done"` within **`rounds <= 3`** (the DS 3-round clarifying ceiling; the test's
   `MAX_CLARIFY_ROUNDS = 4` is headroom, not the target — **if it only passes at 4, report it**).
2. The `guard_judgment` trace (`_guard_judgments(graph)`) shows a **`True`** verdict whose rationale
   **cites the user's actual facts** (checkout / v4.2 / 502 / rollback). A `True` with a generic
   rationale is *not* the right reason — capture the trace in the report either way.
3. `visited == ["intake", …, "research", "answer"]` and the `PRODUCED` reply exists (AC-3/AC-4).

**Do not edit the test's assertions or markers.** The stale failure-message text (lines 320-321,
which explains the U14 finding) may be reworded to point at this plan — that is a message string, not
a contract. Keep it RED-capable: it is the regression guard.

> **⚠ Sequencing reality (R-3):** the live test **cannot** go green on this fix alone if Defect B
> still reproduces — reaching `research`/`answer` for the first time is exactly what exercises the
> unguarded `dispatch` at `executor.py:478`. **Land Defect B first, or verify the live run only once
> both are in.** A live RED after this fix must be triaged against the trace before it is called a
> Defect-A regression.

### Step 5 — baselines
- `cd server && .venv/bin/python -m pytest -q` → green, `1 deselected`.
- `./scripts/test_queries.sh` → **241/241** (untouched by design — zero graph change; run it to prove
  it, and note `test_queries.sh` **deletes the `reference` graph**, so re-run `seed_workflows.sh`
  before any live run — the recovery gotcha in the U14 coordination entry).

---

## 6. Explicit positions on the adjacent items

### 6.1 m-2 (`ctx` never written back) — **not fixed, and not needed here. Justified:**

`_extract_understanding` reads `step_output` **first** (guards.py:143-146), and `step_output` is
`result.output` from the node that **just executed in this same `_drive_loop` iteration**
(`executor.py:333` → `336`). It is always fresh, in-memory, and never round-trips through the graph —
including across suspend/resume, because a resumed run **re-executes** intake and produces a new
output before the guard is re-evaluated (§2.4). So the `understanding` primary path works with m-2
open. Fixing m-2 would need a ctx-write query (**there is none** — verified: `repository.py` has
`record_step_and_advance` (1106) and `fail_run` (1205); no general run-ctx update), i.e. a new §12
query ⇒ **re-opens the graph-dba gate** for zero benefit to this defect. **Leave it open.**

m-2 remains a real prerequisite for two *other* things — the DS 3-round ceiling (`m3-executor.md` §7,
`m3-executor-landing2.md` §U11.2) and OQ-5 below. Both stay tracked with the m-2 work.

### 6.2 m-C (unbounded `read_thread` per agent node) — **not fixed; neutral by construction**

This design issues **zero** additional reads: it reuses the list `_run_agent_node` already fetched.
T8 pins that. The rejected alternative (guard reads for itself) would have **doubled** the reads per
step — that is the main reason it was rejected, and it is a conscious choice, not an oversight. m-C's
own fix (bound the read at the query, a since/last-N thread read) stays with the K-015 / scale work;
when it lands, this design **inherits** it for free — the guard consumes whatever window the node got.
One documented consequence: a `{"kind":"llm"}` guard on a **non-agent** step gets `thread == []` and
degrades to the understanding-only path. Correct for this cut (the only llm guard is on intake); the
`_recent_turns` tolerance (T5) makes it safe, and the `StepResult.thread` docstring says so.

### 6.3 The def-prompt change (S5) — **a genuine complement; I recommend including it** (OQ-2)

Not a substitute, and it does not smuggle option 1 in:
- **The seam fix alone leaves the shipped triage flow permanently on the DS's *degraded* path.** The
  DS note is unambiguous: extract-then-judge is "the single highest-leverage decision"; judging turns
  directly is the risk-#4 **fallback**, "degraded but functional… expect lower κ." Shipping only
  S1–S4 means the one def we have never reaches the primary path — we'd have restored the fallback and
  left the primary dead.
- **S5 alone would be the rejected option 1** — a dead seam for every future def, plus total
  dependence on a 4B's structured emission, which the DS says to *measure*, not assume (Q3/risk #4).
- **Together they are exactly the DS design:** primary path when the model complies, DS-specified
  fallback when it doesn't. That is belt-and-braces in the load-bearing sense — each covers the
  other's failure mode.
- **Cost is trivial and blast radius nil:** `output` is consumed **only** by the guard and the
  `node_rationale` trace — verified: `_record` passes `output=result.output` into `StepRun.output`
  (executor.py:589) and `_assemble_messages` (executor.py:513-534) folds in **only** `systemPrompt` +
  thread turns + `run_ctx`. **No downstream node reads a prior step's `output`.** So making intake's
  final text JSON breaks nothing.

**However — sequence it to keep the evidence clean:** land and live-verify **S1–S4 first**, with the
def untouched. That run proves the *seam* fix on the fallback path (the strongest possible evidence
that the engine is fixed). Then land S5 and re-run to confirm it moves onto the primary path
(`recent_turns == []` in the trace, understanding populated). If S5 landed first, a green live run
would not tell us which change did it — the exact "green for the wrong reason" trap.

### 6.4 Defect B — no interaction, but a sequencing dependency

Disjoint regions: B is `_handle_tool_call` (executor.py:458-482) + `tools.py` mention validation; this
is `StepResult`, `_run_agent_node`'s two returns, line 566, `guards.py`, `app.py`. The judge dispatches
no tools, so nothing this design adds can induce a hallucinated mention. **But** see R-3: this fix is
what first drives the flow *through* B's blast radius, so the live green depends on B.

---

## 7. Why option 2 is the right call (and why I'd still add the complement)

I was asked to say plainly if the stakeholder's choice is wrong. **It is not — option 2 is correct**,
and the stated rationale is exactly right on the evidence: the `thread` parameter is a declared seam
that no callee honors (guards.py:74, verified), so every future `{"kind":"llm"}` guard would inherit a
silent lie, and a prompt-only fix would turn the pin green while the seam stayed broken. My one
amendment is §6.3: on the evidence in the DS note, the def prompt is a **complement**, not a
substitute — landed *second*, after the seam fix is independently proven live. That is option 2 plus
the belt, sequenced so the braces are provably doing the work. **The stakeholder's call to make
(OQ-2); S1–S4 stand alone if they decline.**

---

## 8. Risks

- **R-1 · `_NEGATION_CUES` false-positives on a richer rationale — HIGH, and it looks exactly like
  Defect A.** A transcript-fed judge writes longer rationales; `"…no more info is needed"` contains
  `"more info"` → `_rationale_contradicts` → forced suspend on a correct advance. **Mitigation:**
  T-cues (§5 Step 1) + the S4 prompt rule. This is the single most likely way the fix "doesn't work."
- **R-2 · False-advance (the *dangerous* error, DS Q1) rises on the fallback path.** Judging a
  transcript is lower-κ than judging an understanding. Mitigations: bias-to-suspend + `_coerce_verdict`
  are untouched; S5 moves off the fallback; the real answer is OQ-4 (calibration).
- **R-3 · The live test cannot go green without Defect B.** See §5 Step 4 / §6.4. Triage a live RED
  against the trace before blaming this fix.
- **R-4 · `seed_workflows.sh` idempotence vs. a *changed* def (only if S5 lands).** The def is
  published/materialized by **constraint-backed MERGE, immutable per version** (script header:
  "idempotent by construction… immutable per version"). A workspace that already has `triage@v1` will
  therefore likely keep the **old** systemPrompt — a re-run reports `already present — no-op`.
  **The implementer must verify this and, if so, state the operator step in the plan's exit note**
  (drop/re-materialize `triage@v1`, or bump to `v2` + `config.TRIGGER_DEF_{KEY,VERSION}`). `ws:live`
  is unaffected (the test drops the graph each run) — this is a **`ws:acme` / demo-env** hazard and a
  U15 precondition. Do **not** silently mutate a published def.
- **R-5 · Prompt budget.** 6 × 400 chars ≈ 2.4k, well inside the 6000-char (≈1500-token) DS cap. The
  20-turn node window is unchanged. No RAM/graph impact (rule 6): nothing new is stored.
- **R-6 · Judge-signature change is breaking for any out-of-tree judge.** Only two implementations
  exist (`app._build_llm_judge`, test stubs), both updated here. `Judge = Callable[..., Any]` already
  types it loosely.

---

## 9. Open questions (stakeholder / coordinator calls — my recommendation on each)

- **OQ-1 · Tighten `_NEGATION_CUES`?** → **Recommend yes, test-driven and minimal** (T-cues pins the
  contract; the implementer chooses the smallest change that satisfies it). Deleting the check is
  wrong — it is the DS bias-to-suspend policy.
- **OQ-2 · Include S5 (the intake def prompt) in this change?** → **Recommend yes, landed *second*
  and separately verified** (§6.3, §7). Declining still leaves S1–S4 a complete, correct fix — on the
  DS's degraded path.
- **OQ-3 · Make the guard `text` more concrete** (DS Q1's first remediation lever: enumerate the
  fields "enough information" means)? → **Recommend not now.** Changing the condition *and* the
  evidence channel at once destroys the attribution. Hold it as the lever if the live run still
  under-advances at 3 rounds.
- **OQ-4 · The judge is still wired live UNCALIBRATED.** `golden_guards.jsonl` (20–30 hand-labeled
  cases, κ ≥ 0.6 / false-advance ≤ 10%) was **never built** — DS risk #1 calls it "the executor's
  reliability gate," and the note says *do not wire an uncalibrated judge*. This fix does not change
  that; it makes it visible. → **Recommend: out of scope here; raise as a K-item at M3 close and
  decide explicitly whether U15 may pass without it.** A one-line stakeholder call, not the
  implementer's.
- **OQ-5 · NEW FINDING — the `answer` node never sees the research findings.** Verified:
  `_assemble_messages` (executor.py:513-534) folds in only `systemPrompt` + thread turns + `run_ctx`;
  `run_ctx` never changes (m-2); the research node's `tools` are `graphrag_retrieve` only, so it posts
  nothing to the thread; its findings live **solely** in `StepRun.output`, which nothing reads. The
  answer node's prompt ("grounded in the research findings… cite what you used") therefore has no
  findings to ground in — **AC-4 can pass structurally while being ungrounded in substance.** Not
  caught by the live test (it asserts a `PRODUCED` reply, not its provenance). → **Recommend: a
  separate follow-up with the m-2 ctx-write work** (needs a run-ctx update query ⇒ graph-dba gate).
  Out of scope here; flagged for teco/QA so U15 judges AC-4 with eyes open.

---

## 10. Documentation impact (the implementer's done-condition)

**Changed by this design:**
- **`docs/plans/m3-executor.md` §2.5** — the guard-evaluation section must state the two-tier evidence
  contract (understanding primary; last-6 turns fallback; the omit rule; the thread rides out on
  `StepResult.thread` and costs no extra read). §2.5 is currently silent on where the guard's evidence
  comes from — that silence is what let the seam ship broken. **D5 wording stays untouched.**
- **`docs/plans/m3-executor.md` §8** *(only if S5 lands)* — the intake row's systemPrompt cell.
- **`docs/plans/m3-executor-ml.md`** — a one-line dated pointer under §Q1 that the RECENT-TURNS
  fallback is implemented at `guards._recent_turns` + `app._build_llm_judge` (and that risk #4
  materialized as Defect A). The note's *method* is unchanged — do not rewrite it; it was right.

**Explicitly NOT changed (state this in the handoff):**
- `AGENTS.md` — the `seed_workflows.sh` row points at §8 rather than quoting the prompt, so it stays
  true either way. No new live-verified FalkorDB fact.
- `docs/DESIGN.md`, `docs/QUERIES.md`, `scripts/bootstrap_schema.sh` — **zero** graph/DDL/query change
  (guards are app-side, rule 8). No graph-dba gate.
- `docs/HISTORY.md` / `docs/BACKLOG.md` — **not this implementer's** (teco owns the rollup at close);
  the coordination log entry (`m3-executor-coordination.md`) is likewise teco's.
- `docs/reviews/*` — analyst's.

---

## Ready to implement — summary

**Plan:** `falkor-chat/docs/plans/m3-guard-thread-context.md`

Defect A is a **broken contract across three files**, not a prompt problem: `executor.py:566` hardcodes
`thread=None`, `guards.py:74` declares `thread` and never reads it, and `app.py`'s judge prompt has no
RECENT TURNS block — so with an intake prompt that never emits an `understanding`, the judge is handed
`UNDERSTANDING: {}` every turn and correctly biases to suspend forever.

**The fix (≈40 lines, additive, no graph change, no locked-code change):** the thread turns
`_run_agent_node` **already reads** ride out on a new `StepResult.thread` field; `_select_transition`
passes `thread=result.thread`; `guards.py` gains `_recent_turns(thread, n=6)` and the DS precedence
(*understanding primary; turns only when it's empty*) and hands `recent_turns=` to the judge;
`app._build_llm_judge` builds the DS §Q1 prompt with a capped RECENT TURNS block. **`_drive_loop` is
not edited at all** — the §2.1 A/B/C loop and `record_step_and_advance` are untouched by construction,
and the seam costs **zero extra graph reads** (m-C neutral, pinned by a test).

**Order:** guards unit pins (incl. the **`_NEGATION_CUES` false-positive contract — R-1, the most
likely way this fix silently fails**) → executor seam pin (`thread=None` regression) → judge-prompt
pins → live `-m live` verification (needs **Defect B** landed first) → optionally the intake def
prompt (§6.3) landed **second** and re-verified. Baseline: **312 + 1 deselected** green throughout;
`test_queries.sh` **241/241** untouched.

**Read before starting:** §2 (root cause), §5 T-cues, §6 (m-2 / m-C / S5 / Defect B), §9 OQ-5.
