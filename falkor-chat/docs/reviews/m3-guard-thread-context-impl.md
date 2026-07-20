# Review — K-022 Landing 2, guard-seam (Defect A) + tool-error survival (Defect B) implementation

> **Reviewer:** analyst · **Date:** 2026-07-19 · **Gate:** mandatory analyst implementation gate
> (blocks U15 / qa-engineer acceptance).
> **Target:** commit **`aa8b813`** ("fix(falkor-chat): K-022 Landing 2 — guard-seam (Defect A) +
> tool-error survival (Defect B)"), diffed against its parent **`c3cc239`**.
> **Baselines read:** `falkor-chat/AGENTS.md`, `docs/plans/m3-executor.md` (§2.1/§2.2/§2.5/§7/§8),
> `docs/plans/m3-guard-thread-context.md` (the Defect-A design, S1–S5), `docs/plans/m3-executor-ml.md`
> (§Q1), `docs/reviews/m3-executor-impl.md`, `docs/reviews/m3-executor-landing2-impl.md`,
> `docs/plans/m3-executor-coordination.md` (2026-07-15+ entries, treated as unverified claims).
> **Out of scope (per brief, not reviewed):** uncommitted working-tree changes (`AGENTS.md`,
> `docs/*`, `scripts/seed_workflows.sh`); live/model behavior (`-m live`, qa-engineer's U15);
> everything at or before `c3cc239`.

---

## Verdict: **approve with suggestions**

**0 blocker · 2 major · 3 minor · 3 nit.**

The engine fix is correct, minimal, and conforms to the S1–S4 design. Defect A is genuinely closed at
the seam (not papered over by a prompt), Defect B's catch is provably narrow, the locked §2.1 A/B/C
drive loop is **byte-identical** (hash-verified, not eyeballed), and the commit touches **zero**
graph/DDL/query surface. Both majors are adjacent-surface issues — a doc/def divergence that will
mislead U15, and a new silent-failure mode inside the (otherwise correct) tool-error net — not
defects in the fix itself. Neither needs a redesign; both are small, targeted follow-ups. I recommend
closing **M-1 (doc)** *before* U15 starts, since qa-engineer will read `m3-executor.md` §8 as the
def spec.

---

## Mandatory confirmations

### 1. `_drive_loop` byte-identity — **CONFIRMED, mechanically**

Extracted the function body by AST line range from `514346b`, `c3cc239` and `aa8b813` and hashed it:

| Revision | `_drive_loop` SHA-256 (12) | Bytes |
|---|---|---|
| `514346b` | `71055f756280` | 2844 |
| `c3cc239` | `71055f756280` | 2844 |
| `aa8b813` | `71055f756280` | 2844 |

> *Editorial correction added at the K-022 Landing-2 doc rollup (2026-07-19), not by the reviewer:*
> **the byte figure is wrong — the extraction yielding this hash is 2860 bytes** (an earlier
> coordination entry quotes a third figure, 2839). The **SHA is correct and reproducible** and is the
> only reliable way to verify the lock; a future gate checking the byte count would wrongly report the
> lock broken. Tracked as a carried finding under **K-027** in `docs/BACKLOG.md`. The finding this
> table supports — `_drive_loop` unchanged across the three revisions — is unaffected.

**Nothing changed** — same hash, same length; only its line offset moved (310→324) because of the
additions above it. `_record` is likewise byte-identical (`3e2a1d94638c`), and
`repository.record_step_and_advance` is not in the diff at all (`git show --stat` lists no
`repository.py`). The permissible-under-the-plan question is therefore moot: `m3-guard-thread-context.md`
§3/§4 required "`_drive_loop` is **not edited at all**", and it wasn't. The chosen seam
(`StepResult.thread`) achieves this **by construction** — `_select_transition`'s call site inside the
loop (`executor.py:348-350`) is unchanged; only the callee's body reads the new field.

Also verified: `_drive` (the M-1 fault net) is **code-identical** — AST-equal after stripping
docstrings; the only change is the docstring paragraph at `executor.py:302-306`. `_run_agent_node`,
`_select_transition` and `_handle_tool_call` did change (expected, they are S1/S2/Defect-B).

### 2. Defect-A seam fix (S1–S4) complete and correct — **CONFIRMED**

The full chain exists in source:

- **S1** — `executor.py:106` adds `thread: list[dict[str, Any]] = field(default_factory=list)` to the
  frozen `StepResult`, documented at `executor.py:96-100`. Both `_run_agent_node` exits carry it: the
  final-text return (`executor.py:457-459`) and the `maxIterations`-exhaustion return
  (`executor.py:475-476`). The offline stub path (`executor.py:403`) keeps the `[]` default.
- **S2** — the previously-broken call site is fixed: `executor.py:607` now reads
  `thread=result.thread` (was the literal `thread=None`). `grep -n "thread=None"` over
  `server/falkorchat/` returns nothing.
- **S3 / DS precedence** — `guards.py:136`:
  `recent_turns = [] if understanding else _recent_turns(thread)`. This is the DS omit rule on
  **truthiness**, so an emitted-but-empty `{"understanding":{}}` still falls back to turns (pinned by
  `tests/test_guards.py` T6). `_recent_turns` (`guards.py:195-218`) takes `thread[-6:]`
  (`RECENT_TURNS_N = 6`, `guards.py:39`), normalizes to `{speaker, role, text}`, caps each turn at
  `TURN_TEXT_MAX = 400`, and is total (non-list / non-Mapping rows / empty text → skipped, never
  raises).
- **S4** — `app.py:_render_judge_user` (`app.py:259-289`) renders `CONDITION` / `CURRENT STATE` /
  `RECENT TURNS (context only)`, omitting empty blocks, capped at `JUDGE_USER_MAX_CHARS = 6000` by
  evicting **oldest** turns first; `_JUDGE_SYSTEM_PROMPT` (`app.py:244-256`) was rewritten to describe
  the two tiers **and** carries the R-1 prompt rule ("the rationale must state ONLY the evidence
  supporting your decision … never mention what is absent").
- **No extra read (m-C neutral) — CONFIRMED.** `_read_thread_context` is called exactly once per
  agent node (`executor.py:440`); `guards.evaluate_guard` and `_select_transition` issue no read of
  any kind — `guards.py` imports only `json`/`collections.abc`/`dataclasses`/`typing` and holds no
  repository or services handle. `tests/test_executor_agent.py`
  (`test_agent_node_carries_its_thread_window_out_on_the_step_result`) asserts
  `svc.read_calls == ["t1"]` — exactly one read, which is the T8 pin the design asked for.
- **Signature migration is complete** — every judge implementation and stub takes `recent_turns`
  (`app.py:308`, `tests/test_guards.py:39`, `tests/test_executor.py:37`); no call site passes the old
  4-kwarg shape (grep over `server/`).

### 3. R-1 — the negation-cue trap — **CONFIRMED addressed, with a residual gap (see minor m-1)**

Matching is no longer a bare substring test. `_rationale_contradicts` (`guards.py:171-186`) scans
**every** occurrence of each cue and treats a cue as a contradiction only when `_is_negated`
(`guards.py:189-192`) finds no negator in the 12 characters immediately preceding it. `"no relevant"`
was removed from `_NEGATION_CUES` with a stated reason (it embeds its own negator and is
unresolvable), and `"still need"` was added to keep coverage.

The plan's rationale×expected contract (`m3-guard-thread-context.md` §5 T-cues) is honored and
**both directions are pinned**: `tests/test_guards.py` `ADVANCING_RATIONALES` (4 cases, incl. all
three from the plan's table) assert `decision is True` *and* that the rationale survives verbatim;
`SUSPENDING_RATIONALES` (6 cases, incl. the plan's two) assert `decision is False`. I re-ran them —
green. The prompt-side rule (first line of defense) is in place, and `_coerce_verdict`'s
bias-to-suspend policy is untouched.

Residual: the 12-char window is a heuristic and, as written, its failures fall on the *unsafe*
(false-advance) side, contradicting the comment at `guards.py:65-68`. See **m-1** — minor, because
the backstop only engages on an already-inconsistent verdict.

### 4. Defect-B tool-error survival is narrow, not blanket — **CONFIRMED**

Verified from the live class objects, not from reading:

```
HumanHandoffSignal MRO: ['HumanHandoffSignal', 'Exception', 'BaseException', 'object']
issubclass(HumanHandoffSignal, ServiceError) -> False
ServiceError subclasses -> ChannelNotFoundError, InvalidSearchQueryError,
                           ThreadNotFoundError, UnknownActorError, UnknownMemberError
repository errors (WorkflowRunNotFoundError, StepBudgetExceededError,
  EmbeddingDimensionError, MemberIdCollisionError, WorkflowDefSpecError,
  WorkflowDefNotFoundError) -> all False (plain Exception)
```

- **`HumanHandoffSignal` cannot be swallowed** (`tools.py:308`, `class HumanHandoffSignal(Exception)`);
  it is disjoint from `ServiceError`, so it passes straight through `_handle_tool_call`'s
  `except ServiceError` (`executor.py:501`) to `_drive`'s suspend branch (`executor.py:315-319`).
  Pinned end-to-end by `tests/test_executor_agent.py::test_human_handoff_signal_escapes_the_tool_loop_to_the_suspend_path`
  (raised *through* `dispatch`, i.e. exactly the path the catch wraps).
- **The catch is narrow**, not `except Exception`: repository/driver faults and `redis` errors are not
  `ServiceError` subclasses, so they still reach the M-1 net. Pinned by
  `test_engine_fault_in_a_tool_still_escapes_to_the_m1_net`.
- **The M-1 net is intact** — `_drive` is code-identical to `514346b` (confirmation 1); the
  `HumanHandoffSignal`-before-generic ordering, the `fail_with_note` + re-raise, and `AT_STEP` clearing
  are unchanged. No new path can leave a `running` zombie: the only newly-absorbed exceptions are
  absorbed *inside* the node and turned into an ordinary tool message, after which the drive continues
  normally.
- **Bounded** — each re-prompt costs one iteration (`test_repeated_tool_errors_are_bounded_by_max_iterations`
  asserts exactly `maxIterations` dispatches then a graceful `node_note` exit). No spin risk.
- `tools.PostMessageTool.run` (`tools.py:218-231`) catches only `UnknownMemberError` and returns an
  actionable `error:` string naming the id-vs-displayName confusion; anything else propagates
  (`tests/test_tools.py::test_post_message_lets_unrelated_service_errors_propagate`).

Caveat on breadth: see **M-2** — the catch is narrow w.r.t. control flow and engine faults, but it is
*wider than the class it documents* (it also absorbs non-model-correctable `ServiceError`s), and it
logs nothing.

### 5. No graph / DDL / QUERIES change — **CONFIRMED**

`git show --stat aa8b813 -- falkor-chat/docs/QUERIES.md falkor-chat/docs/DESIGN.md falkor-chat/scripts/`
returns **empty**. The 20 changed files are: 3 kaizen inboxes, `falkor-chat/AGENTS.md`, 5 plan docs, 5
`server/falkorchat/*.py` (`app`, `executor`, `guards`, `tools`) + `pyproject.toml`, 7 test files, and
`tests/eval/golden_guards.jsonl`. `repository.py`, `services.py`, `trigger.py`, `config.py`, `mcp.py`
are untouched. Guards remain parsed app-side only (AGENTS.md rule 8), no new node type, index, or
stored data (rule 6). **No graph-dba gate is re-opened; the 241/241 query suite result stands on the
diff alone.**

---

## Verification actually run (and what I could not run)

| Check | Result |
|---|---|
| `cd server && .venv/bin/python -m pytest -q` | **171 passed, 177 skipped, 1 deselected** in 8.2s. **FalkorDB is down** (`docker ps` shows no containers), so all graph-backed tests self-skipped. Collected total = 171+177+1 = **349**, exactly consistent with the claimed "348 passed, 1 deselected". The 1 deselected is `test_workflow_live.py`'s live marker — correct, not a gap. |
| Affected files only (`test_guards.py test_executor_agent.py test_executor.py test_app.py`) | **61 passed, 17 skipped**. All new offline pins (guards T1–T6b + T-cues, executor-agent Defect A/B, app T10–T13, tools) ran and passed. |
| `./scripts/test_queries.sh` | **Not run, deliberately.** FalkorDB is down and the suite drops the global `reference` graph (AGENTS.md), which would disturb the concurrent unit's live-def investigation. Given confirmation 5 (zero graph surface in the diff), reading the diff is sufficient evidence. |
| Source drift | `git diff aa8b813 HEAD -- falkor-chat/` is **empty** — the reviewed source is what is on disk; the only working-tree deltas are the docs + `seed_workflows.sh` I was told to ignore. |

**Verification gap to note for U15:** because FalkorDB was down, the *drive-level* Defect-B regression
pin `tests/test_executor.py::test_hallucinated_mention_does_not_fail_the_run` (uses the `wf_repo`
fixture) did **not** execute in my run. Its node-level twin
(`test_executor_agent.py::test_tool_level_error_is_reprompted_not_raised`) did, and I verified the
exception hierarchy independently, so I have no doubt about the behavior — but the full 348-green
claim remains unverified by me. Re-run `pytest -q` with FalkorDB up before U15 sign-off.

---

## Findings

### Major

**M-1 · `m3-executor.md` §8 now documents an intake/answer def that the shipped `seed_workflows.sh`
does not contain — the spec QA will read at U15 describes neither the committed nor the current
artifact.**
Evidence: the diff rewrites §8's intake row to *"end each turn with ONLY the
`{"understanding":{request,known,missing}}` JSON object (D8/S5 …)"* and the answer row to *"**MUST**
deliver the grounded answer by calling `post_message` … **never pass `mentions`**"*
(`docs/plans/m3-executor.md` §8, plus the new D8/S5 and Defect-C block beneath it). But
`git show aa8b813:falkor-chat/scripts/seed_workflows.sh` lines 96-99 still seeds the *old* intake
prompt (`"Ask the user clarifying questions until you can state their request precisely; ask one
question at a time."`) and lines 118-124 the old answer prompt — the commit message itself says the
`seed_workflows.sh` work is "NOT included". Why it matters: §8 is the def spec. A QA engineer taking
AC-2/AC-4 from §8 will assume the `understanding` **primary** evidence tier is live, when the shipped
def can only ever exercise the **degraded RECENT-TURNS fallback** — which is precisely the
attribution trap `m3-guard-thread-context.md` §6.3 sequenced around ("land S1–S4 first, with the def
untouched"). The code follows the plan; the doc got ahead of it. Suggested fix: mark §8's intake/answer
cells as **proposed (D8/S5, not yet seeded)** with a pointer to the unit that lands them, or land the
script change — either way reconcile §8 with `scripts/seed_workflows.sh` in one change *before* U15
opens. (Note: the concurrent uncommitted work removes the understanding-JSON instruction from the
script with a "do not re-add" comment, which makes §8 *more* wrong, not less — whoever owns that unit
should own the §8 reconciliation too.)

**M-2 · The `ServiceError` catch absorbs failures the model cannot correct, and logs nothing — a new
silent-degradation mode.** Evidence: `executor.py:499-517` catches all of `ServiceError`, whose
subclasses include `UnknownActorError` (raised at `services.py:226` / `:260` when the *context actor*
— i.e. `FALKORCHAT_AGENT_ID` — resolves to no member) and `ThreadNotFoundError`
(`services.py:217`/`:254`). Neither is a model *argument* problem: `post_message`'s thread comes from
`run["ctx"]`, and the actor comes from deployment config. With a misconfigured agent id, every
`post_message` now fails, burns `maxIterations` re-prompts, and the node exits **gracefully** — the run
can reach `done` having posted nothing at all, which is exactly the AC-4 failure signature. Before this
commit it failed loudly. The only record is `trace.append(("tool_result", "ERROR: …"))`, and
`_trace_step` uses `_NULL_TRACER` unless `run["trace"]` is set (`executor.py:339`), so on a normal
(non-debug) run **the diagnostic disappears entirely** — there is no `_log` call on this path.
Suggested fix, in order of value: (a) add `_log.warning("tool %s failed: %r", call.name, exc)` in the
except block — unconditional, not tracer-gated, one line; and (b) consider letting the
non-model-correctable subset propagate, e.g. `except ServiceError as exc: if isinstance(exc,
(UnknownActorError, ThreadNotFoundError)): raise`, or invert it to catch only the argument-level
errors (`UnknownMemberError`, `InvalidSearchQueryError`). (b) is a judgment call worth stating
explicitly in `m3-executor.md` §2.2 either way; (a) I would not ship U15 without.

### Minor

**m-1 · The R-1 negator window leaks across clause boundaries, and its failures land on the *unsafe*
side — the opposite of what the code comment claims.** Evidence: `guards.py:65-68` states the 12-char
window is "deliberately too narrow to span a clause boundary … erring narrow keeps the failure on the
safe (over-suspend) side", citing *"did not provide the version; more info is needed"* as the case it
handles. It handles that one — but only because the intervening text happens to exceed 12 chars. I
probed `_rationale_contradicts` directly:

| Rationale (judge said `decision:true`) | Should contradict | Actual |
|---|---|---|
| `"The user did not say; more info is needed."` | yes | **False (missed)** |
| `"Alice gave no version; still need the logs."` | yes | **False (missed)** |
| `"She said 'I do not know'; more info is needed."` | yes | **False (missed)** |
| `"The user did not provide the version; more info is needed."` | yes | True ✓ |
| `"No version was given; unclear which release."` | yes | True ✓ |

A missed contradiction is a **false advance** — the dangerous direction under DS Q1, and the direction
the comment says cannot happen. Stakes are limited (the backstop only fires on an already-inconsistent
verdict, and the prompt rule is the primary defense), hence minor, not major. Suggested fix: require
the negator and the cue to be in the same clause — scan the window and reject it if it contains
`[;,.:]` or a conjunction, e.g.
`re.search(r"\b(no|not|nothing|never|n't)\b[^;,.:]{0,12}$", window)` — and add the three rows above to
`SUSPENDING_RATIONALES` so the direction is pinned (today only the safe direction is). At minimum,
correct the comment: the heuristic's residual failure mode is false-advance, not over-suspend.

**m-2 · `_recent_turns` slices before filtering, so malformed rows shrink the evidence window.**
Evidence: `guards.py:207` iterates `thread[-n:]` and *then* skips non-Mapping / empty-text rows
(`:208-212`). A window whose newest 6 rows include, say, 3 empty-text system rows yields 3 turns to the
judge even though older valid turns exist — less evidence exactly when the judge is on its degraded
tier. Suggested fix: filter first, slice second (`valid = [r for r in thread if …][-n:]`); the existing
T5 tolerance tests still pass unchanged.

**m-3 · The judge's evidence tier is invisible in the trace.** Evidence: `_select_transition`
(`executor.py:610-611`) traces `(transition, guard_text, verdict)` only. Nothing records whether a given
judgment ran on the `understanding` (primary) or on `recent_turns` (fallback) — yet that distinction is
the whole point of `m3-guard-thread-context.md` §6.3's sequencing, and `m3-guard-calibration.md` will
need it to stratify results. Suggested fix: have `evaluate_guard` return the tier on `GuardVerdict`
(e.g. `tier: str = ""`, `"understanding"` / `"turns"` / `"none"`) and fold it into the
`guard_judgment` trace payload — additive, no graph change (the payload is already an opaque string).

### Nit

**n-1 ·** Function-local `import json as _json` in both `_render_judge_user` (`app.py:270`) and
`_build_llm_judge` (`app.py:305`); `app.py` has no module-level `json` import and every other module
imports it at the top. Move it up unless the deferral is deliberate (it doesn't appear to be — `json`
is stdlib and cheap).

**n-2 ·** `_render_judge_user`'s cap loop (`app.py:279-287`) re-joins the whole message on every
eviction — O(n²) in turn count. Irrelevant at N=6, but `tests/test_app.py::test_judge_prompt_is_capped_by_dropping_the_oldest_turns_first`
drives it with 50 turns, so the shape is reachable. A running length or a reverse-accumulate pass would
be linear.

**n-3 ·** `tests/eval/golden_guards.jsonl` (26 rows: 15 `expected:false` / 11 `true`, 19 with an
`understanding`, 7 turns-only) is well-formed and correctly labeled, but no test or script reads it yet
(`grep golden_guards tests/` → nothing). That matches the stated plan (the calibration *run* is a later
unit), so this is a bookmark, not a defect: make sure the K-item that consumes it is on the backlog so
the file doesn't rot.

---

## What's solid

- **The root fix is at the seam, not in a prompt.** The three broken links named in
  `m3-guard-thread-context.md` §2 (`thread=None`, `thread` never read, no RECENT TURNS block) are each
  closed in the file where they lived, and the "declared seam that lies" is gone: `thread` now has
  exactly one producer and one consumer, both documented at the contract.
- **The seam choice earns its keep.** Riding the already-read window out on `StepResult` satisfies the
  §2.1 byte-for-byte lock *by construction* and costs zero extra graph reads — and the m-C-neutral
  claim is pinned by an assertion on the stub's read count, not left to inspection.
- **The precedence lives in `guards`, not in the judge**, exactly as §S3 argued; the judge is
  demonstrably a dumb renderer (`test_judge_prompt_omits_recent_turns_when_an_understanding_is_present`
  passes `[]` and gets no block), so a future judge inherits the method decision.
- **Defect B's boundary is argued *and* proven.** Three tests drive the three exception classes
  (`ServiceError` → re-prompt, `RuntimeError` → M-1, `HumanHandoffSignal` → suspend) *through*
  `dispatch`, which is the only way to prove the catch's placement rather than its wording. The
  bounded-by-`maxIterations` claim has its own test.
- **Both defense layers were kept.** The prompt rule (rationale states only supporting evidence) and
  `_coerce_verdict`'s bias-to-suspend are both present; the cue check was tightened, not deleted, as
  OQ-1 required.
- **Prior gates hold.** M-1 fault net, m-3's named `WorkflowConfigError`, n-2's null-guard net, AC-6's
  reject-before-dispatch ordering, and the Option-B deferred `emissions`/`trace` lifecycle are all
  intact — I re-read each; none is touched by this diff (the ungranted/malformed checks still run
  before `dispatch`, and the new except path returns before `_buffer_emission`, correctly, since a
  failed post has no msgId).
- **Docs mostly did their job.** `m3-executor.md` §2.5 now states the two-tier evidence contract whose
  absence let the seam ship broken, §2.2 states the tool-error rule with its narrow boundary, and
  `m3-executor-ml.md` §Q1 is annotated rather than rewritten (its method was right — risk #4 predicted
  this exact failure). The one exception is §8 (M-1 above).

---

## Open questions (for the coordinator, not fixes)

1. **M-2(b) is a design call I should not make alone:** should `UnknownActorError` /
   `ThreadNotFoundError` remain absorbed as re-prompts (simple, uniform rule; silent on
   misconfiguration) or propagate to the M-1 net (loud, but one bad thread id ends a run)? My
   recommendation is propagate + log, but the "one rule, three cases" framing in §2.2 is defensible if
   the log lands. Decide it and write the decision into §2.2 either way.
2. **Who owns the §8 reconciliation (M-1)?** It collides with the concurrent `seed_workflows.sh` unit;
   it should not be done twice or, worse, half.
3. **The 348-green claim is still unverified end-to-end** (FalkorDB down during this review). Someone
   should re-run `pytest -q` with the DB up before U15 — I would not want U15 to be the first place
   the 177 graph-backed tests run against this commit.
