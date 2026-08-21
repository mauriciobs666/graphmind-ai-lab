# Ministral re-probe — review (K-027 item 5, D13 finding 2)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-027 (item 5)

## Scope & verdict

Reviewed `docs/plans/ministral-reprobe-ml.md` (U3's deliverable, `data-scientist`) against: the
real codebase (`server/falkorchat/{modelconfig,app,guards,executor,transport}.py`, `server/tests/
eval/{probe_ministral_judge,guard_calibration}.py`, `server/tests/eval/golden_guards.jsonl`), the
prior Qwen calibration report (`docs/test-reports/guard-judge-calibration-2026-08-17.md`), the
prior D13 probe (`docs/archive/plans/m3-capability-probe-ml.md`), `git status`/`git diff` across
the whole `falkor-chat/` tree, a live LM Studio instance at `localhost:1234`, and the shared
`kaizen_team` graph. `cpg_falkorchat` was **not used** — the coordination doc's own baseline
correctly flags it stale relative to this area, and I read the source directly instead, per its
guidance.

**Verdict: approve.** Every empirical claim I could independently check — the judge-calibration
arithmetic, the workspace-override resolution mechanism, the `_assemble_messages` alternation
defect, and the live HTTP 400 — reproduced exactly as the note describes. I found no blockers and
no majors. Two minors, both about durability/reproducibility of secondary artifacts, not about the
trustworthiness of the reported numbers.

**CPG:** considered, not relevant — the coordination doc's own entry baseline already establishes
`cpg_falkorchat` as stale for this exact area (guard/executor, 3 commits behind, one of them U1's
concurrent edit), so re-deriving that staleness here would be redundant; I read
`guards.py`/`executor.py`/`modelconfig.py`/`app.py` directly instead, as directed.

## Findings

### Verified — judge-calibration numbers (item 1, §3 of the note)

- **`probe_ministral_judge.py` calls the real, unmodified `guards.evaluate_guard`.** Read in
  full (`server/tests/eval/probe_ministral_judge.py`). It imports `guard_calibration` bare and
  monkeypatches only `gc.build_call` (module-global rebind), which `run_case` (`guard_calibration.py:167`)
  resolves via its own module's globals at call time — so the redirection is real, not a stub.
  Every metric function it calls (`false_advance_rate`, `advance_recall`, `cohens_kappa`,
  `confusion_matrix`, `coercion_flip_rate`, `flip_rate`, `per_path_breakdown`,
  `materiality_probe_failed`) is the same code `test_guard_calibration_live.py` uses, untouched.
- **The workspace-override mechanism works exactly as claimed.** `ModelGateway._workspace_override_ref`
  (`modelconfig.py:708-727`) checks `if overrides is not None` first — and the probe always passes
  `overrides={"guardModel": MODEL_REF}` (never `None`) — so `ws="ws:ministral-probe"` is never
  consulted; `_ws_overrides.get(ws, kind)` is unreachable on this path. Confirmed by reading the
  code, and the crosswalk `_KIND_TO_OVERRIDE_KEY["guard"] == "guardModel"` (`modelconfig.py:102`)
  makes the override land on the right kind. `"ws:ministral-probe"` therefore never reaches a graph
  selector — the note's cleanup claim (§8, "no throwaway graph state was ever created") is correct.
- **The arithmetic is independently reproducible from the note's own disclosed breakdown**, not
  just internally consistent. Recomputing by hand from the per-path breakdown (`understanding`
  tp=3/fp=0/tn=7/fn=5, n=15; `turns` tp=2/fp=0/tn=3/fn=1, n=6):
  - G2 = (3+2)/(8+3) = 5/11 = 45.45% → matches the reported 45.5% exactly.
  - κ: po=(5+10)/21=0.714, a_pos=(5+6)/21=0.524, b_pos=(5+0)/21=0.238, pe=0.524×0.238 +
    0.476×0.762=0.488, κ=(0.714−0.488)/(1−0.488)=0.442 → matches every reported figure exactly.
  - FAR_all denominator 15×3=45 calls, boundary conservatism 5/5=100%, flip-rate 1/26=3.8% all
    check out.
  This is strong evidence the numbers were transcribed from a real run, not estimated or
  misremembered — an error anywhere in the chain (a mistyped tp/fn, a wrong case count) would have
  broken at least one of these independent recomputations, and none did.
- **The D13 baseline figures the note cites for comparison** (bare `json.loads` 0/11=0.0%,
  fence-tolerant re-parse 4/11=0.364, Qwen 9/11=0.818) match `docs/archive/plans/m3-capability-probe-ml.md:350-351`
  verbatim.
- **The fixture sha256 the note cites** (`35061c79aa9ae93f5e2350d30e4543a1a64f72b49a4cb0409319cd431d6776b4`)
  matches both `sha256sum server/tests/eval/golden_guards.jsonl` on the live tree and the value
  recorded in the 2026-08-17 Qwen report — the fixture really is byte-identical across both runs.
- **The two disclosed caveats are accurately characterized.** `git status` confirms
  `falkorchat/guards.py`/`tests/test_guards.py` are the only in-flight U1 files; reading U1's diff,
  m-1's fix (clause-boundary negation) lands inside `_is_negated`, used only by
  `_rationale_contradicts`, used only by `_coerce_verdict` — exactly the function
  `coercion_flip_rate` measures. The note's own reported coercion-flip rate of 0.0% (0/78) is
  therefore not just "consistent with" the fix not mattering to this fixture, it is the *direct*
  measurement of whether that code path fired at all, and it didn't. m-2 (filter-before-slice in
  `_recent_turns`) only affects the 6 `turns`-path cases; the note doesn't independently verify this
  one but the reasoning is sound (golden set rows are curated, not malformed). No temperature pin
  for Ministral is disclosed and correctly attributed to `config/models.json` being off-limits per
  brief; `git status`/`git diff` confirm `config/models.json` and `config/opencode.example.json`
  carry no changes anywhere in the tree.

### Verified — the alternation-crash finding (item 2, §4.2 of the note)

- **`_assemble_messages` does what the note says.** Read `executor.py:909-931`: system prompt,
  then thread turns role-mapped `user`/`assistant`, then an unconditionally appended
  `{"role": "user", "content": f"CONTEXT:\n{context}"}` (line 930) — no conditional, no role check
  against the last thread turn.
- **`_drive`'s fault net matches the cited line range exactly.** `executor.py:440` opens the
  `try:`, `447-449` is `except Exception as exc: self._fail_with_note(...); raise` — the cited
  `440-449` bounds the whole guarded block precisely, not approximately.
- **Live-reproduced independently, not just read.** I ran the note's exact minimal 3-message repro
  (`system`, `user`, `user`) directly against the live LM Studio instance:
  - `mistralai_ministral-3-3b-instruct-2512` → **HTTP 400**, body containing the identical Jinja
    error text the note quotes verbatim ("After the optional system message, conversation roles
    must alternate user and assistant roles except for tool calls and results").
  - `qwen/qwen3-4b-2507`, byte-identical message shape → **HTTP 200**, clean completion.
  This is a stronger check than the brief required ("not required... rely on code-reading
  verification instead") — I had LM Studio access and the check was cheap, so I ran it. The finding
  is real, not a misreading of a stack trace.
- **K-039's implicit-dispatch fallback exists exactly where and how the note describes it**
  (`executor.py:723-754`, `not result.is_tool_call` → dispatches `post_message` implicitly when
  granted and non-empty text) — the note's §4.4 argument that this fallback (not native tool-calling)
  is what actually determines Qwen's AC-4 outcome checks out against the real code.
- **Model identity claims (§2) are all live-confirmed.** `curl .../api/v0/models` on the live
  instance today shows both catalog ids (`mistralai_ministral-3-3b-instruct-2512` / publisher
  `bartowski`, `mistralai/ministral-3-3b` / publisher `mistralai`) with identical `arch: mistral3`,
  `quantization: Q8_0`, `max_context_length: 262144` — exactly as reported.
- **The two "durable environment facts" the note claims to have written to `kaizen_team`
  are actually there.** Queried the graph directly: both entries exist, `author: data-scientist`,
  dated 2026-08-20, content matching the note's description (alternation-crash mechanism; aliased
  Ministral catalog ids). The coordinator's "spot-checked" claim in the coordination log is
  independently corroborated, not just trusted.

### Minor — §4.3 replay script is unreproducible by construction

`scratchpad/replay_answer_ministral.py` lives in the U3 session's own scratchpad, not the repo —
it cannot be re-read or re-run by anyone else, including this review. The note itself flags this
("this one is genuinely one-off... not meant to be re-run/maintained") and the brief anticipated
it, so this is not a trust problem for *this* run's numbers (5/5 Ministral, 0/5 Qwen) — the
described method (single merged `user` turn, verbatim `answer`-node systemPrompt and
`PostMessageTool.schema`, n=5 draws) is plausible and consistent with D13's own 3/3 finding and the
already-documented Defect-C failure mode. But the §4.3 numbers rest entirely on trust in the
prose description, with zero artifact for a future re-verifier. **Suggested improvement:** if this
finding is ever cited again (e.g. when the alternation-crash backlog item is filed), route the
n=5 replay through `probe_ministral_judge.py`'s sibling pattern — a small git-tracked,
non-`test_*`-named throwaway under `server/tests/eval/` — the same treatment already given to the
judge-calibration driver, rather than a scratchpad script that stops existing at session end.

### Minor — G1's 0.0% at n=10/30 is a thin true negative, not a strength claim to lean on

The note reports this correctly and doesn't overclaim it, but worth naming for whoever acts on the
verdict: with 0/30 calls advancing, the 95%-ish upper confidence bound on the true false-advance
rate is comfortably above 0% (the golden-set expansion note, U2, makes exactly this point about
n=30 for the same tier). The note's own boxed caveat ("a pass means only that no large defect was
detected... not a calibration certificate") already covers this, so this is not a defect in the
note — just a flag that the eventual golden-set expansion (K-027 item 4) should re-run this probe
too, not just Qwen's, once it lands.

### Nit — comparison table's `advance-recall` D13 row omits which arm's G1 was blocked

`docs/archive/plans/m3-capability-probe-ml.md:350` marks the bare-`json.loads` D13 arm's G1 as
"—" because 26/26 outputs were unparseable, making G1 undefined rather than 0%. The note's own
comparison table (§3, "Comparison to the D13 baseline and to Qwen") correctly reproduces this as
"—" too — no actual defect, just confirming it wasn't silently coerced to a misleading 0%.

## What's solid

- The methodological discipline is real, not asserted: every claim that could be independently
  checked (arithmetic, resolution mechanism, HTTP behavior, kaizen writes) checked out exactly, on
  the first attempt, with no rounding surprises or unexplained deltas.
- Scope discipline is clean. `git diff`/`git status` across the whole tree confirm U3 touched
  exactly one new file (`server/tests/eval/probe_ministral_judge.py`, correctly not `test_*`-named
  so pytest never collects it) plus the two doc deliverables (`ministral-reprobe-ml.md`,
  `golden-set-expansion-ml.md` is U2's, not U3's) — no edits to `config/models.json`,
  `config/opencode.example.json`, `reference`, `ws:acme`, or any of `guards.py`/`executor.py`/
  `app.py` (U1's concurrent, legitimate territory, correctly left alone).
  `falkor-chat/docs/{HISTORY,BACKLOG}.md` also carry no U3 edits, consistent with the coordination
  doc's "advisory note, no BACKLOG/HISTORY entry until acted on" rule.
  The `guards.py` mid-edit caveat is handled exactly right: disclosed, tied to a specific metric
  (coercion-flip rate) that would have caught it mattering, and it didn't.
- The verdict itself is well-supported and not overstated in either direction — it correctly
  separates "Ministral loses at judging" (a capability gap, matches D13, no code implicated) from
  "Ministral cannot be measured as agent/step until `_assemble_messages` is fixed" (a genuine new
  defect, out of scope, correctly deferred to a backlog filing rather than silently fixed or
  silently dropped).
- Holding the alternation-crash filing until U1 lands (to avoid a `BACKLOG.md`/`HISTORY.md`
  collision) is the right call and cites the precedent correctly
  (`m3-followups-coordination.md`'s K-034 race, per the coordination log).

## Open questions

None that block this gate. One forward-looking note for whoever files the alternation-crash
backlog item: the note's §4.4 argument (K-039's fallback already neutralizes Qwen's native
tool-calling weakness, so the axis D13 measured is "close to moot in practice") is a real,
non-obvious finding on its own — worth making sure the backlog item's framing doesn't undersell it
relative to the alternation crash itself.
