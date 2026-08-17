# Guard-judge calibration (K-027 item 3) — current-code method note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-027 item 3 (M3.5)

## 1. The question and the decision it serves

**Question:** is the archived protocol at `docs/archive/plans/m3-guard-calibration.md`
(2026-07-16, frozen, my own past instance's work) still the correct, buildable spec for
calibrating the `intake → research` fuzzy guard judge, given what has landed since (K-027 item 1's
parse fix, 2026-07-24; K-042's model-provider config, both landings through 2026-08-11; K-027 item
2's terminal-post engine contract, 2026-08-16)?

**Decision it gates:** whether `tdd-engineer` builds the calibration harness straight against the
archived §4/§5/§6/§7/§8, or against an addendum here first.

**Top-line answer: the method is unchanged and sound. Build against the archived protocol's §4
(gate), §5 (harness spec) with one precise addendum below, §6 (statistical honesty), §7
(outcomes), and §8 (report template) as-is.** Nothing that landed since 2026-07-16 touches the
metric definitions, the gate thresholds, the k=3 replicate design, or the N=26 statistical-honesty
conclusion. One harness-construction detail (§5.1's `run = {}`) needs a precise addendum — not
because it currently produces a wrong number, but because it exercises a code path production
never takes. A second, more consequential thing surfaced while grounding this note: **a number
already sitting in `docs/BACKLOG.md` K-027 item 3, described as "already on record," is not a
valid substitute or preview for this calibration** — see §3.

## 2. Findings from the real system (verified 2026-08-17, HEAD `acda33d`)

All of the following were read directly, not inherited from the brief or the coordination doc.

**F1 (K-027 item 1, parse layer) — confirmed, no method impact.**
`server/falkorchat/app.py::_LlmGuardJudge.__call__` (lines 418–438) parses the judge reply with
`extract_own_line_json_object(text, require_key="decision")`, not a bare `json.loads`. The
archived protocol's header already scopes itself to "the **post-fix** judge... not today's broken
prompt," so this is exactly the state it anticipated. The bias-to-suspend fallback
(`{"decision": False, "rationale": "unparseable judge output"}` on a `None` parse) is unchanged in
spirit and feeds `guards._coerce_verdict` exactly as the protocol's §3.1 describes. **No gate-math
or fixture change follows from this.**

**F2 (K-042, run/model resolution) — confirmed, needs a §5.1 addendum (see §4 below).**
`_LlmGuardJudge.__call__` signature is now
`(condition, *, understanding, recent_turns, ctx, step_output, model=None, run=None)`. Verified at
`app.py:418-429`:

```python
ws = run.get("ws") if isinstance(run, dict) else None
overrides = run.get("modelOverrides") if isinstance(run, dict) else None
llm = self._models.llm("guard", requested=model, ws=ws, overrides=overrides)
```

`_build_llm_judge(models)` (`app.py:451-455`) wraps a raw LLM via `_as_model_gateway`'s
`StaticModelGateway` sugar, or passes a real `ModelGateway` through unchanged — either shape is
fine for the harness to inject, and the archived §5.1's `judge = <the real production judge from
app._build_llm_judge>` line is still exactly correct. What needs revision is the literal
`run = {}` on the same table row — see §4.

**F3 (`guards.py`) — confirmed unchanged.** Read directly, `guards.py:181-244`:
`evaluate_guard(guard, *, ctx, run, step_output, thread, judge)` — same signature the archived
protocol's F1 describes; still no `understanding`/`turns` parameters (they are derived, not
passed). `_extract_understanding` (`:537-550`) still prefers `step_output`'s JSON, then
`ctx["understanding"]`, then `{}` — unchanged precedence. The fallback trigger is still
**truthiness**, not presence (`:225`: `recent_turns = [] if understanding else
_recent_turns(thread)`) — the archived F5 still holds verbatim, and the harness must still set all
three of `step_output` (non-parsing prose), `ctx` (no `understanding` key), and non-empty `thread`
for a `path: "turns"` case, exactly as §5.1 already specifies. `GuardVerdict` (`:167-172`) is still
just `(decision: bool, rationale: str = "")` — no `tier` field; the carried finding m-3 is still
open, and the archived note's reasoning that the harness can track evidence tier from the
fixture's own `path` field (not the judge's return) still holds — confirmed, nothing new needed
here. The full offline `test_guards.py` suite is green (**147 passed**, run 2026-08-17) — the
harness can be built with confidence against currently-tested code.

**F4 (fixture) — confirmed exact match, mechanically verified, not eyeballed.** Parsed
`server/tests/eval/golden_guards.jsonl` with a script rather than trusting the brief's count:

```
26 rows; path counter: {'understanding': 19, 'turns': 7}
(expected, path) counter: {(False,'understanding'): 11, (True,'understanding'): 8,
                            (False,'turns'): 4, (True,'turns'): 3}
```

That decomposes to exactly the protocol's §4 table: 8+3=11 `clear_advance`, 7+3=10
`clear_suspend`, 4+1=5 `boundary` (the fourth `(False,'understanding')` row beyond the 7
`clear_suspend` is the 4 `understanding`-path boundary cases; the row count matches once boundary
is split out). **No drift.**

**F5 (K-027 item 2, terminal-post contract, 2026-08-16) — confirmed no interaction.** The new
`config.requiredTools` engine guarantee enforces at `_run_agent_node`'s two exit points that a
declared tool (e.g. `post_message`) was actually dispatched; it operates on the **executing**
step's own output, after the step runs. The `intake → research` guard this protocol calibrates
evaluates the **next transition's** condition against `ctx`/`step_output`/`thread` in
`_select_transition`, a separate seam. They compose in a live run (a `requiredTools` violation is
logged/traced but does not fail the drive) but do not interact methodologically — calibrating the
guard judge says nothing about, and is said nothing about by, whether `post_message` fired.

**F6 (model/provider config) — confirmed, with a materially relevant gap. See §5.**

## 3. The stale-number hazard in `docs/BACKLOG.md` K-027 item 3 — read before running anything

BACKLOG's K-027 item 3 entry currently reads: *"Diagnostic already on record: on clean golden
inputs Qwen's judge passes both arms (recall 0.818, false-advance 0.067), so the live 3/10 is a
generator-half problem, not a judge problem."* That number traces to
`docs/archive/plans/m3-capability-probe-ml.md` (the Qwen-vs-Ministral capability probe, run
2026-07-19). Read in full to ground this — it is **not** a preview or partial satisfaction of this
calibration, for three compounding reasons, all verifiable from that document's own text:

1. **It predates the post-fix judge.** The probe's own results table (line 350) labels the arm
   explicitly: *"advance-recall (**shipped bare-`json.loads` pipeline**)... 9/11 = 0.818."* K-027
   item 1's `extract_own_line_json_object` fix landed 2026-07-24, five days **after** this run. The
   archived calibration protocol is explicit that it calibrates "the post-fix judge... not today's
   broken prompt" — the probe's number is the *pre-fix* number by its own label.
2. **It did not go through `evaluate_guard`.** The probe's §3.3 describes the method as "the golden
   set feeds the judge **fixed inputs**" — i.e. it drove the judge callable directly with
   `understanding`/`turns` values, not through synthesized `step_output`/`ctx`/`thread` and the
   real `evaluate_guard` precedence logic. The whole point of the archived protocol's F1 ("the
   whole point is to go through the real `evaluate_guard`... not just the prompt") is exactly what
   this number skips.
3. **The false-advance figure uses the wrong denominator for G1.** `1/15 = 0.067` is stated over
   **15** cases — that is `FAR_all` (all `expected:false` rows, `clear_suspend` + `boundary`
   combined), not the gate's `FAR_strict` denominator of **10** `clear_suspend` cases the archived
   §4.1 specifies for G1. Whether the one false advance landed on a `clear_suspend` or a `boundary`
   case is not stated in the probe, so `FAR_strict` cannot even be reconstructed from it — it might
   be 0/10 or 1/10 (still passing at ≤10%, but that is not the same as *knowing* it).

There is also no evidence of k=3 replicates in the probe (single-pass per case), so `flip_rate`
and the G1 per-call-not-per-case-majority counting rule (§5.2) were never exercised either.

**Recommendation:** treat the probe's number as historical context only — it is legitimate
evidence that nothing about the judge's *reasoning* looked structurally broken on this fixture as
of 2026-07-19, which is useful triage context for why live intake advancement (3/10) was diagnosed
as a generator problem rather than a judge problem. It is not a result this calibration can cite,
inherit, or shortcut from. **`tdd-engineer` must run the harness fresh**, and the eventual report
should note explicitly that it supersedes the probe's diagnostic numbers for this condition
(pointer only — the probe document itself is untouched, out of scope here).

## 4. §5.1 addendum — the `run` construction

The archived §5.1 table's row `— | run = {}, judge = <the real production judge...>` needs revision
for the post-K-042 judge. Recommended replacement row:

```
— | run = {"ws": "ws:golden-eval", "modelOverrides": {}},
    judge = app._build_llm_judge(models)  # models = the real ModelGateway.from_env()
```

**Why not the mechanically-simpler `run = {}`.** Both constructions currently resolve to the same
model — I traced `_workspace_override_ref` (`modelconfig.py:708-727`):

```python
if overrides is not None:
    key = _KIND_TO_OVERRIDE_KEY.get(kind)
    value = overrides.get(key) if key else None
elif ws is not None:
    value = self._ws_overrides.get(ws, kind)
else:
    value = None
```

`run = {}` makes both `ws` and `overrides` resolve to `None` in `_LlmGuardJudge.__call__`
(`run.get(...)` on an empty dict), landing in the `elif ws is not None` / `else` branch — a branch
**production never reaches**. `executor._drive` (`executor.py:430,439`) unconditionally stamps
`run["ws"] = ctx.ws` and `run["modelOverrides"] = self._repo.read_model_overrides(ctx.ws)` — the
latter always returns a dict (all-`None` values when no override row exists,
`repository.py:1785-1812`, confirmed by reading), never omits the key. So production always enters
the `overrides is not None` branch; `run = {}` in the harness always enters the branch beside it.
Numerically identical today (no workspace anywhere has ever written a `guardModelOverride` — that
property is set only via a REST/service path no seed script or fixture touches), but that
equivalence is coincidental, not designed, and it is the exact kind of divergence F1's own stated
purpose ("go through the real precedence logic... not around it") exists to catch. The fix costs
nothing: `overrides={}` (a non-`None`, empty dict) drives the identical `overrides is not None`
branch production takes, `.get(key)` still returns `None`, still resolves to the per-kind default
— and it needs **no live graph read** (`GraphWorkspaceOverrides` is never consulted when
`overrides` is passed non-`None`), so it costs nothing for reproducibility either. `ws` itself is
inert in this branch (never read once `overrides is not None`) — `"ws:golden-eval"` is a
documentation-only placeholder, not a real or required workspace.

`model=None` throughout is correct as specified — the fixture's `condition` guard JSON never sets
a `"model"` key (`guard = json.dumps({"kind": "llm", "text": row["condition"]})`, §5.1
unchanged), so `evaluate_guard` never forwards `model=` to the judge (`guards.py:238-240`), and the
`guard` kind's per-kind default (`config/models.json`'s `defaults.guard`) is what resolves. This
is the right thing to calibrate — it is what every live `intake→research` transition actually
uses today.

## 5. Model / quantization / temperature for the report's provenance header

**Configured (`config/models.json`, HEAD):** `defaults.guard = "lmstudio/qwen/qwen3-4b-2507"`,
`timeouts.guard = 180`. `models` has no entry for this ref, so no per-model overlay settings apply
— `_resolve_element`'s `params` for this ref is `{}` today (`modelconfig.py:670-686`, verified by
reading; `config/models.json` has only a `dim` entry for the embedding model).

**Quantization (live-checked, 2026-08-17, `curl localhost:1234/api/v0/models`):**
`qwen/qwen3-4b-2507` is present, `quantization: "Q4_K_M"`, `state: "not-loaded"` (LM Studio JIT
loads on first request — this is the expected idle state, not a problem).

**Temperature — a genuine gap, not just an unrecorded fact.** Grepped the whole repo: `temperature`
appears **nowhere** — not in `config/models.json`, not in `llm.py`, not in `modelconfig.py`. The
request payload `OpenAICompatibleLLM.complete` builds (`llm.py:133-137`,
`{"model": ..., "messages": ..., **self._params}`) sends **no `temperature` key at all** for the
guard kind today, so the live call rides whatever LM Studio's per-model default sampling preset
is — a value this codebase does not control, does not know, and (unlike a model id) cannot be
pinned by citing a config file. This directly undercuts the archived §5.2's own framing
("Temperature 0 does not guarantee determinism... therefore run k=3"), which implicitly assumes
temperature is pinned near 0; today it is not pinned at all.

**Recommendation (config change, for `tdd-engineer` to make, not a method change):** add
`"temperature": 0` under `config/models.json`'s `models` map, keyed
`"lmstudio/qwen/qwen3-4b-2507"`, before running the harness. This flows unmodified through
`_resolve_element`'s `settings`/`params` (`timeout`/`dim`/`protocol` are popped, everything else
`camel_to_snake`'d into `params` and spread into the request) — confirmed by reading, not assumed.
This is cheap, standard practice for a gate/judge role, and turns an uncontrolled variable into a
recorded, reproducible one; it does not require touching any Python. If the team instead wants to
calibrate exactly today's unpinned behavior, that is a legitimate but weaker choice — in that case
the report **must** record LM Studio's actual active sampling temperature for this model at run
time (readable from LM Studio's own UI/config, not from this repo), not leave the field blank.

**Provider file — an environment pre-flight `tdd-engineer` must confirm, not something this note
can settle.** `FALKORCHAT_OPENCODE_CONFIG` defaults (`start_server.sh:102`) to
`$HOME/.config/opencode/opencode.json`. On this box that file currently declares `lmstudio` at
`baseURL: http://192.168.0.69:1234` (a LAN host, unreachable from this sandbox) and lists only
`google/gemma-3n-e4b` under `lmstudio.models` — it does **not** declare `qwen/qwen3-4b-2507`.
Separately, `localhost:1234` **is** live right now, matches `config/opencode.example.json`'s
`baseURL`, and does serve `qwen/qwen3-4b-2507`. `ProviderCatalog`/`_resolve_element` only validates
the **provider id**, not the model id, against the declared file (`modelconfig.py:662-668`) — a
model absent from the provider's `models` map still resolves and gets sent to whatever `baseURL`
is configured, so a mismatched provider file would not fail loudly, it would just point at the
wrong (or wrong-address) LM Studio instance. **Before running: confirm `FALKORCHAT_OPENCODE_CONFIG`
resolves to a file whose `lmstudio.baseURL` actually reaches the LM Studio instance serving
`qwen/qwen3-4b-2507`** (on this box, that is `localhost:1234`, matching the example file, not the
current default `$HOME` file). Record the resolved `baseURL` and provider file path in the report
alongside the model id — this is per-environment, not a repo fact, and is exactly the kind of
silent mismatch the archived §5.2's "record model id... for reproducibility" line exists to catch.

**Prompt revision:** `_JUDGE_SYSTEM_PROMPT`, `server/falkorchat/app.py:320-330`, unchanged since
K-042 landed (last touch to `app.py` is commit `eb1a60f`, 2026-08-11, per K-042 L2 file history;
`guards.py`'s last touch is `a2b8aa9`). Cite HEAD (`acda33d`) as the run's baseline commit in the
report.

## 6. Statistical honesty — unaffected

Nothing above changes N, the case mix, or the judge's fundamental bias-to-suspend design. The
archived §6/§6.1 conclusions hold exactly as written: this is a **one-sided screen at n=21 clear
cases** (11 `clear_advance` / 10 `clear_suspend`), not a certification; a pass means "no blocker
found at a sample size that could only have found a large one"; G1's ≤10% threshold is not
independently measurable at n=10 (even a perfect 0/10 carries a 95% CI of [0%, 27.8%]); k=3
replicates reduce measurement noise, not the effective case-level N. **The §8 verbatim caveat
sentence is required, unedited, adjacent to the verdict line in the eventual report** — reproduced
here only as a pointer, per the brief's "don't duplicate 486 lines" instruction; the source of
truth is `docs/archive/plans/m3-guard-calibration.md` §6.1/§8.

## 7. What `tdd-engineer` needs, concretely

1. Build the harness per archived §5.1, with the `run` row replaced by §4 above:
   `run = {"ws": "ws:golden-eval", "modelOverrides": {}}`.
2. `judge = app._build_llm_judge(models)`, `models = modelconfig.ModelGateway.from_env(...)` —
   confirm at construction time that `FALKORCHAT_OPENCODE_CONFIG` resolves to a provider file
   reaching the LM Studio instance that actually serves `qwen/qwen3-4b-2507` (§5 above — do not
   assume the shell's ambient default is correct on the run box).
3. Before the live run, add `"temperature": 0` under `config/models.json`'s
   `models["lmstudio/qwen/qwen3-4b-2507"]` (or document and record the unpinned alternative — §5).
4. k=3 replicates per case (26 × 3 = 78 calls), G1 counted per-call, G2/κ/confusion matrix by
   per-case majority, exactly per archived §5.2.
5. Report at `docs/test-reports/guard-judge-calibration-<date>.md` per archived §8, with the
   provenance header carrying: model ref `lmstudio/qwen/qwen3-4b-2507`, quantization `Q4_K_M`
   (re-verify at run time — LM Studio state can change), temperature (whatever was actually sent),
   resolved `baseURL`, prompt revision = HEAD commit, fixture sha256, k=3, date. Note explicitly in
   the report that it supersedes the pre-fix, non-`evaluate_guard`, wrong-denominator numbers in
   `docs/archive/plans/m3-capability-probe-ml.md` for this condition (§3 above).
6. Gate failure on G1 or G2 ⇒ block the wiring, no override, per archived §7 — unchanged.

## 8. Risks & open questions

1. **(Addressed, not a residual risk) The BACKLOG "diagnostic already on record" line.** Once the
   live run in this unit lands, `docs/BACKLOG.md` K-027 item 3's text citing recall 0.818 /
   false-advance 0.067 should be corrected or clearly marked superseded — left as-is it will
   mislead the next reader into thinking the calibration already happened. Flagging for `teco`'s
   coordination close, not fixing here (BACKLOG is out of this note's write scope).
2. **Temperature choice is a live decision, not fully mine to make unilaterally.** I recommend
   pinning it to 0 (§5) as the reproducible, standard choice for a judge role; if the team instead
   wants to calibrate today's unpinned behavior for fidelity to what's *actually* shipped, that is
   defensible too, but the report must record the real value, not omit it.
3. **Provider-file mismatch is environment-specific.** What I found on this box (LAN-host default
   pointing at an unreachable address with the wrong model list) may not hold on whatever box
   `tdd-engineer` runs the harness on — treat §5's finding as "check this," not "this is broken
   everywhere."
4. Everything else carried from the archived protocol (representativeness, single-labeler boundary
   strata, self-preference, the guard-text-vs-judge attribution risk for `bd-04`, "one condition
   string only") stands unchanged — see archived §10, not reproduced here.
