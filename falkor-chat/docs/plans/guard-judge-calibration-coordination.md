# `guard-judge-calibration` (K-027 item 3) — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-027 item 3 (M3.5)

**Closed 2026-08-17.** All units delivered/gated/accepted. VERDICT: **wire** — both gate arms pass
(G1 false-advance 0.0%, G2 advance-recall 81.8%), κ=0.811 (diagnostic). Two independent review
gates both returned **approve with suggestions, no blocker**; both non-blocking suggestions were
folded into the report before close. `docs/BACKLOG.md` K-027 item 3 flipped to ✅ delivered and its
stale superseded number corrected in place; `docs/HISTORY.md` carries the dated entry. See the
final ledger below for the full trail.

## Goal

Run the guard-judge calibration gate (D9/D10) that `docs/BACKLOG.md` K-027 item 3 has been
carrying since D12-B: decide whether the LLM-as-judge behind the `intake → research` fuzzy guard
(`server/falkorchat/app.py::_LlmGuardJudge`) is fit to wire live on the shipped local 4B, per the
two-arm screen — **G1 false-advance ≤ 10%** and **G2 advance-recall ≥ 0.80** — specified in the
existing advisory protocol, `docs/archive/plans/m3-guard-calibration.md` §4/§5, against the
existing 26-row fixture `server/tests/eval/golden_guards.jsonl` (unconsumed before this unit —
nothing read it yet, BACKLOG K-027 item 4 nit n-3).

**Why re-open the protocol before building, rather than handing it straight to `tdd-engineer`:**
the protocol document is dated 2026-07-16 and lives under `docs/archive/` (frozen, pre-dates the
current docs convention, must not be edited in place). Two things landed since that change the
harness's construction details, though not — on a first read — the calibration *method*:

1. **K-027 item 1 (parse-layer slice A, 2026-07-24)** — `_LlmGuardJudge` now parses with
   `llm.extract_own_line_json_object(..., require_key="decision")`, not the bare `json.loads` the
   protocol was written against. The protocol's own header already anticipates this ("calibrates
   the **post-fix** judge... not today's broken prompt") and BACKLOG explicitly says item 3 is
   "unaffected... its false-advance metric (D9) is only meaningful against this settled parse."
2. **K-042 (model-provider config, landed between the protocol and today)** —
   `_LlmGuardJudge.__call__` now has signature
   `(condition, *, understanding, recent_turns, ctx, step_output, model=None, run=None)` and
   resolves the LLM via `self._models.llm("guard", requested=model, ws=ws, overrides=overrides)`.
   The protocol's §5.1 harness table's literal `run = {}` no longer exercised the same code branch
   production does — the exact kind of "protocol vs. current code" drift this unit existed to
   catch before `tdd-engineer` built against a stale spec.

## Scope of this coordination

**K-027 item 3 only** (judge calibration, D9/D10). **Not** item 4 (golden-set expansion, D11 —
separate backlog item) or item 5 (Ministral re-probe). The existing 26-row fixture was used as-is;
per protocol §6 it supports a one-sided *screen* ("no blocker found"), not a certification —
expanding it was explicitly out of scope.

## Final ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `data-scientist` | `a338db0600ba96a84` | accepted | `docs/plans/guard-judge-calibration-ml.md` | teco-verified (spot-checked key claims); no independent-review gate — advisory, low implementation risk, consumed by U2/U3 |
| U2 | `tdd-engineer` | `afa68b0e2f2ad464a` | delivered | `docs/test-reports/guard-judge-calibration-2026-08-17.md` + `server/tests/eval/{guard_calibration.py,test_guard_calibration.py,test_guard_calibration_live.py}` + `config/models.json` (temperature pin) | `analyst` + `data-scientist` → both approve with suggestions |
| U2b | `analyst` | `aefebc0f3fd22126a` | accepted | `docs/reviews/guard-judge-calibration.md` | approve with suggestions, no blocker |
| U3 | `data-scientist` | `a68e3daabd1e90972` | accepted | `docs/reviews/guard-judge-calibration-ml.md` | approve with suggestions, no blocker |
| U4 | `tdd-engineer` | `afa68b0e2f2ad464a` (resumed) | accepted | report revision (explicit `raw_rationale`/`coercion_flip` columns + expanded materiality-probe section) | teco-verified |
| U5 | `tdd-engineer` | `afa68b0e2f2ad464a` (resumed) | accepted | `docs/BACKLOG.md` flip + stale-number correction, `docs/HISTORY.md` dated entry | teco-verified |

## U1 — method review (accepted)

**Top-line: the archived protocol's method is unchanged and sound.** One harness-construction
addendum needed (§5.1's `run = {}` → `run = {"ws": "ws:golden-eval", "modelOverrides": {}}` — both
resolved identically today but the empty-dict form coincidentally took a code branch production
never reaches). Also surfaced: (a) the "diagnostic already on record" recall/false-advance numbers
quoted in `docs/BACKLOG.md` K-027 item 3 were **stale and not a valid substitute** for this
calibration — pre-fix parser, bypassed `evaluate_guard`, wrong G1 denominator (traced to
`docs/archive/plans/m3-capability-probe-ml.md`, 2026-07-19) — flagged for BACKLOG correction at
close (done, U5). (b) No `temperature` was pinned anywhere in the repo for any model kind —
recommended adding `"temperature": 0` under `config/models.json`'s
`models["lmstudio/qwen/qwen3-4b-2507"]` before the live run (a **production config change**, not
test-only — `teco` accepted this recommendation, see below). (c) This box's default
`FALKORCHAT_OPENCODE_CONFIG` pointed at an unreachable LAN host lacking the Qwen model;
`localhost:1234` was the one that actually worked and served `qwen/qwen3-4b-2507` (Q4_K_M).

**teco verification (spot-checked directly, not accepted on data-scientist's word alone):**
confirmed by grep that `temperature` appears nowhere in `server/falkorchat/` or `config/*.json`;
confirmed `executor.py`'s `_drive` unconditionally stamps `run["ws"]`/`run["modelOverrides"]`
(never omits either key); confirmed `$HOME/.config/opencode/opencode.json` pointed at
`192.168.0.69:1234` with only `google/gemma-3n-e4b` listed; confirmed `localhost:1234` was live and
served `qwen/qwen3-4b-2507`.

**teco decision:** accepted the temperature=0 pin as a reasonable, reversible, standard-practice
config change for a judge/guard role (recommended by the domain expert, cheap, one line) rather
than pausing to the user — a technical calibration-fidelity call, not a product/scope trade-off.

## U2 — harness + live run (delivered, teco-verified)

**Verdict: G1 false-advance = 0.0% (0/30 calls, n=10) · G2 advance-recall = 81.8% (9/11) · both
gates pass · WIRE.** teco independently re-derived G1/G2/FAR_all/κ/per-path accuracy/materiality
probe from the report's own raw per-case table (not trusted on the summary alone) — all figures
checked out exactly. Ran `.venv/bin/python -m pytest -q` directly: **1088 passed, 3 deselected**,
matching the claimed count (baseline was 1064 passed / 2 deselected — the delta is the 24 new
offline tests + 1 new deselected live test). Confirmed `config/models.json`'s diff was exactly the
one-line `temperature: 0` addition, nothing else touched.

**Deviations flagged by the agent, accepted:** built `ModelGateway` directly from
`ProviderCatalog.load`/`Overlay.load` rather than `.from_env()` (an autouse test fixture silently
redirects `.from_env()` to the offline dim-4 config in every pytest test, which would have made
the temperature pin unverifiable) — cites an existing precedent in `test_golden_set_integrity.py`
("D7 mechanism 1"); used `config/opencode.example.json` as the provider file since the ambient
`FALKORCHAT_OPENCODE_CONFIG` was unset on the run box, live-confirmed it reaches `localhost:1234`
and serves the right model.

## U2b + U3 — independent review gates (both accepted)

**`analyst` (U2b) — approve with suggestions, no blocker.** Personally reproduced the mutation-test
claim (both mutations independently re-applied, confirmed red, reverted, confirmed green
byte-identical). Hand-recomputed every headline number from the report's own per-case table — all
reproduce. Traced `ModelGateway.from_env()` internals and confirmed the harness's direct
`ProviderCatalog`/`Overlay` construction is functionally identical modulo the deliberately-bypassed
env-var indirection; the cited precedent is real. Findings minor/nit only: the per-case table
omitted explicit `raw_rationale`/`coercion_flip` columns the archived §8 spec asked for, plus
cosmetic lint noise.

**`data-scientist` (U3) — approve with suggestions, no blocker.** Every headline number
independently re-verified a second way (G1/G2/FAR_all/κ/per-path all reproduce exactly from the
per-case table + a fresh fixture sha256). "WIRE" confirmed correct against archived §7's decision
table. §8 report-template compliance clean. **Substantive non-blocking finding:** the report's
materiality-probe line ("Passed") was technically correct per the archived bloc-AND criterion, but
hid that `ca-04`/`ca-08` (the two G2 misses) are themselves the fixture's designed materiality
probes, and the judge's rationale in both echoed the `missing` field almost verbatim rather than
reasoning about sufficiency — a real 2-of-3 individual-probe miss worth stating explicitly (§8 item
8: "the qualitative read is worth more than any aggregate at this sample size"). `tn-07` (the sole
FAR_all contributor) read as a defensible boundary case, not fallback-path over-permissiveness, but
n=1 there was noted as uninformative either way.

Both reviewers converged on the identical non-blocking gap from different angles — a strong,
independently-corroborated signal to fold in before close, even though neither review required it.

## U4 — report revision (accepted, teco-verified)

Sent back to the same `tdd-engineer` agent (resumed by id, not respawned) to fold in both
reviewers' non-blocking suggestions: (1) explicit `raw_rationale`/`coercion_flip` columns added to
the per-case table, derived from already-captured run data (not a fresh live run — the agent
proved every row's `coercion_flip=False` from the already-verified 0.0% aggregate plus a fresh grep
for `_coerce_verdict`'s fallback markers, zero hits); (2) the materiality-probe section expanded
with an explicit paragraph naming the `ca-04`/`ca-08` pattern and quoting both rationales, while
keeping the gate-level "Passed" verdict unchanged (the bloc-AND criterion genuinely didn't trigger).
teco read the revised report directly and confirmed both additions landed correctly, the verdict/
G1/G2/κ lines were untouched, and no code/gate-math was touched.

## U5 — BACKLOG/HISTORY closeout (accepted, teco-verified)

Sent back to the same agent to flip `docs/BACKLOG.md` K-027 item 3 to ✅ delivered (substantive
entry: harness paths, verdict, κ, report path, both review verdicts, the materiality-probe finding,
the §6/§6.1 one-sided-screen caveat) and strike through + mark superseded the stale
recall-0.818/false-advance-0.067 line in the same section (cites `docs/plans/guard-judge-
calibration-ml.md` §3 rather than re-deriving), and to add a dated `docs/HISTORY.md` entry
(What/Result/Mechanism/Tests/Review gates/Supersedes-a-stale-number, matching the file's own
established shape). teco read both diffs directly: correct section, correct citations, no drift to
any number, no `Status:` field touched on any of the four other docs (all remain `active`, correct
per the "item close, not milestone close" scoping — K-027 overall stays 🟡 in-progress, items 4/5
open). Offline suite re-verified once more independent of the agent's own count: **1088 passed, 3
deselected, 0 failures.**

## Follow-ups (not this coordination's scope)

- **K-027 item 4** (golden-set expansion, D11) and **item 5** (Ministral re-probe) remain open,
  tracked in `docs/BACKLOG.md` K-027 as before — this coordination did not touch either.
- The carried finding **m-3** (`GuardVerdict` has no `tier` field) remains open; not a blocker for
  this item, since the harness derives evidence-tier from the fixture's own `path` field.
