# Salesperson tool-orchestration reliability — round 5 (K-062) — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-062 (post-M6, not a milestone gate)

Successor to `docs/plans/salesperson-tool-reliability4-coordination.md` (K-060 diagnosis, closed
2026-08-31, no fix warranted — stays `active` since round 2's own K-061 confirmation item is still
open, not this round's concern). Rounds 2-4 stay `active`, untouched. Picking up **K-062**
(`docs/BACKLOG.md`), whose severity was revised upward this week (pooled 20.4%, Wilson CI
11.5-33.6%, from two independent samples 8.3% and 32.0% too far apart to reconcile without a
dedicated pass) — the last open item in this investigation thread besides K-061's own remaining
live-confirmation follow-up (deliberately left for later, lower urgency, per the user's own
decision this session).

**Prior art (read before dispatching or picking this up cold):**
- `docs/BACKLOG.md` K-062 — the filed item in full: why it exists (the model has the *correct*
  hold-reason available verbatim in the tool result and substitutes a factually wrong one to the
  customer), both samples' rates and CIs, the Owner note (mechanism already understood, priority
  case changed not the diagnosis), the Test-strategy note (a dedicated n≈25-30 isolating pass to
  narrow the 8.3%-32.0% gap before any fix is attempted).
- `docs/reviews/salesperson-tool-reliability-ml.md` §15.4 — the opportunistic re-screen that
  produced the 32.0% figure (method note: "every candidate match read in full, not just
  regex-matched" — this round's dedicated pass should keep that discipline, not loosen it).
- `docs/reviews/salesperson-tool-reliability-ml.md` §9.2/§9.4 (K-058) and §15.1 (harness
  precedent reused every round: real `ModelGateway.from_env()`, `LoggingToolRegistry` ground-truth
  wrapper since executor trace payloads truncate at 200 chars, throwaway per-pass workspace,
  `GRAPH.DELETE` teardown, `reference`/`ws:acme` never written to).
- `server/falkorchat/executor.py:1030-1055` — the K-058 same-turn-mention guard's held-call path:
  the message fed back to the model states the *actual* reason verbatim
  (`f"{target_arg} {target_value!r} was not mentioned anywhere in this turn's own text (K-058)"`,
  ~line 1039) — K-062 is the model receiving this accurate explanation and, in the
  customer-facing reply, substituting a plausible-sounding wrong one ("not recognized as a
  product" / "not recognized in the catalog") instead.
- `docs/reviews/salesperson-tool-reliability-ml.md` §9.3 — the K-058-class fabrication-honesty
  concern this pattern rhymes with (a different failure, same general shape: what the model says
  to the customer diverging from what it was actually told/found).

**Scope discipline (carried from rounds 1-4):** diagnosis first, larger dedicated n, before any
fix attempt. The *mechanism* is already understood (BACKLOG's own note) — this round's unit is a
tighter rate estimate plus whatever finer mechanism detail a dedicated, isolating pass surfaces
(e.g. does it correlate with which product/turn-shape, is it sensitive to hold-reason wording,
does a `systemPrompt`/tool-result-wording lever look promising) — explicitly **not** a fix
attempt yet. Whether a fix is warranted, and its shape, is a decision for after this unit lands.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `data-scientist` | `a3ab455da49118afa` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §16 | `teco` (direct) → accepted | 172.4k tok / 66 tools |
| U2 | `data-scientist` | `a63cae47aba77a5c7` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §18 | `teco` (direct) → accepted | 224.3k tok / 100 tools |

## Notes

- Single-unit start, same shape as every prior round's own first unit — `teco` verifies the
  diagnosis directly (re-checkable stats/ground-truth, no code change) rather than dispatching a
  separate `analyst` gate for a diagnosis-only deliverable. A fix unit, if warranted, follows
  normal implementer + `analyst` review gating, dispatched as its own follow-up unit once this
  lands and is reviewed.
- No parallel dispatch risk (single unit; no other round's coordination has a unit currently in
  flight — round 2 and round 4 are both idle awaiting a future user decision).
- **U1 delivered and independently re-verified by `teco` 2026-08-31.** Diff confirmed scoped to
  exactly `docs/reviews/salesperson-tool-reliability-ml.md` (264 insertions, one file — `BACKLOG.md`
  and this coordination doc both untouched by the delegate, as instructed). Every Wilson 95% CI
  recomputed from scratch and matched exactly (1/28→3.6% CI 0.6-17.7%; 5/28→17.9% CI 7.9-35.6%;
  27/28→96.4% CI 82.3-99.4%; the cited 14/24→58.3% CI 38.8-75.5% from §12.6; 0/13 and 6/14 both
  reproduced). The one-sided Fisher exact test on the 0/13-vs-6/14 broader-defect table
  independently recomputed via a from-scratch hypergeometric implementation: p≈0.0101, matching
  the delegate's reported p≈0.010; the strict-table equivalent (0/13 vs. 1/14) recomputed at
  p≈0.52, also matching. Code citations verified against source: `executor.py:1030-1045`'s exact
  K-058 held-call reason wording, and `proof_defs.py:323-362`'s `systemPrompt` catalog-mismatch
  sentence quoted verbatim and confirmed to never anticipate the K-058-hold scenario. `ws:ds-k062`
  confirmed absent from the live `GRAPH.LIST`; `reference`/`ws:acme` re-verified `OK` via
  `verify_salesperson.sh acme`, `verify_catalog.sh`, `verify_workflows.sh acme` (all three,
  independently re-run by `teco`, not just trusted from the delegate's own report). The `kaizen_team`
  write confirmed present via direct query (temperature-pin-swing fact, correctly attributed to
  `data-scientist`).
  **Verdict: diagnosis accepted as delivered.** Findings folded into `docs/BACKLOG.md` K-062
  directly by `teco` (rewritten in place, still 🟡 in-progress — the strict/broader rate split,
  the precondition-occurrence-swing driver, the `temperature`-pin recommendation, and the two named
  candidate levers were all folded in; header changed to 🟡 in-progress from 🔵 proposed since a
  dedicated diagnosis has now landed). **No fix has been chosen yet** — that decision, and the
  separate `temperature: 0` config-pin question, are `teco`'s own next open items, per the
  delegate's own §16.5 recommendation not to treat this pass as settling severity on its own.
- **2026-08-31 — reproducibility fix shipped directly by `teco`** (trivial single-file no-brainer
  per own routing table: one config entry, mirrors an existing precedent exactly, no design
  judgment): `config/models.json` now pins `temperature: 0` for `lmstudio/mistralai/ministral-3-3b`
  — verified via a direct `Overlay.model_settings()` resolution check and the offline
  `model`/`config`-scoped test selection (204 tests, green). `docs/HISTORY.md` and `docs/BACKLOG.md`
  K-062 both updated. This closes the reproducibility half of §16.5's recommendation; the fix
  decision for K-062 itself (view_cart nudge / systemPrompt addition / no fix) remains open,
  per the user's own explicit choice this round to ship only the config pin for now.
- **2026-08-31 — U2 dispatched, per the user's explicit choice ("Dedicated eval of both levers")**
  on the three-option decision `teco` posed (dedicated eval / no fix / ship the systemPrompt lever
  directly). Brief: three-arm live eval (fresh baseline, lever (a) view_cart-recheck nudge, lever
  (b) systemPrompt scenario-naming line), both variants constructed in-memory in the eval script
  only, `proof_defs.py` untouched, scored under §16.2's strict/broader taxonomy, explicitly
  instructed not to reuse §16's own baseline numbers since those predate the temperature-pin fix
  (that would reintroduce the exact confound just closed). Deliverable: `ml.md` next section
  (delegate confirms the number against the file's own tail at dispatch time), ending in an
  explicit ship/no-ship recommendation per lever, implementation left to normal
  `tdd-engineer`/`coder` to `analyst` gating as a follow-up unit if warranted.
- **U2 delivered and independently re-verified by `teco` 2026-08-31.** Diff confirmed scoped to
  exactly `docs/reviews/salesperson-tool-reliability-ml.md` (342 insertions, one file —
  `proof_defs.py`, `config/models.json`, `BACKLOG.md`, `HISTORY.md` all untouched by the delegate,
  as instructed). Every Wilson 95% CI across all three arms (baseline/lever-a/lever-b, 19 figures
  total) recomputed from scratch via a standalone Python script and matched exactly. Every cited
  Fisher exact p-value recomputed independently (two-sided, matching the delegate's own method):
  baseline recheck-mechanism p≈0.0070, lever-a-vs-baseline occurrence p≈0.011, lever-a-vs-baseline
  unconditional broader p≈0.42, lever-b-vs-baseline conditional p≈0.0028, lever-b-vs-baseline
  unconditional p≈0.011, lever-b-vs-baseline occurrence p≈0.79 — all matched. Code citations
  verified against source: the anchor substring
  ("if a product name does not match anything in the catalog, say so plainly rather than adding it
  anyway. Prices shown ") confirmed present exactly once in the live, imported
  `SALESPERSON_DEF['steps'][0]['config']['systemPrompt']` (`version` confirmed `v5`, via direct
  Python import, not just a source-file grep — the string is built from adjacent literal
  concatenation across lines, so a plain multi-line grep alone would not have found it);
  `WorkflowTrigger.__init__` (`server/falkorchat/trigger.py:35-47`) confirmed to take `def_key`/
  `def_version` directly, independent of `config.TRIGGER_DEF_KEY`/`_VERSION`, exactly as cited;
  `Repository.materialize_snapshot` (`repository.py:2584`) confirmed to write into `self._graph(ws)`
  only, never `reference`, exactly as cited. `ws:ds-k062-lev-baseline`/`-lever-a`/`-lever-b`
  confirmed absent from the live loaded-graphs list; `reference`/`ws:acme` re-verified `OK` via
  `verify_salesperson.sh acme`, `verify_catalog.sh`, `verify_workflows.sh acme` (all three,
  independently re-run by `teco`). Both `kaizen_team` writes confirmed present via direct query,
  correctly attributed to `data-scientist`.
  **Verdict: eval accepted as delivered — ship neither lever.** Findings folded into
  `docs/BACKLOG.md` K-062 directly by `teco` (rewritten in place per the revisit convention): rate
  history consolidated to four samples, the occurrence-rate swing finding foregrounded as a
  standing caution for the whole script family, the mechanism-lead replication noted, both levers'
  results and rejection reasons recorded, and the earlier "materially more comparable from here
  on" claim about the `temperature: 0` pin corrected in place per §18.7's own finding that the pin
  does not reliably narrow the swing. Header changed to reflect closure: no fix shipped, closed
  pending only a re-worded lever (a). No `docs/HISTORY.md` entry — diagnosis/eval only, no shipped
  behavior change beyond the already-recorded config pin, consistent with this thread's own
  established precedent (K-060 round 4, K-062 round 5's own U1) of not logging diagnosis-only
  passes to HISTORY.
- **K-062's own fix-decision thread is now closed** — both named candidate levers evaluated with
  good statistical power and a clear, evidence-backed "ship neither" result; the diagnosis itself
  (U1) and the fix decision (U2) this round set out to produce are both delivered and verified.
  This round's own scope is complete — flipping this coordination doc's own `Status:` to
  `archived` as part of the same close-out, per `teco`'s own ownership of `plans/*-coordination.md`
  archival (root `AGENTS.md`).
