# `workflow-catalog-lookup` — Test Report (Ministral re-verification)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-052, K-056 (M6)

## Summary

Regression re-verification of K-052's full acceptance-test suite, executed 2026-08-29 against the
real running system — FalkorDB (`falkordb-dev`), the M1 server on `http://localhost:8010`, and LM
Studio on `http://localhost:1234` now serving `mistralai/ministral-3-3b` (re-pointed from
`qwen/qwen3-4b-2507` by K-056, commit `03a3c8c`, `salesperson@v2.1`) — per
`docs/test-plans/workflow-catalog-lookup2.md`. Workspace: a fresh throwaway
`ws:qa-catalog-lookup2`, provisioned and verified in sync before any test item ran. Reached
Ministral the same way prior units in this coordination did: a session-scratch
`FALKORCHAT_OPENCODE_CONFIG` copy with `baseURL` corrected to `http://localhost:1234` (the shared
`~/.config/opencode/opencode.json`'s gateway-IP `baseURL`, `http://192.168.0.69:1234`, reconfirmed
unreachable from this WSL2 box this session).

**Verdict: PASS WITH DEFECTS.** All five acceptance criteria (AC-1..AC-5) hold against the live
system on `mistralai/ministral-3-3b`. **The central question this pass exists to answer — does the
qwen-era D-1 fabrication mechanism (`docs/test-reports/workflow-catalog-lookup-report.md`)
reproduce on Ministral — is answered no.** The exact same 9-question, one-continuous-conversation
sequence the parent report's D-1 fabricated on (including the identical rerun-at-the-same-slot
methodology) was replayed turn-for-turn against Ministral: every turn returned the correct fact,
list, or abstention, cross-checked against ground-truth Cypher and against `Message.toolsUsed`
(a genuine tool dispatch on every fact-bearing turn, not a cached/pattern-matched repeat). One new,
non-blocking, cosmetic defect was found (D-1 below: an inconsistent currency symbol, `$` vs. `€`,
across otherwise-identical replies) — informational, does not affect any AC's correctness.
Ministral's own known duplicate-instruction defect (ml-note §9) was watched for opportunistically
(TP-16) but has no direct analogue on this feature's read-only tool surface; not observed here (the
dedicated, higher-signal probe for it lives in the K-053 re-run, which has a write-tool surface).

**CPG:** considered, not relevant — this is a live acceptance re-run against a changed model
dependency; no code under test changed shape (only a `config.model` value did), so this is not a
structural-impact-analysis question. `proof_defs.py`/`repository.py`/`tools.py` were read directly
where relevant rather than queried through `cpg-analysis`.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-01 | PASS | `redis-cli PING` → `PONG`; `GET :1234/v1/models` lists `mistralai/ministral-3-3b` (confirmed fresh this session, not assumed from U43); `server/.venv` imports `falkorchat` cleanly. |
| TP-02 | PASS | `bootstrap_schema.sh qa-catalog-lookup2` (dim 1024) exit 0; `seed_demo.sh qa-catalog-lookup2` exit 0; `seed_catalog.sh` → 15 `Product` rows in `reference` (no duplication, already-present catalog); `seed_salesperson.sh qa-catalog-lookup2` → `reference` def `salesperson@v2.1` already present (no-op, correct — published by U43), `ws:qa-catalog-lookup2` snapshot materialized fresh. |
| TP-03 | PASS | `verify_catalog.sh` → `RESULT: OK` (15 products); `verify_salesperson.sh qa-catalog-lookup2` → `RESULT: OK`, topology "2 steps (assistant/agent, ended/decision), 1 transition" — unchanged from the original v1 pass. |
| TP-04 | PASS | Server started bound to `qa-catalog-lookup2`/`salesperson@v2.1` with the corrected `FALKORCHAT_OPENCODE_CONFIG`; log shows `model provider lmstudio: baseURL http://localhost:1234 -> http://localhost:1234/v1 (rule)`; `GET /health` → `{"status":"ok"}` on first attempt (no connectivity issue this time, unlike the original pass's pre-fix failure). |
| TP-05 | PASS | `POST /channels/demo-general/threads {"title":"qa-workflow-catalog-lookup2"}` → `201`, `threadId 35d641fa66b64a6aa98f8a1f4308a247`. |
| TP-06 (AC-1) | PASS | "How much does the Wireless Mouse Pro cost?" → `toolsUsed:["lookup_product_fact"]`; reply "The Wireless Mouse Pro costs €29.99. It's in the Peripherals category." — value correct ($29.99, matches `reference`), currency **symbol** wrong (`€` not `$`) — see D-1. |
| TP-07 (AC-2, category) | PASS | "What Wearables products do you have?" → `["filter_products"]`; "Fitness Tracker Band: €79.99, Smartwatch Series 5: €249.99" — correct pair, correct prices, no extras (symbol quirk again, see D-1). |
| TP-08 (AC-2, price range — **D-1 repro slot #1**) | PASS | Turn 3 of the continuous conversation (the exact slot the parent report fabricated 2 phantom products on): reply lists exactly the correct 3-item set — Gaming Mouse Pad XL ($19.99), Wireless Charging Pad ($24.99), Wireless Mouse Pro ($29.99) — `toolsUsed:["filter_products"]`. Rerun (same question, same thread): identical correct set, this time with `$` symbols. No fabrication on either attempt — direct disconfirmation of the qwen-era mechanism at this exact conversational position. |
| TP-09 (AC-3, product) | PASS | "How much does the Quantum Toaster 3000 cost?" → `["lookup_product_fact"]`; "I don't have any product named Quantum Toaster 3000 in our catalog." — plain abstention. |
| TP-10 (AC-3, category) | PASS | "Do you have any Furniture products?" → `["filter_products"]`; "We don't currently have any Furniture products in our catalog." — plain abstention. |
| TP-11 (AC-4a — **D-1 repro slot #2**) | PASS | Turn 6 (the exact slot the parent report fabricated $149.99 on): "How much is the Portable SSD 1TB?" → `["lookup_product_fact"]`; "$109.99" — correct. Rerun (same thread): "€109.99" — correct value again, symbol quirk again. |
| TP-12 (AC-4b) | PASS | "What's the price of the Portable SSD 1TB?" (same thread, turn 7) → `["lookup_product_fact"]`; "€109.99" — matches TP-11 exactly; genuinely re-dispatched the tool each time (not a cached repeat — `toolsUsed` non-empty on every one of the three Portable-SSD turns). |
| TP-13 | PASS | Direct Cypher against `ws:qa-catalog-lookup2` (`WorkflowRun`→`StepRun`→`PRODUCED`→`Message`) confirms all 9 turns' persisted `Message.text` match the REST-reported replies exactly, including the `$`/`€` symbol variation (genuine model output, not a display artifact) and ground-truth prices for all 6 named products (`Fitness Tracker Band 79.99`, `Gaming Mouse Pad XL 19.99`, `Portable SSD 1TB 109.99`, `Smartwatch Series 5 249.99`, `Wireless Charging Pad 24.99`, `Wireless Mouse Pro 29.99` — all match `reference` exactly). |
| TP-14 (AC-5) | PASS | Re-ran `seed_catalog.sh` → still 15 `Product` nodes; re-ran `seed_salesperson.sh qa-catalog-lookup2` → **both** `reference` def and `ws:qa-catalog-lookup2` snapshot report "already present — no-op" (genuine idempotence, not just the def side); both verify scripts still `OK` afterward. |
| TP-15 | PASS | `WorkflowRun{runId:13b1851a...}.status` = `waiting` after all 9 turns; `MATCH (sr:StepRun {stepKey:'ended'}) RETURN count(sr)` → `0` across the whole workspace. |
| TP-16 | Not observed (informational) | Reviewed all 9 turns' `toolsUsed`/reply text for an unprompted repeated/duplicated fact — none found; every tool call's target matched the current turn's own question. K-052's read-only surface gives this pattern little room to manifest; the higher-signal write-tool probe is K-053's own TP-16. |

## Defects

### D-1 (MINOR, new — cosmetic, not gating any AC) — inconsistent currency symbol (`$` vs. `€`) across otherwise-identical replies

**Severity:** MINOR — every price *value* observed across all 9 turns was correct; only the
currency *symbol* the model chooses to render is inconsistent.

**Steps to reproduce:** ask the same exact question twice in one conversation (e.g. "What products
do you have under $30?", asked with a literal `$` in the question) — turn 3's first answer rendered
all three prices with `€` (e.g. "Gaming Mouse Pad XL – Peripherals, €19.99"); the immediate rerun of
the identical question rendered the same three correct prices with `$` instead. The same
inconsistency recurred independently on the Portable SSD 1TB question (turn 6: `$109.99`; turn 6's
rerun: `€109.99`; turn 7: `€109.99`).

**Expected:** a consistent currency symbol (the catalog and every tool response use plain numeric
`price` values with no currency marker at all — `$` is the natural convention for a USD-priced
electronics catalog with no internationalization in scope).

**Actual:** the model alternates between `$` and `€` unpredictably, even for the identical fact
re-asked seconds later in the same conversation.

**Impact:** none on any of K-052's AC-1..AC-5 (all are about factual/numeric correctness, not
currency-symbol formatting) — this is a new observation from live testing, not present in the
original qwen-era report (which used `$` consistently in all 8 of its own captured replies) and not
part of this pass's own D-1/D-2 watch list. Flagged because a customer-facing "salesperson" stating
a wrong currency, even with the right number, is a small but real trust/clarity gap worth a
system-prompt tweak (e.g. an explicit "always use `$`" instruction) if the team wants to close it —
not blocking.

## Coverage & gaps

**Covered:** all five ACs, live, on the new model; the exact D-1 repro sequence replayed
turn-for-turn (including both original fabrication slots and both same-thread reruns); graph-level
ground truth for every one of the 9 turns; seed/verify idempotence (genuine no-op on both sides this
time, not just `reference`); the scaffold's safety-critical guard property.

**Gaps, consistent with the parent plan's own disclosed scope:**
- No literal tool-call-argument tracing (same `@mention`-path limitation as the original pass) —
  substituted by `Message.toolsUsed` (available on this build, wasn't leaned on as heavily in the
  original qwen-era report) plus ground-truth-outcome cross-checks.
- TP-16's watch is opportunistic on K-052's read-only surface — it has no write-tool analogue to
  probe directly; K-053's own re-run (`workflow-cart-and-totals2.md`) carries the substantive,
  dedicated duplicate-instruction probe.
- No independent statistical characterization of the currency-symbol quirk's rate — observed on 2
  of 9 turns in this single conversation, not chased further (a display-formatting nit, not this
  pass's central question).

## Feedback & recommendations

1. **Ship the Ministral re-point for K-052's own scope — no regression found.** The qwen-era D-1
   mechanism does not reproduce anywhere in this pass, replayed against the exact sequence that
   found it originally. Every AC holds.
2. **D-1 (currency symbol) is a cheap, worth-fixing polish item**, not a blocker — a one-line
   system-prompt addition ("always express prices with a `$` prefix, never `€` or another symbol")
   would likely close it; not verified here (out of QA's remit to modify the system under test).
3. **Testability note, reused from the parent report's own feedback #4:** replaying a
   previously-failing long-conversation sequence turn-for-turn (rather than a fresh, easier
   sequence) is the right regression-check design when the specific claim under test is "does this
   exact defect still occur here" — worth keeping as the standard technique for any future
   model-swap re-verification in this coordination.
4. **`docs/test-plans/workflow-catalog-lookup2.md` and this report together `Extend`
   `docs/test-plans/workflow-catalog-lookup.md`** per collision rule 5 (ordinal-on-slug,
   `llm-provider-config`/`llm-provider-config2` precedent) — the original test-plan/report stay
   intact as the qwen-era historical record; this cycle is additive, not a replacement.

## Artifacts left in the live demo (disclosed, not cleaned up)

- Workspace `ws:qa-catalog-lookup2` (fresh, this pass's own): `demo-general`/`demo-welcome` from
  `seed_demo.sh`, plus 1 test thread (`qa-workflow-catalog-lookup2`) carrying 1 `WorkflowRun` and 9
  turns of messages. Left in place for `teco`'s own disposition, same precedent the original
  `ws:qa-catalog-lookup` pass set.
- `reference` graph: unchanged by this pass beyond the already-idempotent no-ops TP-02/TP-14 show.
- No shared files were modified — the corrected LM Studio config lives only in this session's
  scratchpad, referenced only via `FALKORCHAT_OPENCODE_CONFIG` for this pass's own server process,
  stopped at the end of this test item sequence.
- `ws:acme`/`reference` were untouched by this pass directly; both were restored and re-verified as
  part of the shared final regression step (see `workflow-cart-and-totals2-report.md`'s own TP-15,
  the single shared final step for both re-runs).
