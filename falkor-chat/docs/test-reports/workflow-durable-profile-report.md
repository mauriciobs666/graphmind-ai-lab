# `workflow-durable-profile` — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-054 (M6)

## Summary

Live acceptance pass for K-054 (durable, workspace-scoped customer name/delivery-address capture)
executed 2026-08-29 against the real running system — FalkorDB (`falkordb-dev`), a fresh M1
server instance on `http://localhost:8021` bound to a throwaway workspace, LM Studio at
`http://localhost:1234` serving `mistralai/ministral-3-3b` — per
`docs/test-plans/workflow-durable-profile.md`. Code under test: `salesperson@v3` (commits
`663093d`/`36096d0`, HEAD at test time). Workspace: fresh throwaway `ws:qa-durable-profile`,
provisioned and verified in sync (catalog, both workflow defs, correct `salesperson@v3` topology)
before any test item ran; zero pre-existing `Customer` nodes confirmed.

**Verdict: PASS.** All three acceptance criteria (AC-1..AC-3) hold against the live system. The
exact BLOCKER axis `analyst`'s plan-gate review caught and `graph-dba` fixed in v2 (a naive
unconditional `SET` nulling an omitted field on a partial update) was re-proven closed live, in
**both** field directions (TP-04: address-only update preserves name; TP-05: name-only update
preserves address) — not just the one direction the task brief named, at negligible extra cost.
Cross-conversation persistence (AC-1) was demonstrated cleanly: a fresh, independent thread,
same actor, correctly recalled the exact stored name/address with **no re-ask**, on the first
attempt. The two already-shipped sibling capabilities (catalog lookup, cart) showed no regression
in a same-conversation spot-check.

**No defects block this capability's own AC-1..AC-3.** One pre-existing, already-tracked catalog
MINOR (not new, not part of this capability, noted for completeness — see Coverage & gaps) was
observed opportunistically.

Final offline regression suite green, matching the last recorded baseline exactly: **1935
passed / 4 deselected** (`pytest`), **408/408** (`test_queries.sh`, including the new §17 profile
assertions); `ws:acme`/`reference` fully restored (`bootstrap_schema.sh acme` →
`seed_demo.sh acme` → `seed_catalog.sh acme` → `seed_workflows.sh acme` →
`seed_salesperson.sh acme`) and re-verified in sync (`verify_workflows.sh`/`verify_catalog.sh`/
`verify_salesperson.sh acme` all `OK`).

**CPG:** considered, not relevant — live acceptance testing of a small, already-read diff (two
nullable scalar properties on an existing node, thin service/tool functions); no
structural-impact-analysis question this capability raises. `repository.py`/`services.py`/
`tools.py`/`proof_defs.py` were read directly where relevant, matching the plan's own reasoning.

## Results

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-01 | — | PASS | `PONG`; `mistralai/ministral-3-3b` listed at `localhost:1234`; `pytest -k profile -q` → 15 passed/1924 deselected; `ws:qa-durable-profile` bootstrapped (dim 1024), demo/catalog/workflow-seeded; `verify_catalog.sh`/`verify_workflows.sh qa-durable-profile`/`verify_salesperson.sh qa-durable-profile` all `OK` (salesperson@v3 topology: 2 steps/1 transition); `MATCH (c:Customer) RETURN count(c)` → `0` before any test item; server up on port 8021 with the corrected `FALKORCHAT_OPENCODE_CONFIG` (log: `baseURL http://localhost:1234 -> http://localhost:1234/v1`); `GET /health` → `{"status":"ok"}`. |
| TP-02 (AC-1 write) | AC-1 | PASS | Thread A: `@assistant Hi, my name is Jane Doe and my delivery address is 123 Main St, Springfield.` → `toolsUsed:["save_profile"]`; ground truth: exactly 1 `Customer{customerId:"u1"}` with `name='Jane Doe'`, `deliveryAddress='123 Main St, Springfield'`, `profileUpdatedAt` set (1788034765564). |
| TP-03 (AC-1 read) | AC-1 | PASS | Fresh, independent Thread B (new `threadId`, same actor `u1`): turn 1 (`"do you have any wireless mice in stock?"`) did not itself probe the profile (see Coverage & gaps for an unrelated catalog observation on this turn); turn 2, a direct probe (`"Before I order anything, do you already have my name and delivery address on file?"`) → `toolsUsed:["get_profile"]`, reply: *"Yes! I already have your name as **Jane Doe** and delivery address as **123 Main St, Springfield**."* — correct values, **no re-ask**. Ground truth confirms `profileUpdatedAt` unchanged from TP-02 (read-only call, no incidental write). |
| TP-04 (AC-2, address-only) | AC-2 | PASS | Same Thread B, next turn: `@assistant My delivery address has changed to 456 Oak Ave, Springfield.` (name not repeated) → `toolsUsed:["save_profile"]`; reply confirms the address update. Ground truth: `name` **unchanged** (`'Jane Doe'`, not nulled — the exact BLOCKER axis), `deliveryAddress` updated to `'456 Oak Ave, Springfield'`, `profileUpdatedAt` bumped (1788034828209). |
| TP-05 (AC-2, name-only — symmetric direction) | AC-2 | PASS | Same Thread B, next turn: `@assistant Actually, please call me Jane Smith instead.` (address not repeated) → `toolsUsed:["save_profile"]`; reply confirms. Ground truth: `name` updated to `'Jane Smith'`, `deliveryAddress` **unchanged** (`'456 Oak Ave, Springfield'`, not nulled), `profileUpdatedAt` bumped (1788034841415) — confirms the `coalesce()` fix holds in **both** field directions live. |
| TP-06 (AC-3) | AC-3 | PASS | `verify_salesperson.sh qa-durable-profile` → `salesperson@v3`, topology 2 steps (`assistant`/agent, `ended`/decision), 1 transition — unchanged from v2.1; `proof_defs.py` confirms `SALESPERSON_DEF["key"] == "salesperson"` (no new def key) — the two profile tools and extended `systemPrompt` are `config` additions to the same existing def. |
| TP-07 | — | PASS | Thread C: `@assistant How much is the Wireless Mouse Pro?` → `toolsUsed:["lookup_product_fact"]`, reply: *"$29.99"*, category Peripherals — matches the fixture catalog exactly. `@assistant Yes, add 1 Wireless Mouse Pro to my cart.` → `toolsUsed:["add_to_cart","view_cart"]`, reply confirms; ground truth: exactly 1 `CartItem{productId:"wireless-mouse-pro", quantity:1}` under `u1`'s `Cart` — no regression from the K-052/K-053 baseline. |
| TP-08 | — | PASS | Re-ran `seed_salesperson.sh qa-durable-profile` — both `reference` defs and both `ws:qa-durable-profile` snapshots report "already present — no-op"; `verify_salesperson.sh qa-durable-profile` still `OK`, same topology. |
| TP-09 | — | PASS | `pytest -q` → **1935 passed / 4 deselected** (matches the K-054 cluster-2 commit's own recorded baseline exactly); `./scripts/test_queries.sh` → **408/408**, including the new §17 profile assertions (both partial-update directions, both `[verified]` in the graph note, now re-confirmed by the suite). `ws:acme`/`reference` restored (`bootstrap_schema.sh`→`seed_demo.sh`→`seed_catalog.sh`→`seed_workflows.sh`→`seed_salesperson.sh`, all `acme`) and re-verified: `verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh acme` all `OK`/in sync (salesperson@v3, correct topology). |

## Defects

None found against this capability's own scope (AC-1..AC-3). No regression found on the two
already-shipped sibling capabilities in the spot-check (TP-07).

## Coverage & gaps

**Covered:** all three ACs, live, against the real model and the real graph — a full name+address
write, cross-thread persistence with a direct "do you have my info" probe eliciting the exact
correct values with no re-ask, and both partial-update directions of the `coalesce()` fix (the
precise axis the plan-gate BLOCKER was found and fixed on), each backed by ground-truth Cypher
taken immediately after the live turn, never trusting reply text alone. AC-3's structural claim
(same combined def) confirmed via both the seeded topology and the source constant.

**An observation, not a defect against this capability:** Thread B's first turn ("do you have any
wireless mice in stock?") did not trigger a `get_profile` call at all — the `systemPrompt`
instructs the model to check the profile "once, early in the conversation" regardless of topic,
but Ministral treated a pure catalog question as not warranting the check. This did not violate
AC-1 (no re-ask happened, because no profile-related question was asked in that turn either), and
the very next turn in the same thread, once the customer's own question actually implicated their
profile ("do you already have my name and address on file?"), correctly triggered `get_profile`
and answered exactly right. Recorded as a prompt-discipline observation per the plan's own §6 risk
note (distinct from a persistence bug), not gating.

**A separate, pre-existing, already-tracked catalog issue, unrelated to K-054, noted for
completeness only:** that same Thread-B turn's `filter_products` call answered "I don't have any
wireless mice in stock right now" even though "Wireless Mouse Pro" (category Peripherals) is in
the fixture catalog — consistent with the already-open MINOR from
`docs/reviews/workflow-catalog-lookup-impl.md` (`filter_products`'s `categoryNormalized`
NULL-coalesce edge, likely a category-token mismatch between "mice" and "Peripherals"). This is
K-052 scope, already tracked, not re-investigated here (out of scope per this pass's brief, which
asked for a cheap regression spot-check, not a re-hunt) — TP-07's own direct-name lookup
(`lookup_product_fact`) on the same product answered correctly, confirming the gap is specific to
the category/filter path, not a broader catalog regression.

**Gaps, disclosed, consistent with the plan's stated scope:**
- **Multi-tenant/distinct-customer profile isolation** not demonstrable until K-016 (M2.5) lands
  — same caveat every prior live pass in this coordination carries.
- **No statistical measurement** of Ministral's own already-characterized duplicate-instruction
  defect was attempted here — out of this capability's scope; none was observed opportunistically
  across the ~5 mutating profile/cart turns in this pass.
- **AC-2's symmetric (name-only) direction (TP-05) was not literally required** by the task
  brief's own AC-2 wording (which names only the address-update case) — included anyway since it
  is the same bug axis and cost one extra live turn.

## Feedback & recommendations

1. **The BLOCKER fix holds live, in both field directions.** This was the pass's central
   question — `analyst`'s plan-gate catch (v1's unconditional `SET`) is now proven closed not just
   at the graph-note probe-script level and the fixture-level repository tests, but through the
   real tool-call path with a real model choosing when to call `save_profile` and with what
   arguments. No further action needed.
2. **AC-1's "does not ask again" held cleanly on the first attempt** — no re-ask observed. Given
   this coordination's history of intermittent reply-completeness gaps on other tools (D-1/D-2 in
   the cart-and-totals re-run), a single clean pass is good evidence but not exhaustive; a future
   pass wanting a stronger statistical claim would repeat TP-03's probe across several fresh
   threads rather than rely on one.
3. **The systemPrompt's "call `get_profile` once, early" instruction is not always honored on a
   turn that doesn't itself implicate the profile** (the Coverage & gaps observation above) — this
   is arguably fine (nothing was lost; the very next relevant turn checked correctly), but if a
   future requirement wants the model to proactively greet a returning customer by name on *any*
   first turn of a new conversation regardless of topic, this prompt wording alone may not
   reliably produce that — worth a dedicated eval if that requirement ever lands.
4. **Testability note, reused from every prior live pass in this coordination:** ground-truth
   Cypher checked immediately after each live turn (never deferred) is what let TP-04/TP-05
   distinguish "the reply says it saved" from "the graph actually preserved the other field" —
   the same discipline that caught the original BLOCKER at design time, now confirmed unbroken on
   the production path.
5. **`docs/test-plans/workflow-durable-profile.md`/this report are the first cycle for this
   topic slug** — no ordinal needed, no `Extends`/`Supersedes` pointer required.

## Artifacts left in the live demo (disclosed, not cleaned up)

- Fresh disposable workspace `ws:qa-durable-profile` — left in place, not deleted, same precedent
  every prior QA pass in this coordination has set. Carries this pass's own test data: 4 threads
  (`demo-welcome` plus three QA threads — A/B/C) under channel `#qa-profile`; one `Customer`
  (`u1`, `name='Jane Smith'`, `deliveryAddress='456 Oak Ave, Springfield'` — final state after
  TP-05); one `Cart` with one `CartItem` (`wireless-mouse-pro`, quantity 1, from TP-07).
- The QA server process (port 8021) was stopped before the final regression run and not
  restarted — no server left running against the throwaway workspace.
- The session-scratch `FALKORCHAT_OPENCODE_CONFIG` copy (corrected `baseURL`) lives only in this
  session's scratchpad, referenced only for this pass's own server process — not written into the
  repo, matching every prior unit in this coordination.
- `ws:acme`/`reference` — restored and re-verified in sync (see TP-09); no lingering test data
  from this pass in the shared workspace.
