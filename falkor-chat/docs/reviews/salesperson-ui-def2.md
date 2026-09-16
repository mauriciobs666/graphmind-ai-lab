# Review: salesperson-ui U-DEF2 (cart-item price field rename) — implementation

> **Status:** active · **Owner:** `analyst` · **Tracks:** U-DEF2 / DEF-2 (`docs/plans/salesperson-ui2-coordination.md`; `docs/test-reports/salesperson-ui-report.md`)

**Deviation from the requested path:** the brief suggested `salesperson-ui-s15-def2.md`. This fix
is not part of S15 (the QA pass that *found* DEF-2, `docs/plans/salesperson-ui2-coordination.md`'s
S15 row) — it is its own follow-up unit, tracked in the same ledger as **U-DEF2**. Naming the file
after S15 would misattribute the fix to the QA unit rather than the fix unit; `salesperson-ui-def2.md`
mirrors the ledger's own unit name and leaves room for a parallel `salesperson-ui-def1.md` etc.
without collision. House style (`falkor-chat/docs/reviews/salesperson-ui-s17-impl.md`) confirms
`falkor-chat/docs/reviews/` is the right directory for this family.

## Scope & verdict

Reviewed the uncommitted `salesperson` working-tree diff fixing DEF-2 (`docs/test-reports/
salesperson-ui-report.md`'s DEF-2 section, read in full) — `CartPanel.tsx` reading `item.unitPrice`
against a server response shape that actually carries `price`, rendering `$NaN` per cart line.
Baseline: `git diff -- salesperson/src/views/CartPanel.tsx salesperson/src/api/endpoints.ts
salesperson/src/views/CartPanel.test.tsx` (the full, only diff — confirmed via `git status --short
salesperson/`, which shows exactly these three files and nothing else). Cross-checked against the
server code the fix's own reasoning cites (`falkor-chat/server/falkorchat/services.py`,
`tools.py`) and grepped the whole client tree for any other consumer of the renamed field.

**Verdict: approve.** The fix is minimal, correctly scoped, and correct. The rename direction
(client → server's actual field name, not the reverse) is the right call given the verified blast
radius on the server side. The fixture-typing hardening is real, not cosmetic — independently
reproduced failing at `tsc -b` under a simulated future drift. No other client code path reads
`.unitPrice` on a cart item. Two non-blocking notes below.

**CPG:** considered, not relevant — `cpg_salesperson` does not exist in this FalkorDB instance
(confirmed directly via `GRAPH.LIST`/`GRAPHS`: only `cpg_deprecated_salesperson` and
`cpg_falkorchat` are loaded, neither of which models this TypeScript client tree); this is a
three-file source diff small enough to read and grep directly, not a call-graph/impact-analysis
question a CPG would answer differently.

## Findings

### None blocking, none major

No blockers or majors found. Two minor/informational notes below, neither gating.

### Informational — verified the fix direction independently, not just accepted the stated reasoning

The implementer's stated blast-radius reasoning checks out on direct inspection, not just as
narrated:

- `falkor-chat/server/falkorchat/tools.py:535` (`ViewCartTool.run`) does
  `json.dumps(self._services.get_cart(ctx))` — `get_cart`'s dict is serialized **verbatim** to the
  live LLM agent as a tool-call result, confirming a server-side rename would change the
  agent-facing tool contract, not just an internal detail.
- `services.py` uses the `"price"` key at 5 internal touchpoints: `_priced_cart_lines` (:2899,
  :2903 — the shared two-graph read both `get_cart` and `place_order` route through),
  `add_cart_item` (:2936, its own independent dict), plus the two callers (`get_cart` :2947,
  `place_order` :3028). A server-side rename would have touched all of these, not a single field.
- `place_order` (:3031-3038) deliberately remaps `line["price"]` → `unitPrice` when building
  `OrderBlock.lines` — confirming `OrderBlock`'s `unitPrice` naming is a genuinely separate,
  already-correct server contract, not something this fix should (or does) touch. Client-side,
  `OrderPanel.tsx` reads `OrderBlock.lines` through its own runtime type guard against
  `endpoints.ts`'s deliberately-`unknown[]`-typed `lines` field — untouched by this diff, correctly.
- Grepped the whole client tree (`grep -rn "unitPrice\|\.price\b" salesperson/src`) for any other
  reader of a cart item's price field: the only production hit is `CartPanel.tsx:61`, now fixed.
  `OrderPanel.tsx`'s `unitPrice` usage is the separate `OrderBlock` contract, correctly untouched.
  No other client code path was missed.

### Informational — the fixture-typing hardening was independently mutation-tested, not just read

Per `claude/analyst/review-techniques.md`'s zero-touch scratch-copy technique: copied
`salesperson/` to a scratch directory, edited only the scratch copy's `CartPanel.test.tsx` fixture
literals from `price:` back to `unitPrice:` (simulating a future fixture drift with the interface
and `CartPanel.tsx` left as the fix landed them), and ran `npx tsc -b` there. Result: **3 real
compile errors** (`TS2353: Object literal may only specify known properties, and 'unitPrice' does
not exist in type 'CartItem'`) at exactly the mutated lines — confirming the excess-property check
fires because the fixture object literals are passed directly into a `CartItem[]`-typed parameter.
This is genuine protection against the exact class of regression DEF-2's own report named ("a
hand-typed `unitPrice` fixture here is what let the real `$NaN` regression through undetected"),
not just a plausible-sounding comment. Separately, reverting only `CartPanel.tsx`'s field read (not
the fixture) in a second scratch copy reproduced the pre-fix bug exactly: `vitest run` failed with
the DOM literally rendering `$NaN` and the test's `expect(screen.getByText('$59.98'))` failing —
the same signature QA's report describes. Real tree confirmed untouched by either mutation
(`git status --short salesperson/` unchanged before/after); real-tree `tsc -b` and `vitest run
src/views/CartPanel.test.tsx` (7/7) also independently reproduced clean.

## What's solid

- Scope discipline: exactly the three files needed, no incidental changes, no drive-by refactor.
- The inline code comments in both `endpoints.ts` and `CartPanel.test.tsx` explain *why* the field
  is named `price` and explicitly flag `OrderBlock.lines`'s different, correctly-untouched
  `unitPrice` contract — a future editor won't have to re-derive this from the git history.
- The fix closes the actual root cause (client assumption vs. server reality), not a workaround —
  matches DEF-2's own "Recommendation" section exactly (rename one side to match the other, harden
  the fixture typing).

## Open questions

None for this review. The coordination ledger (`docs/plans/salesperson-ui2-coordination.md`,
U-DEF2 row) already tracks the remaining process step (closing the gate, committing, and a
`docs/HISTORY.md` entry at delivery) — outside this review's scope.
