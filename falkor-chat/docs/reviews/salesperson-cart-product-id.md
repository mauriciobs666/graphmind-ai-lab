# Review: cart writes keyed by product ID (plan gate)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-078

## Pass 1

### Scope & verdict

**Reviewed:** `falkor-chat/docs/plans/salesperson-cart-product-id.md` (Status `active`, pre-implementation),
checked against the tree at `ac39345d`: `executor.py`, `tools.py`, `services.py`, `repository.py`,
`storefront.py`, `proof_defs.py`, `seed_salesperson.sh`/`verify_salesperson.sh`, the existing guard
tests, and the TP-007 probe. Also read: BACKLOG K-078/K-079, the K-079 RCA, and ml review §9/§12/§15.
The baseline was re-derived from the U3 artifacts plus read-only queries on `ws:demo`/`reference`/
`ws:acme`. No tests were run. The plan changes no code yet, and every claim checked was a static or
data claim.

The stakeholder mandate (writes keyed by a stable ID) was not reviewed.

**Verdict: needs changes.** There are no blockers, but four majors. The biggest is that the K-058
protection the plan keeps by design (D3) is weaker than it says: the hold message itself tells the
model how to get a re-fire through, and the live check cannot observe a re-fire. Each fix is a few
lines of plan text, a test, or one extra live condition. No redesign is needed.

CPG: considered, not relevant — `cpg_falkorchat` is stale (built 2026-10-15, 12+ server commits behind); every symbol was read from source instead.

**Brief questions, short answers:**
1. **Premise true.** `Executor._run_agent_node` builds the opening messages once, via
   `_assemble_messages`, from three things only: `systemPrompt`, the `_read_thread_context` text
   turns (window 20), and `CONTEXT` = `run_ctx`. Tool results are appended to the local `messages`
   list only, and that list dies with the node execution. `_record` persists `result.output` as
   `StepRun.output`, not into `run_ctx`. So an earlier turn's tool result never reaches the model.
   The only cross-turn carrier of an ID is the assistant's own posted *text*, and D3 deliberately
   does not trust text.
2. **The S2+S3 coupling is real** (traced in `_handle_tool_call`). After S2 alone, the target value
   is `None`, the `isinstance(str)` check skips K-058, and K-061 still keys. After S3 alone, every
   call is held. **A second coupling is missing from the plan: S4+S5** (F2). The guard's state
   space is covered in F1, F3 and F4.
3. **The baseline reproduces exactly** under the plan's definition. Other definitions spread from
   0 to 75 for pt-BR (Appendix A). The bar is reachable but not assured, and the latency gate is
   tighter than it looks (F5, "View on the bar").
4. **Callers.** `storefront.py` writes no cart. It only reads `get_cart`/`filter_products`, and no
   SPA button writes the cart; the only write path is chat. Non-test callers of
   `add_cart_item`/`remove_cart_item` are the two tools only (confirmed by grep). The §7 docs list
   is complete apart from F2's pin coupling. Two mutants are defective (F4).
5. **The concurrency question is moot now.** F1 and its numeric-string follow-up landed as
   `ac39345d` while this review was running. The overlap is textual adjacency only, with no
   semantic overlap (F7).

### Findings

#### F1 — major — The hold message tells the model how to get a re-fire through, and §6.3 cannot see re-fires

D3 keeps K-058 for the §9.2 mechanism: an in-turn re-issue of an earlier turn's add, after a
correct add of a different product. But the new hold `reason` (S3 details) says "Look the product up
first with `lookup_product_fact` … then pass the `productId` that result returns". For a re-fire,
that is exactly the recovery path. The model calls `lookup_product_fact("Wireless Mouse Pro")`, gets
the ID back, and the re-add dispatches.

The old reason ended with "only call this tool for something the customer actually asked for in
their current message", and the new one drops that clause. Separately, on the pt/es path, D2's
miss list grounds **all 15 products** for the rest of that turn. So for pt/es the guard holds
nothing after the first lookup.

Test 12 stops at "mouse held", so it never shows the next iteration. §6.3 cannot detect a regression
either: TP-007's five turns contain exactly one add, so there is no add-then-add opportunity. The
"accepted and measured" claim (D3 residual) therefore has no measurement behind it.

The baseline contains one relevant data point: in Q3, all 146 held runs held exactly once and never
retried (`demo-q3-window.log`, 146 distinct run ids). The model obeyed the old "don't" wording.

**Fix:**
- (a) Make the reason conditional: "if the customer asked for this product in their current
  message, look it up first …; otherwise do not add it".
- (b) Add a §6.3 condition that reuses ml §9.1's `same-category-immediate` + `distinct-immediate`
  scripts (n ≥ 24 conversations), and report in-turn re-fires against §9.2's 1/24.
- (c) Add an offline test that scripts hold → lookup(held product) → add. It pins the accepted
  behaviour (dispatched), so the depth of the guard is explicit rather than implied.
- (d) Put the residual in §8 as a stakeholder-accepted risk.

#### F2 — major — S4 and S5 must land in one commit

`seed_salesperson.sh` publishes `{**SALESPERSON_DEF, "key": …, "version": SALESPERSON_VERSION}`,
with the version taken from env and defaulting to `v8`. `verify_salesperson.sh` checks the stored
config against the **current** constant.

After S4 (constant = `v9` content) and before S5 (defaults still `v8`), a seed onto a freshly wiped
`reference` stores `v9` content under the name `v8`. A default `pytest` run wipes `reference`, so
this sequence is likely. It is the burned-`v6` shape (`falkor-chat/AGENTS.md` trap 2), and it would
burn `v8`. On the existing `ws:demo`, verify fails loudly instead.

The plan's coupling paragraph names only S2+S3, and S5's done-condition runs seed and verify only
after the fact.

**Fix:** state "S4 and S5 land in one commit". Add a done-condition: `grep -n 'v8' …` on the three
scripts shows history mentions only, **in the same commit** as the `version` bump.

#### F3 — major — The prompt and the `add_to_cart` schema name `view_cart` as an ID source for adds, but D3 holds that

D7 item 1 ("copied exactly from … `lookup_product_fact`, `filter_products` or `view_cart`") and the
suggested `productId` description (S2 details, the same text for **both** tools) both tell the model
that a `view_cart` ID is valid for an add. D3/M1/test 13 hold exactly that.

The natural case is §9.4's legitimate repeat: "add one more mouse" → `view_cart` → `add_to_cart(id)`
→ held, plus one wasted iteration on a 3B model, with the gate already tight.

**Fix:**
- Split the wording per tool: `add_to_cart` sources are `lookup_product_fact`/`filter_products`;
  `remove_from_cart` sources are those plus `view_cart`.
- Do the same for the prompt sentence.
- Add a scaffold assertion that the `add_to_cart` schema text does not contain `view_cart`.

#### F4 — major — Two mutants are wrong: one is equivalent, and the one that matters is missing

- **M5 is an equivalent mutant.** No non-success content carries a `productId` key:
  - the K-058 and K-061 hold JSON is `{held, reason}`;
  - AC-6 and malformed-argument results are plain strings;
  - the `ServiceError` path returns a string.

  Collecting from those contributes nothing, so test 19 cannot redden. S3's done-condition (M1–M6
  red) is therefore unsatisfiable as written.
- **Missing: the "this turn" scope itself.** A mutant that hoists `surfaced` to `self` survives
  every test 12–24, because each test runs one node execution. The executor is a process-wide
  singleton (see `_drive`'s docstring), so that mutant would also share grounding across customers.

**Fix:**
- Replace M5 with "the guard treats the call's own `productId` argument as surfaced" (the
  model's argument grounding itself). That mutant is killed by test 15.
- Add M10, "`surfaced` outlives the node execution", killed by a new test: two `_run_agent_node`
  calls on one executor, where the first surfaces `x` and the second calls `add_to_cart(x)` with
  no lookup → held.
- Fix S3's mutant range: M1–M6 and M9–M10 belong to S3, M7–M8 to S2.

#### F5 — minor — The §6.3 gates need definitional tightening before they can be measured

- **Turn-2 p95 is gated pooled, but the baseline is language-masked.** en turn-2 p95 is 20.1 s;
  pt-BR and es are 10.1 s, because a held turn ends early. The "14.1 s" figure hides that en
  already sits about 9 s from the 29 s line, and every add now costs at least one extra iteration.
  Report turn-2 p95 per language, and keep the gate pooled.
- **M3 misses a doubled purchase that is split across order and cart** (an order line ×1 plus a
  re-added cart line ×1). Define M3 as the participant's total mouse quantity across `CartItem` and
  `OrderLine` being greater than 1. The baseline is still 0 (en "both" = 0, Appendix A).
- **"Wrong-language ≤ baseline CI" is ambiguous.** State a count, e.g. ≤ 3/375 per language
  (the Wilson upper bound for 0/375 is about 1.0%).
- **The coordination DoD says "no rise in held … writes".** By design, holds will rise (guessed
  slugs, R1). Reword the DoD to "no rise in wrong-product/doubled writes", which is what §6.3
  gates.

#### F6 — minor — A failed `lookup_product_fact` returns names without prices, which invites a fabricated price (K-079 class)

For a pt/es price question ("quanto custa o Mouse sem fio Pro?"), the miss now returns the product's
real name but no price. The prompt forbids guessing, but trial 14's $45.99 shows that this 3B model
invents a price when none is in context (RCA §3).

**Fix:**
- Make the miss shape `{"found": false, "hint": "call lookup_product_fact again with the exact
  catalog name for its price or category", "catalog": […]}`.
- Add to §6.3, reported only, the RCA's name→price corpus sweep (baseline 31 wrong of 79 stated).

#### F7 — minor — §4, R6 and §9 describe F1 as in flight, but it has landed

`ac39345d` put the blank-category rule in `Repository.filter_products`, not in
`FilterProductsTool.run` as §4 says. It also added `_coerce_price_bound` to `tools.py`, directly
after `LookupProductFactTool.run`. S2's edit to that `run()` is adjacent, so it would conflict
textually only if it were developed from a pre-`ac39345d` base. There is no semantic overlap:
`catalog_index` calls the repository with `category=None`.

**Fix:** rebase the wording ("F1 landed `ac39345d`; S2 branches from it") and drop the
"serialize" step.

#### F8 — minor — D4's `remove_from_cart` key ("raw quantity or None") turns 0 into "remove the whole line"

`RemoveFromCartTool.run` passes `quantity` through verbatim, and `remove_cart_item` tests
`quantity is None`. A `0` is therefore a different request from "omitted", but `0 or None` keys the
two the same.

**Fix:** project `arguments.get("quantity")` exactly. The point of D4 is to mirror `run()`.

#### Nits

- **n1.** D3 says "the probe's log greps depend on" the WARNING prefix, but no script greps it: a grep over
  `salesperson/scripts/` and `falkor-chat/scripts/` finds nothing (only `executor.py` and its
  tests). Keeping the prefix is still right for the ad-hoc
  analysis; just correct the reason.
- **n2.** S2's file list omits `tests/test_executor_agent.py`, although S2's details update its
  `FakeCartServices`, which wraps the real `AddToCartTool`. Without it, S2 cannot be reviewed green
  on its own.
- **n3.** Normalizing case in the service relies on lowercase IDs, and only `seed_catalog.sh`
  `_slugify` guarantees that (all 15 live IDs are lowercase, checked on `reference`). Say so in the
  `normalize_product_id` docstring.

### What's solid

- **The premise analysis.** Its "IDs only from current-turn results" conclusion is correct, and
  the alternatives it rejected are argued from evidence: a per-turn catalog index costs about 250
  tokens on an 8K model, and the localized-names option still matches strings.
- **D1's back-compat argument.** `config.tools` stores names only, and a `productName`-only call
  hits the existing required-arg re-prompt (confirmed in `_handle_tool_call`).
- **D5** reuses `lookup_products_by_id` (no new Cypher), keeps one normalizer, and keeps the
  "resolve before `ensure_*`" ordering, which mutant M8 pins.
- **D4** keys K-061 on the honoured arguments, closing the `productName`-extra variant of the
  §15.2 loophole.
- **The step table, docs list and test list are concrete.** The §7 sweep matches a repo-wide grep
  for `v8` pins (`AGENTS.md`, `README.md`, `salesperson/README.md`, three scripts).

### View on the bar

**Reachable by a faithful implementation? Plausibly yes, not assured.** pt-BR and es already
*attempt* the add about as often as en succeeds: about 75 and 70–72 of 75, against en's 70. All
five en misses are "share your delivery address first" replies with no add attempted, which is
prompt policy rather than a cart-tool failure. So the practical ceiling is about 70/75 in every
language.

60/75 therefore needs about 83% of pt/es attempts to complete a three-decision recovery: guess or
hold, then lookup with a miss and the list, then pick the ID and add. Two signals point in opposite
directions:
- *For:* the model knows the English name. 32 of the es holds literally passed
  "Ratón inalámbrico profesional (Wireless Mouse Pro)", and 10 of the 25 mouse lookups used the
  English name.
- *Against:* 0 of 146 held runs retried, though that was under the old "don't" message.

No data exists on picking an ID from a list. A single n=75 run cannot tell a true 80% from 85%: at
80% it fails the ≥60 gate about half the time.

**Measurable as stated?** Mostly yes. The outcome script's definition (cart ∪ order per
participant) is exactly what I re-ran. F5's four edits are needed first.

**My independent view on es/pt-BR ≥ 65:** keep **≥ 60/75 as the K-078 close gate**. K-078's defect
is "carts end up empty", and 60 separates fixed from not fixed beyond doubt. Do not gate on 65.
Against a ~70 ceiling, 65 demands about 93% recovery from a 3B model, and a miss would be
indistinguishable from n=75 noise. If the stakeholder wants parity, express it as a **gap** gate:
(en − es) and (en − pt-BR) ≤ 10 points in the same run. That measures "language no longer matters"
directly and is robust to en itself moving. Keep en ≥ 65 as the no-regression line: every en add
now pays the hold-and-lookup path too.

### Open questions

1. **Does the stakeholder accept F1's residual?** Under ID grounding, K-058 stops a re-fire for one
   iteration (en) or not at all after a failed lookup (pt/es). If not, D3 needs a different
   mechanism. Fix (a) restores the old guard's main lever, which is model compliance.
2. **Does the stakeholder want the gap-gate formulation** in place of, or alongside, an absolute
   ≥ 65?

### Stopping condition for Pass 2

Approve when the revised plan has all of the following:
- (1) a conditional hold reason and a live add-then-add condition in §6.3, or an explicit
  stakeholder acceptance recorded in §8;
- (2) S4+S5 in one commit;
- (3) per-tool `productId` source wording in the schema and the prompt;
- (4) M5 replaced and a cross-node-execution mutant and test added;
- (5) F5's four definitional edits.

Minors and nits may stay open.

## Appendix A — Baseline re-derivation (Q3, n = 75 per language)

**Sources:**
- `…/scratchpad/u3/single-model.jsonl`, 1,125 turns, mapping participant to language;
- read-only `ws:demo` queries over `Cart-[:HAS_ITEM]->CartItem` and
  `Customer-[:PLACED]->Order-[:HAS_LINE]->OrderLine` per language id list;
- `demo-q3-window.log`;
- `lms-server-q3.log`, parsed with `scratchpad/rca/preds.py`.

| Definition | en | pt-BR | es |
|---|---|---|---|
| A. mouse in cart **or** order (plan M1) | **70** (67 cart + 3 order, 0 both) | **1** (order) | **0** |
| B. open cart line only | 67 | 0 | 0 |
| C. `Customer` node exists | 75 | 75 | 75 |
| D. successful `add_to_cart` dispatch (LM log: 71 calls with English name, 0 held) | 70 | 1 (it used the English name) | 0 |
| E. add attempted at all (dispatched + K-058-held, by name language) | 70 | ≈ 75 (74 held + 1) | 70–72 (70 held + 2 unattributed `Mouse Wireless Pro`) |
| Other-product lines / qty > 1 | 0 / 0 | 0 / 0 | 0 / 0 |

**Holds and latency:**
- Holds: 146 `off-turn write held`, all `add_to_cart`, in 146 distinct runs, each run held once
  (0 retries).
- K-061 `same-turn` holds: 0. `possible fabrication`: 0.
- Dead turns: 0 of 1,125. Turn p95 is 14.1 s overall and 14.1 s at turn index 2. Per language at
  turn index 2: en 20.1 s, es 10.1 s, pt-BR 10.1 s. Medians 4.0, 8.0, 6.0 s. The probe polls at
  about 2 s granularity.
- en misses: 5 of 75, all "need your delivery address before adding" (trials 7, 11, 61, 67, 69).

**Note on `ws:demo`:** it holds 206 carts in total, all `wireless-mouse-pro`, including carts from
other runs. One non-Q3 cart has quantity 2, and one non-Q3 order has a `bluetooth-speaker-mini`
line. Neither is in the Q3 participant set, so the outcome script must filter by participant id, as
the plan says.

## Pass 2

**Reviewed:** plan Version 2 (revised in place), against the Pass 1 stopping condition. Code claims
were checked against `ac39345d`. No tests were run.

**Verdict: approve with suggestions.** All five stopping-condition items are met. The fixes were
also checked against each other (below), and none of them undoes another. One new minor finding
and three nits remain; none blocks implementation.

CPG: considered, not relevant — `cpg_falkorchat` is still stale; the code claims were verified against source at `ac39345d`.

### Dispositions

- **F1: fixed.**
  - The hold reason is now conditional: `_HELD_REASON` per tool in D3, asserted by test 12 ("current
    message", "do not add it").
  - The new test 26 pins the depth of the guard.
  - C2 (add-then-add, n = 24 for each of en and pt-BR, escalation at ≥ 3/24) measures re-fires,
    which TP-007 cannot.
  - R3's acceptance is recorded in §8.
- **F2: fixed.** "S4 and S5 land in one commit" appears in the §5 table, the coupling paragraph
  and §9.
- **F3: fixed.**
  - Each tool's description names its own sources: D1 and the S2 details.
  - The `v9` prompt sentence is split into an add part and a remove part.
  - Tests 8 and 25 assert that the add wording never mentions `view_cart`.
- **F4: fixed.**
  - M5 is replaced by a self-grounding mutant. Test 15 kills it: an add with nothing surfaced must
    be held, and the mutant would dispatch it.
  - M10 and test 27 are added, and the steps own the right mutants: M1–M6, M9 and M10 in S3, M7
    and M8 in S2.
  - Code claims re-checked: the old M5 really was equivalent. In `_handle_tool_call` at `ac39345d`,
    the K-058 and K-061 holds return `{held, reason}`, and the AC-6, malformed and absorbed
    `ServiceError` paths return strings. The singleton claim also holds: `app.py` builds a single
    `WorkflowExecutor` per app process, and `_drive`'s comment says so. Turns also run
    concurrently on worker threads, so state on `self` would be a data race as well as a leak
    across customers.
- **F5: fixed.**
  - Turn-2 p95 is now reported per language.
  - M3 sums the mouse quantity across `CartItem` and `OrderLine`.
  - Wrong-language is gated at ≤ 3/375 per language.
  - The coordination doc's DoD no longer mentions holds.
- **F6: fixed.** The miss now returns a `hint`, and the RCA's name→price sweep is a reported
  measure in C1.
- **F7: fixed.** §4, R6 and §9 now state that F1 landed at `ac39345d`.
- **F8: fixed.** D4 uses `arguments.get("quantity")` verbatim and explicitly forbids `or None`.
- **n1, n2, n3: fixed.**

### How the fixes interact

- **F6's hint, F1's conditional hold, and the re-fire path.** These do not conflict:
  - The hint concerns price and category, and the hold concerns whether the customer asked.
  - On a re-fire, the hint gives the model nothing new: after a miss, the catalog list has already
    grounded every product (R3, accepted).
  - The only effect is on cost. A model that obeys the hint on an add turn makes one more lookup
    call. That pushes the worst pt/es add path from 5 to 6–7 iterations (see n4), still under
    `maxIterations` 8.
- **F3's per-tool wording, `_GROUNDING_SOURCES`, and K-061.** These are consistent:
  - The prompt and schema sources for add (`lookup_product_fact`, `filter_products`) are a subset
    of the grounding set.
  - The two extra grounding sources are harmless. `add_to_cart` re-grounds an ID the call already
    had to have. `query_graph_data` carries no ID until K-079 F2.
  - K-061 still runs only after K-058, so it only ever sees grounded calls.
  - Test 26's re-add is not a K-061 duplicate, because the earlier mouse add was held, and
    `dispatched_writes` records only successful dispatches.
- **F4's local `surfaced` and loops with several iterations.** The scope is right:
  - The variable lives in `_run_agent_node`, so it spans every iteration of one node execution.
    That is exactly what lookup-then-add needs, and tests 12, 16 and 17 rely on it, which also
    kills a "reset each iteration" mutant.
  - Test 27 covers the other edge, a second turn on the same executor.
  - The salesperson topology has two steps and one transition. A node that re-runs within one
    drive starts with fresh messages, so starting with an empty `surfaced` is consistent.

### New findings

#### P2-1 — minor — The stakeholder answers are recorded, but the text still says "pending" in seven places

The answers are recorded at the end of §8, but these passages still read as open:
- the D3 residual ("stakeholder acceptance pending");
- C1 ("These numbers are pending the stakeholder's answer to §8 Q2");
- C2 ("Not gated while the stakeholder decides on R3");
- R3 ("Acceptance is the stakeholder's call");
- test 26 ("If the stakeholder rejects R3");
- the §8 preamble ("both must be answered before S7's verdict");
- §9 ("S7 verdict waits on …", "pending the stakeholder's bar decision").

The implementer and `qa-engineer` read those passages, not the answer block.

**Fix:** rewrite them in place as the settled state:
- R3 accepted with the mitigations;
- C2 reported, with ≥ 3/24 in either language escalated before close;
- the bar as stated.

Per the root `AGENTS.md` rule, an open item is rewritten, not appended to.

#### Nits

- **n4.** D7's "worst path … is 5" no longer holds. A model that follows the new hint adds a
  lookup, and a call sending only `productName` adds a re-prompt, so the path reaches 6–7. It is
  still within 8, but correct the count.
- **n5.** The add hold reason says the ID "was not returned by `lookup_product_fact` or
  `filter_products`", while the grounding set for add also includes `query_graph_data` and
  `add_to_cart`. This is intentional for the model's benefit; add one comment line beside
  `_HELD_REASON` saying so, so that a later edit doesn't "fix" one of them to match the other.
- **n6.** The S2 details are missing a line break: "source list.Tool descriptions". §2's CPG line
  still cites `41ce1c2b` as the tree that was read.
