# Cart writes keyed by product ID, not the model's rendering of the name — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-078, K-079

## Goal

Close `falkor-chat/docs/BACKLOG.md` K-078: Spanish and Portuguese carts stay empty because
`add_to_cart`/`remove_from_cart` take `productName`, the model translates it, and the K-058 off-turn
guard (`executor.py`, `_WRITE_TARGET_ARG`) holds the call.

**Stakeholder decision (2026-10-22):** the cart write tools identify the product by a stable ID,
not by the translated name. Rejected: mapping a translated name back to the catalog name, or
loosening the guard's name matching. Where the ID comes from, and what the K-058 guard grounds an
ID-keyed write against, are the architect's to design (`docs/plans/salesperson-cart-product-id.md`).

K-079 (fabricated prices) runs alongside as an `analyst` root-cause analysis. Its fix is sequenced
after the K-078 plan, because both may change what the catalog tool results show the model.

## Definition of done

- Plan, `analyst` plan gate passed.
- Implementation, offline tests mutation-tested, `analyst` code gate passed.
- Live check against LM Studio: the plan's §6.3 gates met (es/pt-BR ≥ 60/75 filled carts, en ≥ 65/75,
  wrong-product/doubled ≤ 2/225 — stakeholder 2026-10-22), C2 re-fire reported.
- Docs: `SERVER.md`/`DESIGN.md`/`QUERIES.md` where they describe the cart tools; `HISTORY.md` and
  `BACKLOG.md` at close (teco).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 plan: ID-keyed cart writes + guard grounding | `architect` | `a8bdb39689b10277c` | accepted (v3) — `productId` required, `productName` dropped; guard grounds on same-turn tool results; failed lookup returns ID+name list; `v9` incl. F3 currency; F2 separate follow-up; S1–S7 (§5) | `falkor-chat/docs/plans/salesperson-cart-product-id.md` | `analyst` (`aad2f6769bc435f75`, 215k tok / 68 tools) → pass 1 **needs changes** (F1 re-fire via hold message, F2 S4+S5 one commit, F3 per-tool ID sources, F4 M5 equivalent + shared-executor mutant; F5–F8 minors) → v2 by `architect` (230k tok) + stakeholder answers recorded §8 (R3 accepted w/ mitigations; bar unchanged) → pass 2 **approve w/ suggestions** (P2-1 settled-wording, n4–n6) → v3 text fixes by fresh `architect` `a3cef4d686dcaeb39` (done, 65k tok; teco grep: no pending wording left) · `falkor-chat/docs/reviews/salesperson-cart-product-id.md` | 189k tok / 43 tools |
| U2 K-079 price RCA | `analyst` | `ad4ca71ea5b9eda4b` | accepted — premise wrong: real prices mis-paired (RC-1); teco spot-check CF-1 (`coalesce($categoryNormalized…)`, `repository.py`) holds; F2/F3 relayed to U1 2026-10-22 | `falkor-chat/docs/reviews/salesperson-fabricated-prices-rca.md` | RCA (no gate) | 148k tok / 30 tools |
| U3 K-079 F1: blank `category`/`minPrice`/`maxPrice` = no filter | `tdd-engineer` | `a016211940d9020ea` | accepted (`ac39345d`) — category rule in `Repository.filter_products`, price rule `_blank_as_none` in `tools.py` (departs from F1, argued); 2 new tests green (teco re-run), 357 reported; teco mutant (`max_price` reads `minPrice`) → `test_filter_products_returns_items` red, restored (`cmp`) | `repository.py`, `tools.py`, 2 test files, `docs/QUERIES.md` §15.2 | `analyst` (`ad4ca71ea5b9eda4b`) → pass 1 approve w/ suggestions (m1 numeric-string bound, n1 parens, n2 HISTORY note) → m1/n1 fixed by `tdd-engineer` (`_coerce_price_bound`; 89 green teco re-run) → pass 2 **approve** (n2 → HISTORY at close) · `falkor-chat/docs/reviews/salesperson-fabricated-prices-impl.md` | 96k tok / 29 tools |
| U4 plan S1–S3 (`4dac501e`) (normalize, services/tools by ID, guard + dedup — S2+S3 one commit) | `coder` | `a9b81039424228f01` | accepted — 496 green reported, 493 teco re-run (executor/extraction/tools/services); M1–M10 + X1/X2 killed; teco mutant (per-call copy of `surfaced`, main loop) → 18 red, restored (`cmp`); storefront test files NOT run (fixtures wipe `reference`) — run before S7 with re-seed | per plan §5 | `analyst` (`a570c18d126e2f842`, 196k tok) → pass 1 **approve w/ suggestions** (F1 self-comparing hold tests, F2 single-response grounding untested, F3 table key-set test, F4 → S7 brief, F5 scaffold db tests unrun; n1–n3); `verify_catalog.sh` + `verify_salesperson.sh demo` OK after gate (reference intact) → F1–F3/n1/n3 to fresh `coder` `aa9ba42b5c68f7555` (84k tok; F1 folded into hold tests, F3 dedup-subset dropped, n1 self-source rows dropped — all confirmed by execution) → pass 2 **approve** (p2-n1 comment fixed by teco; D3 table/D2 wording → later plan revision) · `falkor-chat/docs/reviews/salesperson-cart-product-id-impl.md` | 242k tok / 73 tools |
| U5 plan S4–S6 (`v9` + pins in one commit, docs; seed+verify by teco after commit) | `coder` | `a1a11d39a8ec09aeb` | accepted (`8a72f08d`; `v9` published to `reference`/`ws:demo` from the clean committed tree, `verify_salesperson.sh demo` OK) — `v9` prompt (D7 + USD), pins, docs; 4 scaffold + 496 targeted green reported, 4 scaffold teco re-run; 9 prompt mutants killed; teco sweep: no live `v8` pins left; nothing published | per plan §5 | `analyst` (`a3456850c405a6572`) (128k tok) → pass 3 **approve w/ suggestions** (F1 7/9 prompt mutants survived → test tightened by `coder`, 7/7 + 5 earlier killed via in-memory plugin; teco mutant negation-flip of the `view_cart` sentence → red, `proof_defs.py` sha unchanged; F2 publish reads the tree → `v9` prompt frozen, diff-clean check before seed; n1 fixed) · `...-impl.md` Pass 3 | 147k tok / 42 tools |
| U6 plan S7 live check (needs LM Studio free; brief must carry impl-review F4: tag name→price misjoins following a miss + `query_graph_data`; storefront + scaffold db tests run by teco 2026-10-23: 265 passed; `reference` re-seeded (catalog + `salesperson@v9`/`order-fulfillment@v1`), both verify scripts OK; also report first-lookup miss rate per language and confirmation-turn adds — impl review Pass 3) | `qa-engineer` | — | queued | `falkor-chat/docs/test-reports/salesperson-cart-product-id-report.md` | — → — | — |

## Documentation-impact scan

To be completed from U1's plan. Candidates: `falkor-chat/docs/SERVER.md` (tool catalog),
`falkor-chat/docs/DESIGN.md`/`QUERIES.md` (cart queries), `salesperson@v<n>` def text in
`proof_defs.py`/`seed_salesperson.sh`, `falkor-chat/AGENTS.md` if the pinned def version changes,
`salesperson/README.md` (pinned `salesperson@v8`).
