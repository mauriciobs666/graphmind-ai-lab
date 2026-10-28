# `query_graph_data` row identity and category-miss recovery — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-079 (F2), K-086

## Goal

Close two `falkor-chat/docs/BACKLOG.md` items that share the `QueryGraphDataTool.run` → `querygen`
hand-off, designed together as the backlog directs:

- **K-079 F2** — `query_graph_data` never returns non-aggregate attributes without the label's
  identity column (RC-1, `docs/reviews/salesperson-fabricated-prices-rca.md` §F2).
- **K-086** — browsing a category by its translated name through `query_graph_data` dead-ends in
  "no matching data found" (`docs/test-reports/salesperson-cart-product-id2-report.md` DEF-C).

Stakeholder go: 2026-10-28 ("go ahead with K-079 F2 and K-086"). Out of scope: K-079 F4/F5, K-085,
K-087.

## Definition of done

- Design note `docs/plans/salesperson-query-graph-data.md`, plan-gated by `analyst`.
- Implementation with the RCA's red-first coverage probe, code-gated by `analyst`.
- NLQ eval Layer 1 re-run on ministral with no regression (RCA F2 done-condition), and a live
  `refire_probe.py --schedule browse` re-measure for DEF-C (`qa-engineer`).
- Docs: `HISTORY.md` entry; K-079 narrowed to F4/F5 and K-086 removed from `BACKLOG.md`; NLQ eval
  Layer 1 contract (`server/tests/eval/nlq_scoring.py` header, or wherever the plan locates it)
  and any `QUERIES.md`/`DESIGN.md`/`SERVER.md` surface the plan names.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `aeb4c751ec7dddcf9` | delivered | `docs/plans/salesperson-query-graph-data.md` §1–§10 (teco checked headings, no placeholders, 2026-10-28); stakeholder questions §9: `name` vs `productId` identity (§3.2), LM Studio time | `analyst` → — | 205k tok / 59 tools |
| U2 | `analyst` | `a21f2c6fb2f5cbd70` | gated | `docs/reviews/salesperson-query-graph-data.md` Pass 1 (plan gate): B1, M1, M2, m1–m7, n1–n3; `name` uniqueness confirmed | gate on U1 → **needs changes** | 200k tok / 62 tools |
| U1b | `architect` | `aeb4c751ec7dddcf9` (resumed) | delivered | plan v2: B1, M1, M2, m1–m7, n1–n3 addressed; departs from review on M1 name-side trigger, probe totals, m4 (en control dropped); new R7 (pre-existing ORDER BY on non-projected var) | U2b → — | 267k tok / 27 tools (this run) |
| U2b | `analyst` | `a21f2c6fb2f5cbd70` (resumed) | gated | review Pass 2: all Pass 1 fixed (executed); new N1, N2 minor, N3 nit; R7 confirmed pre-existing (+ `avg()` on strings); stopping condition for v3 stated | gate on U1b → **approve with suggestions** | 260k tok / 22 tools (this run) |
| U1c | `architect` | `aeb4c751ec7dddcf9` (resumed) | accepted | plan v3: N1–N3 + R7 text only | Pass 2 stopping condition, checked by teco 2026-10-28: v2→v3 diff hunks all inside the fence (header, revision line, §6.2 stub/T-B10/17/18, §7 refire row, §8.2 Run 0, §8.3 browse-type/B5, R7); guard rc 1 on `a5d8e807^ a5d8e807`, rc 0 on an empty range; `probe_v2.py` re-run 44,323 / 1,239 → **met, no Pass 3** | 281k tok / 7 tools (cumulative run) |
| U3a | `tdd-engineer` | `aff2487d2b206344d` | delivered | Stage A (F2): A1–A3 committed (see `git log`); probe red 1,239/44,323 then green; MA1–MA9 + argued-down mutants killed; full offline suite 3331 passed / 1 xfailed; demo re-seeded, both verify scripts rc 0 (teco re-ran 2026-10-28). teco checks: targeted suites 162 passed; own argument mutant (KB schema passed to `ensure_row_identity` in `QueryGraphDataTool.run`) killed by 3 tool tests. Departures: T-A3 gained a `[p.price, count(p)]` case (plan's shape did not kill MA1) | `analyst` → **not yet dispatched** | 180k tok / 57 tools |
| U3b | `tdd-engineer` | — | queued | Stage B (K-086): plan §6 B1–B3 — after U3a (both edit `QueryGraphDataTool.run`) | `analyst` → — | — |
| U4 | `analyst` | — | queued | `docs/reviews/salesperson-query-graph-data-impl.md` | — | — |
| U5 | `qa-engineer` | — | queued | `docs/test-plans/salesperson-query-graph-data.md`, `docs/test-reports/salesperson-query-graph-data-report.md` | — | — |

## Checkpoint — paused 2026-10-28 (stakeholder: session near limit)

Next, in order: (1) U4a — `analyst` code gate on Stage A's commit, review at
`docs/reviews/salesperson-query-graph-data-impl.md`; (2) U3b — Stage B (`tdd-engineer`), same brief
shape as Stage A; (3) U4b — code gate on Stage B; (4) U5 — `qa-engineer` Stage C (plan §8), Run 0 at
`e8798351`; (5) close-out docs. Agent ids above resolve only in the session that spawned them —
in a new session, dispatch fresh with the plan, the review and this ledger as inputs.

## Notes

- CPG `cpg_falkorchat` is stale (built 2026-10-15, 18 server commits behind). Proceeding without
  a rebuild: the change is local to `tools.py`/`querygen.py`, and callers can be found with grep.
- Stakeholder decisions (2026-10-28): F2 identity column is `name`, per plan §3.2, provided the
  U2 gate confirms uniqueness; LM Studio live runs approved for U5.
- **Run 0 baseline sha (plan §8.2): `e8798351`.** `falkor-chat/server` and `falkor-chat/config`
  are byte-identical to `03328dc1` (`git diff --quiet` rc 0); Stage A had not started.
- Follow-up to file at close (plan §9 R7): two pre-existing `query_graph_data` shape classes that
  compile but fail at execution: aggregate with `order_by` outside `returns`, and `avg()` over a string.
