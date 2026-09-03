# The one salesperson UI — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M<n> TBD)

## Goal

Deliver the business-facing salesperson UI specified in `docs/requirements/salesperson-ui.md`
(FR-1…FR-11, AC-1…AC-11): a modern, mobile-usable, multi-participant chat product surface built
against the workflow-engine-backed `salesperson` agent (falkor-chat M6; `v5` today, bumped to `v7`
by plan step S1), replacing the retired
standalone `salesperson/` Streamlit app.

**Definition of done:** AC-1…AC-11 verified; the old `salesperson/` app retired; documentation
(root `AGENTS.md`, component READMEs, `HISTORY.md`, a `tico` user manual) reflects the delivered
surface.

## Standing decision — the deprecated CPG is ignored (stakeholder, 2026-09-02)

**`cpg_deprecated_salesperson` is not maintained, not documented, and not rebuilt. No unit of this
or any future coordination spends effort on it** — no freshness check at dispatch, no rebuild, no
doc that explains it. It is left in place (not deleted: a drop is destructive and irreversible, the
graph has no rebuild path short of re-running Joern over a retired tree, and keeping it costs
nothing but ~17.5k nodes of RAM).

**The U8 rename still stands and was not wasted** — it existed to stop the *new* `salesperson/`
component's conventional CPG name from resolving to the *old* app's contents. That trap is closed
either way; ignoring the graph does not un-close it.

**What this cut mid-flight:** `tico` (U11) was redirected to strip the stale example from
`docs/manuals/graph-ontology.md` without teaching anything about the deprecated graph, and
`architect` (U12) was told to drop the plan's CPG-fate discussion entirely.

**One thing deliberately kept**, flagged for the stakeholder rather than decided silently:
`skills/cpg-analysis/references/freshness.md`'s U8b addition stays. It documents a **general,
empirically verified hazard** — `git log --since=<unparseable>` returns zero commits with exit 0,
silently, so a staleness check against any dateless marker reads as a false "unchanged" — which is
true of any hand-written marker, not just this graph. The deprecated graph appears in it only as one
citation. Trimming that citation is a one-line edit if preferred.

## Decisions taken by the coordinator

- **Doc family home = repo-root `docs/`**, matching where `tico` filed the requirements doc.
  Collision rule 2 (same slug across kinds) keeps `requirements/salesperson-ui.md` →
  `plans/salesperson-ui.md` → `plans/salesperson-ui-coordination.md` → `reviews/salesperson-ui.md`
  → `test-plans/` → `test-reports/` in one tree. Root `docs/` already carries cross-component
  topics (`doc-reference-convention`, `kaizen-*`) alongside the CPG component's own
  `BACKLOG.md`/`HISTORY.md`. **Where the new UI's *code* lives is a separate question**, delegated
  to the architect (U1) and escalated to the stakeholder at the plan gate.
- **Implementation units are not drawn yet.** They are decomposed from U1's plan step table
  (one unit per step or small adjacent-step cluster) once the plan is gated.

## Context the units inherit

- `K-056` (agent skipping tool calls / fabricating catalog facts), which AC-10 gates the first
  live demo on, was **resolved 2026-08-30** by a model swap to `mistralai/ministral-3-3b`
  (`falkor-chat/docs/HISTORY.md`). `K-060` (rarer synthesis-time omission on mixed-category
  `filter_products` results) is still open and in-progress — related, but not AC-10's gate.
- falkor-chat M6 closed 2026-08-30 (`scripts/seed_salesperson.sh` publishes the def; see the
  version note below).
- **FalkorDB is running** (`falkordb-dev`, `localhost:6379`, v4.18.11, detached `--rm`, data
  volume persisted) — started 2026-09-02 at the stakeholder's direction so U1 could verify live.
  17 graphs loaded, including `reference` (K-052 catalog), `ws:acme`, and the M6 QA-pass
  workspace graphs (`ws:qa-salesperson-demo`, `ws:qa-cart-totals`, `ws:qa-durable-profile`, …).
- **The current def is `salesperson@v5`, not `v4`** (`server/falkorchat/proof_defs.py:301`; K-057
  bumped v4→v5). `falkor-chat/AGENTS.md` claimed `v4` — **fixed as U1c**, see the note at the end.
- **CPG freshness (checked by `teco` at dispatch, per `skills/cpg-analysis/references/freshness.md`):**
  `cpg_falkorchat` was **stale** — built `2026-08-26T22:27:22Z`, with 29 commits to
  `falkor-chat/server` since (all of K-053/K-054/K-055/K-057/K-058/K-059/K-061/K-035). Rebuild
  dispatched as U1b. `cpg_salesperson` has **no freshness marker at all** (zero rows) — no signal;
  deliberately *not* rebuilt, since `salesperson/` is the component this coordination retires.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 — Implementation plan for the whole feature | `architect` | `a27a11937187341b7` | delivered | `docs/plans/salesperson-ui.md` (17 steps, S0-S16) | `analyst` (U2) → — | 197k tok / 58 tools |
| U1b — Rebuild the stale `cpg_falkorchat` code graph | `graph-dba` | `a700bd17926f5df2a` | accepted | live `cpg_falkorchat` + `cpg/.cpg-artifacts/` | teco-verified | 113k tok / 44 tools |
| U1c — `falkor-chat/AGENTS.md` v4→v5 drift fix | `teco` (trivial, single-file) | — | accepted | `falkor-chat/AGENTS.md` rows 82-83 | none (trivial; see note) | — |
| U2 — Plan gate (Pass 1) | `analyst` | `a53c28ffa5049fa67` | delivered | `docs/reviews/salesperson-ui.md` — **needs changes**: 4 blockers, 9 majors, 15 minors | — | 198k tok / 55 tools |
| U3 — Stakeholder decisions on plan §8 OQ-1…OQ-6 | stakeholder | — | delivered | all six answered, see below | — | — |
| U4 — Revise plan for U2 findings + U3/U7 decisions | `architect` | `a27a11937187341b7` (resumed) | delivered | `docs/plans/salesperson-ui.md` **v1.1**, 19 steps (S12 split a/b/c) | `analyst` re-gate (U2 Pass 2) → — | 282k tok / 19 tools cumulative |
| U5 — Move old app to `deprecated/salesperson/`, free the name | `coder` | `a49cf0a28cf09a417` | accepted | `deprecated/**`, root `AGENTS.md`, `cypher-mcp/README.md` | teco-verified (mechanical) | 91k tok / 27 tools |
| U6 — Update `claude/frontend-engineer/frontend-engineer.md` (M9) + cross-agent sweep | `cobb` | `a87f90e01200c51dd` | accepted | 9 files in `claude/` + `skills/` | **gate skipped, justified below** | 123k tok / 25 tools |
| U7 — Re-decide presenter identity after B1 | stakeholder | — | delivered | **reverted to `FALKORCHAT_PRESENTER_KEY`** (the plan's original design); B2's fix left to the `architect` with a close-structurally constraint | — | — |
| U8 — Rename `cpg_salesperson` → `cpg_deprecated_salesperson` | `graph-dba` | `aaafc289ac73b80b9` | accepted | live graph + `skills/joern-cpg/references/cpg-model.md` | teco-verified | 118k tok / 22 tools |
| U8b — Document the third `:CpgBuildInfo` shape U8 created | `graph-dba` | `aaafc289ac73b80b9` (resumed) | accepted | `skills/cpg-analysis/references/freshness.md` (+31 lines) | **gate skipped, justified below** | 124k tok / 5 tools |
| U11 — Fix `cpg_salesperson` as a live example in a user manual | `tico` | `a07c97e2644028f80` | accepted | `docs/manuals/graph-ontology.md` | teco-verified (scope cut to a correction pass) | 78k tok / 34 tools |
| U11b — Same drift in `docs/manuals/cpg-getting-started.md` (3 lines) | `tico` | `a07c97e2644028f80` (resumed) | accepted | `docs/manuals/cpg-getting-started.md` | teco-verified (trivial) | 84k tok / 4 tools |
| U9 — Re-derive `skills/joern-cpg/SKILL.md`'s per-file CPG scaling rule of thumb | `cobb` | — | queued (**after U8**) | `skills/joern-cpg/SKILL.md` only (U8 owns `references/cpg-model.md` — disjoint, no collision) | `graph-dba` → — | — |
| U10 — Re-gate the revised plan (`## Pass 2`, same reviewer, revised in place) | `analyst` | `a53c28ffa5049fa67` (resumed) | delivered | `docs/reviews/salesperson-ui.md` Pass 2 — **approve with suggestions** | — | 255k tok / 16 tools cumulative |
| U12 — Fold N1-N3 + 2 nits into the plan (v1.2) | `architect` | `a27a11937187341b7` (resumed) | accepted | `docs/plans/salesperson-ui.md` **v1.2**, 1129 lines, 32-row file map | teco-verified; N1 confirmed closed | 316k tok / 17 tools cumulative |
| **S0** — Participant provisioning + reset Cypher design note | `graph-dba` | `a5e1bc3d8b68384f5` | delivered | `docs/plans/salesperson-ui-graph.md` (679 lines) | `analyst` (S0-gate) → — | 265k tok / 75 tools |
| **S0-gate** — Review the reset Cypher design | `analyst` (fresh) | `a1fb4168b116a1f4e` | delivered | `docs/reviews/salesperson-ui-graph.md` — **approve with suggestions** (4 major, 5 minor) | — | 219k tok / 60 tools |
| **S0b** — Revise the design note for the gate's findings | `graph-dba` | `a5e1bc3d8b68384f5` (resumed) | delivered | `docs/plans/salesperson-ui-graph.md` **v1.1**, 941 lines (was 679) | `analyst` re-gate (S0-gate Pass 2) → — | 370k tok / 50 tools cumulative |
| **S0-gate Pass 2** — Re-gate the revised design note | `analyst` | `a1fb4168b116a1f4e` (resumed) | delivered | Pass 2 — **needs changes** on 1 blocker (P1) + 2 minors | — | 313k tok / 33 tools cumulative |
| **S0c** — Close P1-P3 + 2 nits | `graph-dba` | `a5e1bc3d8b68384f5` (resumed) | delivered | `docs/plans/salesperson-ui-graph.md` **v1.2**, 1043 lines | `analyst` Pass 3 (narrow) → — | 431k tok / 29 tools cumulative |
| **S0-gate Pass 3** — Narrow closeout on P2/P3 only | `analyst` | `a1fb4168b116a1f4e` (resumed) | **accepted — APPROVE** | `docs/reviews/salesperson-ui-graph.md` Pass 3 | — | 332k tok / 8 tools cumulative |
| **S4** — Repository + service primitives, implementing S0's Cypher verbatim | `coder` | — | queued (**behind S2 — shared live DB**) | `repository.py`, `services.py`, `QUERIES.md` §18, 2 test files | `analyst` → — | — |
| **S5** — Node toolchain + `salesperson/` SPA scaffold | `devops` | `a770537aafab6b123` | accepted | `salesperson/**` (new) | teco-verified by execution | 113k tok / 63 tools |
| **S1** — `salesperson@v6`→**`v7`** def bump + `AGENTS.md` rows 82-83 | `coder` | `aef74d44be60f1cff` | **gated → needs changes** (F-1 blocker: `v6` collides with a reverted K-060 experiment in `ws:acme`; **S1's original report was correct, teco's rebuttal was wrong** — see below) | `proof_defs.py`, 2 scripts, scaffold test, `falkor-chat/AGENTS.md` | `analyst` → — | 130k tok / 42 tools |
| **S2** — chat-path `run_ctx` merge | `tdd-engineer` | `a1aa5c430de8da50d` | delivered (**re-dispatched 2026-09-02** after the first dispatch was lost — no agent id recorded, never ran) | `services.py`, `trigger.py`, `test_services.py`, `test_trigger.py`, **+`test_process_input.py` (5th file, outside the plan's S2 column)**, `QUERIES.md` §12.1/§12.12, `HISTORY.md` | `analyst` (S1+S2 impl gate) → — | 113k tok / 45 tools |
| **S1+S2 impl gate** — review both delivered diffs | `analyst` (fresh) | `a3a1f90613439b23c` | delivered — **needs changes** (1 blocker, 1 major, 4 minor, 2 nits) | `docs/reviews/salesperson-ui-impl.md` | — | 164k tok / 79 tools |
| **S1b** — F-1 blocker: `v6`→`v7` bump; F-8 verify-script detection; F-5 HISTORY entry; re-seed wiped `reference` | `coder` | `aef74d44be60f1cff` (resumed) | delivered — suite 2330 teco-verified; F-8 proven by a **constructed negative control** (every pre-existing check green, new check still fails) | `proof_defs.py`, 2 scripts, scaffold test, `falkor-chat/AGENTS.md`, `HISTORY.md` | `analyst` re-gate (Pass 2) → — | — |
| **U13** — Plan v1.3: `v6`→`v7` sweep + F-4 stale §5.0 file map | `architect` (**fresh** — prior instance at 316k tok, follow-up self-contained) | `abeef0ec5b77cc45f` | delivered — **11 sites swept, not the 3 the review named**; also flagged a contradiction in *this* doc (fixed) | `docs/plans/salesperson-ui.md` v1.3 | `analyst` re-gate (Pass 2) → — | 109k tok / 27 tools |
| **U13b** — Plan v1.4: F-6 clause on S8; pin S1's done-condition to a throwaway probe (+ class sweep); §2.2 baseline wording | `architect` | `abeef0ec5b77cc45f` (resumed) | delivered — **class sweep found 2 more unpinned-workspace instances**, incl. **S4's, a false-evidence trap** (verify with no arg asserts against `ws:acme`, not the reset probe — passes green proving nothing) | `docs/plans/salesperson-ui.md` v1.4 | `analyst` re-gate (Pass 2) → — | 131k tok / 10 tools |
| **U13c** — Plan v1.5: impl review added to header `Reviews:`; §8 outstanding-obligations clause; §6.1 marked deliberately unpinned | `architect` | `abeef0ec5b77cc45f` (resumed) | **accepted** — teco-verified: header conformant, both done-conditions read `<that same probe graph>` (not a bare placeholder), no bare verify call left in any done-condition | `docs/plans/salesperson-ui.md` **v1.5** (+40/−21 over v1.2) | `analyst` re-gate (Pass 2) → — | 137k tok / 3 tools |
| **S2b** — F-2 major: service-side `run_ctx` size bound; F-3 invariant test; `HISTORY.md:68` `v6`→`v7` | `tdd-engineer` | `a1aa5c430de8da50d` (resumed) | delivered — suite **2336** teco-verified; **F-3 closed the real gap** (reversing the merge now fails a test; it previously left all 2330 green). **Two items referred to the re-gate:** F-2's bound is caller-only where both siblings bound *merged*, and `test_workflow_timers.py` (K-028's file) was edited outside column | `services.py`, `test_services.py`, **`test_workflow_timers.py`**, `HISTORY.md` | `analyst` re-gate (Pass 2) → — | — |
| **Pass 2 re-gate** — S1b + S2b, revise review in place | `analyst` | `a3a1f90613439b23c` (resumed) | **accepted — APPROVE WITH SUGGESTIONS** (no blockers, no majors). Answered all 3 teco questions **by mutation, not argument**; upheld the author's F-2 deviation against teco's inclination to overrule | `docs/reviews/salesperson-ui-impl.md` `## Pass 2` + Appendices D/E | — | 227k tok / 39 tools |
| **S4** — Repository + service primitives, implementing S0's Cypher verbatim | `coder` (**fresh**) | `ad18a6012575d8d7b` | delivered — suite **2379** teco-verified; **5 note blocks confirmed byte-identical in `repository.py`**; `ws:acme` inventory unchanged. **Ninth method `ensure_participant` added** (note §12 mandates §3 verbatim; sole writer of `Channel.participantId`) — referred to the gate | `repository.py` (+636), `services.py` (+107), `QUERIES.md` §18 (+598), 2 test files (+1222), `DESIGN.md`, `HISTORY.md` | `analyst` (Pass 3, **fresh**) → — | 284k tok / 110 tools |
| **S4 gate** — Pass 3 on the largest, most safety-critical diff | `analyst` (**fresh** — Pass 1/2 reviewer at 227k tok) | `a4fed35b842be85c5` | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 1 major, 6 minor, 2 nits). Attacked both guards **7 ways**, none broke; found forging closes *harder* than the note claims (`Channel` UNIQUE). Upheld the 9th method — **the plan's 8-method list was the defect** | `docs/reviews/salesperson-ui-impl.md` `## Pass 3` + Appendices F/G | — | 231k tok / 89 tools |
| **S4b** — M-1 major + M-2/3/4/5/7 + 2 nits | `coder` (**fresh**) | `a18dc58f5d7dc983c` | delivered — **M-1 parameter dropped, not gated** (teco-verified signature); suite **2381** teco-verified; **re-measured the reviewer's own N-7 citation table by ablation and found it short by two** | `repository.py`, `services.py`, `QUERIES.md`, `DESIGN.md`, `HISTORY.md`, `test_repository.py` | `analyst` Pass 4 → — | 173k tok / 95 tools |
| **S4b gate** — Pass 4 on the S4 findings | `analyst` | `a4fed35b842be85c5` (resumed) | **accepted — APPROVE**, 1 nit. Re-measured N-7 unfiltered and **corrected itself**: its Pass 3 `-k` filter was not the error source. Confirmed `CREATE`→`MERGE` is an *equivalent mutant*, no coverage gap | `docs/reviews/salesperson-ui-impl.md` `## Pass 4` | — | 277k tok / 33 tools |
| **U20** — Pass 5 re-gate of plan v1.13 | `analyst` | `ab94a9b40db374063` (resumed) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 1 major). **P5-1: the undifferentiated-handler defect recurred inside §5.3, the section built to fix it** — `409` this time. P5-2 beat both teco-offered options: "storefront disabled" has **no code path** | `docs/reviews/salesperson-ui.md` `## Pass 5` | — | 227k tok / 18 tools |
| **U21** — Plan v1.14: all 6 Pass 5 findings + both rulings; **added a §5.2-response → C-rule completeness table** | `architect` | `a3ff2db4359dbebc2` (resumed) | delivered — **the table caught a 3rd instance of the defect class within minutes** (`403` on two presenter responses meaning different things, C2 covering both with one no-op action). Also **rejected half the reviewer's timer suggestion** with reasoning | `docs/plans/salesperson-ui.md` **v1.14** | `analyst` Pass 6 → — | 172k tok / 17 tools |
| **U22** — Pass 6: convergence check, not just a re-gate | `analyst` | `ab94a9b40db374063` (resumed) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 1 major, 4 minor). **Cleared S3+S6 for dispatch on evidence**: hashed every step row across v1.2/v1.10/v1.14 — S3 and S6 byte-identical in all three, S7/S8/S10/S12a changed at every version. **P6-1: the completeness table caught the 3rd instance and created the 4th** — `(response → rule)` keying still lets one row span two routes | `docs/reviews/salesperson-ui.md` `## Pass 6` | — | 253k tok / 10 tools |
| **U23** — Plan v1.15: table re-keyed on **(route, response)** (9→36 rows) + P6-2/3/4/5, new C10/C11/C12 | `architect` | `a3ff2db4359dbebc2` (resumed) | delivered — **re-keying surfaced a 5th instance, a *blank cell***: `reset` → `5xx` asserted "never retried" since v1.10 with **no client rule**; "never retried" was defended at the library and application layers, never the browser. Also found P6-2 undercounted (5 `422` routes, not 3) | `docs/plans/salesperson-ui.md` **v1.15** | `analyst` Pass 7 → — | 198k tok / 11 tools |
| **U24** — Pass 7: is the class closed? + dispatch judgment | `analyst` (**fresh** — the Pass 3-6 reviewer had *prescribed* the fix under test) | `a4f0457bda1615d13` | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 3 major). **Class NOT closed — instance 6 found on two axes the table cannot key** (below it: the discriminator is the error body's *field*; beside it: TanStack's `shouldRetry` is status-blind). **P7-3 would have shipped**: `FalkorDBUnreachableError` has no handler, so a query-time timeout escapes as a bare `500` on every poller at once. **Reproduced Pass 6's undocumented hash method** to authenticate its v1.14 column | `docs/reviews/salesperson-ui.md` `## Pass 7` | — | 126k tok / 38 tools |
| **U25** — Plan v1.16: **close the class at runtime, not in the table** — total server error map + client loud-default C13 | `architect` | `a3ff2db4359dbebc2` (resumed) | delivered — **building the guard surfaced instance 7**: F8 was reset-scoped only by accident of context; `/messages` and `/order/advance` are writes with the same may-have-committed ambiguity, so the map now splits **read-vs-write**, not reset-vs-other. **Scored its own fix honestly: closes *unruled* (1,3,4,5,7), does NOT close *mis-ruled* (2,6)** — and flagged a bounded hole in its own guard (`return` vs `raise`) | `docs/plans/salesperson-ui.md` **v1.16**; S6/S7 row hashes re-verified unchanged | `analyst` Pass 8 → — | 240k tok / 16 tools |
| **U26** — Pass 8: final plan gate — is the class closed **enough to ship**? | `analyst` | `a4f0457bda1615d13` (resumed; asked to argue against its own Pass 7 prescription) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 4 major, 3 minor, 1 nit). **Confirmed the architect's own scoring by producing three mis-ruled instances inside the v1.16 delta itself** (P8-1/2/3): generalising F8 from "either reset" to "every write" extended C4's *domain* faster than its *content*, and the cross-cutting `504` row hid the cells that opened. **Argued against its own Pass 7 prescription** — C13 detects an *absent* rule, is silent on a rule that matches and is wrong, and that residual appears nowhere in the shipping document. **Gave the stopping rule teco asked for** (below) | `docs/reviews/salesperson-ui.md` `## Pass 8` — committed `0efc014` | — | 169k tok / 9 tools |
| **U27** — Plan v1.17: the **one consolidating touch** — P8-1…P8-N1, no Pass 9 | `architect` (**fresh** — the v1.14-v1.16 architect was at 240k tok and this task is self-contained) | `a8e01a3759dafabd0` | **accepted — committed `0ba772b`** (253/72, one file). **Fixed the root cause, not the four symptoms**: a route-class table (5/4/2) every "every route" phrase is re-keyed onto, the `504` row split five ways, and the licence tightened to "one meaning **and** one action". **Corrected Pass 8's arithmetic twice** (five writing routes, not six; four-of-five with one *wrong* and one *missing*, two different defects). **Qualified P8-7 while adopting it** — a declaration is itself an enumeration, so it narrows the residue rather than closing it. Absorbed two mid-run teco relays (S6's env contradiction, the reversed `SERVER.md` routing) and both S6-gate carry-forwards | `docs/plans/salesperson-ui.md` **v1.17**; S3/S7/S9 rows **teco-verified byte-identical** | **none — plan gates stopped** | 191k tok / 102 tools |
| **S6** — Storefront core: participant registry, join, token verify, turn-state map | `coder` (**fresh**) | `a5db169a0966bad59` | delivered — **committed `2f7938d`**, suite **2439** teco-verified **solo**. Mutation-tested both danger-zone assertions teco flagged at dispatch: a cache-first branch reddens the deleted-participant test, an in-process-authoritative registry reddens the restart-survival test. **Found a contradiction between the plan's prose and its own S6 env table** (`FALKORCHAT_PRESENTER_KEY` vs `FALKORCHAT_STOREFRONT_PRESENTER_KEY`) — relayed to U27 in flight. Pinned the `compare_digest("", "")` trap for S10 | `storefront.py` (new), `config.py` (+61/-0), `test_storefront.py` (new), `SERVER.md`, `HISTORY.md` | `analyst` Pass 6 → — | 202k tok / 66 tools |
| **S6 gate** — Pass 6 on the storefront core: can the cache reach an auth decision? | `analyst` | `a24e4bcbd0b9a1f8e` (resumed — the S3-gate reviewer, adjacent surface) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 2 major, 2 minor) — committed `a38090a`. **Answered the auth question structurally, not by inspection**: all 8 `_records` touch sites enumerated, exactly 2 reads, `resolve_token` reaching the map only via write-side helpers, no caller outside the module, exception route **fail-closed**. **Found the pre-planted vacuous assertion again** (S6-3, 3rd time this reviewer has). **Disagreed with teco's routing of the `SERVER.md` §1.5 carry-forward and was right** → follow-up 15 | `docs/reviews/salesperson-ui-impl.md` `## Pass 6` + Appendix J | — | 175k tok / 23 tools |
| **S6b** — close Pass 6: pin `_cache_put` (S6-1), give the package scan a control (S6-3), reshape the constant-time tripwire (S6-4) | `coder` | `a5db169a0966bad59` (resumed — its own review findings, same two files) | **accepted — committed `5594134`**, suite **2441** teco-verified solo; `storefront.py` +12/-0, comment only. **Probed for the *false* positive, not just the true one** — a benign local rename that the over-tight tripwire used to redden now passes, which is what "over-tight" actually means and almost nobody tests. Re-grounded the stale env-var docstring on something **executable** (the test reads `SERVER.md` and asserts all seven names appear), so a rename that misses the doc reddens instead of drifting | `storefront.py`, `test_storefront.py` | `analyst` Pass 6 → **closed** | 232k tok / 24 tools |
| **S6c** — S6+S6b close-out in `falkor-chat/docs/HISTORY.md` (S3's "Review close-out" precedent) | `coder` | `a5db169a0966bad59` (resumed — **holds the observed figures**; a fresh agent would reconstruct them, which is the fabrication risk) | **accepted — committed `62aa638`**, +74/-0, one file; storefront files verified untouched and no suite run, so S7's DB window was never contended. **The resume-for-figures call paid off exactly as intended**: it attributed the three *pre-fix survival* counts to the review's Appendix J rather than claiming them, and **left pass counts out** where it only had them against a different test-count denominator — declining to reconstruct rather than producing a plausible total | `falkor-chat/docs/HISTORY.md` | — | 235k tok / 5 tools |
| **S7** — Storefront state, reset, catalog, images | `coder` (**fresh** — see note) | `a26100cb193d95085` | delivered — **committed `dd78e70`**, suite **2473** teco-verified solo; +465/-9 and +950, two files. **13 mutations killed, 4 benign refactors kept green** (S6b's false-positive discipline, adopted unprompted). **Found a delivered-code gap that blocks two of its own deliverables** — `services.filter_products` projects no `productId` (teco confirmed at `repository.py:2762`) — and shipped a documented `1+n` workaround rather than editing a delivered step's file. **Proved a plan statement false**: the post-reset profile re-write `MERGE`s the `Customer` back, so §4.8's delete inventory is true of the delete and false of the end state. **Answered the `lookup` question: no** | `storefront.py`, `test_storefront.py` | `analyst` Pass 7 → — | 227k tok / 81 tools |
| **S7 gate** — Pass 7 on the largest impl diff + **three rulings** teco will not self-decide | `analyst` | `a24e4bcbd0b9a1f8e` (resumed — reviewed S6 in this same module) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, **0 major**, 3 minor, 1 nit) — committed `e9b0363`. **Dissolved Ruling 1's blocker instead of weighing it** (below). **S7-1: proved the quiesce tests don't assert the wait** by running the worker-finishes-first ordering — all four stayed green; safety was timing, not assertion. Verified the `lookup` grep and found the **sharper** reason not to delete (below) | `docs/reviews/salesperson-ui-impl.md` `## Pass 7` + Appendix K | — | 253k tok / 36 tools |
| **S7b** — close Pass 7: assert the wait (S7-1), pin the two error-path evictions (S7-2), stop a broken deadline hanging (S7-3) | `coder` | `a26100cb193d95085` (resumed — its own findings, same two files) | delivered — **committed `d9d2f2b`**, suite **2476** teco-verified solo; `storefront.py` **byte-identical** to `dd78e70`, so all three findings were about what the tests assert, not code defects. **Overrode the reviewer's suggested fix on two of three, with argument** — and on S7-3 **showed the reviewer's own fix does not catch the reviewer's own mutant** (a call that never returns is never followed by its assertion) | `test_storefront.py` only | `analyst` Pass 8 → — | 267k tok / 23 tools |
| **S7b gate** — do the two overrides deliver what the findings asked, or only look more rigorous? | `analyst` (**fresh** — the Pass 7 reviewer ended at 253k tok; this check is self-contained) | `a893ac5083e24f334` | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 0 major, 1 minor, 2 nits) — committed `6464b32`. **Both overrides upheld by execution**: it re-ran Pass 7's own suggested fix against Pass 7's own mutant and watched it **hang** (terminated 32 s, exit 143), and proved S7-1's substitute detects with the stub sleep set to **zero** (8/8), so detection genuinely does not depend on a duration. **S8-1: found the silent margin inside the fix that was justified by rejecting a margin** — `started_at` is stamped on the calling thread before `worker.start()` (teco confirmed at `test_storefront.py:103`) | `docs/reviews/salesperson-ui-impl.md` `## Pass 8` + Appendix L | — | 113k tok / 52 tools |
| **S7b2** — close Pass 8: stamp `started_at` on the worker thread (S8-1) + two nits | `coder` (**fresh** — the S7b author ended at 267k tok; the review fully specifies the work) | `a1af13ddaad935f25` | delivered — **committed `6fbe541`**, suite **2476** (baseline exactly) teco-verified solo; `storefront.py` byte-identical; 11 executable lines. **Reproduced the reviewer's false-green at HEAD, then killed it.** The adverse-ordering margin **grew** 54 µs → 142 µs, so the fix strengthens detection rather than merely not weakening it. **Rejected the suggested `expect_error` kwarg** for the literal `pytest.raises` idiom, deleting two bookkeeping asserts and a dead third notion of elapsed time. **Refused a widening that looked free** — a blind-sleep mutant showed `seconds=10` would turn the idle test green, trading a live detection for a flake that has never fired. **Sized a margin its own change created** (max thread-start skew 0.1446 ms vs 150 ms) | `test_storefront.py` only | **folded into the S7c gate** (same file; see note) | 119k tok / 38 tools |
| **S7c** — Ruling 1's catalog projection + removal of S7's `1+n` workaround | `coder` (**fresh**) | `a0633c22b2d2eaba5` | delivered — **committed `f5291e6`**, suite **2478** teco-verified solo; five files, both test files **pure insertions**, S7's catalog tests green **unedited** as the plan predicted. **Caught a false negative in its own first-draft tripwire** — the fixture slugs were exactly `slugify(name)`, so a `_catalog_rows` fabricating ids would have passed; one row is now `opaque-sku-42`/`Widget 007`. **Found the query gate is structurally blind** (finding below) and **sharpened the plan's counterweight at the mechanism level**: no test in this repo *can* observe what `FilterProductsTool` hands the model | `repository.py`, `storefront.py`, `test_repository.py`, `test_storefront.py`, `QUERIES.md` §15.2 | `analyst` → — | 139k tok / 65 tools |
| **S7c2** — the stale query-gate constant + §15.1's pre-existing drift (**teco-authorized scope widening**) | `coder` | `a0633c22b2d2eaba5` (resumed — it measured both fixes) | **accepted — committed `8aaeca3`**. **Refused to treat 408/408 as the evidence** and proved fidelity by AST instead — extracting the code's real query text, so the code side is the string the engine receives rather than a re-typing. **Corrected teco's overstatement of the defect** (the script self-checks its own header; it is blind across the code boundary only) and **sized the whole-document audit**: 109 blocks, 66 matching, 43 leads. **Pushed back on teco's scope line and was right** → S7c3 | `scripts/test_queries.sh`, `QUERIES.md` §15.1 | folded into the S7c gate | 164k tok / 16 tools |
| **S7c3** — `$LOOKUP`'s two coupled constants: the other half of the same K-053 instance | `coder` | `a0633c22b2d2eaba5` (resumed) | **accepted — committed `83af07c`**, two lines. All four §15 cells now agree. **Delivered the follow-up-16 design** (fence marker, three rules, both alternatives rejected with reasons) and **the comparator in prose rather than as a file** — teco rebuilt it from that description and reproduced both `MATCH` results, which is the test of whether a description is durable | `scripts/test_queries.sh` only | folded into the S7c gate | 173k tok / 7 tools |
| **S7c gate** — Pass 9 over **four** commits: S7b2, S7c, S7c2, S7c3 | `analyst` | `a893ac5083e24f334` (resumed — wrote Pass 8 on this same file) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 0 major, 1 minor, 2 nits) — committed `cc3a9e0`. **"This surface is ready to build S8 on."** Verified all three disputed claims rather than accepting them, and explained *why* the deleted silent-drop branch is safe: `Product.productId` is **UNIQUE but not MANDATORY**, so a null row is representable today and was identical under reconstructed S7 code. **S9-1: the tripwire pins a method *name*, not a read *count*** — restoring the `1+n` loop via `_repo` leaves it fully green. **Corrected an earlier pass of its own review**: S7-4 did not vanish, 32 reads → **2, not 1** | `docs/reviews/salesperson-ui-impl.md` `## Pass 9` + Appendix M | — | 211k tok / 60 tools |
| **S7c4** — close S9-1: make the tripwire pin the read *count*, not an attribute name; §15.1's date nit | `coder` | `a0633c22b2d2eaba5` (resumed) | in-flight (**holds the live DB**) | `test_storefront.py`, `QUERIES.md` §15.1 | folded into S8's gate | — |
| **U28** — Plan v1.18: the four **proved** corrections (Rulings 1-3 + S7's `storefront_dir` wiring) | `architect` (**fresh** — the v1.17 architect ended at 102 tool uses) | `a29f3ebb7c1908730` | **accepted — committed `039cae3`** (50/18, one file); all seven step-row hashes **teco-re-derived independently** and matching. **Improved two of teco's four framings** (below) and **closed a pre-existing §5.0 map gap** — S9 listed no test file at all, despite every S9 done-condition being a test. **Refused a coordination decision rather than taking it** — see the split, next row | `docs/plans/salesperson-ui.md` **v1.18** | **none — plan gates stopped** | 147k tok / 62 tools |
| **U29** — Plan v1.19: **split Ruling 1 out of S8** into `S7c` ahead of it; carry S9's cache decision in the S9 row | `architect` | `a29f3ebb7c1908730` (resumed ×2) | **accepted — committed `732f5e0`** (27/19). **Decided S9's cache question rather than parking it** — remove `_records` whole, with S7-2 banked as *dissolved rather than fixed* and S8's now-vacuous tripwire recorded as the **correct** end state. **Self-reported a miss in its own v1.18 delivery** (§9 never received v1.18's map changes). Renamed off the `S7b` collision teco caught, and **argued for keeping two mentions as tombstones** rather than a clean grep — the sequence gap is otherwise unexplained plan-side, and closing it would recreate the collision | `docs/plans/salesperson-ui.md` **v1.19**; S7c `2f03c064`, S8/S9 unmoved by the rename | **none — plan gates stopped** | 181k tok / 3 tools (rename) |
| **S3** — Two wiring switches: responder kill switch + §4.9's `dev_surface` un-mounting | `tdd-engineer` | `adebab5c261838206` | delivered — suite **2391** teco-verified. **Caught the `_IncludedRouter` trap while writing the test**: FastAPI 0.139 keeps an included router as ONE opaque entry, so the naive `app.routes` read sees 7 of 37 paths and the obvious assertion passes *while the router is mounted*. Added a **positive control** so the empty-table assertion can't pass vacuously. 7 mutations, 7 killed | `config.py`, `app.py`, `test_app.py`, `SERVER.md`, `HISTORY.md` | `analyst` Pass 5 → — | 175k tok / 49 tools |
| **S3 gate** — Pass 5 on the impl review | `analyst` (**fresh**) | `a24e4bcbd0b9a1f8e` | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers). **Found the 7th instance, *pre-planted***: `_route_paths` is prefix-blind, harmless in S3, but S8 is told to reuse it for a route table whose whole content **is** a prefix. Proved S3's vacuity mode empirically (renamed the traversal attr → assertion still passed, control failed) | `docs/reviews/salesperson-ui-impl.md` `## Pass 5` + Appendix I | — | 109k tok / 45 tools |
| **S3b** — P5-1 prefix threading, P5-2 raise-don't-skip, 2 nits | `tdd-engineer` | `adebab5c261838206` (resumed) | **accepted — committed `673342b`**. Suite **2394** + prefix fix teco-verified on an S8-shaped 2-level app (`/shop/api/join`). **Went past the review**: the gate called P5-4 unreachable, its own mutation confirmed the fix was *unpinned*, so it wrote the test anyway — 3 distinct prefix mutants, "three independent ways to be wrong" | `app.py`, `config.py`, `test_app.py`, `SERVER.md`, `HISTORY.md` | `analyst` Pass 5 → **approve w/ suggestions, closed** | 186k tok / 15 tools |
| **P5-3** — `SERVER.md` is in no row of §5.0's file map (**3rd map gap this review has found**) | `architect` | — | queued (**held until Pass 7 returns** — the plan is under review; editing it now hands that gate a moving target) | `docs/plans/salesperson-ui.md` §5.0 | `analyst` → — | — |
| **U14** — Plan v1.6: S4 row corrected to **nine** methods (M-6) | `architect` | `abeef0ec5b77cc45f` (resumed) | delivered — also found **M-6's structural root**: S0's Interfaces column has named `ensure_participant` since v1.0, so the S4 row contradicted **the row above it**, not the note | `docs/plans/salesperson-ui.md` v1.6 | `analyst` → — | 157k tok / 51 tools |
| **U14b** — Plan v1.7: *cite, don't re-list* applied to S7/S10 | `architect` | `abeef0ec5b77cc45f` (resumed) | delivered — **the re-list was also WRONG**: S7/S10's quiesce done-condition was **vacuously true** (writes anchored on deleted nodes match zero rows whether quiesce works or not) on the demo's most destructive op | `docs/plans/salesperson-ui.md` v1.7 | plan re-gate → — | 174k tok / 60 tools |
| **U14c** — Plan v1.8: absorb S0's unactioned §12 hand-offs + full §12 audit | `architect` | `abeef0ec5b77cc45f` (resumed) | delivered — **4 items unabsorbed** (teco's 3 + the anomaly-response contract), 2 absorbed in code but not plan text, 3 correctly S4-scoped. **Found: no step owns the presenter view** | `docs/plans/salesperson-ui.md` v1.8 | plan re-gate → — | 199k tok / 8 tools |
| **U14d** — Plan v1.9: presenter view given an owner — **new row `S12d`** (`frontend-engineer`), numbering stable, no renumber; §10's AC-5 row corrected from 2 wrong owners to 3 right ones; §6.3 #8, §5.0, S15, §9 swept | `architect` | `abeef0ec5b77cc45f` (resumed) | delivered — also handed over **6 ranked self-flagged uncertainties**, incl. asking the gate to diff its own most-compressed edit against the source | `docs/plans/salesperson-ui.md` **v1.9** (20 steps) | **plan re-gate** → — | 212k tok / 5 tools |
| **U16** — Independent re-gate of the plan delta, v1.2→v1.9 (**+84/−39 accepted on teco's own verification alone** — producer self-verification, a teco process gap) | `analyst` (**fresh**) | `ab94a9b40db374063` | **accepted — NEEDS CHANGES** (1 blocker, 3 major, 8 minor, 2 nits). **Vindicated the gate outright**: the blocker was caused by a teco instruction | `docs/reviews/salesperson-ui.md` `## Pass 3` | — | 146k tok / 48 tools |
| **U17** — Plan v1.10: M-1 blocker (F8 → server-side S7/S10 + new `504`), M-2, M-3 (narrow), M-4 + 8 minors + 2 nits | `architect` | `abeef0ec5b77cc45f` (resumed) | delivered — **took M-3's origin on itself**: the six-field roster was the plan's own v1.0 invention, not a shortfall in delivered S4 | `docs/plans/salesperson-ui.md` **v1.10** (+106/−50 over v1.2) | `analyst` Pass 4 → — | **253k tok** / 20 tools — **now over the resume threshold; next architect unit dispatches fresh** |
| **U18** — Pass 4 re-gate of plan v1.10 | `analyst` | `ab94a9b40db374063` (resumed) | **accepted — APPROVE WITH SUGGESTIONS** (0 blockers, 1 major, 3 minor). Blocker + all 3 majors fixed; **2 architect fixes judged better than the reviewer's own proposals** | `docs/reviews/salesperson-ui.md` `## Pass 4` | — | 187k tok / 26 tools |
| **U19** — Plan v1.11: P4-1 partial sweep + P4-2/3/4 | `architect` (**fresh** — prior instance at 253k tok) | `a3ff2db4359dbebc2` | delivered — swept all 4 hits of the removed fields (1 defect, 3 legitimate); **found an undocumented gap nobody had seen**: S12a's `504` re-read calls `/state` on the reset-all path, which answers `401` once the sweep invalidates the presenter's participant token | `docs/plans/salesperson-ui.md` v1.11 | `analyst` Pass 5 → — | 96k tok / 26 tools |
| **U19b** — Plan v1.12: reset-all `504` re-read → roster not `/state`; S12d negative-control fixture explained | `architect` | `a3ff2db4359dbebc2` (resumed) | delivered — **found a second, worse defect one layer up**: S12a's `401 → rejoin` is undifferentiated across two credentials, so a **successful** reset-all yanks the presenter off `/shop/presenter`, breaking §4.3's explicit promise (teco-verified, lines 466-470) | `docs/plans/salesperson-ui.md` v1.12 | `analyst` Pass 5 → — | 107k tok / 10 tools |
| **U19c** — Plan v1.13: client credential contract consolidated into a **new §5.3** (C1–C8) — teco's call to fix the class, not add a third clause | `architect` | `a3ff2db4359dbebc2` (resumed) | delivered — surfaced a **pre-existing hole: §6 had no client tier at all** (the SPA's whole test-strategy presence was "`npm test` green" in 4 step rows). Made + labelled one decision (`localStorage`); flagged `503` as carrying the same defect `504` had | `docs/plans/salesperson-ui.md` **v1.13** | `analyst` Pass 5 → — | 143k tok / 18 tools |
| **U20** — Pass 5 re-gate of plan v1.13 | `analyst` | `ab94a9b40db374063` (resumed) | in-flight | `docs/reviews/salesperson-ui.md` `## Pass 5` | — | — |
| **U15** — Graph note v1.3: close §12 open item 2 (storefront does **not** advance cursors — `architect`-confirmed); N-5 `v6`→`v7` | `graph-dba` (**fresh** — S0 instance at 431k tok) | `ad82106cf50e24e44` | in-flight | `docs/plans/salesperson-ui-graph.md` v1.3 | `analyst` → — | — |
| **S1c** — N-1: extend the drift check to transition `guard` (also create-only); N-4 burn note in `seed_salesperson.sh` | `coder` | `aef74d44be60f1cff` | queued (**behind S4 — shared live DB**) | `verify_salesperson.sh`, `seed_salesperson.sh` | `analyst` → — | — |
| **S2c** — N-2: comment says "~20 chars", measured **46**; N-3: one sentence noting the timers test is now coupled to the start bound | `tdd-engineer` | `a1aa5c430de8da50d` | queued (**behind S4 — `services.py` same-file collision**) | `services.py`, `test_workflow_timers.py` | `analyst` → — | — |
| S2 · S3 — `run_ctx` merge, responder kill switch | `tdd-engineer` | — | queued (**serialized behind S1 — shared live DB**) | — | — → — | — |
| S4…S16 — remaining implementation | per plan v1.2 §5.1 | — | queued | — | — → — | — |

## Stakeholder decisions, 2026-09-02 (plan §8)

| OQ | Decision | Effect on the plan |
|---|---|---|
| **OQ-1** AC-3 acceptance basis | **Stub-LLM pass + a *published* live latency curve + a staggered demo script.** A live pass/fail concurrency threshold is explicitly **not** the bar. | As the plan proposed (§6.4 stands). R1 is accepted as a stated residual, not engineered away. |
| **OQ-3** code home | **Neither option offered.** Move the existing Streamlit app to a new **`deprecated/salesperson/`** directory **now**, and give the new client component the freed **`salesperson/`** name. Server half stays inside `falkor-chat` (as the plan argued). | **Plan change.** §4.1 rewritten; no `salesperson-ui/`; S5 scaffolds into `salesperson/`; S16's `git rm -r salesperson/` becomes a *preserving* move done early, not a delete done last; §2.4's parity citations must be re-pointed at `deprecated/salesperson/*.py`; R11 is materially weakened as a risk. |
| **OQ-5** presenter identity | **`FALKORCHAT_PRESENTER_KEY`, rate-limited** — the plan's original env-var operator secret. The localhost-only binding chosen in this round was **reverted at U7/B1**: it was *weaker* than the key it replaced, because uvicorn 0.49.0 defaults `proxy_headers=True` and trusts `FORWARDED_ALLOW_IPS`, so behind the TLS proxy §3 promises every peer is loopback. The analyst's startup-printed-token option was also declined. | No net plan change from the original design. §4.3 records **why** the loopback variant was rejected so a later reader doesn't "simplify" it back; R6 names the standing shared secret as the accepted residual. See the U7 row below for the full round trip. |
| **OQ-6** product images | **An agent sources ~15 permissively-licensed stock images** (Unsplash/Pexels-class), commits them keyed by `productId` slug, and records the licence in the component README. | As proposed. `dist/` stays gitignored with a documented build (plan default). |
| **OQ-2** locales | **en / pt-BR / es** — plan default accepted. | No change. |
| **OQ-4** order advance | **Participant self-serve only**; no presenter-driven variant. | No change. |

## Documentation impact (scanned at decomposition; refined after U1)

| Document | Why it is touched | Owner |
|---|---|---|
| root `AGENTS.md` | `salesperson/` row now describes the **new** UI; a new `deprecated/` row for the retired Streamlit app; component-docs table; "Working in this repo" bullet | U5 (move) + S16 (final pass) |
| `deprecated/README.md` (new) | states what `deprecated/` means and that nothing in it is maintained | U5 |
| `salesperson/README.md`, `salesperson/AGENTS.md` | retired app | implementer |
| `falkor-chat/README.md` / `AGENTS.md` / `docs/QUERIES.md` | only if new REST routes / graph reads are added | implementer |
| `docs/HISTORY.md` (whichever tree owns the new component) | one entry per delivered change | implementer |
| `docs/BACKLOG.md` | new `K-`/`C-` items filed out of gates | `teco` reports, human applies |
| `falkor-chat/AGENTS.md` | v4→v5 drift **fixed 2026-09-02 (U1c)**; plan step S1 bumps to `v7` (**not `v6` — burned, see the S1 section**) and must carry the same two rows forward | `teco` (done) → S1 implementer |
| `docs/manuals/salesperson-ui.md` | end-user manual for the shipped UI (FR-1…FR-11 walkthroughs) | `tico` |

## U1b verification (teco, 2026-09-02)

`cpg_falkorchat` rebuilt and independently verified, not accepted on the delegate's word:

- Marker: `builtAt 2026-09-02T12:38:21Z`, `sourceCommit 4bb96e1` (= `HEAD`), parse root
  `cpg/.cpg-artifacts/src/falkor-chat-server`. **First build of this graph ever to carry a real
  `sourceCommit`** — `graph-dba` staged the pruned copy inside the repo (gitignored) instead of
  `/tmp`, so future freshness checks can run `git log <sourceCommit>..HEAD` instead of raw-age
  guessing.
- Spot-checked 4 post-staleness symbols live against the working tree — `services.add_cart_item`
  (2653), `services.advance_order` (2812), `querygen.compile` (275),
  `executor._resolve_add_to_cart_dedup_args` (334) — **all line numbers match exactly**.
- 285,546 nodes / 1,935,681 edges (was 234,396 / 1,583,246). Data-flow layer present
  (`REACHING_DEF` 477,889), so the `cpg-analysis` RCA/taint and test-gap recipes work.
- **`SOURCE_DIRTY: true` is a false alarm and must not be read as "the parsed source was
  modified"** — `pipeline.sh` runs `git status --porcelain` repo-wide with no pathspec, so it
  stamped `true` because of unrelated untracked files. `git status --porcelain -- falkor-chat/server`
  was empty; teco re-confirmed this independently.

## Follow-ups filed (not this coordination's scope)

1. **`cpg/.cpg-artifacts/MANIFEST.txt` had drifted** — its last recorded baseline was a 2026-08-17
   build, but the graph actually replaced was a 2026-08-26 one from a run that never appended.
   Consider gating future rebuilds on a manifest append. Owner: `graph-dba` / `devops`.
2. **`skills/joern-cpg/SKILL.md`'s per-file scaling rule of thumb under-projects by ~18-20%** —
   documented ~2,700-2,800 nodes / ~18,000-18,600 edges per Python file; this run measured
   3,245 / 21,996. Owner: `cobb` (skill owner); not edited by `graph-dba`, correctly.
3. **`SOURCE_DIRTY` is repo-wide, not source-scoped** — worth fixing in `pipeline.sh` (add a
   pathspec) or documenting in `skills/cpg-analysis/references/freshness.md`, since as-is it
   produces a permanently-`true` field that readers will learn to ignore. Owner: `graph-dba`.

## Note on U1c (teco's own trivial fix)

`falkor-chat/AGENTS.md` rows 82-83 documented `salesperson@v4` as current; source is `v5`
(`proof_defs.py:301`, K-057). Both seed/verify scripts already defaulted to `v5`, so the drift was
doc-only and confined to one file — inside `teco`'s trivial-fix exception, taken at the
stakeholder's explicit direction. The replacement prose mirrors `proof_defs.py`'s own module
docstring and `seed_salesperson.sh`'s header. **Skipped the independent-review gate by
construction** (root `AGENTS.md`: trivial, low-risk units may, stated explicitly). An unfiltered
sweep confirmed no other live document claims v4 as current — remaining `v4` mentions are in
frozen `plans/`, `reviews/` and `test-reports/` documents, where they are correct history.
**Plan step S1 bumps the def to `v7` — not `v6`, which is burned (see the S1 section); whoever takes S1 must carry these same two rows forward.**

## U2 gate outcome (2026-09-02) — **needs changes**

`docs/reviews/salesperson-ui.md`. The analyst re-verified ~20 source claims and 2 live FalkorDB
claims from the plan and **found no false one** — the design's grounding is sound; the failures are
at its edges.

**Blockers, and where each routes:**

| # | Finding | Routes to |
|---|---|---|
| **B1** | The **new, stakeholder-chosen** localhost-bound presenter routes are *weaker* than the key they replace. uvicorn 0.49.0 defaults `proxy_headers=True`, trusting `FORWARDED_ALLOW_IPS` (default `127.0.0.1`). Behind any TLS-terminating proxy — which §3's own diagram promises — every peer *is* loopback; or a LAN client sends `X-Forwarded-For: 127.0.0.1` and takes the presenter surface. `::1` isn't in uvicorn's default trust list either. | **stakeholder (U7)**, then `architect` |
| **B2** | The unauthenticated legacy REST + `web/` + MCP surface is an unaddressed **AC-2 read path**. If `FALKORCHAT_WS_ID` lands on the demo workspace, any phone that trims `/shop` off the link reads every participant's transcript and can post as `u1`. §4.3, S8, §6.2 and §10 are all silent on it. | `architect` |
| **B3** | S11 never seeds the `Agent` into the demo workspace (`seed_demo.sh` defaults to `acme`; S11 calls it bare), so `_validate_and_derive_role` raises `UnknownMemberError` **before any write** and every participant's first message 500s. B2 and B3 are the same trap from opposite sides — the obvious fix for one opens the other. | `architect` |
| **B4** | Two repository primitives the design needs (resolving a customer's orders; `get_order` is by `orderId` only) **don't exist and are in no step's scope**. §5.0 pins `repository.py` away from S7, so the S7 delegate would have to stall or write Cypher into `storefront.py`, breaching the layering rule. | `architect` |

**Teco-verified independently** (not accepted on the reviewer's word): uvicorn `0.49.0` /
`proxy_headers` default `True` (B1); `seed_demo.sh:42` defaults to `acme` and
`services._validate_and_derive_role` is pre-write validation (B3).

**Majors worth naming here:** M1 (§4.8 and §5.2 flatly contradict each other on whether "reset
mine" invalidates the token — a real design fork, back to `architect`); M2 (no per-participant
turn serialization — two rapid posts start two concurrent runs on one thread); M3 (the image
manifest points at the *source* dir, not the served one, so AC-11 silently degrades to all-text-only
while §6.3 #9 still passes); M4 (AC-8/FR-10 not actually covered — the join display name never
reaches the profile); M5 ("roughly halves LLM load" is off by 3-9×, and it feeds OQ-1's hardware
conversation); M6 (§5.0's shared-file map — *what dispatch is gated on* — is incomplete in three
places); M9 (S16's acceptance command can never pass, and its file list misses a live agent prompt
this work invalidates → **U6, routed to `cobb`, not S16's `coder`**).

**OQ-1 caveat the analyst added, and I accept:** the chosen AC-3 basis makes AC-3's literal wording
("no noticeable degradation **for any participant**") unmeetable for agent turns. The test report
must say so plainly rather than record a pass against wording it doesn't satisfy.

## Stakeholder decisions, round 2 (2026-09-02, post-gate)

| Question | Decision | Consequence |
|---|---|---|
| **B1 / OQ-5 re-decided** | **Revert to `FALKORCHAT_PRESENTER_KEY`** — the plan's original env-var operator secret, rate-limited. Both the hardened-loopback variant and the analyst's startup-printed-token option were declined. | The localhost-binding design is dropped after one round trip. §4.3 must **record why** (B1's uvicorn `proxy_headers=True` inversion) so a later reader doesn't "simplify" it back. R6 names the standing shared secret as the accepted residual. |
| **B2** | **`architect` decides, but must close it structurally** — AC-2 has to hold by construction, not because an env var happens to be right. Stakeholder explicitly declined to prescribe the fix. | Solved together with B3 (the reviewer's warning: same trap from opposite sides; the obvious fix for one opens the other). |

**Process note worth keeping.** The presenter design round-tripped: plan proposed a key → stakeholder
chose localhost-only → gate proved localhost-only strictly weaker → stakeholder reverted to the key.
The revision must preserve that trail as a rejected-option-with-evidence in §4.3, not silently
present the key as if it had never been questioned. This is the `BACKLOG.md`/`DESIGN.md` rule from
root `AGENTS.md`: a rejected option with a reversal trigger is a live constraint on the system, and
belongs on the design surface that owns it.

## U6 verification + gate decision (teco, 2026-09-02)

**Accepted. Independent review gate deliberately skipped — recorded here explicitly** per root
`AGENTS.md`'s "say so in your report" rule, on the grounds that every factual claim in the change
was verifiable directly and I verified all of them, and that a prompt edit is trivially reversible:

- **The load-bearing new claim — "this lab's CPGs are Python-only, so no front-end source is in
  one" — verified live**, not taken on report: `cpg_falkorchat` `METHOD.FILENAME` extensions are
  `py` × 4084 and extension-less × 521. **Zero** `.js`/`.ts`/`.html`/`.css`. So
  `falkor-chat/web/app.js` — the one front-end already in this repo — is not in a CPG and never was.
- **Diff scope verified**: `git status` shows the change confined to `claude/` and `skills/`, as
  briefed. No `salesperson/`, `falkor-chat/` or `docs/` file touched.
- **All five substantive diffs read in full** (`frontend-engineer.md`, `analyst.md`,
  `data-scientist.md`, `claude/README.md`, `skills/cpg-analysis/SKILL.md`) — each matches its
  reported description exactly; no unreported edits.

**The judgment call worth preserving.** `cobb` declined to swap `cpg_salesperson` → `cpg_falkorchat`
in the `frontend-engineer` prompt, because the agent's *own* `kaizen/plan.md` had predicted this
exact rot on 2026-08-24 ("`cpg_salesperson` now lives in three places that rot together"). A fresher
pointer would have been a fourth site of the same fragility. It replaced the pointer with a fact
that cannot rot on a rename, and named **no directory** for the new component — so whichever way the
`salesperson/` rename lands, there is no further site to update. `/shop` is the only path pinned,
and the plan's mount design fixes it.

**Sweep was not a clean negative** — two other live prompts carried the same stale assumption and
were fixed in the same pass: `claude/data-scientist/data-scientist.md:56` described `salesperson` as
the retired LangChain/LangGraph app (an ML-method question about "the salesperson agent" would have
been answered against the wrong system), and `claude/analyst/analyst.md:84` used `cpg_salesperson` as
a one-token example. `claude/AGENTS.md` verified as a genuine negative — nothing in it became false.

## Follow-ups filed (round 2)

4. **`claude/scripts/audit-team.sh` reports 3 pre-existing FAILs on check 7 (personal identifiers)** —
   `claude/docs/plans/bypass-permissions-subagent-gap{,-coordination}.md`,
   `claude/docs/reviews/bypass-permissions-subagent-gap.md`, and `docs/plans/doc-reference-convention.md`.
   None is in U6's diff and U6 added no new FAIL line. Three of the four sit in `claude/docs/`, so
   they are arguably `cobb`'s on a later pass. **Not this coordination's scope.**
5. **`claude/devops/kaizen/plan.md:46-56`** carries two forward-looking backlog items that die with
   the Streamlit app ("Extend Compose coverage to `salesperson`", "a `salesperson` Streamlit app
   image"). Real drift, but it is another agent's backlog and the retirement has not landed —
   correctly left alone by `cobb`; belongs in the milestone closeout list.
6. **`skills/joern-cpg/references/cpg-model.md:34`** cites `(cpg_falkorchat, cpg_salesperson)` as
   evidence for a dated, live-verified caveat. True as written and still true; rewriting it would
   falsify an evidence trail. It follows the graph's actual fate → **U8**.

## Not this coordination's work — seen and deliberately untouched

`docs/requirements/small-model-benchmarking.md` (untracked, `Status: Interviewing`, owner `tico`,
dated 2026-09-02) is an **in-progress requirements interview from another session**. It is not part
of this coordination, no unit of mine created or modified it, and nothing here should touch it. Noted
only so a later reader doesn't mistake it for stray output of this work.

## U4 plan revision accepted for re-gate (teco-verified, 2026-09-02)

`docs/plans/salesperson-ui.md` **v1.1**, header block carries `Version:` + `Reviews:` and one dated
revision line (not a stacked `Update:` narrative), per root `AGENTS.md` rule 5. Spot-verified by
teco: **19 step rows** (S12 split into S12a/b/c), `dev_surface` present, and — the one I actually
distrusted — **all 4 `FALKORCHAT_DEMO_WS` occurrences are explicit rejections, not usages**, so the
two-variable trap really is gone rather than renamed.

**How B1-B4 were closed:**

- **B1** — `FALKORCHAT_PRESENTER_KEY` reinstated per the stakeholder. The loopback variant is
  recorded as *tried and rejected on executed evidence*, in a paragraph opening **"Do not 'simplify'
  this to a localhost check"** — exactly the rejected-option-with-reversal-trigger treatment root
  `AGENTS.md` asks for.
- **B2 + B3 — solved together, structurally, in a new §4.9, by making the dangerous configuration
  inexpressible rather than merely checked.** (1) With the storefront enabled, `create_app` doesn't
  mount the unauthenticated surfaces *at all* — no `api.build_router`, no `/` static mount, no
  `/mcp`; `dev_surface` is a **function parameter for tests, never an env var**. (2) `FALKORCHAT_DEMO_WS`
  is deleted outright: once the unauthenticated readers are gone, `WS_ID` has no security role, so a
  second variable buys nothing and creates the only thing that made B3 possible — two values that can
  disagree.
- **`architect` declined one of the reviewer's own B2 fixes and said so with reasoning**: refusing to
  start when `WS_ID == DEMO_WS` is correct only if the legacy surface stays mounted, and adopting it
  alongside the real fix would *mandate* the two-variable split and therefore mandate B3's trap. This
  is the behaviour I want from a revision — reasoned disagreement in the document, not silent
  compliance.
- **B4** — the two missing primitives added to S0 **and** S4, with the dispatch allocation fixed:
  §5.0's `services.py` row is now `S2 → S4`, so the S7 delegate can't be stranded.

**M1 fork settled** (reset-mine keeps the token; re-joining would orphan `User`/`Channel` nodes the
presenter roster reads, and `customerId == participantId`). **M2 partially declined** with reasoning
(per-participant single-flight adopted as a correctness fix; the extra "one pending" slot declined as
a second queue-position concept in both API and UI).

**AC-3's honesty clause is now in the plan**, §6.4 and the §10 row: met for all read paths at 50
participants, **not** met as literally worded ("for any participant") for agent-reply latency. The
recording rule is written down so nobody has to decide it under pressure on demo week.

## U5 verification (teco, 2026-09-02)

- **25 tracked files registered as renames** (22 `R`, 3 `RM` for banner/path edits) and **zero
  deletions** — history followed the move, nothing was recreated. `salesperson/` is gone;
  `deprecated/salesperson/` holds 23 entries.
- **The plan's four `deprecated/salesperson/*.py` citations all resolve** (`cart.py`, `chatbot.py`,
  `customer_profile.py`, `session_manager.py`) — which is the specific thing that had to be true
  before the Pass 2 re-gate could check §2.4's parity evidence. This is why the re-gate was held.
- Root `AGENTS.md` describes `deprecated/` and **correctly does not describe a new `salesperson/`
  component**, which does not exist yet. Its "Retired components" bullet adds a genuinely useful
  guard: a request to work on "the salesperson app" is almost certainly about the not-yet-built
  replacement, so confirm before touching `deprecated/`.
- Unfiltered post-move sweep: the only surviving live references to the old path are
  `docs/requirements/salesperson-ui.md` (where `salesperson/` is the **statement of the problem
  being solved**, not a stale pointer — correctly left) and `claude/frontend-engineer/kaizen/plan.md`
  (forward-looking and accurate). Both correct leaves.

## Follow-ups filed (round 3)

7. **`.claude/settings.local.json:13` carries a now-stale permission-allowlist entry** — an absolute
   path ending `.../graphmind-ai-lab/salesperson --graph cpg_salesperson`, i.e. **both** halves are
   now wrong (directory moved; graph being renamed by U8). Impact is benign — an allowlist entry that
   no longer matches fails closed, costing at most one extra permission prompt. **Deliberately not
   touched by anyone**: `coder` correctly judged it entangled with U8, and `teco` is deliberately not
   editing a permissions file — that is the user's own domain, not an agent's. **Relayed to the
   stakeholder as a finding.**
8. **The moved `.venv/` is dead** — `git mv` on a directory silently carried a gitignored virtualenv
   whose console-script shebangs and `pyvenv.cfg` still point at the pre-move path, so
   `deprecated/salesperson/README.md`'s `./.venv/bin/python …` invocation will fail. Correctly not
   fixed (gitignored, app retired, and recreating it is an environment mutation needing approval).
   Anyone wanting to run the retired app needs a fresh
   `python -m venv .venv && pip install -r requirements.txt`. Captured to `kaizen_team` by `coder`.

## U8 verification (teco, 2026-09-02)

Renamed via plain Redis **`RENAMENX`** on the `graphdata` key — a true in-place key move, **no
destructive copy-and-drop**, so the stop-and-ask fork in the brief was never reached. `graph-dba`
proved the semantics on a disposable probe graph *before* touching the only copy of a graph with no
rebuild path: `GRAPH.LIST` follows the key, indexes stay `OPERATIONAL`, and even the compiled
query-plan cache follows (the internal `GraphContext` is moved, not rebuilt).

Teco-verified independently: `cpg_deprecated_salesperson` at **17,549 nodes / 359 `METHOD`**
(identical to pre-rename), `EXISTS cpg_salesperson → 0`, and the marker reading
`builtAt=unknown · sourcePath=deprecated/salesperson · status=retired-component · renamedFrom=cpg_salesperson`.

**The marker is deliberately not a normal one.** `BUILT_AT` is the literal string `"unknown"` — not
NULL, not a plausible timestamp — so it fails ISO parsing *loudly* instead of being silently
coalesced, and `sourcePath` puts `deprecated/` in front of any reader using the standard recipe.
`MARKER_ORIGIN` states plainly that it is hand-written, not a pipeline stamp. This is the right
trade: zero rows is indistinguishable from "the pipeline failed".

`MANIFEST.txt` needed **no** change — verified: it never referenced the graph key at all.

## Follow-ups filed (round 4)

9. **`docs/plans/kaizen-team-sandbox.md`** (`Status: active`, owner `architect`) asserts as observed
   live state that "only `cpg_falkorchat` and `cpg_salesperson` are loaded". Now false on both the
   name and the count. Low priority — an active plan's observed-state note, not a live constraint.
   Owner: `architect`.
10. **`falkordb-quirks.md` promotion pending.** `graph-dba` logged the `RENAME`/`RENAMENX`
    non-destructive-graph-rename technique to `kaizen_team` (`entryId
    a79dd064-a17e-4f91-a174-09d42eda1e6f`, `suggestedHome: knowledge base`) rather than writing it
    directly, because the quirks file lives under `claude/`, outside that unit's scope. Owner:
    `cobb`, on its next distillation pass.

## Outstanding for the stakeholder (teco will not do these)

- **A disposable probe graph is still loaded:** `probe_u8_rename_dst` (3 synthetic `:Foo` nodes, 1
  edge), created by U8 solely to prove the rename semantics. Cleanup command:
  `docker exec falkordb-dev redis-cli GRAPH.DELETE probe_u8_rename_dst`. **Neither `graph-dba` nor
  `teco` will run it** — `graph-dba` correctly returns destructive commands rather than executing
  them as a subagent, and destructive graph ops are outside `teco`'s Bash grant (read-only
  investigation, project suites, integration commits). Blast radius is nil; it is litter, not risk.
- **`.claude/settings.local.json:13`** — stale permission-allowlist entry (see follow-up 7). A
  permissions file is the user's own domain; no agent in this coordination will edit it.

## U8b verification (teco, 2026-09-02) — and why its finding matters beyond this coordination

**Accepted; independent gate skipped, recorded explicitly.** The unit's load-bearing claim is an
empirical one about `git` behaviour, and I reproduced it directly rather than trusting the report:

```
git log --oneline --since=unknown -- skills/   → 0 commits, exit 0
git log --oneline --since=2026-08-01 -- skills/ → 31 commits   (control)
```

**Git silently accepts an unparseable approxidate and returns zero commits with exit 0.** It does
not error. So the standard staleness check — `git log --since=<builtAt> -- <sourcePath>` — run
against a hand-written marker produces a confident, wrong "the source hasn't moved" answer. That is
a **false-negative that reads as a clean bill of health**, which is the worst possible failure shape
for a freshness check, and it was verified by running it, not reasoned about.

`graph-dba` also placed the guard **on check 2 itself** — where the harmful command is literally
written — rather than only in the two sections I named, on the reasoning that a `teco` following the
recipe top-down must hit the warning *before* the command, not after. That is a better call than my
brief and I'm adopting it as written. It further noted that the scratch-build escape hatch does
**not** carry over (there is no `builtAt` to anchor a `--since` on at all), and that for a retired
component "frozen snapshot" is the *correct* reading — so the doc's usual "ask `graph-dba` to
rebuild" reflex is wrong here.

**Beyond this coordination:** this hazard applies to *any* CPG whose marker lacks a parseable date,
not just the one graph renamed here. Every consumer of the freshness recipe — `teco` at dispatch
time, per the recipe's own stated audience — was exposed to it before this fix.

## U10 Pass 2 outcome — **approve with suggestions**, all 4 blockers closed

**The one gate remaining is an ordering constraint, not another review:** **N1 must land in §4.8/S0
before S0 is dispatched**, because S0 is first out of the door and is the step that would otherwise
bake the defect into the reset Cypher.

**N1 (Major) — teco-verified live before acting on it.** `config.WS_ID` defaults to `"acme"`
(`config.py:16`) and S11 never *pins* `FALKORCHAT_WS_ID`, so the storefront's workspace silently
becomes `ws:acme` — which I confirmed holds **2 `Channel`, 2 `Thread`, 52 `Message`, 1 `User`** plus
544 `Entity`/87 `Chunk`/29 `Document`. `reset-all` would run its multi-label sweep against that.
The design's *intent* is safe; the **test** is not: victims and survivors share the labels
`Channel`/`Thread`/`Message`, so S4's "assert every survivor by label" is **structurally incapable**
of catching an over-broad channel delete — it would pass while the data went. Fix: pin
`FALKORCHAT_WS_ID` in `start_demo.sh` (still one variable, §4.9 survives), add a non-label survivor
clause to §4.8, and — the part that actually closes it — have S4 seed a **non-participant**
channel/thread/message and assert it survives `reset_all`. A positive test, not another label
assertion.

**Notable dispositions.** The reviewer **withdrew its own B2 fix (a)**, stating it had the logic
backwards and that `architect`'s decline was correct — the strongest possible validation of writing
reasoned disagreement into the document rather than complying silently. It also credited a real
catch: `GET /health` lives **inside** `api.build_router` (`api.py:55`, teco-confirmed) and would have
disappeared with the un-mounting, so §4.9's route-table claim rests on S3's bare liveness route
rather than an assumption. M9 was verified by execution: the replacement `grep` returns zero now, and
returned exactly the two `frontend-engineer.md` lines at `HEAD` — so the plan's parenthetical was
true when written and U6 has since closed them. M2's decline accepted (the correctness half —
server-side single-flight, `409` before the write — was what mattered).

**Dispatch judgment recorded (deviation from teco's own heuristic).** `architect` is at ~282k
cumulative tokens, past the ~250k threshold at which a small follow-up would normally go to a fresh
delegate. I resumed it anyway: N1 changes reset semantics that interlock with **M1's settled fork**
(reset-mine keeps the token) and **M8's label inventory**, both decided by this agent last pass with
part of the reasoning necessarily in its head. A fresh agent re-deriving whether a non-label survivor
clause conflicts with those risks incoherence worse than the context bloat. Its tool use is also low
(19), so the context is document content, not tool churn.

## Dispatch deviation from the plan's stated parallelism (teco, 2026-09-02)

The plan's §9 dispatch order opens **"S0 · S1 · S2 · S3 · S5 in parallel"**. **S1, S2 and S3 cannot
safely run concurrently — with each other or with S1 — and I have serialized them.** This is a
`teco`-level dispatch constraint, not a plan defect: the plan's own R8 names the mechanism, and
§5.0's file map is file-scoped, so it correctly shows these steps as disjoint *on disk*. The
collision axis is **live database state**, which a file map cannot express.

**Verified in source before acting** (`falkor-chat/server/tests/conftest.py:101-110`):

```python
@pytest.fixture()
def wf_repo(conn) -> Repository:
    """A Repository over `ws:test` **and** the global `reference` graph, both wiped."""
    db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")
```

So **any `pytest` run wipes the global `reference` graph**, and `conn` wipes `ws:test`. Therefore:

- **S1** seeds and verifies `salesperson@v7` + `order-fulfillment@v1` into `reference` and asserts
  `verify_salesperson.sh` exits 0 — a concurrent suite run pulls that graph out from under it.
- **S2 and S3** each end in "pytest green". Two concurrent full-suite runs wipe `reference` and
  `ws:test` under one another.

Either failure mode is **transient and mutually corroborated** — the exact shape my standing
instructions warn is most likely to be misread as a real defect and burn a debugging cycle.

**Dispatched now (genuinely disjoint on both axes):** S0 (`graph-dba`, works only in a uniquely-named
throwaway probe graph — explicitly barred from `ws:test`, `ws:acme` and `reference`) and S5
(`devops`, Node toolchain only, needs no database and is barred from running the suite).

**S1 → S2 → S3 follow serially.** The cost is small: they are three of nineteen steps, and S4 gates
on S0 anyway.

## U11 verification (teco, 2026-09-02)

`grep 'salesperson\|deprecated' docs/manuals/graph-ontology.md` → **0**. The stakeholder's scope cut
landed: the manual names no deprecated graph, and the `BUILT_AT = "unknown"` / hand-written-marker
material `tico` had drafted was removed rather than kept.

Two judgment calls worth preserving. The "Loaded right now" cell was rewritten as a claim about
*the live CPG* rather than an exhaustive inventory — so it cannot rot the same way again when the
graph list next changes. And the durable lesson was kept but **decoupled from any specific graph**,
landing as an FAQ entry framed as *"the CPG mistake that returns confident wrong answers rather than
an error"* — which is the failure mode that actually costs someone a day.

## Follow-up 11 — for the stakeholder to decide, NOT actioned

**`docs/manuals/graph-ontology.md` §2 (`kaizen_team`) is materially wrong, and it is pre-existing
drift — not caused by this coordination.** Found incidentally by `tico` while live-verifying, and
correctly left alone by it.

**Teco-verified:** `CALL db.labels()` on `kaizen_team` → `['KaizenEntry', 'Agent']`, and
`db.relationshipTypes()` carries `PRODUCED`/`MENTIONS`. The manual states at line 202 that the graph
is *"deliberately **flat today** — one node type, no edges"* and at line 243 that the richer ontology
is *"Ready for design, not yet built"*. **The M8 ontology shipped 2026-08-22** (root
`docs/HISTORY.md`).

**Why this is worth a unit rather than a footnote:** it is a *user-facing manual* telling readers the
write shape is a flat `author` string, when the live, enforced shape is
`(:Agent)-[:PRODUCED]->(:KaizenEntry)`. Every agent on this team writes kaizen entries; a reader
following §2 would write the superseded shape. Scope is a **section rewrite, not a correction pass** —
it falsifies the label table, the relationships line, the Mermaid diagram, both "Try it" queries, a
gotchas bullet and two FAQ entries. Owner `tico`, with `docs/requirements/kaizen-agent-ontology.md`'s
status checked in the same pass, and an `analyst` gate on the factual claims.

Deliberately **not** folded into this coordination: my standing rule is to report pre-existing drift
rather than silently expand scope into it. Logged to `kaizen_team` by `tico` with the live evidence.

## S0 delivered — the ablation evidence is the deliverable

`docs/plans/salesperson-ui-graph.md`. The design puts safety in **two `MATCH` guards**, not in caller
discipline:

- **G1** `WHERE u.tokenHash IS NOT NULL` on the anchor — rejects a non-participant `User` as a reset root.
- **G2** `WHERE ch.participantId = u.userId` on the channel hop — a **provenance marker written only by
  `ensure_participant`**, deliberately *not* an id-equality check. `demo-general` carries no
  `participantId`, and `null = anything` is `null`, so it is **structurally unreachable** as a delete
  target regardless of membership edges or `User` properties.

**`graph-dba` proved N1 literally rather than asserting safety.** It seeded two adversarial fixtures —
a real participant who is *also* a genuine `MEMBER_OF demo-general`, and a non-participant `User`
carrying `channelId:'demo-general'` plus a `MEMBER_OF` edge — then ablated one guard at a time:

| Variant | Deleted | `demo-welcome` | A label-based survivor check reports |
|---|---|---|---|
| CONTROL (shipped) | 17 | **alive** | `Channel 4, Message 9, Thread 3` |
| G2 removed | 23 | **GONE** | `Channel 4, Message 6, Thread 2` |
| G1+G2 removed | 6 | **GONE** | **`Channel 4, Message 9, Thread 3`** |

The last row is N1 made concrete: **label counts byte-identical to the control while a thread, three
messages and two read-cursors are destroyed.** Every "assert survivors by label" check passes on that
destructive run — which is exactly why S4's positive non-participant survivor test exists.

**DDL verdict: no new indexes or constraints.** One additive nullable unindexed property
(`Channel.participantId`); a `UNIQUE` on it was considered and rejected with per-property reasoning.
`reset_participant` 4.5 ms, `reset_all` 236 ms at 50 participants × 40 messages, all nine
participant-scoped reads index-anchored. A measured 2.2× `reset_all` tuning lever was **deliberately
not shipped** — it would make correctness depend on the nullable, unindexed `Message.threadId`.

**Teco-verified:** `ws:acme` is untouched at 2 `Channel` / 2 `Thread` / 52 `Message` — identical to
its pre-S0 inventory. The unit stayed inside its probe graph.

**One finding handed forward to S4:** the presenter roster's bare `tokenHash IS NOT NULL` label-scans
`User`; an always-true `u.userId > ''` conjunct upgrades it to an index scan. The S0 gate is asked to
confirm that is sound rather than a fragile trick.

## S1 — code accepted; its "pre-existing defect" report was **correct**, and teco's rebuttal was wrong

**What S1 reported:** `ws:acme` already held a `salesperson@v6` snapshot before its run, diverging
from `reference`, so `./scripts/verify_salesperson.sh` with no argument (defaults to `acme`) exits 1.
It filed this as a **pre-existing defect** and declined to fix it. **That was right.**

**The two `v6`s are different definitions that collided on a version number** — established by the
S1+S2 impl gate (`docs/reviews/salesperson-ui-impl.md`, F-1) and independently re-verified by teco
against the live graph:

| | K-060 lever string | `language` |
|---|---|---|
| `ws:acme` `salesperson@v6` assistant step | **present** | absent |
| `proof_defs.py` v6 (S1's work) | absent | 7 occurrences |

The `ws:acme` copy is v5 plus one paragraph — the K-060 synthesis-time safety net that
`falkor-chat/docs/BACKLOG.md:67` records as **"Reverted, never shipped."** It was live-tested against
`ws:acme` from an uncommitted working tree, published to the graph, then reverted in the tree. It is
in **no file and no commit**.

**Why teco's rebuttal failed.** teco ran `git log -S '"v6"' -- .../proof_defs.py` (no commits) and
`git show HEAD:... | grep -c '"v6"'` (0), and concluded v6 could not predate the unit. Both commands
are true and both are irrelevant: they search **commit history** for an artifact that never entered
it. A graph snapshot published from a working-tree-only experiment is structurally invisible to
`git log -S`. The absence of evidence was read as evidence of absence, in the one place where the
search could not have found anything.

**The double-seed hypothesis is withdrawn.** `seed_salesperson.sh:215`'s `snap_pre` probe was
reporting accurately. S1's own scratch workspace `ws:s1v6` holds a v6 byte-identical to the file, so
the author verified cleanly against a clean surface; the `ws:acme` copy was never theirs.

**Impact is contained:** `MATCH (r:WorkflowRun) WHERE r.defKey='salesperson'` in `ws:acme` returns
**zero rows**, so nothing is bound to the orphan snapshot, and S11 pins the demo at its own
workspace. `reference@v6` and `ws:s1v6@v6` are both correct.

**Resolution: S1 bumps to `v7`** (gate option (a), the reviewer's recommendation) — string edits
only, no graph surgery, no approval-gated destructive op. `v6` is a **burned version number**: it
denotes the reverted K-060 experiment that exists only in `ws:acme`, and it must never be reused.
The v5→v7 gap is deliberate and must be documented where a reader would otherwise "fix" it.

**Lessons worth carrying.**

1. **Commit history cannot falsify a claim about live graph state.** This lab live-tests from
   uncommitted working trees as a matter of course, so the graph routinely holds artifacts that were
   never committed. To check what is in a graph, query the graph.
2. **teco told the gate not to re-examine this** — the brief said "the misattribution is already
   established; you don't need to re-litigate it." The gate re-litigated it anyway and caught the
   error. A coordinator's own conclusion handed to a reviewer as settled fact is the one input a
   reviewer has no independent reason to check; never mark teco's own reasoning as out of scope.

## Outstanding cleanup for the stakeholder (teco will not run destructive ops)

Three items, all benign, all needing a human hand:

```
docker exec falkordb-dev redis-cli GRAPH.DELETE probe_u8_rename_dst    # U8's spent rename probe (empty)
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:probe-s0-reset      # S0's spent reset probe (wiped, 0 nodes)
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:s1v6                # S1's scratch seed/verify workspace (holds the CORRECT v6 - F-1 evidence, keep until S1c closes)
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:s1v7                # S1b's scratch seed/verify workspace
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:probe-s0r3          # S0 v1.2's throwaway probe - teco-verified EMPTY (0 nodes); the note's own disposal list never accounted for it
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:probe-s4b           # S4b's FalkorDB-quirk isolation probe (2 :User nodes)
```

Plus the orphan divergent snapshot in `ws:acme` described above — deleting the `salesperson@v6`
snapshot there restores `verify_salesperson.sh` (no arg) to exit 0. Nothing is bound to it, but it is
a delete inside a populated workspace, so it is explicitly the stakeholder's call, not mine.

## S0 gate outcome — guards held, but **one row of S0's own evidence does not reproduce**

**Verdict: approve with suggestions.** The reviewer states plainly it **could not defeat either
guard**, and verified *why* rather than reasoning about it: `Repository.create_channel`
(`repository.py:184-198`) writes a fixed three-property map with no caller-controlled extras, and
**no query anywhere in the codebase deletes a `MEMBER_OF` edge** — so G2's provenance marker is
unforgeable by any shipped path. §2.3's load-bearing variant B reproduced **byte-identically on an
independent fixture**. The core design stands.

### Correction to the record — teco relayed a wrong evidence row upstream

**S0's §2.3 variant A ("G2 removed → 23 deleted, `demo-welcome` GONE") does not reproduce.** With G2
removed the query row-multiplies per matched channel and the re-mint `FOREACH` raises `unique
constraint violation on node of type Thread`, **writing nothing**. The reviewer reproduced the
published "23 deleted" only after *dropping* the `Thread.threadId` UNIQUE constraint — which S0's own
§7 proves was present on its probe. **Variant B — the row that actually demonstrates N1 — is
unaffected and reproduces perfectly.** `teco` had already relayed the full table to the stakeholder
and has corrected it there.

### F1 — the tip S0 handed forward to S4 is withdrawn, and teco verified why

S0 recommended adding an always-true `u.userId > ''` conjunct to upgrade a label scan to an index
scan. It is wrong twice. Verified live by `teco` on the engine:

```
42 > ''  ->  null        'abc' > ''  ->  true        null > ''  ->  null
```

A participant `User` whose `userId` is not a string therefore **survives `reset_all` completely while
the status row reports success** — a silent under-delete. And it buys nothing: measured label scan
**0.0036 ms** vs index scan **0.055 ms**, a **15× slowdown**. The recommendation is withdrawn and
will not reach S4.

### The other three majors

- **F2** — `scoped=false` is a *partial* reset, not a no-op: the commerce block is unguarded, so it
  deletes `Customer`/`Cart`/`CartItem`/`Order`/`OrderLine` while keeping thread/messages/runs/cursors,
  which `reset_all` then orphans permanently while reporting success. Not reachable today; a
  consistency defect, and exactly the shape that becomes a live bug later.
- **F3** — §7 named the wrong orphan classes and gave S7/S10 a done-condition that **cannot fail**.
  The one real orphan producer is `QUERIES.md` §9.3's member-anchored `advance_cursor`, minting a
  `ReadCursor` for a dead thread that neither reset can collect. One-clause fix executed by the
  reviewer.
- **F4** — the evidence-row correction above.

**Agreed and closed:** the DDL `NO` verdict — with the useful nuance that a `UNIQUE` on
`Channel.participantId` would in fact have been *safe* (FalkorDB exempts absent/null; re-join is
clean), so S0's rejection was a correct **scope** call rather than a safety one; the unshipped tuning
lever (it carries **two** nullable dependencies, not one); the `ReadCursor` label-scan trade; and §8's
profile numbers.

## S0b — the revision found two defects in its own v1.0 that the gate had not

Both are the kind that only surface when you stop trusting your own prior output:

1. **The published Cypher did not parse as printed.** v1.0 used `--` for its guard comments, and
   `--` is **not a comment on this build**. Verified independently by `teco`:
   `MATCH (u:User) -- x` → `Invalid input 'a': expected '>' or '('`. Since **S4 implements this note's
   Cypher verbatim**, a note whose queries cannot run is a defect of the first order — and it survived
   a full `analyst` gate, because a reviewer reproducing behaviour naturally adapts the query rather
   than pasting it. v1.1 now closes the loop mechanically: every ```cypher block is extracted from the
   finished note and executed (5/5 run; 4/5 byte-identical to the verified text).
2. **F3's first cut cost 3×** — leaving the `ReadCursor` stream open before the `users` `UNWIND`
   multiplied it, taking `reset_all` to 684-692 ms; collapsing `tcur` into its own `WITH` restored
   235-247 ms, matching v1.0. So F1's anchor change and F3's sweep are both free.

**It also disclosed more than the gate had caught on F4:** its published "23 deleted" came from *a
delete-only ablation it failed to disclose*. Both rows plus a correction note are now in the table.
And it strengthened the load-bearing half — **variant B needs no ablation at all**, running clean as
shipped because `u2` belongs to one channel so no row multiplication occurs. Pass 2 is asked to
confirm that, since it is now the sole demonstration of N1.

**One decline, argued:** it substituted a **structural** `(u)-[:HAS_CURSOR]->(own)` traversal for the
reviewer's `rc.memberId IN pids` (same coverage, no property dependency), and declined the complete
`t IS NULL` sweep because it would also collect a *non-participant's* dangling cursor — outside
§4.8's scoping rule. The `Agent`-owned residual is documented as bounded, quiesce-preventable and
read-path-harmless, with the complete form recorded for a future GC job.

**One trade-off escalated into the re-gate rather than accepted silently:** F2's fix means an
*unscoped* participant keeps a valid token — `graph-dba` names this a visible **FR-7 deviation**
(recoverable-and-loud over permanent-and-silent) and requires S10 to surface the counter. FR-7 says
the presenter's control clears **every** participant's state, so Pass 2 is asked to rule on whether
this degrades gracefully or breaches the requirement. **If it breaches, it goes to the stakeholder.**

**Teco-verified:** F1's withdrawal is real, not cosmetic — the 7 surviving `userId > ''` mentions are
all the documented rejection, including an explicit "**Do NOT add `u.userId > ''` anywhere**"
instruction. `ws:acme` re-confirmed intact after both S0 passes.

**Cleanup queue grows by one:** `ws:probe-s0r2` (emptied, synthetic only).

## S5 verification (teco, 2026-09-02) — accepted, verified by running it

- **Node `v24.20.0` / npm `11.19.0`**, installed per-user with SHA256 verification against
  `nodejs.org` (no sudo available on this box), pinned in `.node-version`/`.nvmrc`, reproducible via
  `salesperson/scripts/install_node.sh`. Confirmed live: `~/.local/node/current/bin/node` → `v24.20.0`.
- **Build output is correct**: `dist/index.html` references only `/shop/…`; I grepped for
  root-absolute references outside `/shop/` and found **none**. `dist/` and `node_modules/` confirmed
  gitignored via `git check-ignore`.
- **Scope clean**: nothing outside `salesperson/**` (the one other modified file, `cypher-mcp/README.md`,
  is U5's, not S5's).

### The R5 risk was real, and worse than the plan's wording — teco reproduced it

The plan said "node is not on `PATH`". The actual trap is nastier, and I confirmed both halves:

```
command -v npm   ->  /mnt/c/Program Files/nodejs/npm     (the WINDOWS shim)
command -v node  ->  (absent)
```

So a naive `command -v npm` probe **passes**, then installs Windows-native `esbuild`/`rollup`/
`lightningcss` that a Linux build cannot load — failing much later, deep inside the bundler, where
the cause is unrecognisable. `build.sh` names this exact case and refuses, and rejects any `node`
resolved under `/mnt/`.

**Verified by execution, not by reading:** running `build.sh` under
`env -i HOME=$HOME PATH="/usr/bin:/bin:/mnt/c/Program Files/nodejs"` — i.e. with the Windows npm
first on PATH and no native node — **exits 0** and builds correctly, because the resolution order
(`$NODE_BIN_DIR` → `$NODE_PREFIX/current/bin` → PATH) puts the pin ahead of the caller's shell.

### One nit, not worth a unit

`build.sh:31` reads `${NODE_PREFIX:-$HOME/.local/node}` with `set -u`, so under a **totally** empty
environment (`env -i` with no `HOME`) it dies with `HOME: unbound variable` instead of its own
diagnostic. `HOME` is set in every realistic invocation, so this is cosmetic — recorded, not
dispatched. A `${HOME:-/root}`-style default would close it if `build.sh` is ever called from a
systemd unit or a minimal CI shell.

### Handoffs S5 recorded for later steps

- Dependencies for the whole SPA track are **pre-installed** (TanStack Query v5, i18next +
  react-i18next, Tailwind, Vitest + Testing Library, Playwright), so no later step needs to touch
  `package.json` — which §5.0 makes S5-owned. `AGENTS.md` records adding a dependency as a sanctioned
  exception to that row.
- **No router chosen** — deliberately left to S12a as its design call.
- **S12a must delete `passWithNoTests: true` from `vite.config.ts`** when the first real test lands,
  or an empty suite silently stays green.
- **S12b**: `playwright.config.ts` has one `Pixel 7` project and no `webServer` (it drives a running
  server, `SALESPERSON_E2E_BASE_URL`). The `chromium-headless-shell` binary was verified to actually
  **launch** on this WSL2 box at 412×839 — which matters because `playwright install-deps` would need
  sudo that isn't available. Note `npm ci` does not install browsers; `npx playwright install chromium`
  is a documented one-time step.

## S0-gate Pass 2 — everything substantive approved; the blocker is that the **document** doesn't parse

All nine Pass 1 findings confirmed genuinely fixed, both self-caught defects confirmed correctly
repaired, and all three claims teco asked to be re-derived reproduce. One blocker.

### P1 — the closed loop proved the wrong property

**Teco-verified independently:**

```
opening fences: 6    bare closing fences: 4    GLUED closers: lines [347, 496]
```

Two closing fences are glued to the last code line, which is not a valid CommonMark fence. Parsed
with `markdown-it`: **4 blocks instead of 6, 7 tables instead of 11, one 455-line block** running
from §4's reset query to §8 — swallowing §5 (including `reset_all` itself), **§6's keep/delete
inventory** (the deliverable S0's own scope row mandates), §7 and most of §8.
`reset_all_participants` is **not extractable at all** under a conformant tool.

**The generalisable lesson**, and why the S0b loop missed it: an extract-and-execute loop over fenced
blocks proves *the queries run*; it does **not** prove the document delivers them. The extractor's own
tolerance for a malformed fence hid the defect. The fix is two assertions — **block count** and **max
block length** — since a 455-line "Cypher block" is self-evidently wrong. The reviewer found the
identical defect in **its own review document** and fixed it before shipping, so this is a general
trap for any agent publishing verified queries, not a one-off slip.

### F2's FR-7 trade-off — ruled graceful degradation, NOT a breach. Not escalated.

The reviewer settled it with a fact S0's note *has* (§1.1 line 76) but doesn't deploy where the
behaviour is defined: **the unscoped branch is unreachable on a healthy graph** (`ensure_participant`
is atomic, `create_channel` cannot set the marker, nothing deletes a `MEMBER_OF` edge). So
`unscopedCount` is always 0 in practice and both behaviours are dead branches; the only question is
which failure is better on an already-corrupt graph. **v1.0 satisfied FR-7 nominally while stranding
a transcript permanently and silently — an AC-2 leak reported as success. v1.1 misses FR-7 for one
participant and counts it.** AC-2 is the stronger requirement. Teco accepted this and did **not** take
it to the stakeholder; S0c must add a stated reversal trigger.

### Dispatch judgment (second deviation, same delegate)

`graph-dba` is at ~370k cumulative tokens, well past the ~250k threshold. Resumed anyway: the
remaining work is five precisely-specified edits to a 941-line note whose value is its **live-verified**
Cypher, so a fresh delegate would have to re-read ~1,600 lines and re-verify every query to take
ownership responsibly. The evidence also runs against the degradation worry — v1.1 *found two defects
in its own v1.0* that a full gate had missed. If S0c's output shows drift, that is the signal to switch.

## S0c — no drift, and the fix went past what was asked

**P1 teco-verified independently on the finished file:**

```
opening fences: 6   closing fences: 6   glued closers: NONE
blocks paired : 6   lengths=[31, 77, 60, 4, 9, 3]   max=77
```

Byte-for-byte the delegate's own figures. **It did not just add newlines.** The loop now asserts
block count, max block length, table count and "no line ends in a non-bare fence" *before* executing
anything — and it ran a **negative control**: re-gluing exactly those two fences produces
`blocks 4, max 513, tables 9, GATE: FAIL` on all three shape assertions. That proves the gate catches
the defect rather than assuming it, which is the difference between a fix and a fix you can trust.
The generalisation is written into §12 so S4 inherits it, and captured as a `kaizen_team` entry
flagged `suggestedHome: prompt` — the delegate correctly judged this belongs in its standing practice,
not just this document.

**P2 was narrowed rather than merely documented**, on a real design insight the reviewer's own fix had
missed: the two resets have **different node lifecycles**. `reset_participant` leaves the `User`
alive, so a cursor on a *surviving* thread is live read-state for a membership that still exists;
`reset_all` deletes the `User`, so the wide sweep is *required* or the remnant is unowned. Treating
them alike was the actual defect. Verified with before/after numbers on the same fixture
(`cursorCount: 3` vs v1.1's `4`), with F3's orphan class still collected and no unowned cursor left
after `reset_all`.

**P3 became a real contract** — a five-row table S4/S7/S10 must implement (`scoped=false` → `409`,
never `200`; `unscopedCount > 0` → `200` with `incomplete: true` and `unresolved: unscopedIds`;
`Thread` constraint violation → `5xx`, **do not retry**) — and `reset_all` now returns `unscopedIds`
so `unresolved` is populatable without a second query. Prose in a design note does not survive into
an implementation; a contract does.

**It found and fixed a live bug in its own first cut of P2**: `OPTIONAL MATCH (liveT:Thread
{threadId: own.threadId})` raises `_AR_EXP_UpdateEntityIdx: No record was given to locate a value
with alias own` when the participant holds **no** cursors — caught by the regression's second-reset
case, *not* by its own P2 test. Logged as §11 row 35.

**The resume-vs-fresh judgment held.** Third consecutive pass from this delegate at 431k cumulative
tokens, and no degradation: it narrowed a design rather than papering over it, built a negative
control unprompted, and self-reported a bug its own new test had missed.

**Pass 3 is deliberately scoped to P2 and P3 only** — P1 is teco-verified, and the reviewer is told
not to re-spend on it. A narrowing's risk is **under-delete**, the F2/F3 failure mode from the other
side, so that is what Pass 3 is pointed at.

**Cleanup queue grows to five:** `ws:probe-s0r3` (emptied, synthetic only).

## S0 CLOSED — Pass 3 verdict: **approve**. S4 is unblocked (on the DB, not on design).

Three passes, one design note, and the gate earned every one: Pass 1 found 4 majors, Pass 2 found a
blocker that made the document undeliverable, Pass 3 confirmed the behavioural narrowing was safe.

**The reviewer built a *harder* fixture than the note's** — `p-ccc` holding five cursors instead of
three — specifically to hunt for what the narrowing might now miss. Results:

- **F3's orphan class is still collected**, and a bonus nobody claimed: `liveT IS NULL` also catches a
  cursor whose `threadId` is unset. The note's own three-cursor numbers reproduce **byte-for-byte**
  (`cursorCount: 3`, `deletedCount: 18`).
- **Nothing unowned after `reset_all`, and the real result is *stronger* than the note claims** — the
  dangling check comes back empty too, and the `Agent`-owned residual §7 documents doesn't arise
  here at all (the Agent's cursors for just-deleted threads are caught by the thread-scoped half; the
  residual only exists when the thread died in an *earlier* reset). **§7 is pessimistic, not wrong** —
  a one-clause narrowing, filed below rather than reopening an approved note.
- **The surviving `demo-welcome` cursor is live read-state, not leaked state.** It is owned by the
  participant's own `User` and names a thread §4.8 *mandates* survives. The one shape that would carry
  cross-participant state — a cursor on another participant's live thread — is **unreachable** (§4.3
  resolves thread ids server-side from the token) and **self-heals**: once that thread dies the cursor
  goes dangling and the `liveT IS NULL` branch collects it on the next reset.
- **The zero-cursor raise is fixed four ways** (no cursors at all; second reset with cursors already
  gone; `scoped=false` with and without cursors).
- **P3 is implementable with no gap** — every field the five contract rows key on is present and
  correctly typed, and the constraint violation propagates as an *exception* rather than a status row,
  which is what makes "propagate as 5xx, don't retry" the only thing a caller can do.

## The M-1 blocker — caused by a teco instruction, not by the architect

**What shipped into the plan:** F8 ("a client-side timeout on a reset means *unknown*, not nothing
changed") was routed to **S12a**, the SPA transport step.

**Why that is wrong.** `FALKORDB_SOCKET_TIMEOUT` is the **server's** Redis socket timeout to
FalkorDB — teco-verified: `falkorchat/config.py:29` feeds `db.py:44`'s
`FalkorDB(socket_timeout=…)`. The graph note assigns the rule to **S7/S10**, and the delivered
`QUERIES.md` §18.7 carries it correctly. Only the plan got it wrong.

**Consequence had it shipped:** no server step carries the rule, and S12a's rule can never fire —
the browser receives a clean `503` meaning "nothing changed" while the delete has committed. A
participant is told their data survived when it did not.

**Cause: teco's brief.** It read: *"Route it wherever it actually belongs — the SPA step that owns
reset UX (S12-something or S13), not S8."* That asserted the client side as a **premise**, in a
brief whose stated purpose was absorbing the note faithfully. The architect followed it. A
delegate has no standing to doubt a coordinator's factual premise, and an isolated-context
delegate has no cheap way to check one.

**The rule this yields:** when routing a mandate to an owner, **state the mandate and ask where it
belongs** — do not supply the answer as background fact. teco's routing guesses are the least
reviewed input in the whole pipeline: no gate reads the briefs, and the delegate treats them as
given. Where teco does have a view, mark it as a steer to be overridden, not as a premise. The M-3
brief was written that way deliberately ("**My steer, and it is a steer, not an instruction**").

**This is the second finding of its class this coordination.** The first: teco told the S1/S2 gate
that its own misattribution conclusion was settled and not to re-litigate it — and it was wrong.
Both are teco's reasoning entering an artifact through a channel nothing reviews.

## CORRECTION — the S1b "phantom concurrent writer" was real, and teco was wrong

**What teco recorded earlier:** that the S1b `coder`'s report of a concurrent rewrite of
`falkor-chat/AGENTS.md` was a confabulated collision — the delegate re-reading its own edit and
failing to recognise it — and that its deference to the imagined other agent was the real risk.

**That was wrong. There was a second writer.** `git log -- falkor-chat/AGENTS.md` shows
**`ef02c7a` "docs: context-file convention + repo-wide AGENTS.md bloat sweep"**
(2026-09-02T19:10:15), an ancestor of `HEAD`, authored by **no unit of this coordination** — the
separate Claude Code session that has been committing to this repo throughout. Its diffstat for
that file (26 changed lines) matches S1b's uncommitted work exactly: **that session committed this
coordination's in-progress file along with its own sweep.**

**Why teco got it wrong.** teco checked only which of *its own* delegates were in flight, found the
`architect` fenced off that file and reporting one file touched, and concluded nobody could have
written it. *"None of my agents did it"* is not *"nobody did it"* — and teco had **already
discovered the concurrent session earlier in the same coordination**, then failed to apply that
knowledge to the next diagnosis that needed it.

**Content verified intact after the sweep**, so nothing was lost: `salesperson@v7` ×2, the burned-`v6`
note ×2, F-8's full drift-check clause, and row 73's DDL-only safety fact are all present in both
`HEAD` and the working tree. The sweep compacted prose without dropping substance.

**The delegate was right and its instinct was sound.** teco's earlier note framed its deference to
the "phantom" as the lesson; the actual lesson is that teco dismissed an accurate field report
because teco's own model of who could be writing was incomplete. A `kaizen_team` correction entry
is filed (`5d8a1c34-…`).

## Follow-up 13 — a real gap in the doc convention, for the doc-standard owner (`cobb`/human)

Root `AGENTS.md`'s header block defines an optional **`Reviews:`** field but never says what it
ranges over: *reviews **of this document*** (here, only `docs/reviews/salesperson-ui.md`) or
*reviews **in this family*** (also `docs/reviews/salesperson-ui-impl.md`, which reviews the
implementation but amended the plan twice — F-4's file map, F-6's S8 clause). Both readings are
defensible from the text.

Decided **for this coordination only**: list both, because a reader following only the first
citation cannot see why the plan says what it says. That is a local call on one document, **not a
convention change** — the convention itself is genuinely ambiguous and should be settled by whoever
owns the doc standard.

## Follow-up 14 — the unpinned-workspace trap is a repo-wide class, not a plan defect

U13b's class sweep of `docs/plans/salesperson-ui.md` found **three** done-conditions that invoked a
`*.sh` script with no workspace argument, where the default resolves to `ws:{FALKORCHAT_WS_ID}` →
`acme`:

| Step | Shape | Why it mattered |
|---|---|---|
| S1 | `seed_salesperson.sh <ws>` unpinned | **Destructive-ish** — how a working-tree def reached `ws:acme` and burned `v6` (F-1) |
| S4 | `verify_salesperson.sh` no arg, post-`reset_all` | **False evidence** — asserts against `ws:acme`, not the graph the test just reset; passes green proving nothing |
| S11 | verify scripts exempted from the row's own "every seed script gets the workspace explicitly" rule | The rule is what a `devops` implementer copies |

All three are fixed in plan v1.4. **The class is not plan-specific** — any done-condition, script
docstring or runbook line in this repo that invokes these scripts bare has the same shape, and the
read-only ones are the dangerous kind precisely because they cannot corrupt anything and so never
announce themselves. Worth a repo-wide sweep outside this coordination.

Deliberately left unpinned, as a decision rather than an oversight: plan §6.1's *"Re-run the seed
sequence after any default pytest run"* — that is guidance to a human restoring their own dev
workspace, and pinning it would make it wrong for its purpose.

## Follow-up 12 — one clause, not worth reopening an approved note

§7's `Agent`-owned orphan residual is described more broadly than it is. Pass 3 established it only
arises when the thread died in an *earlier* reset. Fold into the next natural touch of this document
rather than a dedicated unit. Owner: `graph-dba`.

## Follow-up 16 — the canonical query gate cannot detect the drift it exists to prevent

`falkor-chat/AGENTS.md` names `./scripts/test_queries.sh` as *the* query gate, and its 408/408 is
cited across this coordination as evidence a query change is safe. **It re-types each canonical query
as a shell constant rather than executing `repository.py`** — so it verifies the *transcription*, and
a code change that diverges from the constant passes it green.

This is not hypothetical: **it has already fired once, silently.** K-053 added `p.productId` to
`Repository.lookup_product` and did not update `QUERIES.md` §15.1, which still documented three
columns until S7c2. The gate reported green throughout. S7c hit the identical shape at §15.2 and
`test_queries.sh:1387`, which is how it was noticed at all.

**The precise defect, corrected by the implementer after teco overstated it.** The script is *not*
blind in general: `assert_no_data_row` compares the full returned header, and mutating either of
S7c2's two edits without the other fails 407/408 in both directions — so it does self-check its own
`RETURN` list. It is blind **across the code boundary specifically**. The accurate statement is:
*the script verifies that its transcription is internally consistent and runs on the live engine;
nothing verifies that the transcription is the query the code sends.* That is narrower, and
actionable in a way "the gate is decorative" is not.

**Fixed:** §15.2's two constants, §15.1's body (S7c2), and `$LOOKUP`'s two coupled constants (S7c3 —
the other half of the same K-053 instance; see the scope note below). **Not fixed:** the property
itself. A gate whose passing is independent of the code it gates will drift again, and the two
candidate answers — generate the constants from the code, or execute the repository methods — are a
design question for `graph-dba` and `architect`, not a patch. Outside this coordination.

**The audit is bounded, and that is the useful part.** The implementer built a ~30-line AST
comparator (extracting each `ro_query` literal, `literal_eval`-ing the concatenation, normalizing
whitespace, comparing token by token) and ran it across the whole document: **109 fenced `cypher`
blocks, 66 matching a `repository.py` literal exactly, 43 not — 3 of those DDL.** That 43 is a **lead
count, not a defect count**: most will be legitimate (services-level composition, illustrative
shapes, multi-statement examples). Triage is the audit. The same comparator is also the shape of a
real code-vs-doc gate — the thing `test_queries.sh` structurally cannot be — so the open design
question is its **false-positive discipline** (allowlist? fence marker? naming convention?), because
a gate that cries wolf on forty legitimate blocks is abandoned in a week, which is precisely how this
one came to be trusted for something it does not do.

**The design answer, worked out and recorded here so the follow-up starts from it rather than from
scratch.** The implementer's judgment, and teco's, is that the mechanism is a **marker in the fence**
— ` ```cypher verbatim=Repository.filter_products ` — with three rules:

1. every **marked** block is compared against that symbol's literal and fails on mismatch;
2. every **marked** block fails if the symbol does not exist (this is what catches a **rename** — the
   failure a doc gate most needs and the one a naming convention silently misses);
3. **unmarked** blocks are *counted and reported, never failed*.

Rule 3 is the false-positive discipline, and it is the whole design. A legitimately illustrative
block costs nothing and never trains anyone to ignore output, while the unmarked count is a visible,
monotonically-improving figure (**43 → 0**) that makes the audit incremental work anyone can pick up
rather than one large triage nobody schedules. The claim lives *inside* the block it governs, one
line above the text — impossible to change the query without seeing it.

Both alternatives were considered and rejected with reasons worth keeping: an **allowlist** puts the
claim in a fourth artifact that can itself drift and **fails open** (a new section with no entry is
silently unchecked, so coverage erodes invisibly — the same defect class being fixed); a **naming
convention** is nearly free and already roughly honoured, but it is *inference rather than a claim*,
and it breaks silently on a section documenting two methods, on a query held in a module constant,
and on anything composed in `services.py`.

**The caveat that must ship with it, so the gate is not oversold the way 408/408 was:** this can only
ever check **verbatim transcriptions**. Much of `QUERIES.md`'s value is the prose *around* the blocks
— §15.2's `GRAPH.PROFILE` deviation note is the clearest case — and nothing mechanical will notice
when that reasoning goes stale.

**The pairing rule is the part that does not generalize**, and knowing that is what makes the sizing
number honest: the sweep that produced "43" asked only *is this block's text in the set of all query
literals?* — set membership, no pairing — so it can say a block has no counterpart but not which
method it was meant to transcribe, and therefore cannot separate real drift from a block that never
claimed to be a transcription. The marker is precisely the missing pairing, written down.

Discovered by the S7c implementer, which also filed it to `kaizen_team` (`entryId 16ab10b7-…`)
because the property is durable and written down nowhere. The comparator itself lived only in `/tmp`;
teco asked for it **in prose rather than as a file**, and then rebuilt it from that description and
reproduced both `MATCH` results — so the description, not the script, is the durable artifact.

## Two scope calls, opposite directions, same reasoning (teco, 2026-09-03)

**Widened:** `QUERIES.md` §15.1's pre-existing drift, normally a report-don't-chase follow-up, was
folded into S7c2 — because it sits **one section above** the §15.2 S7c just corrected, describing the
same defect. A document left half-right, with the wrong half adjacent to the right half, is worse
than either fixing or not fixing both.

**Held:** any *other* `QUERIES.md` drift of the same class is to be **reported and left**. A
whole-document audit may well be worth doing; it is not worth doing inside a step whose neighbour
(S8) is about to be gated on one clean subject.

The line between them is not size — it is whether leaving it creates a *new* inconsistency in
something this coordination just touched.

**And teco then drew that line in the wrong place, and the implementer pushed rather than complied.**
Capping S7c2 at two edits left `$LOOKUP` in `test_queries.sh` still three-column — so after `8aaeca3`,
`repository.py` and `QUERIES.md` agreed about `lookup_product` while the script alone dissented:
*exactly* the half-right state the widening was authorized to prevent, one file over. The
report-don't-chase rule was aimed at **other** drift; applying it to the other half of the instance in
hand was a misapplication. Fixed as **S7c3**.

Worth keeping as the general form: **the unit of "one instance of drift" is the fact, not the file.**
A rule that stops at a file boundary will keep splitting instances in half.

## S7b2's gate is folded into S7c's, not skipped (teco, 2026-09-03)

S7b2 is test-only, 11 executable lines, closing an already-gated finding, with mutation evidence in
both directions and a suite count matching baseline exactly. On the usual test it is the justified
skip. **But this coordination's record argues the other way** — three consecutive gates each found a
real flaw in the fix below them, including one *inside* the fix that a gate had just prescribed.

So it is **gated, but not separately**: S7c edits `test_storefront.py` too, so its reviewer reads
that file regardless. Folding the check in costs one paragraph of brief instead of a ~110k-token
dispatch. What the S7c gate must cover on S7b2's behalf: the `started_at` relocation, and the
**third override in a row** — the rejected `expect_error` kwarg in favour of the literal
`pytest.raises` idiom, which changed two call sites.

**The general rule this instance is an example of:** a gate can be *merged into the next one* when a
later unit already opens the same file — that is a real saving. It is not an excuse to skip; the
saving is the dispatch, never the review.

## The strongest result in the build so far — a suggested fix that provably does not work

Worth stating on its own, because it changes how much weight a review's *suggested fix* should carry
relative to its *finding*.

Pass 7 found that a broken quiesce deadline **hangs** instead of failing, and suggested an
elapsed-time assertion. S7b refused it and argued structurally: a call that never returns is never
followed by its assertion, so an elapsed form converts *slow-but-returning* into a failure and cannot
touch a hang. Pass 8 then **ran Pass 7's suggestion against Pass 7's own mutant** and watched it hang
— terminated at 32 s, exit 143 — while the substitute failed in 3.34 s with both test names printed.

Twice now in this coordination an implementer has improved on its reviewer's suggested fix, and this
time the suggestion was not merely weaker but **structurally incapable** of catching the defect that
motivated it. **The finding was right and the fix was wrong**, and only execution could tell them
apart — which is Pass 8's stopping-rule argument arriving from the other direction.

The corollary teco is carrying forward: brief a reviewer's suggested fix to an implementer as *a
candidate to beat*, never as the deliverable. Both units that did so produced something better.

**And the symmetry is worth keeping.** Pass 8's own finding (S8-1) is that `_call_bounded` stamps its
start instant on the *calling* thread — so the fix justified by rejecting a thin margin quietly
contains the same thread-start skew, only unmeasured. Nobody in this chain has been right by default;
each has been right where the next one actually looked.

## Step ids and unit ids are one namespace — the `S7b` collision (teco, 2026-09-03)

**A plan step id and a coordination unit id collide the moment either reaches a commit message**, and
the `<step><letter>` carry-forward convention makes it likely rather than freak: the review document
uses `S1b`, `S4b`, `S6b` for a fix against a step's own surface, so a *fix unit* consumes exactly the
id a later plan revision would want for a *new step* in the same neighbourhood.

That is what happened. While the architect was writing v1.19, `d9d2f2b` shipped with a body opening
*"salesperson-ui S7b, closing Pass 7's three minors"*; two further commits reference it, this ledger
carries two rows under it, and an `analyst` was mid-run writing a review section about it. The
architect chose `S7b` for the new split-out step by sound reasoning from the same convention — and
**could not have seen the clash**, because its brief scoped it to the plan file alone.

**Commit messages are history and are not rewritten**, so the plan renames to `S7c`. The check that
would have caught it costs nothing and is now the rule: **before accepting a new step id, `grep` the
git log and this ledger, not just the plan.** The plan is the one place the id is *not* yet in use.

## The Ruling 1 split (teco, 2026-09-03) — one gate should judge one thing

The architect scoped Ruling 1's fix into S8 as instructed, then **declined to restructure the plan
around it** and handed the decision back, correctly: creating a step is a coordination call.

**Taken: split it into its own step, sequenced ahead of S8.** The argument is not "S8 is big" — it is
that **Pass 8's stopping rule names S8's gate as the specific place where review resumes**, because
that gate proves the `{handlers} × {routes}` assertion actually fails when a handler has no row.
That gate is the payoff for closing eight plan passes without a Pass 9. Handing it an unrelated
catalog refactor — five delivered artifacts, two of them previously gated — dilutes precisely the
review that trade bought.

Scope of the new step: `repository.filter_products`'s projection + row mapping, `_catalog_rows`'s
simplification (dropping the second read **and** the `if product is None: continue` silent-drop
branch), `QUERIES.md` §15.2's `RETURN` line, and one tripwire test, with a done-condition that fails
if the projection lands without the simplification, so the two cannot drift apart.

## Two framings the architect corrected, both better than the adjudication

**1. `storefront_dir`.** Teco wrote *"or every `imageUrl` is `null`"*. That holds **only when
`FALKORCHAT_STOREFRONT_DIR` is unset** — and S11 sets it, so in the real demo deployment a
`create_app` that forgets to forward still works by fallback. The failure that actually bites is the
**mismatch**: `/shop` serving tree A while the manifest is built from tree B, which yields *wrong*
URLs rather than null ones and is **invisible to the obvious test** (one tmp dir, config unset). The
done-condition is now written against a second, also-populated directory.

**2. Ruling 1's cost.** Teco called it "the one with a real trade-off", meaning technical risk. The
architect relocated the cost: the technical call is right, but "reach into a delivered file" is five
delivered artifacts landing on the largest step in the plan. That reframing is what produced the
split above.

**The pattern across both:** teco stated a failure mode in the form that made the point vividly, and
in each case the vivid form was the *less likely* failure. An implementer or reviewer working from
the vivid form writes the test that catches it — and misses the real one.

## Ruling 1 was dissolved, not weighed — and that is the reusable lesson

Teco asked the gate to weigh a trade-off: take a one-line fix inside a **delivered** file and accept
that `tools.FilterProductsTool` would start feeding product slugs into the salesperson agent's LLM
context, or keep S7's correct `1+n` workaround.

**The reviewer refused the framing and checked the premise instead.** `services.lookup_product` has
projected `productId` since **K-053**, and `LookupProductFactTool.run` returns `{"found": True,
**row}` — so the agent's context **already contains product slugs today**, from the sibling catalog
tool. The fix does not introduce a new exposure; it makes two sibling tools consistent. Teco verified
both facts independently (`repository.py:2681`, `tools.py:428`).

The trade-off teco spent a ruling on **did not exist**. The general form, worth carrying: *before
weighing a cost, check whether the system already pays it.* A cost that is already being paid is not
a cost of the change.

**What keeps this honest rather than merely clever** is that the reviewer then argued *against* its
own conclusion, on the record: applying the fix live gave 2473 passed with zero test edits, but zero
test breakage measures **code, not model behaviour** — and the 14 deselected `live` tests are AC-5
grounding, querygen NLQ and triage, **none of them a salesperson catalog conversation**. So no
harness observes this either way, and the evidence for "safe" is the K-053 precedent, not a passing
test. That distinction is now in the plan, not just in the review.

## Held: one consolidated plan touch (v1.18) — now dispatched

Four corrections to `docs/plans/salesperson-ui.md` are known or pending. They are being **batched
into one architect edit**, not dispatched as they arrive — the same discipline Pass 8 prescribed for
the review findings, and for the same reason: a plan edited four times is four chances to move a
step row that a dispatched agent is building against.

All four are now adjudicated, so v1.18 lands them in one edit:

1. **§4.8 gets a footnote, not a correction** (Ruling 2). The post-reset profile re-write `MERGE`s a
   name-only `Customer` back — but §4.8's column is *Deletes*, and the delete does delete. The gate
   also corrected teco's reading of which assertion is load-bearing: it is **not** the `PLACED`/`Cart`
   emptiness but `profile == {"name": "Ada", "deliveryAddress": None}` — the `None` address is what
   proves the name is a **re-write and not a survivor**, which is exactly what the inventory stood for.
2. **S8 must pass `storefront_dir` from `create_app(storefront_dir=…)`** into the `Storefront`
   constructor, or every `imageUrl` is `null` — an S7-introduced wiring obligation that exists in no
   plan row, and precisely the §4.7 failure where AC-11 passes with everything null.
3. **Cancellation is an S9 obligation, in front of S7's wait, never in place of it** (Ruling 3) —
   plus the half S7 did not state: `quiesce_s` is 30 s against a 180 s agent timeout, so a slow turn
   turns reset-mine into a `503` where cancellation would have succeeded. That is *why* §4.8 wanted
   cancellation, and it is what gets forgotten at S9.
4. **S8 takes the `productId` fix** (Ruling 1 — see the section above for why the objection
   dissolved) and drops `_catalog_rows`'s second read with it, deleting the `if product is None:
   continue` silent-drop branch.

**Note that plan-gate closure does not mean plan-edit closure.** What Pass 8 stopped was commissioning
another *review pass* per revision; a factual correction proved by an implementation still lands, and
is verified by `diff` plus the step-row hashes rather than by a Pass 9.

## A `HISTORY.md` entry body is not corrected when a later step supersedes it (teco, 2026-09-02)

S6c offered a one-line fix: the S6 entry body still says the constant-time property is "pinned by an
explicitly static source assertion", which S6b reshaped into a `compare_digest` spy. **Declined, and
the agent's instinct to ask rather than quietly rewrite was right.**

`HISTORY.md` is a **dated log read by lookup**, not a living document read whole — so an entry
records what was true *on its date*, and the close-out beneath it records the supersession. Editing
the body would erase the fact that the tripwire shipped in a weaker form and was reshaped after
review, which is the single most useful thing that entry now carries. The same reasoning is why
root `AGENTS.md` puts `HISTORY.md` in the may-grow-without-bound class and exempts it from the
rewrite-don't-append rule that governs `BACKLOG.md` and the context files.

The test for the next person facing this: **is the document read whole, or by lookup?** Read whole
⇒ rewrite in place. Read by lookup ⇒ append the correction and leave the record standing.

## Open decision, teco's, to be taken at S7's close — does `Storefront.lookup` survive?

`lookup(participant_id)` is the read-through cache's only reader, and **plan v1.17 gives it no
caller**: S9's `enqueue_turn(ctx, participant, posted)` receives the record from the authenticated
route, and S7's `get_state(ctx)` / `reset_participant` take a `ctx`. If S7 and S9 both turn out not
to need it, **deleting it removes the confusable surface entirely** — `lookup` and `resolve_token`
return an identical `ParticipantRecord`, so a call site cannot distinguish an unauthenticated read
from an authenticated one — which beats detecting that confusion with the source tripwire S8
currently carries.

**Why it is not decided yet, and not decidable by argument:** deletion contradicts S6-1's premise and
would retire the `_cache_put` refresh S6b has just pinned. The step that answers it is S7, so S7's
brief carries the question as an explicit deliverable: *did you need it?* — evidence from having
written the code, not a prediction.

**S7's answer, 2026-09-02: no — and the reasoning is better than the grep.** `grep '\.lookup('` over
the package returns only the definition. But the interesting part is `reset_participant`, the one
S7 method that *does* need `displayName` and `language`: `lookup` is the **wrong source** for them,
because its cached `thread_id` is stale the instant the reset returns, so using it would require a
`forget` first — a plain graph read with extra steps. It takes the authenticated `ParticipantRecord`
instead, which S8 has just re-read via `resolve_token` on that same request, and whose two needed
fields the reset does not touch. So **S7 writes *through* the cache and reads it never** (the
post-reset `_cache_put` is pinned by mutation M6).

**The decision therefore moves to S9's close, not S7's.** S9's `enqueue_turn(ctx, participant, posted)`
receives the record from the route, so if it also needs nothing, `lookup` ends the build with no
production caller at all — and S8's source tripwire would be guarding a method nothing calls. The
S7 gate has been asked to verify the grep and the staleness argument before that decision is taken.

**Why no type-level fix was taken instead** (S6b's argued "no", verified rather than asserted):
nothing type-checks this repo — no mypy or pyright config anywhere, no pre-commit hook, ruff selects
only `E,F,W,I`, and `falkor-chat/docs/SERVER.md` §1.7 already records that ruff is not a wired gate.
So a `NewType`, a subclass or a `Protocol` would be enforced by nothing at edit, commit or run time.
And a genuinely separate `ParticipantScope` dataclass would be *structurally identical*, so it passes
anywhere the original is expected: it buys a reviewer a name to notice, not an impossibility. Every
candidate is detection, not prevention — which is the whole argument for deleting the surface if it
proves unused.

## R12 — a product-visible residual the architect accepted rather than engineered away

Flagged here because it is the one v1.17 decision with a **user-visible** consequence, and the
stakeholder may overrule it cheaply. Teco did not escalate it as a blocker: the risk is Low, the
decision is documented with a reversal trigger, and reversing it *later* costs one new step.

**Join is not idempotent.** A FalkorDB socket timeout (default 10 s) during `POST /shop/api/session`
can commit the write while the token never reaches the browser — leaving a `User` with a `tokenHash`
nobody holds, owning a `Channel` and `Thread`, in the presenter roster, while the person re-joins as
a second identity.

**Rejected alternative:** a client-supplied idempotency nonce, which §5.2's invariant does permit.
It was rejected because it reopens **delivered** S6 — a new `join()` parameter, a uniqueness
constraint and an S0 amendment — to close a window that requires a socket timeout on the single
write a participant makes before holding any state. The client reports *"your join may not have
completed — join again"*, the presenter is warned a stale roster row may appear, S12d renders it as a
participant who never speaks, and `reset-all` sweeps it, since it is an ordinary participant `User`.

**Reversal trigger (in the plan, `R12`):** join acquiring a side effect beyond the roster — payment,
external provisioning, a quota — or use outside a controlled demo. Then the nonce lands as its own
step.

## Follow-up 15 — `SERVER.md` §1.5's layout block is five milestones stale (NOT S8's debt)

The block is headed *"Layout (as built, M1)"* and lists **8 modules against the package's 27** — it
omits nineteen modules across M2–M6. S6 proposed hanging the refresh on **S8**, since S8 adds
`storefront_api.py`; teco relayed that to the architect, and the Pass 6 reviewer **overruled both of
us with the better argument**: routing it to S8 makes S8 the owner of five milestones of debt it did
not create, inside the largest remaining step. Teco reversed the instruction mid-run.

Standalone item, outside this coordination. Owner: whoever next touches `falkor-chat/docs/SERVER.md`
substantively. **Not** to be folded into any `salesperson-ui` step.

## Why S7 dispatches fresh while S6b resumed the same agent

Both decisions come from the same rule and land on opposite sides of it, which is worth recording
because the rule is easy to apply mechanically and get wrong.

**S6b resumed** `a5db169a0966bad59` (the S6 author, ~202k tok / 66 tools): the findings are *its own
code*, in the two files it just wrote, and two of the three turn on reasoning it never wrote down —
why three docstrings say the cache is never read, and what the tripwire was meant to catch. A cold
agent would re-derive that at a cost exceeding what the resume spends.

**S7 dispatches fresh**: it is a *new step* against a module that is now committed, reviewed and
documented, specified by its own §5.1 row. That is self-contained by construction — the definition
of work that does not need the incumbent's undocumented reasoning — and the incumbent will be past
250k tokens once S6b closes, where continuing trades tokens and hallucination risk for no benefit a
good brief cannot supply.

## The plan gates are stopped at Pass 8 — the stopping rule, and who set it

Eight review passes on `docs/plans/salesperson-ui.md` end here. The rule was set by the **reviewer**,
not by teco's patience, and teco asked for it in those words: *"if you believe further plan passes
have negative expected value, say so — I would rather stop on your recommendation than on my
patience."*

Its answer, and the reason it is more than an opinion: passes 5-8 each returned roughly one major
plus a short tail in one surface, and by Pass 8 **the marginal instance was being produced by the
fixes rather than found in the original** — P8-1, P8-2 and P8-3 are all mis-ruled instances created
by the v1.16 delta that was supposed to close the class. That pattern converges slowly under review
and quickly under execution. Both halves of the class now have an owner:

- **Unruled responses** (a response with no client rule) — closed *structurally*, by S8's total-by-type
  error map bounding the producible set and C13 making any survivor loud in the demo. Neither depends
  on anyone having enumerated correctly, which is what four table re-keys failed to achieve.
- **Mis-ruled responses** (a rule that matches and is wrong) — open, and carried by two mechanisms
  rather than by more review: each rule stating its own discriminator, and **S12a's per-rule tests
  enumerating the routes each rule spans**. That last clause is Pass 8's highest-value line: C4's test
  then names all six writing routes, and P8-2 fails at implementation time, mechanically, with no
  reviewer in the loop.

So the coordination resumes at the two **implementation** gates, where the evidence is runnable:
S8's `{handlers} × {routes}` assertion (checking that adding a handler with no row actually fails it)
and S12a's per-rule tests. Plan revisions after v1.17 are verified by `diff` plus the step-row hashes,
not by commissioning a Pass 9.

## Dispatch state — the critical path is still the live database, one step at a time

S0-S6 are closed and committed (`2f7938d` is S6). The constraint has not changed and will not: every
implementation step's done-condition is integration tests on `ws:test`, and the suite wipes both
`ws:test` and the global `reference` graph, so **two agents cannot run it at once** — a second run
produces mutually-corroborated spurious failures, which this coordination has already seen once.

That serialization now costs more than it did, because the **review** gates want to run mutations
too, and Pass 8's whole argument is that runnable evidence is where the remaining value is. So the
gate and the next implementation step alternate rather than overlap: the S6 gate holds the database
while S7 waits, then S7 holds it. S8-S10 and S11/S12a sit behind that same chain. One at a time by
construction, not by choice.

The only work genuinely parallel to it is **document** work — the v1.17 plan touch is running
concurrently with the S6 gate precisely because it touches no database and no source file.
