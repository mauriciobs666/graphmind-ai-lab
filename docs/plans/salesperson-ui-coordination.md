# The one salesperson UI — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M<n> TBD)

## Goal

Deliver the business-facing salesperson UI specified in `docs/requirements/salesperson-ui.md`
(FR-1…FR-11, AC-1…AC-11): a modern, mobile-usable, multi-participant chat product surface built
against the workflow-engine-backed `salesperson` agent (falkor-chat M6; `v5` today, bumped to `v6`
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
| **S1** — `salesperson@v6` def bump + `AGENTS.md` rows 82-83 | `coder` | `aef74d44be60f1cff` | delivered (code accepted; **one self-inflicted side effect, see below**) | `proof_defs.py`, 2 scripts, scaffold test, `falkor-chat/AGENTS.md` | `analyst` → — | 130k tok / 42 tools |
| **S2** — chat-path `run_ctx` merge | `tdd-engineer` | — | in-flight | `services.py`, `trigger.py`, 2 test files | `analyst` → — | — |
| S2 · S3 — `run_ctx` merge, responder kill switch | `tdd-engineer` | — | queued (**serialized behind S1 — shared live DB**) | — | — → — | — |
| S4…S16 — remaining implementation | per plan v1.2 §5.1 | — | queued | — | — → — | — |

## Stakeholder decisions, 2026-09-02 (plan §8)

| OQ | Decision | Effect on the plan |
|---|---|---|
| **OQ-1** AC-3 acceptance basis | **Stub-LLM pass + a *published* live latency curve + a staggered demo script.** A live pass/fail concurrency threshold is explicitly **not** the bar. | As the plan proposed (§6.4 stands). R1 is accepted as a stated residual, not engineered away. |
| **OQ-3** code home | **Neither option offered.** Move the existing Streamlit app to a new **`deprecated/salesperson/`** directory **now**, and give the new client component the freed **`salesperson/`** name. Server half stays inside `falkor-chat` (as the plan argued). | **Plan change.** §4.1 rewritten; no `salesperson-ui/`; S5 scaffolds into `salesperson/`; S16's `git rm -r salesperson/` becomes a *preserving* move done early, not a delete done last; §2.4's parity citations must be re-pointed at `deprecated/salesperson/*.py`; R11 is materially weakened as a risk. |
| **OQ-5** presenter identity | **`FALKORCHAT_PRESENTER_KEY` rejected.** Presenter routes are **bound to localhost only** — no operator secret, no key exchange, no rate-limiting. | **Plan change.** §4.3/§4.8 rewritten; `POST /shop/api/presenter/session` and S10's rate-limiting largely disappear. **This design has not been reviewed** — `analyst` asked mid-run to assess it, incl. `request.client.host` spoofing, `X-Forwarded-For`, and `::1` vs `127.0.0.1` under `--host 0.0.0.0`. |
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
| `falkor-chat/AGENTS.md` | v4→v5 drift **fixed 2026-09-02 (U1c)**; plan step S1 bumps to `v6` and must carry the same two rows forward | `teco` (done) → S1 implementer |
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
**Plan step S1 bumps the def to `v6`; whoever takes S1 must carry these same two rows forward.**

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

- **S1** seeds and verifies `salesperson@v6` + `order-fulfillment@v1` into `reference` and asserts
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

## S1 — code accepted, but its "pre-existing defect" report is **wrong**, and teco corrected it

**What S1 reported:** `ws:acme` already held a `salesperson@v6` snapshot *before* its run, diverging
from `reference`, so `./scripts/verify_salesperson.sh` with no argument (defaults to `acme`) exits 1.
It filed this as a **pre-existing defect** and declined to fix it.

**The symptom is real — I reproduced it:**

```
./scripts/verify_salesperson.sh          -> exit 1   "salesperson@v6: reference def and ws:acme snapshot diverge (1 differences)"
./scripts/verify_salesperson.sh s1v6     -> exit 0   "OK — 2 defs in sync"
```

**But "pre-existing" is false, and the reasoning behind it does not hold.** `v6` has **never existed
in any commit**:

```
git log -S '"v6"' -- falkor-chat/server/falkorchat/proof_defs.py   -> no commits
git show HEAD:falkor-chat/server/falkorchat/proof_defs.py | grep -c '"v6"'  -> 0
```

The version was created by S1 itself, in today's uncommitted work. A `salesperson@v6` snapshot in
`ws:acme` therefore **cannot** predate the unit. What actually happened is that S1 ran
`seed_salesperson.sh acme` **more than once**: an early run materialized v6 into `ws:acme`, the
`systemPrompt` then changed (the §4.10 assertion was rewritten mid-run during mutation testing), and
because materialize is **create-only** the `ws:acme` snapshot froze at the earlier content while
`reference` moved on.

**The delegate's evidence was misread, not fabricated.** `seed_salesperson.sh:215` probes `snap_pre`
*before* publish/materialize, so "already present — no-op" is true — but it only proves the snapshot
existed before **that invocation**, not before the unit. That is exactly the inference gap between
"my own earlier side effect" and "someone else's pre-existing defect".

**Impact is contained, and the fix is safe:** `MATCH (r:WorkflowRun) WHERE r.defKey='salesperson'` in
`ws:acme` returns **zero rows**, so nothing is bound to the orphan snapshot, and S11 pins the demo at
its own workspace anyway. `reference@v6` and `ws:s1v6@v6` are both correct.

**Lesson worth carrying:** a create-only materialize plus an in-run content change is a silent
divergence generator. Any unit that both seeds a def and then edits that def in the same run should
seed **only** into a scratch workspace, never a shared one.

## Outstanding cleanup for the stakeholder (teco will not run destructive ops)

Three items, all benign, all needing a human hand:

```
docker exec falkordb-dev redis-cli GRAPH.DELETE probe_u8_rename_dst    # U8's spent rename probe (empty)
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:probe-s0-reset      # S0's spent reset probe (wiped, 0 nodes)
docker exec falkordb-dev redis-cli GRAPH.DELETE ws:s1v6                # S1's scratch seed/verify workspace
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

## Follow-up 12 — one clause, not worth reopening an approved note

§7's `Agent`-owned orphan residual is described more broadly than it is. Pass 3 established it only
arises when the thread died in an *earlier* reset. Fold into the next natural touch of this document
rather than a dedicated unit. Owner: `graph-dba`.

## Dispatch state — everything is now blocked on one thing: the live database

S0 is closed on design, but **S4 cannot start until S2 finishes**. S4's done-condition is integration
tests on `ws:test`, and the suite wipes both `ws:test` and the global `reference` graph — the same
serialization constraint recorded earlier. S3, S6-S10 and S11/S12a all sit behind that same chain.
This is the coordination's real critical path now, and it is one-at-a-time by construction, not by
choice.
