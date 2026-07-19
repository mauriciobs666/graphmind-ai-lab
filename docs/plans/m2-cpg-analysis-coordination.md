# M2 — `cpg-analysis` skill · teco coordination doc

> Live orchestration record for M2 (CPG consumer skill). teco drives; edited as units complete.
> Entry points: [`../BACKLOG.md`](../BACKLOG.md) (C-201…C-208), [`../requirements/joern-cpg-pipeline.md`](../requirements/joern-cpg-pipeline.md) (FR-9…FR-14, AC-6…AC-8).
> Started 2026-07-18.

## Goal / definition of done
One `cpg-analysis` skill (lean `SKILL.md` core + four bundled `references/*.md` recipes: impact,
RCA, code-review, test-gap) lets `analyst`/`architect`/`qa-engineer` query a loaded CPG via
`redis-cli GRAPH.QUERY`. Recipes cite the single canonical schema (`skills/joern-cpg/references/cpg-model.md`),
cobb-vetted against skill standards, agent descriptions wired, catalogs + HISTORY synced — and the
skill is **live-verified against a real loaded CPG** (not just authored).

## Locked decisions (do not re-litigate)
- Shape: one skill = core + 4 recipes (not 4 sibling skills).
- Ownership: graph-dba builds; cobb vets + wires agents; analyst is the independent reviewer; teco coordinates.
- Scope: all four recipes in. `qa-engineer` a named consumer. Runtime coverage excluded (test-gap = structural reachability only).
- Naming: `cpg-test-gap`, not `cpg-test-coverage`.

## Environment note (coordination-critical)
FalkorDB is **not running** and **no CPG is loaded** at kickoff. joern (Wave 1) must build+load a
representative CPG and **leave FalkorDB running** with it, reporting the graph key. FalkorDB is
in-memory — if the container stops, the CPG is lost; graph-dba/analyst must be able to reload via
the `joern-cpg` pipeline. If provisioning blocks, escalate to devops.

## Units, owners, gates

| Unit | Owner | Depends on | Done-condition | Review gate |
|---|---|---|---|---|
| W1a — M2 build plan + OQ2 (structure/naming) | architect | — | Plan at `docs/plans/m2-cpg-analysis-skill.md`: OQ2 resolved, skill layout, per-recipe FR/AC map, live-verification strategy. **Frames** recipes; does not write Cypher. | analyst reviews plan |
| W1b — OQ3 + live CPG substrate | joern | — | Joern Python + JS/TS frontend adequacy confirmed; a representative CPG (exercises callers/callees, data-flow, input→sink, prod-vs-test entrypoints) built+loaded into a running FalkorDB; graph key + start method reported | — (verification unit) |
| Gate 1 | analyst | W1a | Review of the plan at `docs/reviews/m2-cpg-analysis.md`; verdict | — |
| W2 — Build the skill (C-201…C-206) | graph-dba | Gate 1 approved + W1b CPG loaded | `skills/cpg-analysis/` (SKILL.md core + 4 recipes) + C-201 schema-contract confirmation/fill in `joern-cpg/references/cpg-model.md`; each recipe **live-verified** against the loaded CPG; `skills/README.md` catalog row added | analyst review + cobb vet |
| Gate 2a | analyst | W2 | Correctness review at `docs/reviews/m2-cpg-analysis-skill.md`; verdict | — |
| Gate 2b + C-207 wiring | cobb | W2 | Skill vetted vs. agent-standards lint; CPG-capability lines added to `analyst`/`architect`/`qa-engineer` descriptions; `claude/README.md` synced | — (cobb is the standards gate) |
| W3 — doc sync (C-208) | folded into W2/Gate2b + teco verify | W2, Gate2b | root `AGENTS.md` skills section, `claude/README.md`, `skills/README.md`, BACKLOG→HISTORY entry all landed in the same change | teco integration read |

## Sequencing
```
W1a architect ─▶ Gate1 analyst ─┐
                                 ├─▶ W2 graph-dba ─┬─▶ Gate2a analyst ─┐
W1b joern (CPG substrate) ───────┘                 └─▶ Gate2b cobb + C-207 ─┴─▶ W3 doc sync ⇒ M2 ✅
```

## ⏸ PAUSED 2026-07-18 (user out of credits) — resume checklist
State at pause:
- ✅ W1a architect plan (`docs/plans/m2-cpg-analysis-skill.md`); ✅ Gate-1 analyst review (`docs/reviews/m2-cpg-analysis.md`, approve-with-suggestions).
- 🟡 **devops** was mid-install of Joern v4.0.579 → `$HOME/joern/joern-cli`; a 2.13 GB download was running as a background job set to auto-resume→checksum→unzip→smoke-test (agentId `aae3bb1dead1cc4a4`). **On resume: check whether the download/unzip/smoke-test completed** before doing anything else.
- ⏸ **joern** (agentId `ab09bdb3884f6769f`) is staged and waiting: once Joern is callable, resume it to run the one-shot pipeline (`pipeline.sh <staged falkorchat+tests> --graph cpg_falkorchat --load`), **confirm the reported test anchors exist post-load** (real FULL_NAMEs), and report node/edge counts. FalkorDB `falkordb-dev` v4.18.11 on :6379 was up at pause but is **in-memory** — re-verify it's still running (`redis-cli ping`; restart via `falkor-chat/scripts/start_falkordb.sh -d`).
- ⛔ **W2 graph-dba** not yet started — gated on the live CPG. When launching, its brief must carry: the plan path, the review path (+ folded majors M1/M2/M3 & nits), joern's substrate facts (graph key `cpg_falkorchat`, anchors, real labels via `CALL db.labels()`), and the two teco decisions below.
- Then Gate-2a (analyst correctness review + independent cold AC-6 invocation) ∥ Gate-2b (cobb standards vet + C-207 agent wiring), then C-208 doc-sync (skill + `skills/README.md` + agent descriptions + `AGENTS.md` skill-count + BACKLOG→HISTORY, one change).

## ▶ RESUMED 2026-07-19
State verified on resume: FalkorDB up on :6379 but **empty** (no CPG loaded); Joern zip **fully downloaded** (`~/joern/joern-cli.zip` 2.13 GB + `.sha512`) but **not unzipped** (`joern-parse` absent); disk `/` at 94%, 5.5 GB free. Launched fresh **devops** (prior context not resumable) to verify checksum → unzip to `$HOME/joern/joern-cli` → smoke-test → reclaim the zip. Next: resume/re-brief joern to load `cpg_falkorchat`, then graph-dba build.

## Status log (resume)
- 2026-07-19 — devops ✅ Joern v4.0.579 installed at `$HOME/joern/joern-cli`, smoke-test green, Python frontend bundled/warm, zip reclaimed → **5.0 GB free** (disk remains the constraint for CPG export). Quirk: use `joern --version`, not `joern-parse --version`.
- 2026-07-19 — W1b ✅ joern: **`cpg_falkorchat` live** in FalkorDB :6379 — 79,581 nodes / 522,182 edges (counts match export). 20 labels / 19 edge types confirmed; anchors resolved with real FULL_NAMEs. Key findings passed to graph-dba: (a) pysrc2cpg direct `(:METHOD)-[:CALL]->(:METHOD)` edges are **unreliable** → use `(:METHOD)-[:CONTAINS]->(:CALL)` + match CALL node; (b) framework entrypoints (17 routes + 9 MCP tools) not statically modeled → test-gap recipe must **seed** prod+test entrypoints; (c) sink disambiguation (exclude `FakeRepo` doubles); (d) reload artifacts in **session-temp** scratchpad → graph-dba to copy to durable spot.
- 2026-07-19 — ⚠️ **FOLLOW-UP (out of M2 scope):** joern found a real bug in the M1 `joern-cpg` loader `cpg-to-falkordb.py --load` — passes each 500-node UNWIND batch as one `redis-cli` argv, hits Linux 128 KiB `MAX_ARG_STRLEN` on large `CODE` props → `OSError: Argument list too long`, yet `pipeline.sh` reported exit 0. Worked around via stdin (`redis-cli -x`). **User decision 2026-07-19: leave as a separate follow-up AFTER M2** — do NOT fold into M2 scope. Action: file as a new `C-`item in `docs/BACKLOG.md` (producer-skill `joern-cpg` fix) during the M2 doc-sync step so it isn't lost.
- 2026-07-19 — W2 graph-dba launched (build the `cpg-analysis` skill) against live `cpg_falkorchat`, with plan + analyst review + all joern substrate facts + teco decisions folded in.
- 2026-07-19 — W2 ✅ graph-dba delivered: `skills/cpg-analysis/` (SKILL.md 145L, `description` 811 chars, `allowed-tools: Bash, Read` + 4 recipes impact/rca/code-review/test-gap), C-201 additive "Consumer-query facts" section in `joern-cpg/references/cpg-model.md`, `skills/README.md` catalog row + stale-count fix. **Live-verified** per-AC vs `cpg_falkorchat` (79,581 nodes/522,182 edges): AC-2 callers=21; AC-3 transitive reach; AC-4 REACHING_DEF backward slice; AC-5 both `get_context` def sites; AC-7 taint both directions; AC-8 test-gap=30 untested methods (`_serialize_opaque` correctly excluded). Deliverables + live graph re-confirmed on disk by teco. graph-dba flags: (a) AC-6 gate = analyst cold invocation (pending Gate-2a); (b) root `AGENTS.md` "all 7 skills" stale (now 9 folders) — C-208; (c) C-207 + `claude/README.md` remain for cobb; (d) verification was **Python-only** (JS/TS not exercised) — record as a known limit. Durable reload artifacts in `cpg/.cpg-artifacts/` (cpg.bin, load.cypher, load_stdin.sh).
- 2026-07-19 — Gate-2a (analyst correctness review + cold AC-6 invocation) ∥ Gate-2b (cobb standards vet + C-207 wiring) launched in parallel against the delivered skill.
- 2026-07-19 — Gate-2b ✅ **cobb: accept** (no must-fix). Standards/§7 lint PASS (name==folder, description 811 chars, `allowed-tools: Bash, Read`, core-plus-references, FR-14 single-schema citation, no PII leak). C-207 wired: CPG-capability clause added to `analyst`/`architect`/`qa-engineer` descriptions + `claude/README.md` catalog cells + each agent's `kaizen/{history,plan}.md`. `skills/README.md` confirmed consistent (9 folders / 9 rows, cpg-analysis row present). Two **minor optional** polish notes routed to graph-dba (non-blocking): (1) §3 clarify `$fn`/`$full` are literal text substitution, not redis-cli bound params; (2) optionally restate route-to-`joern` at missing-graph failure point. **teco C-208 note from cobb:** stale skill-count is NOT only in root `AGENTS.md` — `claude/README.md:81` also reads "all 7 skills" (now 9); fold both into the one C-208 count fix.
- 2026-07-19 — Gate-2a ✅ **analyst: approve with suggestions** (`docs/reviews/m2-cpg-analysis-skill.md`), no blocker. **AC-6 cold invocation PASSES on all four recipes** (ran verbatim as schema-naive caller vs `cpg_falkorchat`; impact/rca/code-review/test-gap all correct using only skill + cited `cpg-model.md`). AC-7 clean=none confirmed a *true* clean with honest coverage caveat (not a false negative). **One Major (owner graph-dba, evidence-accuracy, NOT a query rewrite):** test-gap recipe records "**30** methods flagged" but the exact query yields **39 rows / 32 distinct names** on the identical graph — teco re-ran and confirmed (40 lines incl. header = 39 data rows; count(DISTINCT)=32). Must be corrected before the number is copied to HISTORY at C-208. Minors: code-review Pattern A flags every param incl `self` as taint source (over-reports); AC-5 demoed on symbol w/ ~no cross-file usage + ref query drops LOCAL nodes; test-gap satisfies AC-8 via test-reach *complement* (sound adaptation — state explicitly); + Python-only coverage caveat.
- 2026-07-19 — Re-brief graph-dba (single fix pass): correct the test-gap recorded number to 39 rows / 32 distinct names + fold in analyst minors + cobb's 2 polish notes ($fn/$full literal-substitution clarification; optional route-to-joern on missing graph). Low-risk evidence/precision polish, no query rewrite → **ceremony trim: teco verifies the corrected number itself at integration (already re-ran: 39/32), no second analyst gate.** Both gates already approve-with-suggestions.
- 2026-07-19 — W2-fix ✅ graph-dba: all four findings landed & re-verified live. test-gap now records **39 rows / 32 distinct names** (row-vs-name distinction made explicit); code-review Pattern A over-report documented (filter `self`/non-external params); rca AC-5 anchor swapped to `hybrid_search` (real cross-file def/ref: defs `repository.py:658`+`services.py:396`, refs `responder.py:97`/`services.py:408`/`tools.py:293`) + LOCAL-node boundary noted; AC-8 test-reach-complement framing stated; Python-only coverage boundary added to SKILL.md §1; $fn/$full literal-substitution clarified; missing-graph→joern pointer added. **teco integration check ✅** — re-ran aggregate: `count(*)`=39, `count(DISTINCT g.NAME)`=32; polish edits confirmed on disk. No query logic rewritten. All ACs (AC-2…AC-8) live-verified; AC-6 cold-invocation passed at Gate-2a. **Skill is DONE and accepted.**
- 2026-07-19 — ✅ **M2 CLOSED.** C-208 doc-sync landed (cobb) + **teco integration-read verified all 4 edits on disk**: root `AGENTS.md` "all 9 skills" + cpg-analysis bullet; `claude/README.md:81` "all 9 skills"; `docs/HISTORY.md` M2 entry prepended above M1 (39/32 figure, AC-6 pass); `docs/BACKLOG.md` M2 ✅ + C-201…C-208 ✅ + Last-reviewed 2026-07-19 + **C-101** loader-bug filed under "Follow-ups (post-M2)". All C-201…C-208 done; every AC (AC-2…AC-8) live-verified; AC-6 passed independent cold invocation. Nothing committed (user hasn't asked). Deferred follow-ups: (a) archive frozen M2 plan/review docs → `docs/archive/`; (b) **C-101** joern-cpg loader fix in backlog.
- 2026-07-19 — Superseded note: **C-208 doc-sync** (delegated — outside teco write-guard): root `AGENTS.md` + `claude/README.md:81` "7 skills"→9; `docs/HISTORY.md` M2 entry (39/32 figure); `docs/BACKLOG.md` M2+C-201…C-208 ✅ + file the M1 `joern-cpg` loader `MAX_ARG_STRLEN` bug as a new post-M2 C-item. Archiving of frozen M2 plan/review docs → deliberate follow-up, not bundled into the close commit.

## teco decisions (Gate-1 follow-ups, within coordinator/requirements remit — not user forks)
- **AC-3 scope** → impact recipe does **call-graph reachability over `CALL`** (faithful to FR-10's explicit "over CALL"); type/import-edge dependency reach is an out-of-M2 enhancement, not built now.
- **AC-6 gate** → the AC-6 usability claim ("without hand-knowing the schema") is verified by an **independent cold invocation by `analyst` at Gate-2a** (schema-naive reviewer), not solely by graph-dba self-verification.
- **Anchor manifest** (analyst M3) → satisfied by joern's reported example entities; joern must **confirm these anchors exist in the loaded CPG** (real FULL_NAMEs) when it runs the pipeline — added to the joern resume instruction.
- Analyst majors M1 (name the intraprocedural `REACHING_DEF` boundary; require ≥1 interprocedural anchor so AC-7 "clean=none" isn't a false negative) + nits (`METHOD_FULL_NAME`/`IS_EXTERNAL` confirmed & documented in C-201; skill-count wording made definite, folder count now 9 with cpg-analysis) → folded into the graph-dba build brief. Plan NOT sent back to architect (approve-with-suggestions).

## Status log
- 2026-07-18 — Coordination doc created; launching W1a (architect) + W1b (joern) in parallel.
- 2026-07-18 — W1a ✅ architect plan delivered (`docs/plans/m2-cpg-analysis-skill.md`); OQ2 resolved = no new `code-graph/` dir. Gate 1 (analyst plan review) launched in parallel with W1b (joern still running).
- 2026-07-18 — W1b ⚠️ **blocked**: Joern toolchain missing from machine (present at M1, now gone). joern completed everything else: OQ3 = Python `pysrc` frontend (JS/TS negligible/moot); target = `falkor-chat/server/{falkorchat,tests}` (exclude `.venv`); FalkorDB up+verified (`falkordb-dev`, v4.18.11, :6379); one-shot load command staged; intended graph key `cpg_falkorchat`. Routed toolchain restore → **devops** (also flagged `/` at 93%, 7.1 GB free). After devops: **resume joern** (agentId ab09bdb3884f6769f) to run the staged pipeline, confirm anchors, report node/edge counts.
- 2026-07-18 — devops (Joern restore) in progress: reclaimed ~600 MB (pip cache), 7.2 GB free; Joern v4.0.579 confirmed available; 2.13 GB download running as background job (~20 min), will auto-resume → checksum → unzip → smoke-test.
- 2026-07-18 — Gate 1 ✅ **approve with suggestions** (`docs/reviews/m2-cpg-analysis.md`); no blockers. 3 majors + nits folded into graph-dba brief; 2 caller-questions resolved by teco (see decisions above). Plan not re-sent to architect. graph-dba (W2) still gated on CPG load (devops→joern).
