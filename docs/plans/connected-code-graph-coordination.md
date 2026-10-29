# Connected code graph — expert review & architecture proposal — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (`docs/requirements/connected-code-graph.md`)

## Goal & definition of done

The requirements doc `docs/requirements/connected-code-graph.md` is still **Interviewing**. Before
the stakeholder reviews its acceptance criteria, the stakeholder asked (2026-10-25) for `architect`,
`graph-dba`, `data-scientist`, and `analyst` to review what has been agreed and suggest an actual
implementation architecture.

Done when: an architecture proposal (`docs/plans/connected-code-graph.md`), a graph note
(`-graph`), an ML note (`-ml`), and an independent review (`docs/reviews/connected-code-graph.md`)
exist and have been checked, with every finding that touches requirements surfaced back to the
stakeholder. This is a docs-only chain: if any unit would need source, tests, or config changed, the
remaining chain goes to `teco`.

## Phase P0 — goal & definition of done (`teco`, from 2026-10-29)

Docs chain U1–U14 (owned by `tico`) is complete; the stakeholder asked `teco` (2026-10-29) to run
phase P0 of `docs/plans/connected-code-graph.md` §4. The plan is a proposal and states each phase
gets its own implementation plan, so P0 starts with one.

Done when: every P0 deliverable (a)–(h) of plan §4 exists and is gated; every §6.2 "for P0 to
settle" decision is recorded with its measurement; bundle schema v1 and the graph model are frozen
(or explicitly left provisional per §4 P0(b)); `docs/HISTORY.md` and the C-1001 entry of
`docs/BACKLOG.md` reflect it. Further units (U17+) are drawn from the P0 plan's step table once it
is gated.

Documentation-impact scan: `docs/BACKLOG.md` (C-1001), `docs/HISTORY.md` (entry per delivered
unit), `claude/graph-dba/falkordb-quirks.md` (plan §6.2 graph-dba item), root `AGENTS.md` `cpg/`
bullet (rewritten at P1, not P0, per plan §3.13).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `afd672aabfb7c2986` | delivered | `docs/plans/connected-code-graph.md` v1: §0–§7 complete, no placeholders (tico checked 2026-10-25); requirement findings R1–R17, questions for graph-dba §6.2 and data-scientist §6.3. Run 1 stopped early after §0–§1; resumed same agent | U2/U3/U5 → — | run 1: 115k tok / 28 tools; resume: 140k tok / 37 tools (cumulative) |
| U2 | `graph-dba` | `ab82775fc08487089` | delivered | `docs/plans/connected-code-graph-graph.md`: §0–§5 complete (tico checked 2026-10-26). Verdict: sound with changes (G1–G13; G1–G3 High); M4 answered (compatible, Q9); requirement questions Q-R1–Q-R6 | U5 → — | 220k tok / 50 tools (incl. resume) |
| U3 | `data-scientist` | `a14a2240f9d417411` | delivered | `docs/plans/connected-code-graph-ml.md`: §0–§7 complete (tico checked 2026-10-25). Verdict on proposal: sound with changes (M1–M5 major, m1–m8 minor); requirement findings RQ1–RQ11 | U5 → — | 131k tok / 20 tools |
| U4 | `architect` | `afd672aabfb7c2986` (resumed) | delivered | `docs/plans/connected-code-graph.md` v2: every G/M finding adopted, none rejected; §2.7 cross-reference table; new R18, D4, D5, K8–K10 (tico checked 2026-10-26) | U5 → — | 227k tok / 41 tools (cumulative) |
| U5 | `analyst` | `a7528783f26ffa234` | gated | `docs/reviews/connected-code-graph.md`: findings B1–B6 major + minors/nits; §5 lists 38 deduplicated stakeholder decisions S1–S38 (tico checked 2026-10-26) | gate on U4 → **approve with changes** (no blockers; B1–B6 to architect as v3 before P0 freezes contracts) | 190k tok / 40 tools |
| U6 | `graph-dba` | `ab82775fc08487089` (resumed) | delivered | `docs/plans/connected-code-graph-graph.md` revised in place: B1 edge-direction fix + graph impact of requirement changes since `3c48a092`. Delivered as Version 2, new §6 with B1 fix and probes V15–V18 (tico checked 2026-10-26); stakeholder questions Q-R7–Q-R11 | U8 → — | 288k tok / 17 tools (this run) |
| U7 | `data-scientist` | `a14a2240f9d417411` (resumed) | delivered | `docs/plans/connected-code-graph-ml.md` revised in place: method impact of requirement changes since `3c48a092` (AC-5, AC-12, AC-14, AC-15, AC-24). Delivered as Version 2, new §8 (tico checked 2026-10-26); stakeholder questions SQ1–SQ8 | U8 → — | 189k tok / 15 tools |
| U8 | `architect` | `afd672aabfb7c2986` (resumed) | delivered | `docs/plans/connected-code-graph.md` v3: B1–B6 + S1–S38 + new requirements + U6/U7 input. Delivered v3 (tico checked header, no placeholders, 2026-10-26); §2 cross-reference tables; stakeholder questions V3-Q1–V3-Q5 | U9 → — | 339k tok / 20 tools (this run) |
| U9 | `analyst` | `a7528783f26ffa234` (resumed) | gated | `docs/reviews/connected-code-graph.md` Pass 2 (tico checked 2026-10-26): Pass 1 14 fixed + 1 superseded; new P2-1–P2-7 (2 major) | gate on U8 → **approve with changes**; stakeholder questions SQ-A–SQ-G | 301k tok / 18 tools (this run) |
| U10 | `architect` | `afd672aabfb7c2986` (resumed) | delivered | `docs/plans/connected-code-graph.md` v3.1: P2-1–P2-7 + SQ-A–SQ-G + V3-Q1–Q5 marked decided. Delivered v3.1, no stakeholder questions; one recorded reading of FR-29 "moved" (tico checked 2026-10-26) | U11 → — | — |
| U11 | `analyst` | `a7528783f26ffa234` (resumed) | gated | `docs/reviews/connected-code-graph.md` Pass 3 (tico checked 2026-10-28): all 8 Pass 2 findings fixed; P3-1 minor, P3-n1 nit | gate on U10 → **approve** | — |
| U12 | `architect` | `afd672aabfb7c2986` (resumed) | accepted | plan v3.2: P3-1 (candidates kept as possible, folded) | tico check (stakeholder: no further review pass) | — |
| U13 | `graph-dba` | `ab82775fc08487089` (resumed) | accepted | graph note: P3-n1 preview sizing at default 2 | tico check | — |
| U14 | `architect` | `a785c7d812bd2b44a` (fresh; earlier id not resumable) | accepted | `docs/plans/connected-code-graph.md` amendment for Arduino `.ino` (FR-1, AC-26; later, not phase-blocking) | tico check | — |
| U15 | `architect` | `a5c76dc11e6c767df` | delivered | `docs/plans/connected-code-graph-p0.md`: §0–§14, ~46 steps S01–S51, findings F1–F13, conditional SQ-P0-1/2, K-P0-1 (RAM cap) (teco checked 2026-10-29) | U16 → — | 282k tok / 39 tools |
| U16 | `analyst` | `a3b6193fa11b2bc2a` | gated | `docs/reviews/connected-code-graph-p0.md` Pass 1: B1 blocker (ground-truth isolation, §6), M1–M6, m1–m8, 3 nits | gate on U15 → **needs changes** | 222k tok / 55 tools |
| U17 | `architect` | `a5c76dc11e6c767df` (resumed) | delivered | plan v1.2 (§15 dispositions): Pass 1 all dispositioned; B1 → headless sessions, new S52 (`cobb` defs + guard hook), S53 (`devops` launcher), S54 (planted-breach rehearsal) (teco checked 2026-10-29) | U18 → — | 405k tok / 20 tools (B1 run) |
| U18 | `analyst` | `a3b6193fa11b2bc2a` (resumed) | gated | review Pass 2: P2-1, P2-2, P2-5 major; P2-3, P2-4, P2-6 minor (all fix-induced); six-item stopping condition for v1.3 | gate on U17 → **approve with suggestions** (S52–S54 held for v1.3; rest dispatchable) | 286k tok / 19 tools |
| U19 | `architect` | `a5c76dc11e6c767df` (resumed) | accepted | plan v1.3: P2-1–P2-4, P2-6, nits; no encryption (decision); new S55 (headless capture) | teco mechanical check of stopping items 1–6 → **met** (2026-10-29; items 1–3 are S53 done-conditions, verified at S53) | 444k tok / 27 tools (cumulative) |
| U20 | `devops` | `a700c23d1d6f306b4` | accepted | P0-01 (S01): `cpg/infra/ccg-falkordb.sh`, `cpg/infra/README.md`, `cpg/infra/tests/test-ccg-falkordb.sh` (stub harness, 160-combination ownership probe); M1 label guard, m1–m3, 2g memory cap. Instance up on 127.0.0.1:6390 | `analyst` (`a72f2b8992f574958`) → approve with suggestions; fix round re-verified by teco 2026-10-29: test 327/0 green, `status` rc 0, teco's own argument mutants (HOST=0.0.0.0; label value 0 at `docker run`) both red | 83k + 137k tok |
| U21 | `devops` | `a29f3b01257683422` | in-flight | P0-02 (S02): `cpg/infra/toolchain/{Dockerfile,build.sh,README.md}` | `analyst` → — | — |
| U22 | `architect` | `a2e0c5036a2c7050d` | delivered | P0-03a (S03): `docs/plans/connected-code-graph-bundle-spec.md`: 19 record types, path-based `qname`, 14 golden vectors, findings BF1–BF15 (teco recomputed GV-1/3/4/6/10/14: match) | `analyst` (`a4d13f17331545c53`) + `graph-dba` (`aff0bb6064e845bdf`, resumed), p0-impl §S03 → — | 203k tok / 40 tools |
| U23 | `architect` | `aa663dbdaf3c5ec8f` | delivered | P0-03b (S04): `docs/plans/connected-code-graph-egress-spec.md` v1.0: escape axes A1–A7 (A7 AF_VSOCK added), AC-8 map 14 rows, 11 forbidden flags + launcher-option prefix refusal; findings 1–8 | `analyst` (`a82b9ea39a27bd7b3`, p0-impl §S04) → — | 148k tok / 47 tools |
| U24 | `data-scientist` | `a24f61274678bd8b3` | in-flight | P0-04 (S05): `docs/plans/connected-code-graph-p0-ml.md`: 150 in-corpus sites/language (cap 400), tree-sitter frame, scip/vta truth + blind label; 0.98 recall gate reworded as a screen; open Q1 (no step makes golden flow sets), Q2 (who adjudicates). teco spot-checked 4 headline figures | `analyst` (`a72f2b8992f574958`, p0-impl §S05) → **needs changes** (M1 labeller brief, M3 S24↔S25 circular, M4 flow criteria, m1–m4, n1 → data-scientist; M2 rule authority, M5 missing steps, m5 → architect v1.4) | 132k tok / 30 tools |
| U25 | `graph-dba` | `aff0bb6064e845bdf` | accepted | P0-06 (S07): `claude/graph-dba/falkordb-quirks.md`, 7 entries re-verified on v4.18.11/41811 (V2, V5+V16, V9, V10, V17, V18); V9 old entry corrected, TIMEOUT entry scoped to `TIMEOUT_MAX=0` | `teco` doc check → **pass** (2026-10-29) | 116k tok / 36 tools |

## Notes

- U2 and U3 run in parallel after U1. U4 runs only if U2/U3 raise changes to U1.
- Requirement-level findings go back to the stakeholder; `tico` never edits requirements on a
  delegate's say-so alone.
- 2026-10-25: U3's M4 (no approximate vector search on the fact path) relayed to in-flight U2 as a question, not a premise.
- 2026-10-26: U2 stopped on an API usage limit with its note partly written (§1, §3 done; §2, §4, §5 empty); the same agent was resumed via SendMessage.
- 2026-10-26: U2 delivered. U2 and U3 both raise changes to U1, so U4 runs (same architect agent, resumed).
- 2026-10-26: U5 gated approve-with-changes. Next is a stakeholder pass over S1–S38 (and the AC review), then U6/U7, so v3 is written once against settled decisions rather than twice.
- 2026-10-26: stakeholder decided S1–S38 and reviewed every AC, then re-checked all five use cases (review, security, impact, test map, RCA gained requirements). Requirements stay Interviewing at the stakeholder's request; U6/U7 wait for the Ready-for-design flip. Interim commit of the chain so far on the stakeholder's request.
- 2026-10-26: stakeholder asked for the expert round before the Ready-for-design flip; requirements stay Interviewing. Round shape: U6 ∥ U7 → U8 → U9. AC-12's "reinforce each other" reading (trace + log together rank the true cause at least as high as either alone) is still unconfirmed — delegates treat it as an assumption.
- 2026-10-26: U7 delivered §8 with stakeholder questions SQ1–SQ8, relayed to the stakeholder. It flagged rule 5 (a gated note gets a successor). tico keeps the in-place revision: U5's verdict was approve-with-changes, so this round completes that gate cycle and is not a new design.
- 2026-10-26: stakeholder accepted all of SQ1–SQ8; folded into the requirements (AC-5, AC-12, AC-14, AC-24). U6 delivered with Q-R7–Q-R11, relayed to the stakeholder.
- 2026-10-26: stakeholder answered Q-R7–Q-R11 (preview default lowered to 2); folded into FR-11, FR-15, FR-23. U8 dispatched.
- 2026-10-26: U8 delivered v3 with V3-Q1–V3-Q5, relayed to the stakeholder. The architect did not re-read v3 end to end after its edits; U9 must check internal consistency.
- 2026-10-26: stakeholder accepted V3-Q1–V3-Q5; folded into FR-9, FR-13, FR-15. U9 dispatched.
- 2026-10-26: U9 gated approve-with-changes (P2-1, P2-2 major). P2-2d (FR-14 risk factors out of step with AC-14) was a wording fix and tico applied it; SQ-A–SQ-G relayed to the stakeholder.
- 2026-10-26: stakeholder accepted SQ-A–SQ-G (folded into FR-9, FR-15, FR-21, FR-29, AC-8, AC-21a, AC-24) and chose a fix-up round (U10 → U11) before one commit of the whole round.
- 2026-10-28: U11 stopped on an API usage limit before writing anything; same agent resumed via SendMessage. Stakeholder confirmed FR-29 "moved" = qualified name changed (folded into FR-29).
- 2026-10-28: U11 approved. Stakeholder chose P3-1 → keep candidates as possible, folded; quick fixes U12 ∥ U13 with tico's check only, then one commit of the whole round.
- 2026-10-28: U12 (v3.2) and U13 (graph note v2.1) checked by tico and accepted. Round closed; whole chain committed together. Requirements remain Interviewing until the stakeholder says to flip.
- 2026-10-28: readback reopened for Arduino `.ino` support (FR-1, AC-26, later/not blocking), then confirmed; requirements flipped to **Ready for design**, filed as C-1001. Plan v3.2 predates the `.ino` addition and needs a small amendment (architect) before P0 goes to `teco`.
- 2026-10-28: U14 (plan v3.3: Arduino `.ino` as C++ via an in-house sketch preprocessor and line map; optional slot after P2, gates no phase; K14–K15) checked by tico and accepted, plus the stale "requirements still Interviewing" opening line fixed. Graph and ML notes unchanged. Docs chain complete; P0 onward needs implementers → `teco`.
- 2026-10-29: handed to `teco` for P0 (stakeholder). U15 (P0 phase plan) → U16 (analyst gate) before any implementer. `cpg_falkorchat` is stale (21 server commits since 2026-10-15); U15 proceeds on it — planning leans on it only lightly, and the P0 slice builds its own bundles.
- 2026-10-29: U15 delivered. teco's check found step ids S01–S51 share a namespace with stakeholder decisions S1–S38 (e.g. step S36 vs decision S36); put to U16 as a question, not a premise.
- 2026-10-29: U16 needs changes. Isolation design for B1 (separate headless key-writer session launched by teco vs. a tool-restricted agent definition with path hooks by cobb) put to the stakeholder; U17 revises everything else meanwhile. U16's two learning captures were denied by the auto-mode classifier; surfaced to the stakeholder, not re-written by teco.
- 2026-10-29: stakeholder chose a **separate headless session** for the key writers (B1, option a); relayed to U17. Note for U18: v1.1 removed the passphrase-custody decision point (n3) on the architect's own call; the gate should re-check it.
- 2026-10-29: U18 approve-with-suggestions. Stakeholder rejected encrypting the held-out set ("too much"): no gpg, no passphrase. Held-out keys stay plaintext outside the repo (`~/.local/share/ccg-eval/`), off every builder brief, with a hash manifest in the repo for tamper evidence; hiding is by convention only. This supersedes P2-5 and the n3 thread. Relayed to U19.
- 2026-10-29: plan v1.3 committed (`e3a01931`). Wave 0 dispatched in parallel (U20–U25 = P0-01, -02, -03a, -03b, -04, -06). U22 and U23 both add one pointer to the proposal's `Extended by:` line and were told to re-read it before editing; check for a lost pointer at integration. RAM: S02's image build runs at the same time as S01's 1.5 GB instance on the 6 GB VM.
- 2026-10-29: all six wave-0 units (U20–U25) stopped on the 5-hour API usage limit before writing anything (tree clean, `falkordb-dev` unchanged). Each was resumed on its own agent id via SendMessage at the stakeholder's request.
- 2026-10-29: U20 reported a foreign container `gdba-s07-probe` on `0.0.0.0:6394` (all interfaces), probably U25's. Relayed to U25: bind to loopback or remove it, and report it.
- 2026-10-29: U20 gate approve-with-suggestions; M1 confirmed by teco in code (`docker stop/rm "$NAME"`, label set but never checked). Fix round, with a stub-docker regression test written first, sent to U20's agent.
- 2026-10-29: U25 accepted. `gdba-s07-probe` was U25's (moved to loopback after the relay, now removed). Carry to S22: V18 shows a nonzero `TIMEOUT_MAX` also kills the loader's long writes unless they pass an explicit `timeout`. S22's m8 probe already covers this; cite the quirks entry in its brief.
- 2026-10-29: U24 delivered. Its truth-tool list belongs to S24, not S02; relayed to U21 as information only (structure the Dockerfile for an S24 layer, and keep dependency caches out of fetch-off mounts). S05's Q1/Q2 and its rule changes (screen wording, fold/list, `flowAvailable` threshold) go to the gate first, then to the architect as a plan amendment if needed.
- 2026-10-29: U23 delivered. Its finding 1 was confirmed by teco: the binfmt entry on this host is `WSLInterop-late`, not the `WSLInterop` the P0 plan names (D6, §4.1 test 1, §7), and `/dev/vsock` exists. **Queued plan amendment (architect, v1.4):** WSLInterop detection, S05 Q1 (golden flow sets need a step) and Q2 (adjudicator placement), plus anything the S04/S05 gates route to the plan. S03 seam (bundle-header audit-id = egress `auditId`) relayed to U22.
- 2026-10-29: U22 delivered; the proposal's `Extended by:` line carries all five pointers (no lost write). S03 gate split by claim: analyst = usability for source-only key writers, ids, authority, freeze; graph-dba = the graph-model mapping and BF3/4/6/7/13/14. Open question for the gate: GV-11 tags a `.ts` file `javascript`. S06 (P0-05) is now unblocked on S03's text but waits for the S03 gate, since it encodes S03's locators.
