# Connected code graph — expert review & architecture proposal — Coordination

> **Status:** active · **Owner:** `tico` · **Tracks:** — (`docs/requirements/connected-code-graph.md`)

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
