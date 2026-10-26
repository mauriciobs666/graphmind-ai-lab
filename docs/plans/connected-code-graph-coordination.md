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
| U6 | `architect` (+ `graph-dba` on B1) | — | queued | plan v3: B1–B6 + stakeholder decisions S1–S38 | U7 → — | — |
| U7 | `analyst` | — | queued | `docs/reviews/connected-code-graph.md` Pass 2 | — | — |

## Notes

- U2 and U3 run in parallel after U1. U4 runs only if U2/U3 raise changes to U1.
- Requirement-level findings go back to the stakeholder; `tico` never edits requirements on a
  delegate's say-so alone.
- 2026-10-25: U3's M4 (no approximate vector search on the fact path) relayed to in-flight U2 as a question, not a premise.
- 2026-10-26: U2 stopped on an API usage limit with its note partly written (§1, §3 done; §2, §4, §5 empty); the same agent was resumed via SendMessage.
- 2026-10-26: U2 delivered. U2 and U3 both raise changes to U1, so U4 runs (same architect agent, resumed).
- 2026-10-26: U5 gated approve-with-changes. Next is a stakeholder pass over S1–S38 (and the AC review), then U6/U7, so v3 is written once against settled decisions rather than twice.
- 2026-10-26: stakeholder decided S1–S38 and reviewed every AC, then re-checked all five use cases (review, security, impact, test map, RCA gained requirements). Requirements stay Interviewing at the stakeholder's request; U6/U7 wait for the Ready-for-design flip. Interim commit of the chain so far on the stakeholder's request.
