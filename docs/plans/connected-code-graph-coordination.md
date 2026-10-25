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
| U1 | `architect` | `afd672aabfb7c2986` | in-flight | `docs/plans/connected-code-graph.md` (requirements review + phased architecture proposal) | U2/U3/U5 → — | — |
| U2 | `graph-dba` | — | queued | `docs/plans/connected-code-graph-graph.md` (graph note + findings on U1) | U5 → — | — |
| U3 | `data-scientist` | — | queued | `docs/plans/connected-code-graph-ml.md` (ML method note + findings on U1) | U5 → — | — |
| U4 | `architect` | — | queued | revision of U1 folding U2/U3 (skipped if none needed) | U5 → — | — |
| U5 | `analyst` | — | queued | `docs/reviews/connected-code-graph.md` (requirements + plan + notes) | — | — |

## Notes

- U2 and U3 run in parallel after U1. U4 runs only if U2/U3 raise changes to U1.
- Requirement-level findings go back to the stakeholder; `tico` never edits requirements on a
  delegate's say-so alone.
