# python-web-quirks on the knowledge base — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Requirements: `docs/requirements/python-web-quirks-kb.md` (Interviewing; stakeholder answers to 11 questions recorded in its decision log, 2026-11-09). Docs-only chain so far: verified deliverables stay uncommitted and land in one commit at the chain's terminal state.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `tico` | `abd8a75b922c4b3e3` | accepted | `docs/requirements/python-web-quirks-kb.md` (OQ-1 open) | `analyst` (U3) → approve with suggestions (Pass 3); Ready for design, stakeholder-confirmed 2026-11-09 | 117k tok / 41 tools (cumulative) |
| U2 | `data-scientist` | `a873a36e21d841c7c` | accepted | `docs/plans/python-web-quirks-kb-ml.md` (golden set, recall target, floor, chunking; OQ-1) | stakeholder approved targets 2026-11-09 → approved (AC-3 reworded) | 126k tok / 29 tools |
| U3 | `analyst` | `a3f02cdad3fbd9599` | accepted | `docs/reviews/python-web-quirks-kb.md` Pass 1–3 | — → Pass 1 needs changes, Pass 2 needs changes (narrow), Pass 3 approve with suggestions (applied) | 111k tok / 28 tools (cumulative) |

Next: commit the docs-only chain; U4 `architect` plan `docs/plans/python-web-quirks-kb.md` (decides: durable copy location, degraded-guidance home, header delimiter, listing call and bound; re-derives the method note's counts over the unit list; manifest no-change ruling), then `analyst` gate.

- 2026-11-09: stakeholder answers recorded in the requirements' decision log. New obligations from them: `cobb` edits `tico` and `security-expert` tool grants; `cobb` lists the 4 raw captures that mention the skill and the stakeholder deletes them; Set B author `qa-engineer`, storage outside implementers' reading path (teco decides at plan time).
