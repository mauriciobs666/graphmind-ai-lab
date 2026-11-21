# K-093 message→entity MENTIONS (build or drop) — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-093

Goal: resolve K-093 (`docs/BACKLOG.md`, its own done-condition). Stakeholder chose (2026-11-21) to have
`architect` weigh build vs. drop first; the stakeholder picks the path from
`falkor-chat/docs/plans/message-entity-mentions.md`. Path picked: **B, drop** (2026-11-21). Close: U4 → analyst close check → teco applies REVIEW.md proposals → `HISTORY.md` entry + K-093 removal from BACKLOG.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a2a11875cbb4347f0` | delivered | `falkor-chat/docs/plans/message-entity-mentions.md` (decision note; recommends B, drop) | stakeholder → **B (drop)**, 2026-11-21 | 137k tok / 52 tools |
| U2 | `analyst` | `a58ff3c036d192b44` | accepted | `falkor-chat/docs/reviews/message-entity-mentions.md` (plan gate on B) | Pass 1 → needs changes; Pass 2 (plan v2) → **approve with suggestions** (N1 fixed by teco; N2 folded into U3 brief; N3 tolerated) | 117k+138k tok / 51 tools |
| U3 | `tdd-engineer` | — | queued | drop diff per plan §4/§7-B | `analyst` impl gate → — | — |
| U4 | `tdd-engineer` | — | queued | spec close: QUERIES §6, DESIGN §5.1/§8 (+ reversal trigger) | `analyst` close check (`reviews/message-entity-mentions-impl.md`) → — | — |

Documentation impact (feature close, not build units): DESIGN §5.1, §8, §11; QUERIES §6; SERVER (MCP tool
output) if the response field changes; `docs/HISTORY.md`; `docs/BACKLOG.md` (remove K-093 at close).
