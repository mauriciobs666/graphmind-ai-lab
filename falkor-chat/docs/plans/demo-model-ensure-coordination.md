# Demo bring-up ensures LM Studio has the demo's models loaded — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-066

## Goal

Build K-066's remaining item (1), as specified in `falkor-chat/docs/BACKLOG.md` K-066 (stakeholder
decision 2026-10-22): an ensure step in `falkor-chat/scripts/start_demo.sh` that loads whichever of
the two demo models is missing at startup, with the demo settings, never unloads or reloads, and
stops with a clear message on a foreign model or wrong settings. Items (2)/(3) stay in the backlog.

No architect plan: the backlog item is the spec, and the change is one script plus its wiring.
The plan gate is skipped for that reason. The code gate (`analyst`) and a live check remain.

## Definition of done

- Ensure step built, wired into `start_demo.sh`, offline-tested with a stub `lms` and a stub
  models API, mutation-tested.
- Live-verified against LM Studio: no-op when both are loaded; loads a missing model; stops on a
  foreign model. End state: ministral + embedder loaded, `qwen` unloaded.
- Docs: `falkor-chat/AGENTS.md` (scripts table + demo paragraph), the `start_demo.sh` header.
  `BACKLOG.md`/`HISTORY.md` at close (teco).
- `analyst` code gate passed.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 ensure step + wiring + tests + docs + live check | `devops` | `ac42a013cc2e6b775` | accepted (`23b516f0`) — pass-1 minors landed (22 green, teco re-run; m1/m3 mutants red); first delivery: 12 offline green, 10/10 mutants red, live 3/3; teco argument mutant (`requested=None`) → 9 red, restored (`cmp`) | `falkor-chat/scripts/` (ensure step, `start_demo.sh`), tests, `falkor-chat/AGENTS.md`, `salesperson/README.md` | `analyst` (`a9530bf0b87a04770`, 93k tok / 19 tools) → pass 1 approve w/ suggestions (m1 untested post-load re-check, m2–m5, n1–n4) → minors fixed by `devops` → pass 2 approve w/ suggestions → U2 (P2-m1 post-load `loading` refused, P2-m2 every HTTP error → bypass hint, P2-n1 non-JSON 200; 116k tok) · review `falkor-chat/docs/reviews/demo-model-ensure.md` | 159k+196k tok / 52 tools |
| U2 pass-2 minors (P2-m1/m2/n1) | `tdd-engineer` | `a67cca13bf4bd73e7` | accepted (`23b516f0`) — 28 green (teco re-run), 3/3 mutants red; teco mutant (drop 405) → `[405-3]` red, restored (`cmp`) | `falkor-chat/scripts/ensure_demo_models.{py,sh}`, test file | `analyst` (`a9530bf0b87a04770`) → pass 3 **approve** (leftovers → K-083) | 88k tok / 16 tools |

## Documentation-impact scan

`falkor-chat/AGENTS.md` (scripts table row for `start_demo.sh` + the "pin the demo's models"
paragraph), `start_demo.sh` header comment, `salesperson/README.md` if it describes demo bring-up,
`falkor-chat/docs/BACKLOG.md` K-066 + `HISTORY.md` (teco, at close).

## Close (2026-10-22)

Definition of done met: ensure step built and wired (`23b516f0`), 28 offline tests with mutants,
live no-op/load/refuse verified, docs updated, `analyst` approve after 3 passes. K-066 removed from
`falkor-chat/docs/BACKLOG.md`; record in `falkor-chat/docs/HISTORY.md` 2026-10-22. Follow-ups: K-082
(optional QA reruns carried from K-066), K-083 (default opencode config unreachable here; review
leftovers).
