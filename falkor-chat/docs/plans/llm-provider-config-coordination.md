# LLM Provider & Model Configuration — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-042 (M4)

Coordination record for delivering `docs/requirements/llm-provider-config.md`.
This document, not any agent's context window, is the state of record.

## Stakeholder decisions (2026-08-10)

1. **Requirements are settled.** The `Interviewing` → `Ready for design` flip is a
   mechanical unit routed to `tico` (U0); design proceeds in parallel, not behind it.
2. **Two landings**, not one push:
   - **Landing 1** — FR-1..FR-6, FR-11..FR-15, FR-20: the two config files, the single
     resolver seam, per-kind defaults, per-model settings, env-var cutover. Shippable and
     demonstrable on its own.
   - **Landing 2** — FR-7..FR-10, FR-16..FR-19: roles, ordered fallback chains, workspace
     override + precedence, resolved-model trace recording, embedding-dimension guard.
3. **No cloud API key available.** AC-2 (`{env:...}` secret substitution against a real
   hosted provider) and AC-3 (three provider kinds end-to-end) are **deferred / model-gated**
   for the acceptance pass — the design must still support them, and QA records them
   structurally verified rather than end-to-end, as K-025 handled its gated ACs.
4. **Milestone M4**, backlog item **K-042**.

## Environment facts established at decomposition

- The stakeholder's real `~/.config/opencode/opencode.json` declares **one** provider:
  `lmstudio`, `baseURL: http://192.168.0.69:1234` (LAN IP, **no `/v1` suffix**), one model,
  no `{env:}` substitution currently in use, no cloud provider. This is the only shared-file
  sample that exists; a second OpenAI-compatible LAN host does not exist today.
- FR-20 blast radius (sites setting the replaced env vars):
  `server/falkorchat/config.py`, `server/falkorchat/app.py`, `scripts/start_server.sh`,
  `compose.yaml`, `README.md`, `server/tests/test_workflow_live.py`, plus docs.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U0 | `tico` | `ae8daf5e6fb321743` | accepted | requirements `Status:` → `Ready for design` | none (mechanical) |
| U1 | `architect` | `a6607eea81e284553` | delivered | `docs/plans/llm-provider-config.md` + K-042/M4 backlog entry | `analyst` → — |
| U2 | `graph-dba` | `a9469ba9c6b47c56b` | delivered | `docs/plans/llm-provider-config-graph.md` | `analyst` (with U1) → — |
| U3 | `analyst` | `a87afc398f73067b8` | in-flight | `docs/reviews/llm-provider-config.md` | — |
| U4+ | TBD | — | queued | Landing 1 implementation (from U1's sequencing) | `analyst` re-gate |
| U5+ | TBD | — | queued | Landing 2 implementation | `analyst` re-gate |
| U6 | `devops` | — | queued | env-var cutover + secret hygiene (FR-12/FR-20) | `analyst` |
| U7 | `qa-engineer` | — | queued | `docs/test-plans/llm-provider-config.md` + `-report.md` | — |

Unit shapes for U4..U7 are provisional — U1's plan sets the real sequencing.

## Documentation impact (scanned at decomposition)

Every item below is a done-condition on the unit that invalidates it, not a cleanup pass:

| Document | Why it changes | Rides with |
|---|---|---|
| `falkor-chat/docs/DESIGN.md` §1.3 (M2 stack), §14 (config seam) | the single-model stack and the env-var config seam are replaced | U1 (design), implementers (as built) |
| `falkor-chat/AGENTS.md` | key-scripts env table + M1-server run instructions reference the old vars | U6 |
| `falkor-chat/README.md` | documents the replaced env vars | U6 |
| `falkor-chat/scripts/start_server.sh` (header comment + body) | sets the replaced env vars | U6 |
| `falkor-chat/compose.yaml` | sets the replaced env vars | U6 |
| `falkor-chat/docs/BACKLOG.md` | K-042 + M4 milestone row | U1 |
| `falkor-chat/docs/HISTORY.md` | one entry per delivered landing | each landing's closing unit |
| `falkor-chat/docs/manuals/` | admin-facing "how to configure models" is a plausible new manual — decide at Landing 1 close | flagged for `tico`, gated `qa-engineer` + `analyst` |

## Milestone close

At M4 close, every document this coordination froze flips to `Status: archived`, routed by
kind per root `AGENTS.md` (plans → `architect`, `-graph` → `graph-dba`, reviews → `analyst`,
requirements/manuals → `tico`, test-plans/reports → `qa-engineer`, this coordination → `teco`).

## Findings re-verified by `teco` at integration

The plan rests on four live claims. Each was independently re-checked, not accepted on report:

| Claim | Verdict |
|---|---|
| A missing `/v1` prefix fails **silently** | **Confirmed live.** `POST localhost:1234/chat/completions` → HTTP **200** with body `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}`. Surfaces as `KeyError: 'choices'`. |
| `TraceEvent`s are debug-only, so FR-8 cannot live on them | **Confirmed.** `executor.py:390` — `tracer = self._tracer if run["trace"] else _NULL_TRACER`. (Plan cites `:388`; 2-line drift, mechanism exact.) |
| `_urllib_transport` is duplicated with no timeout | **Confirmed.** Two copies: `llm.py:77`, `embedding.py:37`. |
| The stakeholder's declared endpoint is unreachable | **Confirmed.** `192.168.0.69:1234/v1/models` → connection failure (HTTP 000); `localhost:1234/v1` answers. |
| "37 `llm=` / 23 `guard_judge=` injection sites" | **Off by one each** — my count is 38 / 24. Rationale detail, not load-bearing; no action. |

## Stakeholder decisions taken mid-flight

- **2026-08-10 — FR-10's "suspends" ⇒ `failed`-with-cause. SETTLED, stakeholder's words: "failed is
  fine, go with that."** `architect`'s reasoning stands: `waiting` means "a human can unblock this",
  which an unresolvable model is not. The requirements doc's FR-10 wording ("the run suspends") is
  now narrower than the implemented behaviour — **`tico` should reconcile FR-10's text** at the next
  requirements touch so the document does not read as contradicting the build.

## Open questions still outstanding

- **The `{kind|"*"}` wildcard for the workspace override (FR-16).** FR-16's literal text says a
  workspace may override "everything running in it"; `graph-dba` rejected a true blanket because it
  would force the embedding worker onto a chat model and break FR-19 by construction, and proceeded
  with per-kind overrides. Routed to the `analyst` gate to rule on whether per-kind is a faithful
  reading or a silent narrowing. Stakeholder call only if `analyst` says it is a narrowing.
- **`StepRun.model` vs `StepRun.resolvedModel`** — the two design documents diverge on this one
  name. At the `analyst` gate to settle before an implementer sees it.

## Log

- **2026-08-10** — Opened. Requirements read at `f78b824`; working tree clean against it
  (the reported modification was a stale mtime, `git diff` empty). U0/U1/U2 dispatched.
- **2026-08-10** — U0 accepted, committed `57ac4ec`. Diff verified: exactly the four intended
  edits, no requirement prose touched.
- **2026-08-10** — U1 delivered and committed `4be53e6` (plan + M4 row + K-042 item; BACKLOG diff
  is 71 insertions / 0 deletions, so nothing existing was disturbed). Claims re-verified as above.
  Awaiting the `analyst` gate, which is held until U2 lands so both documents are reviewed together.
- **2026-08-10** — U1's debug-only-trace finding was sent to the **in-flight** U2 agent, because
  U2 was dispatched before the finding existed and its brief had pointed it at the execution trace
  generically. Also pointed U2 at the plan's §8 interface items. Correcting a running agent's
  premise beat discarding its work.
