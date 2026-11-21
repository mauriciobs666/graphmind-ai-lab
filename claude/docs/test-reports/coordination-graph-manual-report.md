# coordination-graph manual walkthrough: test report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** coordination-graph (S9 manual)

Target: `claude/docs/manuals/agent-team.md`, section "Asking 'what's going on?' across the whole repo", the two FAQ entries about the mirror/hook, and the "Learnings store" paragraph (uncommitted working tree, 2026-11-21). Spec = the manual's own claims; no separate plan (short manual, short report).

CPG: not applicable — a user-manual walkthrough, no code-level change under test.

**Verdict: pass with one medium defect (D-1) and three low/minor notes.** Every command, output shape, exit code and recovery text the manual claims was observed, except the two items that could not be run (Mermaid parser, `Hook` firing, both listed under Blocked).

## Method

- Read-only against the shared mirror (`--confirm coordination`): `inflight`, `inflight --waiting`, `unit`, `planned`, `planned --all`, `propose`, `drift`, `kb-status`. Invoked as `PYTHONPATH=coordination-graph cypher-mcp/.venv/bin/python -m coordgraph …` from the repo root. Never ran `sync`, `rebuild`, `bootstrap`, `kb-*` writers, `hook-sync` or `retry-failed` against the shared instance.
- Write-side walkthroughs ran on an ephemeral `falkordb/falkordb:v4.18.11` container (127.0.0.1, random port 47870, `COORDGRAPH_EPHEMERAL_PORT` set, no volume) with `COORDGRAPH_REPO_ROOT` pointing at a scratch repo holding 4 copied files (`coordination-graph-coordination.md`, `identifier-leak-cleanup-coordination.md`, the review `coordination-graph.md`, `falkor-chat/docs/BACKLOG.md`). The container was stopped (`docker ps -a` shows 0 left). Scratch repo: `<session scratchpad>`.
- The hook was not registered and no settings file was touched.

## Results

| # | Manual step / claim | Observed | Result |
|---|---|---|---|
| 1 | `coordination-graph/coordgraph.sh <sub> --confirm coordination` (every table row) | `coordgraph.sh: no venv at …/coordination-graph/.venv — run coordination-graph/setup.sh first.` exit 1. The `.venv` does not exist in this checkout; the manual never mentions `setup.sh`. The `python -m coordgraph` form works. | **FAIL (D-1)** |
| 2 | `inflight` (open coordinations, verdicts, requirements) | Header + "Units in open coordinations", "Waiting on the stakeholder (0)", "Open 'needs changes' verdicts (14)", "Requirements not yet picked up (9)". exit 0. | pass |
| 3 | `inflight --waiting` and the "nothing waiting" FAQ | Prints `Waiting on the stakeholder (0)` / `(none)`, exit 0. Drift header listed `falkor-chat/docs/plans/live-spec-defects-coordination.md` with a `⚠ drift` row, so the FAQ advice ("check `Drift:` first, then open the ledger") is exactly the right move. On the scratch mirror a `paused` unit row appeared as `Waiting on the stakeholder (1)` after `sync`, and was absent (stale) before it: confirms the FAQ. | pass |
| 4 | `unit <slug>` / `unit <slug> U4` / `agent: —` | Whole-ledger block list; `U4 … status: gated, agent: a4a4…, deliverable: …`; `U1 … agent: —`. | pass |
| 5 | `unit coordination-graph S9` prints "no unit matches …" and exits 1 | `no unit matches coordination-graph S9`, exit 1. A bogus slug: same message, exit 1. | pass |
| 6 | `planned falkor-chat`: header line, "48 items", 🔵/🟡, two `-` columns, title, file | `Backlog falkor-chat/docs/BACKLOG.md (48 items)`, `Planned (proposed / in progress) (48)`, rows `<id> 🔵 proposed - - <title> <file>`; `falkor-chat:K-079 🟡 in-progress`. Matches. | pass |
| 7 | `planned … --all` includes deferred/rejected | Extra section `Deferred / rejected (a record, not planned work) (0)`; none exist today. Unknown scope: lists known scopes, exit 1. | pass |
| 8 | `propose`: "as recorded", "Carried — in progress", "Not plannable yet … missing Priority, Depends on, Touches", `--per-milestone N` | All three sections present with the stated reason text; `--per-milestone 2` accepted, exit 0. | pass |
| 9 | `kb-status`: "17 rows, all `degraded`, none `failed-doc`", row shape, exit 0 | `KB mirror: last sync 2026-11-21T12:16:25Z` + 17 rows `degraded <file> § <section> <id>: extraction failed, still searchable`; exit 0; no `failed-doc`/`stale`/`unmirrored`. | pass |
| 10 | Sync header: three lines, `Hook: active\|disabled`, drift states, `file wins` | Exactly three header lines; drift lines read `changed\|unsynced\|gone  <path>  → file wins; run coordgraph sync`; `⚠ drift` row marks. States `changed`, `unsynced`, `gone` all observed; `stale-parser` not exercised. | pass |
| 11 | `drift` exits non-zero when it finds drift; `Drift: none` after sync | Shared: `Drift: 16 source(s)`, exit 1 (16 files edited in the uncommitted tree). Ephemeral: exit 1 with drift, exit 0 and `Drift: none` after sync. | pass |
| 12 | `sync --changed`, `sync <path>`, `sync --all`; only `--all` drops gone files | Empty mirror: `Synced: never – never (0 sources)`, 4 `unsynced`. `sync --changed` → 4 synced, `Drift: none`. Edit → `changed` + `⚠ drift` on `#U1`; `sync <path>` → none. Delete file → `gone`; `sync --changed` leaves it `gone`; `sync --all` prints `deleted … file is gone` and `Drift: none`. | pass |
| 13 | Kill switch: `touch coordination-graph/.state/disabled` → `Hook: disabled`; `rm` → active; `COORDGRAPH_HOOK_DISABLED=1` | Touch → `disabled`; rm → `active`; env `1` and `yes` → `disabled`, `0` → `active`. The hook script with the switch present exits 0 silently. | pass |
| 14 | "this update is not complete" recovery: run `sync <file> --confirm coordination`, then `drift` | With the DB port closed, `hook-sync <file>` printed `…not synced: FalkorDB unreachable at …` and `this update is not complete — say so; it is retried automatically, or run coordgraph sync <file> once FalkorDB is back`, exit 2, queued the file in `.state/failed.json` (attempts, `nextRetryAt` = growing delay). `sync <file>` then `drift` → `Drift: none`. | pass |
| 15 | "refused by the guard" is a config fault, not retried | With `COORDGRAPH_EPHEMERAL_PORT` set plus `--confirm`: `refused by the guard: refusing shared access while COORDGRAPH_EPHEMERAL_PORT is set … nothing is queued or retried: fix the configuration, or set COORDGRAPH_HOOK_DISABLED=1 …`, exit 2. | pass |
| 16 | FAQ "refused or can't reach the database": missing `--confirm` | `inflight: refusing shared FalkorDB port 6379`, exit 1. `kb-status` same family. | pass |
| 17 | Wrong `--confirm` | `refusing shared FalkorDB localhost:6379: --confirm / COORDGRAPH_SHARED_CONFIRM must equal the graph key 'coordination', got 'foo'`, exit 1. | pass |
| 18 | `COORDGRAPH_SHARED_CONFIRM=coordination` as the standing form | `drift` worked without the flag, exit 1 (drift). | pass |
| 19 | FalkorDB unreachable | Shared mode: all three shared ports are live here, so a truly closed shared port could not be produced (a shared-port point-at-3000 returned `Protocol Error: HTTP/1.1 400`, exit 1; 6390 returned `NOAUTH`, exit 1, both clean one-line errors). Ephemeral mode, closed port: `Error 111 … Connection refused`, exit 1. | pass (partial) |
| 20 | Hook honesty: "not registered", "where it fires is unverified", header `active` ≠ installed | Manual states all three; nothing in the working tree registers the hook (`.claude/settings*.json` untouched); `Hook: active` appears with no hook registered. The manual's wording is honest. | pass |
| 21 | Mermaid 1 (`flowchart LR`: files → sync → mirror → views; dashed "edited since last sync" → drift) | Syntax read by eye: valid node/edge/label forms. Behaviour agrees with step 11/12. Parser run **blocked** (see below). | pass (static) |
| 22 | Mermaid 2 (`flowchart TD`: edit → hook registered and not disabled? → no: drift until sync / yes: copy → ok / could not copy: agent warned + retry) | Syntax valid by eye. The "could not copy" branch agrees with step 14; the "no" branch agrees with steps 12/13. | pass (static) |

## Defects

**D-1 (medium): the headline command form fails on a fresh checkout and the manual does not say why.** Every table row uses `coordination-graph/coordgraph.sh …`. That script requires `coordination-graph/.venv` (created by `coordination-graph/setup.sh`); here it is missing, so each command exits 1 with `no venv … run coordination-graph/setup.sh first`. A stakeholder who follows the manual literally gets nothing. Repro: from the repo root, `coordination-graph/coordgraph.sh drift --confirm coordination`; exit 1. Expected per manual: the answer. Fix options (for `tico`): add one line "first time only: `coordination-graph/setup.sh`", or tell the reader to ask `teco`/`devops` if the command says there is no venv. The `python -m coordgraph` form that the brief gave works and produced every result above.

**N-1 (low): rows the manual does not describe.** A waiting row reads `<component>  <ledger>  <unit>  asked <date|unknown>  <title>`; "asked unknown" appeared for a unit with no recorded date. The manual describes waiting questions but never shows a row, so the reader meets this format cold. Optional: one example line.

**N-2 (low): `planned` example vs real output.** The example shows only `falkor-chat:K-079` / `K-082`; the real list starts with `GAL-I02-K0002`-style IDs (new work-item form) before the `falkor-chat:K-…` ones. The example is labelled trimmed, so not wrong, but a reader may wonder what `GAL-…` is.

**N-3 (info): the shared mirror is 16 sources behind today.** `drift` on the shared copy lists 16 changed or unsynced sources (uncommitted edits including the coordination-graph ledger and review that this manual is about). The manual explains this correctly (`file wins`); this is a data-freshness note, not a manual defect. A stakeholder reading `inflight` today sees the `coordination-graph-coordination.md` ledger unit statuses from the last sync.

## Blocked / not verified

- **Mermaid parser:** no `mmdc`/Linux node in this environment, and installing `mermaid` through the Windows `npm` was refused by the permission classifier (not worked around). Diagrams were checked statically only; recommend rendering both once in any Mermaid viewer.
- **Hook firing** (`PostToolUse`): by design not registered and out of scope; the full `post-tool-sync.sh` run against the scratch repo exited 0 silently (inconclusive: no stamp/changed-file detection exercised). `hook-sync` was exercised directly (steps 14, 15).
- `stale-parser` drift state and a genuinely closed *shared* port: not producible without mutating shared state.
- The sentence "edits made meanwhile are copied on the next hook call" (kill-switch removal) depends on hook firing; not testable here.

## Feedback

The manual's behavioural claims are accurate to the tool; the section is honest about the hook. The only real reader-facing break is D-1. Exit-code statements (`drift`, `unit`, `kb-status`) all held.

## Artifacts

- This report: `claude/docs/test-reports/coordination-graph-manual-report.md`. Nothing else in the repo was changed; the ephemeral container is gone.
