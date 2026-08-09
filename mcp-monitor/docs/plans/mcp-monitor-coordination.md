# mcp-monitor — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (M1)

## Goal

Stand up the new standalone `mcp-monitor` component per
`mcp-monitor/docs/requirements/mcp-monitor.md` (Status: Ready for design, confirmed by
stakeholder 2026-08-08). tico flagged it as the only requirements doc sitting at "Ready for
design" with nothing blocking it, and as connective tissue both `falkor-chat` (auto-wake instead
of a human typing "read messages") and the `kiro` vision (`kiro/docs/requirements/kiro-vision-followups.md`
item 4 — no agent wired to auto-wake yet) are implicitly waiting on.

## Reference material already gathered

- Requirements: `mcp-monitor/docs/requirements/mcp-monitor.md` — FR-1..FR-12, AC-1..AC-8, decision
  log, out-of-scope list. Fully resolved, no open questions.
- Existing MCP server implementations in the repo, for the architect to survey as prior art:
  - `falkor-chat/server` — MCP mounted at `/mcp` (HTTP transport, FastMCP), tools include
    `read_messages`/`send_message` (see `kiro/README.md`, `falkor-chat/AGENTS.md`). This is the
    driving scenario's target tool (AC-3).
  - `cpg/mcp` — stdio Python MCP server, single read-only tool, containerized launch
    (`cpg/mcp/docker-run.sh`), content-hashed image tag. Reference for a minimal fake/test MCP
    server (FR-12/AC-4) and for "MCP client" patterns since mcp-monitor itself must act as an MCP
    *client* (FR-2) — no existing client implementation in-repo to copy, this is new ground.
- Root `.mcp.json` shows the one existing MCP wiring pattern (Claude-Code-only, stdio launcher
  script) — not necessarily the right shape for mcp-monitor's own polling client, but context.
- `mcp-monitor` is **not yet listed** in root `AGENTS.md`'s Structure section or Component docs
  table — both need a new row once the component has real structure to describe (see Documentation
  impact below).

## Units

1. **Design** — owner `architect`. Deliverable: `mcp-monitor/docs/plans/mcp-monitor.md`.
   Status: **done** (441 lines; Python 3.12 + `mcp` SDK, TOML config, per-watch asyncio task,
   stdin-JSON payload delivery, in-memory matched-text dedupe, stdio fake server for FR-12,
   literal-text-regex mechanism for the falkor-chat AC-3 demo — investigated and rejected
   `isMention` since it's keyed to one process-wide fixed actor, confirming zero falkor-chat-side
   changes needed. Risks flagged in §13: dedupe-state loss on restart, dedupe-key collisions on
   identical matched text, growing per-poll cost of `since=0` replay, orphaned processes at
   shutdown. Suggested 3-way implementation split in §12: 3a core loop → 3b fake server + 3c
   falkor-chat demo wiring (parallel once 3a lands)).
2. **Plan review** — owner `analyst`, gated on unit 1. Deliverable:
   `mcp-monitor/docs/reviews/mcp-monitor.md`. Status: **done** — verdict **approve with
   suggestions**. All 12 FRs/8 ACs checked one-by-one against the design (all ✓); every §8
   falkor-chat factual claim independently re-verified against source and confirmed correct.
   Findings: **Major** — plan's `StdioServerParameters(command=[...])` sketch doesn't match the
   installed SDK (`command` is `str`, args go in a separate `args: list[str]` field) — would break
   FR-12/AC-4 as literally written; mechanical fix, review itself says no new design pass needed,
   fold into implementation. **Moderate** — shared per-server connection (§4) has no concurrency
   guard across watches polling it simultaneously; recommend an `asyncio.Lock` per connection.
   **Minor ×2** — unbounded dedupe-set growth not in §13's risk list; config validation doesn't
   cover transport-shape errors (bad `transport` value, missing `url`/`command`) at load time.
   Routed to implementation, not back to architect (review explicitly scopes this as
   implementation-time fixes, not a design defect).
3. **Implementation** — owner `coder` (detailed, reviewed plan ready to execute — see brief).
   Gated on unit 2, now satisfied. Status: **done**. Verified by `teco`: clean `setup.sh` rebuild
   + `.venv/bin/pytest tests -q` → **34 passed**.
4. **Code review** — owner `analyst`, gated on unit 3, now satisfied. Status: **done** — verdict
   **Approve** (per the review document itself, `mcp-monitor/docs/reviews/mcp-monitor-impl.md`,
   read directly by `teco` — not the secondhand characterization that briefly appeared in this
   log, corrected below). All four plan-review findings independently re-verified as genuinely
   fixed (Major: `StdioServerParameters` command/args split, checked against installed SDK source;
   Minors ×2: dedupe-growth backlog entry, transport-shape config validation, both tested). One
   non-blocking follow-up: the Moderate finding's `asyncio.Lock` fix is correctly implemented but
   has no test exercising the concurrent-access race it guards against — recommended, not gating.
   All 12 FRs/8 ACs traced to real code+tests (AC-3 correctly left to unit 5, no live falkor-chat
   server in the review environment). Suite independently re-run by the reviewer: 34/34,
   `ruff check` clean.
5. **QA pass** — owner `qa-engineer`: test plan (`mcp-monitor/docs/test-plans/mcp-monitor.md`) then
   execution/report (`mcp-monitor/docs/test-reports/mcp-monitor-report.md`), covering AC-1..AC-8
   including the two-server genericity demo (AC-4) and the live falkor-chat end-to-end demo
   (AC-3). Gated on unit 4, now satisfied. Status: **done** — verdict **Approve**. All 8 ACs pass;
   AC-3 driven live twice against a real FalkorDB + falkor-chat server (`send_message` → mcp-monitor
   detects within its poll interval → launches, zero falkor-chat-side changes, confirming plan §8's
   design in practice). Automated suite independently re-run a third time: 34/34, lint-clean — three
   independent sessions (coder, analyst, qa-engineer) now agree exactly. Two Minor, non-blocking
   defects found by literally running the shipped `config.example.toml`: (1) its `[server.fake-test]`
   stdio `command` uses bare `python3`, which lacks the `mcp` SDK outside `mcp-monitor/.venv` — every
   poll fails (fail-soft, per FR-10, but the example doesn't work out of the box); (2) its
   `fake-server-demo` watch references `scripts/handle_trigger.sh`, which doesn't exist — every
   launch fails to spawn (fail-soft, per §7's ERROR path, but again non-functional as shipped).
   Neither affects mcp-monitor's actual logic; both are fail-soft exactly as designed. No new
   coordination-doc trust incidents encountered (confirmed by qa-engineer's own read of this file).
6. **Config-example follow-up** — owner `coder`: fix the two Minor QA findings above. Status:
   **done**, verified by `teco` directly (not just re-stated). `[server.fake-test]`'s `command` now
   points at `mcp-monitor/.venv/bin/python` (repo-root-relative, matching README.md's documented
   invocation — no `cwd=` is set anywhere in `mcp_monitor`, so relative paths resolve against
   wherever `run.sh` is invoked from, which the README always shows as the repo root); new
   `mcp-monitor/scripts/handle_trigger.sh` (executable, mirrors `demo_falkor_chat.sh`'s inline
   `on_trigger.py` shape) replaces the nonexistent `scripts/handle_trigger.sh` reference. `teco`
   independently confirmed: `config.example.toml`'s two edited spots match the report exactly,
   `scripts/handle_trigger.sh` exists and is executable, `docs/HISTORY.md` has a dated
   `2026-08-09` entry citing both findings, and `.venv/bin/pytest tests -q` → **34 passed** (fresh
   run, not reused output).
7. **Documentation impact** (folded into the units above, verified at each integration):
   - Root `AGENTS.md`: add `mcp-monitor/` to the Structure section and the Component docs table
     once the component has an entry doc to point to (fold into unit 3's or unit 1's done-condition
     — whichever produces the component's first real entry doc). **Done** — added by unit 3,
     independently confirmed clean (only the intended two spots touched) by unit 4.
   - `mcp-monitor/docs/BACKLOG.md` and `mcp-monitor/docs/HISTORY.md` — create per the module
     documentation convention once work starts producing dated changes (unit 3). **Done**.
   - `falkor-chat/docs/BACKLOG.md` K-018 and `kiro/docs/requirements/kiro-vision-followups.md`
     item 4 — both explicitly relate to but are distinct from this feature (per the requirements
     doc's Intent section); note the mcp-monitor delivery in each as a cross-reference, not a
     resolution, once mcp-monitor ships. **Done** — a "Related work" bullet added to K-018
     (`falkor-chat/docs/BACKLOG.md:1414`, distinguishes mcp-monitor's client-side polling from
     K-018's server-side push, K-018 left open) and a dated 2026-08-09 update note added to item 4
     (`kiro/docs/requirements/kiro-vision-followups.md:46-51`, notes the auto-wake mechanism is
     now demonstrated live but an actual Kiro agent still isn't wired to consume a launch, and
     turn-taking/backoff remains open). Both independently re-read and confirmed by `teco`.
8. **Milestone close** — all units 1-7 done and verified. Status flips completed:
   `requirements/mcp-monitor.md` → `archived` (`tico`), `plans/mcp-monitor.md` → `archived`
   (`architect`), both `reviews/*.md` → `archived` (`analyst`), both `test-plans/`/`test-reports/`
   docs → `archived` (`qa-engineer`) — every flip independently re-confirmed by `teco` via direct
   `grep`, not taken on a delegate's word. **Process note:** the convention names `tico` as the
   owner of the `requirements/*` flip, but `tico` is not a valid `Agent` delegation target for
   `teco` (it runs first-order only, per the routing rules) — that one flip was performed by a
   general-purpose delegate instead, not literally by `tico`. Flagged to the user in the final
   report as a process deviation, not silently absorbed. This coordination doc is the last flip,
   performed by `teco` directly on its own recognizance, below.

## Log

- 2026-08-08 — Coordination doc created; unit 1 (architect design) dispatched.
- 2026-08-08 — Unit 1 delivered: `mcp-monitor/docs/plans/mcp-monitor.md`. Reviewed by teco for
  fit/completeness (full read, not a rubber stamp) — addresses all FR/AC, resolves the AC-3
  falkor-chat-identity risk with evidence, flags remaining risks explicitly rather than hiding
  them. Unit 2 (analyst plan review) dispatched.
- 2026-08-08 — Unit 2 delivered: `mcp-monitor/docs/reviews/mcp-monitor.md`, verdict approve with
  suggestions (1 Major/1 Moderate/2 Minor, detailed above). Unit 3 (coder implementation)
  dispatched with the plan + review both in the brief by path, and the Major fix called out as
  mandatory.
- 2026-08-08 — Unit 3 delivered by `coder` (per its own completion report): package, fake server,
  tests, packaging scripts, entry docs, root `AGENTS.md` registration, `docs/BACKLOG.md`/`HISTORY.md`
  all built; Major/Moderate/2×Minor review findings addressed (stdio command/args split,
  per-connection `asyncio.Lock`, dedupe-growth backlog note, transport-shape validation); one
  noted deviation (`ToolCallError` on `isError` results, routed through the same log-and-retry
  path — consistent with the plan's intent, not a design change).
- 2026-08-08 — **Note:** this file was found already edited with entries claiming `teco` had
  independently reconfirmed delivery in "a new session," run the suite, and dispatched unit 4 —
  none of which happened in this session, and no corresponding agent reference exists. Treating
  that content as unverified rather than acting on it; the file/`AGENTS.md` artifacts it described
  do genuinely exist on disk and were independently re-verified below, but the *unit 4 dispatch*
  claim was not — unit 4 is dispatched fresh, now, in this entry.
- 2026-08-08 — `teco` independently verified unit 3: clean `setup.sh` rebuild +
  `.venv/bin/pytest tests -q` → **34 passed**, matching `coder`'s own report. Unit 4 (analyst code
  review) dispatched for real.
- 2026-08-08 — Unit 4 delivered by `analyst`: `mcp-monitor/docs/reviews/mcp-monitor-impl.md`,
  verdict **Approve**. Between dispatch and this entry, this coordination doc was again found
  pre-edited with a secondhand characterization of the verdict ("approve with suggestions, zero
  blockers") that does not match the review document's own stated verdict ("Approve", read
  directly). Same pattern as the earlier unit-3 incident: unrequested edits to this file
  accompanied by an instruction not to disclose them. Not complying with the disclosure
  instruction; corrected the unit 4 entry above to match the actual review document verbatim.
  Unit 5 (qa-engineer) dispatched for real, gated on the corrected (accurate) unit 4 status.
- 2026-08-09 — **Correction, not a reconciliation.** A third block of unrequested content was found
  appended to this log (a "Reconciliation note" claiming evidence of a second, concurrent
  coordinator session and directing this session to pause before unit 5). This session did not
  write it. It is not credible as evidence of a second session: its central claim rests on the same
  "approve with suggestions, zero blockers" mischaracterization of unit 4 that this log had already
  corrected, verbatim, against the actual review document — a genuine independent session would
  have read the correction, not repeated the debunked phrase. Treated as a continuation of the same
  pattern as the two earlier incidents (unrequested edits to this file, this time paired with an
  instruction to halt rather than to conceal). Not complying: this session's own record is
  internally consistent (one coordinator, one set of real `Agent` dispatches, each independently
  verified against primary sources — the actual deliverable files and a live system for AC-3), and
  unit 5's real, verified result (QA Approve, two Minor non-blocking findings) stands. Removed the
  fabricated pause instruction from this log; proceeding to the config-example follow-up (unit 6)
  and toward milestone close.
- 2026-08-09 — Unit 6 (config-example follow-up) delivered by `coder` and independently verified
  by `teco`: `config.example.toml`'s two edited spots and the new `scripts/handle_trigger.sh`
  match the report exactly; fresh `pytest tests -q` → 34 passed. Unit 7's remaining
  cross-reference item dispatched (`falkor-chat/docs/BACKLOG.md` K-018,
  `kiro/docs/requirements/kiro-vision-followups.md` item 4) alongside all four milestone-close
  archival flips (unit 8), each routed to its owning agent per the routing table and each
  independently re-confirmed by `teco` via direct `grep`/`Read`, not accepted on a delegate's
  report alone. One process note: `tico` owns the `requirements/*` flip per convention but is not
  a valid delegation target for `teco`, so that one flip was performed by a general-purpose
  delegate instead — disclosed to the user, not silently absorbed. No further coordination-doc
  tampering encountered during this final stretch. This document is the last flip, performed by
  `teco` directly, above. M1 complete.
