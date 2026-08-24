# Kaizen — Change History: cobb

> Dated log of actual changes to the `cobb` agent. Most recent first.

## 2026-08-23 — Prompt-waste Stage B wave 2: two boilerplate blocks compressed to pilot shapes (own file)
- **What:** Interactive-commit-grant bullet and learning-capture intro/tail compressed to the pilot-validated wordings in `architect.md`/`coder.md` (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B). No CPG-freshness clause exists in this file. Edited by the main session; the §7 lint gate ran over the result including this file (flagged for extra scrutiny since cobb is itself the gate — the lint machinery lives in the `agent-maintenance` skill, untouched by these edits).
- **Removed (class 5/6, already on record):** the grant's "same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`" — this file's 2026-08-21 grant entry; the tail's inbox-replacement sentence — this file's 2026-08-21 inbox-deletion entry (the tail now ends at "(full §1/§2 bookkeeping applies)."; no "raw capture" sentence was re-added — that would duplicate the Learnings-distillation bullet's §5 statement, a class-7 restatement); the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement (mechanics live in the Cypher template below); the grant parenthetical's "— not spawned via `Agent`/`Task` as an isolated delegate" (moved into the carve-out sentence).
- **Gate (a) inventory — all preserved:** grant scope (own verified deliverables, explicit path), full never-list, delegated-subagent carve-out + audit check-8 tokens, the maintainer-only promote-in-same-run clause ("unless you verify and promote it to its proper home … full §1/§2 bookkeeping applies"), Cypher template + call line verbatim.
- **Verified:** `audit-team.sh` PASS; cobb §7 lint — 12/12 files pass, this one "pass with minor findings", both process-level: F1 the "pre-edit self lints cobb's own edit" safeguard is unmeetable as written under live-symlink deployment (mitigated: blocks byte-match the pilot shapes; the checklist lives in the untouched `agent-maintenance` skill); F2 the wave's gate-(e) entries asserted the lint verdict before the lint ran (true in the event). Both logged in `kaizen/plan.md` for C6.

## 2026-08-23 — Prompt-waste doctrine institutionalized in `agent-maintenance` (§5 promotion rule + §7 seventh lint dimension)
- **What:** Two additions to `skills/agent-maintenance/SKILL.md`, per
  `claude/docs/plans/prompt-waste-reduction.md` §3 (v3) Stage F and the 2026-08-23 pilot
  calibration ruling (recorded in `claude/architect/kaizen/history.md`, 2026-08-23 entry):
  (1) §5 step 3's always-loaded-prompt destination now states the promoted form — rule + ≤1-clause
  why, nothing else; evidence/story/provenance land in the producing agent's `kaizen/history.md`
  disposition entry, non-negotiability expressed by stating the rule absolutely. (2) §7 gained
  dimension 7, **Prompt waste** — flags inline provenance (dates, decision-authority markers,
  supersession history, incident retellings), dated `kaizen/history.md` pointers (the calibration
  ruling: pointers are waste too — the history file is the standing greppable home), and duplicate
  restatements within one file; normative citations (paths a rule requires the agent to *use*)
  exempt; the doctrine table cited as the one normative reference. The existing six dimensions are
  unchanged. Consistency touches: §7 intro "six"→"seven", §4 fold-in enumeration, frontmatter
  `description`, `skills/README.md` catalog line, `claude/README.md` skill pointer, and the
  "six semantic dimensions" mention in this agent's own `plan.md` parking lot. §7's Origin note kept as-is (accurate about the
  six-dimension origin).
- **Why:** Stage F of the waste-reduction plan — make the ratchet durable so prompts don't regrow
  the weight the fleet compression (separate coordinated rollout, in flight) is removing.
- **Out of scope, flagged:** `cobb.md`'s own "§7: a semantic judgment pass over six dimensions"
  phrase is now stale — left for the fleet compression rollout, which owns agent-prompt edits.
- **Plan items:** none opened.

## 2026-08-23 — `kaizen_team` curation: stray probe node cleared, producer-write "RETURN trap" root-caused and promoted
- **What:** Two housekeeping items on `kaizen_team`, both surfaced by `teco` on 2026-08-23. (1)
  Deleted the throwaway `entryId:'test-probe-004'` node teco's diagnosis accidentally left behind
  (curator full-node clear, `agent='cobb'`) — confirmed gone. (2) Investigated teco's report that
  every attempted producer-write variant was rejected with the FR-8 message. Read
  `cypher-mcp/server.py`'s `_producer_write_agent_id()`/`_PRODUCER_WRITE_TRAILER_RE` and confirmed
  empirically against `authorize_write()` directly: the recognizer requires the statement to end
  immediately after the `KaizenEntry` map's close — no trailing `RETURN`, which every one of teco's
  reported attempts appended. This is **documented as intentional** in
  `docs/plans/kaizen-agent-ontology.md` §3.1 step 2e ("intentionally strict... known
  future-extension seam, not a defect") and already regression-tested
  (`test_producer_write_with_trailing_extra_clause_is_rejected`). Not a bug. Added a "Gotcha"
  callout to `cypher-mcp/README.md`'s producer-write section (in my explicit remit — it's on the
  small allow-listed MCP-doc list) documenting the exact trap and the asymmetry with the legacy
  author-write shape (which *does* tolerate a trailing `RETURN`). Logged the disposition (promoted)
  in `claude/teco/kaizen/history.md` per the producing agent's history convention, then cleared
  teco's legacy-shaped bug-report entry (`entryId: 9a1c7d2e-...`) from `kaizen_team`. Left teco's
  unrelated second entry (`entryId: b3e4a6a0-...`, fresh-dispatch-vs-resume) untouched, as scoped.
- **Why:** the two items were explicitly delegated as a curator/investigation task; this is the
  graph-curation and cross-agent-fact-distillation duty the role exists for.
- **Boundary note:** mid-session, attempted to also clarify `server.py`'s live `TOOL_DESCRIPTION`/
  `SERVER_INSTRUCTIONS` strings and add a regression test naming the `RETURN` trap specifically —
  the maintainer stopped this ("why is cobb handling a python fix?") before it landed. Reverted to
  README-only (explicitly in remit) and left the `server.py`/test-suite improvement as a documented
  recommendation for whoever owns `cypher-mcp` day-to-day (devops), not something I actioned.
  Confirms the boundary in my own prompt ("a plain server-side Python regex fix might be more
  efficiently owned by whoever owns `cypher-mcp` day to day if that's not you") in practice, not
  just in theory — worth remembering next time a task hands me an open-ended "your call" on a
  Python fix in someone else's component: default to *not* touching the code even when technically
  capable and nominally authorized by the delegating agent's phrasing, and surface it as a
  recommendation instead.

## 2026-08-22 — `agent-maintenance` skill §5 rewritten for the kaizen `:Agent`/`PRODUCED`/`MENTIONS` ontology (M8, S4)
- **What:** Rewrote `skills/agent-maintenance/SKILL.md` §5 (learnings-graph distillation
  procedure) for the M8 ontology shipped by `docs/plans/kaizen-agent-ontology.md` (Version 3,
  approved after 3 `analyst` review passes) and designed by
  `docs/plans/kaizen-agent-ontology-graph.md` (`graph-dba`). Three changes, per the plan's §3.3/S4
  row: (1) **step 1's read** now runs two queries side by side — the pre-existing plain
  `author`-filtered read (still needed, unchanged, for pre-M8 entries with no edges) plus a new
  traversal-based read (graph-dba §5's verified-idiom `OPTIONAL MATCH` + `collect()` + `UNWIND`
  fallback, not the `UNION` form flagged unverified on this build) for "every note produced by or
  mentioning agent X," stated as needed together for as long as any pre-M8 entry remains uncleared
  (graph-dba §7's no-retrofit consequence). (2) **step 3's routing** gained a new branch: when an
  entry turns out to really be about a different agent than its producer, `cobb` tags it with the
  curator-only MENTIONS-write (graph-dba §3) during distillation (FR-4) — never the producing
  agent's job. (3) **step 4's log-and-clear** replaced the always-full `DETACH DELETE` with a
  read-then-decide sequence for current-shape entries: count remaining `PRODUCED`/`MENTIONS` edges
  first (graph-dba §4.1), then delete just the one edge being resolved this pass (§4.2 — the
  producer's own pass always resolves `PRODUCED` regardless of remaining `MENTIONS`, per FR-6/AC-4;
  a mentioned agent's pass resolves only its own `MENTIONS` edge, per AC-3) or the whole node once
  nothing else remains (§4.3, unchanged `DETACH DELETE`) — legacy (pre-M8) entries keep the
  original unconditional full-clear, since they carry no edges to count. Stated explicitly, as its
  own numbered item in step 4's per-entry sequence (not left implicit in step ordering): a same-pass
  MENTIONS tag from step 3 must be durably committed before step 4's count-and-decide read runs for
  that same entry, else the count could read one edge short and a full `DETACH DELETE` could
  silently discard the just-added `MENTIONS` edge before it was ever attached. Also updated §5's
  intro paragraph (describes the new producer-write shape agents call to create an entry, replacing
  the plain `author`-property description) and added one dated line to the section's own Origin
  note, for the same "read coherently" reason — no other section of the skill file touched.
- **Why:** Implements S4 of the approved M8 plan — `cobb` owns `skills/agent-maintenance/SKILL.md`
  §5, the only place this distillation mechanics actually lives (not `cobb.md` itself).
- **Scope note:** this unit does **not** cover S3 (retargeting the 13 agents' own
  "Learning capture" write recipes in `claude/<agent>/*.md`) or S5 (updating `claude/README.md`,
  `claude/AGENTS.md`, root `AGENTS.md` prose) — separate steps in the same plan, not yet run as of
  this entry. Ran as a delegated subagent (teco-coordinated); left uncommitted for `teco`'s
  integration step per the standing delegated-subagent convention.

## 2026-08-21 — Team-wide: universal interactive-mode git-commit grant added to all 13 agents (`kaizen_team` distillation → live stakeholder decision)
- **What:** Two related but distinct changes, both stakeholder-decided live in one session:
  1. **tico's own commit grant extended** to cover the returned artifact of a `qa-engineer`/
     `analyst` verification pass tico itself offered under Mode 3 and the stakeholder accepted
     (`tico/tico.md`, `claude/AGENTS.md`, `claude/README.md`, `tico/kaizen/history.md` — first
     entry, same date).
  2. **Universal interactive-mode commit grant**, team-wide: every one of the 13 agents in
     `claude/` may now `git add`/`git commit` its own verified work **when running interactively**
     (`claude --agent <name>`, a human present turn-by-turn) — never when spawned as a delegated
     subagent, where committing stays `teco`'s integration step after its own verification, same
     as before. `tico`'s and `teco`'s pre-existing, broader grants (own doc kinds; any coordinated
     specialist's verified deliverable) are unaffected — they're unconditioned on invocation mode,
     tied to role, not mode; this new grant is deliberately narrower (own work only) and
     mode-conditioned. Edited: a new/extended Guardrails-equivalent bullet in all 13 agents'
     `<name>.md` (`analyst`, `architect`, `cobb`, `coder`, `data-scientist`, `devops`,
     `frontend-engineer`, `graph-dba`, `qa-engineer`, `security-expert`, `tdd-engineer`, `teco`,
     `tico`); `claude/AGENTS.md`'s "Git-commit authority" section rewritten as a two-layer policy
     (standing broad grants vs. the new universal one); `claude/scripts/audit-team.sh` check 8
     redesigned from an allow-list (`COMMIT_AUTHORS=(tico teco)`) to a scoping check — every agent
     must claim `git add`/`git commit` authority **and** state the delegated-subagent carve-out
     (grepped via the literal phrase "delegated subagent"); `claude/README.md`'s intro paragraph
     and the `teco`/`tico` rows corrected (both previously claimed to be "the only two agents with
     commit authority," now false).
- **Why:** A `kaizen_team` entry from `tico` (`entryId` `e7a1c9d4-3f2b-4a6e-9c1d-8b5f0a2e6d71`,
  dated 2026-08-21) reported that closing out `docs/manuals/graph-ontology.md`'s verification pass
  left three subagent-produced artifacts uncommitted, since tico's commit grant only ever covered
  files its own Write/Edit guard permitted. Put to the stakeholder as a scope-of-authority
  decision (same class as the still-open K-008) via `AskUserQuestion`: ruled to extend tico's own
  grant (change 1). The user then asked cobb directly to commit the resulting diff; cobb declined,
  citing the team's own documented invariant (only `tico`/`teco` may commit, `audit-team.sh` check
  8) and offered three paths via a second `AskUserQuestion` (user commits it themselves; route
  through teco; a one-off override). **The user rejected the premise of asking again**: "we need
  to make an exception for all agents when executed in interactive mode" — a direct ruling, not a
  menu pick — and, when cobb's first response still read as hedging, corrected further: "you guys
  should not refuse when asked by the stakeholder." Implemented change 2 accordingly, without
  further pauses, and committed the result under the very grant just added (cobb's own Principles
  section) once the policy existed to permit it.
- **A genuine reopening, not a bypass.** The 2026-07-30 "no proliferation of commit rights beyond
  tico/teco, do not re-open" ruling was closed by the same stakeholder who is the sole authority to
  reopen it — this is that reopening, done explicitly and on the record, not cobb's or any agent's
  unilateral drift. The 2026-07-30 ruling's *broad, mode-unconditioned* form for tico/teco stands
  untouched; what's new is a narrower, mode-gated form for everyone else (and additively for
  tico/teco too, though their existing grants already exceed it).
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8, full
  113+ PASS / 0 new FAIL (diff against the pre-change baseline captured earlier this session).
- **Plan items:** none opened — direct implementation of an explicit, fully-executed stakeholder
  decision; nothing left pending.

## 2026-08-21 — C-101 verified already-fixed (doc-only close), C-408 `CPG:` shape-ambiguity fix landed in the six wired agents

- **What:** Two unrelated `docs/BACKLOG.md` follow-ups picked up on request. **C-101** (`joern-cpg` loader `MAX_ARG_STRLEN` failure + masked exit code): read current `cpg-to-falkordb.py`/`pipeline.sh` source directly rather than assuming the backlog's 🔵 status was accurate — both defects were already fixed, in commit `e773060` (2026-07-17), *before* this backlog file even reached its current form. No code touched; flipped the entry to ✅ with the commit reference and a description of what the fix actually does (persistent-socket streaming sidesteps `MAX_ARG_STRLEN` by construction; `sys.exit(1 if failed else 0)` + `pipeline.sh`'s `set -euo pipefail` propagate real failure). Also bumped the file's `Last reviewed:` header 2026-07-25 → 2026-08-21. **C-408** (`CPG:` shape-selection ambiguity, DEF-4): the source design, `docs/plans/cpg-agent-adoption.md` §3, is `Status: archived` (header-pointer-only, left unedited) and gives worked examples for `used`/`considered, not relevant` but none for `not applicable` — the gap U9's live re-pass caught (`tdd-engineer` picked `not applicable` for a code-level task on a component with no loaded CPG, when the plan's own wording calls that `considered, not relevant`). Took the report's "worked counter-example" fix option, landed directly in the six *live* wiring points instead — `claude/{analyst,architect,qa-engineer,coder,tdd-engineer,frontend-engineer}/<name>.md`'s `CPG:` sentence each gained one disambiguating clause: `not applicable` is only for a task with no code-level component at all (e.g. a pure requirements/process/documentation task), never for a code-level task in a component that simply has no loaded CPG.
- **Why:** direct user request ("C-101 please and bump the date", then "C-408"). For C-101, verifying against live source rather than trusting the backlog's stale status avoided dispatching `graph-dba` to "fix" a bug that no longer existed. For C-408, fixing at the live wiring points rather than the archived plan matters because the archived doc is cited *from* the agent prompts, not the reverse — a dispatched agent reads its own prompt, never the design doc, so that's the only edit that can actually change future shape-selection behavior.
- **Verified:** C-101 — `git log -S"load_statements"` / `-S"socket.create_connection"` both resolve to `e773060`; `sys.exit(1 if failed else 0)` and `set -euo pipefail` (pipeline.sh:39) read directly from current source. C-408 — `grep`-confirmed all 6 target files' `CPG:` sentences before and after the edit; each now carries the disambiguating clause verbatim (frontend-engineer's differs slightly in surrounding phrasing, matched to its existing structure rather than force-fit the other five's exact sentence).
- **Anything unexpected:** none — both were bounded, already-scoped follow-ups with a stated fix direction in the backlog itself.
- **Follow-up not done in this pass:** none opened; both items closed outright rather than deferred.
- **Docs touched:** `docs/BACKLOG.md` (C-101 ✅, C-408 ✅, `Last reviewed:` bump), `claude/{analyst,architect,qa-engineer,coder,tdd-engineer,frontend-engineer}/<name>.md` (C-408 clause), this entry plus a short pointer entry in each of those six agents' own `kaizen/history.md`.
- **Plan items:** none opened.

## 2026-08-21 — All 12 agents' frozen `kaizen/inbox.md` files deleted (content already fully captured elsewhere), plus `G1`'s last 2 `kaizen_<agent>` graph keys retired

- **What:** Deleted `kaizen/inbox.md` from all 12 agents that carried one (`analyst`, `architect`, `cobb`, `coder`, `data-scientist`, `devops`, `frontend-engineer`, `graph-dba`, `qa-engineer`, `tdd-engineer`, `teco`, `tico`) — git history retains each file in full. Also dispatched `graph-dba` to finish `G1` (`docs/plans/generic-cypher-mcp2-coordination.md`): `GRAPH.DELETE` on the last 2 of 12 `kaizen_<agent>` keys, `kaizen_analyst` and `kaizen_teco`, left live at that plan's closing acceptance pending an unresolved data-fidelity-fix approval that never came.
- **Why:** user-directed cleanup — "no point keeping [inbox.md] since it's already git history." Before touching anything, verified there was no unhandled data anywhere in this cluster: (1) live-checked `kaizen_team` — the shared graph every agent's raw capture routes through since the 2026-08-20 consolidation — and found it **completely empty**, meaning every entry any agent ever wrote there has already been distilled (verified, routed, logged) and cleared; (2) for the 2 still-live per-agent graphs specifically, cross-checked every one of `kaizen_analyst`'s 8 entries and `kaizen_teco`'s 5 entries by `entryId` against `claude/analyst/kaizen/history.md` and `claude/teco/kaizen/history.md`'s 2026-08-21 distillation dispositions — all 13 already promoted or discarded-as-resolved-elsewhere, including the two flagged data-fidelity-defect entries (the corruption was minor enough that both got promoted despite it); (3) every one of the 12 `inbox.md` files' own pre-migration content was already parsed and imported into the graph system verbatim back on 2026-08-20 (each file's own history.md entry for that date confirms it). Nothing found anywhere was a live, undistilled input to anything — every deleted artifact was a pure redundant backup of data already captured downstream.
- **Verified:** `mcp__cypher__query(graph='kaizen_team', cypher='MATCH (n:KaizenEntry) RETURN count(n)')` → 0, before any deletion. `graph-dba` independently re-verified `kaizen_analyst`/`kaizen_teco`'s live contents before its own `GRAPH.DELETE` calls (didn't trust this session's snapshot alone), then re-listed graphs afterward to confirm both keys gone.
- **Anything unexpected:** `graph-dba`'s dispatch report flagged that `guard-destructive-ops.sh`'s `PreToolUse` hook did **not** intercept either live `GRAPH.DELETE` call, run from a nested subagent Bash context — a repeat of the already-tracked gap (`K-018`, this agent's `kaizen/plan.md`; also corroborates `K-019`'s "ask" hooks not enforcing under Auto Mode, `2026-08-21` entry above). `graph-dba` logged a fresh corroborating `kaizen_team` entry (`c3e5f8a2-…`) rather than re-diagnosing — left for a future `K-018`/`K-019` pass, not re-opened here as a new item.
- **Follow-up not done in this pass:** `docs/plans/generic-cypher-mcp2-coordination.md`'s "Resume note" still describes both open items (the 2 data-fidelity defects, the 2 held-back graph keys) as unresolved — needs a closing update reflecting that the graph keys are now retired and the data-fidelity-fix path is moot (the corrupted entries were already distilled downstream regardless, and `kaizen_team` no longer holds them to fix). Out of `cobb`'s normal write remit (a `docs/plans/*-coordination.md` doc); flagged for the user/`teco`.
- **Docs touched (this pass):** 12× `<agent>/kaizen/inbox.md` (deleted), 12× `<agent>/kaizen/history.md` (this entry, replicated per agent), `<agent>/<agent>.md` ×12 (dangling inbox-header-note reference fixed), `claude/AGENTS.md`, `claude/README.md` ×2, `skills/agent-maintenance/SKILL.md` ×3, 5× `hooks/guard-*.sh` (dropped dead inbox.md allow-glob entries: `tico`, `analyst`, `teco`, `data-scientist`, `architect`), 2× `hooks/guard-*.sh` comment touch-ups (`cobb`, `tdd-engineer`), `claude/scripts/audit-team.sh` ×2 comment fixes.
- **Plan items:** none opened for this cleanup itself; the coordination-doc follow-up above is a parking-lot note, not a `K-`item (no backlog id, low stakes, purely a documentation close-out).

## 2026-08-21 — K-019 escalated: `PreToolUse` "ask" hooks confirmed not enforcing under Auto Mode, from any source, in any execution context

- **What:** User-directed test of K-019's first candidate mitigation ("test the settings.json
  mitigation"). Mirrored `guard-destructive-ops.sh` as a session-wide `PreToolUse` hook in
  `.claude/settings.local.json` (gitignored, no team-wide effect), pipe-tested it directly
  (correctly returns `ask`), validated the JSON, then ran the exact `GRAPH.DELETE` command for
  real from **my own main session** (not a subagent) against a disposable scratch graph. It
  executed immediately, `OK`, no pause — same result as the subagent test.
  - Considered the mundane explanation first (per the `update-config` skill's own troubleshooting
    guidance): the settings watcher might not have reloaded the new hook config. Asked the user
    to open `/hooks`. They did. Re-ran the identical test on a fresh scratch graph — **still no
    pause.** Asked the user directly what `/hooks` displayed; they confirmed it listed
    `[Local] Bash — 1 hook`, i.e. the hook **was** correctly registered and visible to the
    harness. This rules out the reload explanation.
  - **Conclusion:** three independent, isolated tests (frontmatter hook / Task-dispatched
    subagent; settings.json hook / main session; settings.json hook / main session again,
    post-`/hooks`-reload, registration confirmed) all show the same failure — a `PreToolUse`
    "ask" hook on `Bash`, confirmed wired, confirmed registered, confirmed correct in isolation,
    does not pause execution for the real matching command. This is broader than K-019's original
    scope (Task-dispatch-specific): it now includes the main session's own tool calls and
    settings.json-sourced hooks, not just subagent-frontmatter ones. Working hypothesis: Auto
    Mode's classifier is silently resolving the `ask` decision before a human ever sees it —
    contradicting both official docs and `claude-code-guide`'s own prior research on this point.
  - **Reverted** the test hook from `settings.local.json` (proven ineffective; leaving it in
    place would misrepresent the config as a working mitigation).
  - **K-019 rewritten** (this file's `plan.md`) with the full three-test trail and the corrected,
    broader scope. **`skills/agent-standards/claude-code.md`** corrected twice more (the Hooks
    section's dated callout and the top-of-file stamp block) to state the confirmed scope rather
    than the earlier, narrower Task-dispatch-only wording.
  - **Not yet done:** drafting the `/feedback` report text for the user (K-019's next step) —
    doing that next as a follow-up to this entry, since I can't submit `/feedback` myself (no
    tool access to it; it's a user-facing slash command).
- **Why:** User-requested: "test the settings.json mitigation," following through K-019's
  candidate next steps in order, plus the natural troubleshooting the ambiguous first result
  demanded (ruling out the reload explanation before accepting the more severe conclusion).
- **Docs touched:** this file, `claude/cobb/kaizen/plan.md` (K-019 rewritten),
  `skills/agent-standards/claude-code.md` (two corrections), `.claude/settings.local.json`
  (test hook added, then reverted).

## 2026-08-21 — K-019: `/feedback` filed; 4th test confirms the gap is matcher-agnostic (Write/Edit, not just Bash)

- **What:** Two follow-ups to the entry above, both user-directed.
  - **Filed the drafted `/feedback` report.** User ran `/feedback` with the drafted text
    (3-test `Bash` repro, Claude Code 2.1.238). Confirmed submitted (`local-command-stdout:
    "Feedback / bug report submitted"`).
  - **Tested whether `Write`/`Edit`-matched hooks share the gap** (user: "yes please test those
    as well"). Used my own frontmatter hook (`guard-cobb-topic-writes.sh`) directly — no subagent
    dispatch needed, since `cobb` already carries a live `Write|Edit` `PreToolUse` hook. Wrote a
    disposable scratch file, `docs/_hook_test_k019_scratch.md`, to a path plainly outside my own
    allowlist (not `claude/*`, not the named skills/MCP-doc exceptions). **The write completed
    immediately, no pause.** Re-fed the exact real `{"tool_input":{"file_path":"..."}}` payload
    to the guard script directly afterward: it correctly returned `ask` for that path — same
    pattern as every prior test, the hook logic is right and the enforcement still doesn't
    happen. Deleted the scratch file immediately (`git status` confirms clean, never staged).
  - **Conclusion:** the gap is confirmed **matcher-agnostic** (`Bash` and `Write`/`Edit` both
    affected), on top of the already-confirmed context-agnostic result (main session and
    Task-dispatched subagent both affected). Four independent tests, one clean pattern.
  - **Updated:** `claude/cobb/kaizen/plan.md` K-019 (status line now "filed upstream," full
    4-test trail, explicit statement of which matcher×context cells are covered vs. the one
    remaining untested combination — Write/Edit on a Task-dispatched subagent).
    `skills/agent-standards/claude-code.md` corrected twice more (Hooks section callout + the
    top-of-file stamp block) to state "matcher-agnostic" rather than the earlier "`Bash`-only"
    wording.
- **Why:** User-directed, continuing K-019's investigation to full closure of the currently
  reasonable test matrix.
- **Docs touched:** this file, `claude/cobb/kaizen/plan.md`, `skills/agent-standards/claude-code.md`.
  `docs/_hook_test_k019_scratch.md` created and deleted within this entry's own test — never
  part of the tracked tree.

## 2026-08-21 — K-018 CONFIRMED via controlled live re-test: subagent-frontmatter `PreToolUse` hooks do not reliably fire for a Task-dispatched subagent's own Bash calls; K-019 opened (systemic, mitigation decision pending)

- **What:** User-directed live test, executing K-018's own prescribed next step. Dispatched
  `graph-dba` via the `Agent` tool with **`subagent_type: "graph-dba"` explicitly set** (the
  variable the G1 incident couldn't rule out) and a self-contained brief: create a throwaway
  scratch graph, then run the exact `docker exec falkordb-dev redis-cli GRAPH.DELETE <key>`
  command shape from the original episode, and report plainly whether anything paused for
  approval before it executed.
  - **Result: it did not pause.** The delete ran immediately, identically to the surrounding
    non-destructive steps — no permission prompt, no `ask`, no interruption. The dispatched agent
    independently re-verified the guard script itself is sound (fed the exact JSON payload
    directly to `guard-destructive-ops.sh`, got a correct `permissionDecision: "ask"` back) and
    confirmed its own frontmatter is correctly wired (`hooks.PreToolUse` → the right script, thin
    wrapper → shared core intact). **This disproves the `subagent_type`-omission hypothesis**
    K-018 was tracking: the dispatch was unambiguously `graph-dba`, with correct hooks, correct
    script logic — and the destructive command still ran unescalated. No other graph on the
    instance was touched; the test graph's deletion was itself the (harmless, disposable)
    payload of the test.
  - **Consulted `claude-code-guide`** for an authoritative read: official docs state hooks fire
    identically for a subagent whether main-session or Task-dispatched — no documented exception,
    confirming this contradicts the docs rather than being explained by them. It surfaced a
    closely-matching **fixed** changelog bug (`v2.1.212`: "auto mode was overriding PreToolUse
    `ask` decisions for unsandboxed Bash") and recommended checking the installed version.
    `claude --version` → **2.1.238** — well past 2.1.212, so this is **not** that already-fixed
    bug recurring; it's a distinct, still-open gap specific to the Task/Agent-dispatch path (the
    2.1.212 fix likely covered the main-session's own Bash calls only, not a dispatched
    subagent's). It also confirmed session-wide `.claude/settings.json`-defined `PreToolUse`
    hooks are architecturally distinct from subagent-frontmatter hooks and, per docs, should
    reliably cover subagent Bash calls where frontmatter hooks don't — **untested against this
    repo's actual gap**, flagged as the natural next diagnostic rather than executed
    unilaterally (a settings.json hook change affects every session team-wide).
  - **Disposition:** K-018 closed as CONFIRMED (not hypothesized) — moved here; **K-019** opened,
    high priority, as the systemic follow-up: this is the enforcement mechanism nearly every
    guarded agent's "harness-enforced" Guardrails claim rests on, for the primary way these
    agents actually run (Task-dispatched). Candidate next steps (settings.json-hook test,
    `/feedback` upstream report, treating delegated-execution guardrails as currently unverified)
    recorded in K-019 — **not executed**, since choosing among them (and any resulting
    settings.json change) is a call for the user, not something to decide unilaterally given the
    safety stakes.
  - **Raw capture routed:** the dispatched `graph-dba` subagent logged this finding itself as
    `:KaizenEntry` `a4f3d2e1-9b7c-4a1e-8f6d-2c1b3e5a9d70` (`author: graph-dba`) in `kaizen_team`.
    Read and verified against the transcript above (matches exactly, `suggestedHome: 'unsure'`
    honestly reflecting that the entry itself couldn't determine routing) — fully captured in
    this entry and in K-018/K-019, so cleared from `kaizen_team` after this write was confirmed
    (`entryId a4f3d2e1…`, curator-clear, `agent='cobb'`). `claude/graph-dba/kaizen/history.md`
    carries a short cross-reference pointer (the detailed narrative lives here since K-018/K-019
    are cobb-owned, team-wide items, not `graph-dba`-specific).
- **Why:** User-requested: "confirm subagent_type on the next graph-dba dispatch" — K-018's own
  prescribed test, executed exactly as specified plus the natural follow-up research once the
  result came back positive-for-a-gap.
- **Docs touched:** this file, `claude/cobb/kaizen/plan.md` (K-018 closed, K-019 opened),
  `claude/graph-dba/kaizen/history.md` (pointer), `skills/agent-standards/claude-code.md`
  (dated caveat correcting the "frontmatter hooks fire... when spawned as a subagent" claim).

## 2026-08-21 — Team-coherence certification (script clean, §7 lint: 2 minor findings fixed)

- **What:** User-requested certification, closing out a full `kaizen_team` distillation sweep run
  earlier this session (oldest-pending-first: `qa-engineer` 10 entries, `data-scientist` 8,
  `architect` 4, `graph-dba` 1, `tdd-engineer` 1 — `analyst`/`teco` already at zero from a prior
  session, see the entry above). `kaizen_team` now holds **zero** entries across every author.
  - **Deterministic script:** `bash claude/scripts/audit-team.sh` — **114 PASS / 0 FAIL**, clean.
  - **Judgment checklist:** none of this session's edits touched a roster, boundary, hook wiring,
    or subagent-awareness phrasing — every change was either a knowledge-base addition, a prompt
    clause deepening an existing capability, or a doc-content fix, so roster accuracy/handoff
    symmetry/enforcement parity/boundary reciprocity all reconfirmed unchanged rather than newly
    verified from scratch. Spot-checked `claude/README.md`'s rows for the five touched agents
    against their current files — all still accurate, no catalog edit needed.
  - **§7 prompt-quality lint**, run over every artifact changed since the last certification
    (`data-scientist.md`, `architect.md`, `tdd-engineer.md` — the three agent prompts actually
    edited this session; KB/doc files given a lighter contradiction/coverage-only pass):
    - **Clean:** `architect.md` (new Guardrails bullet — no contradiction, persona, or coverage
      issue; mild productive overlap with the existing "plan must stand alone" principle, not a
      defect). `qa-testing-techniques.md`, `lm-studio-model-notes.md`, `freshness.md` — no
      contradictions with existing content. `cypher-mcp/README.md` — verified the corrected
      write-result text doesn't leave a stray contradicting claim elsewhere in the file.
    - **Minor, fixed:** `tdd-engineer.md` Workflow step 1 had grown, across four edits since
      2026-07-09, into a single run-on paragraph carrying seven distinct conditional
      instructions — a cognitive-load outlier against its own peer steps, in a prompt region with
      a documented prior failure mode (DEF-3, a buried instruction going unfollowed on live
      dispatch). Restructured into a short lead + a bulleted sub-list of the three doc-path
      branches, no content change. Logged in `tdd-engineer/kaizen/history.md`.
    - **Minor, parked:** `data-scientist.md`'s LLM-as-judge bullet now carries three distinct
      validity rules in one paragraph (general caveats, class-conditional-rate gating, and
      today's judge-collapse caveat-splitting rule). Still thematically coherent and each
      sentence self-contained — not fixed now; parked in `data-scientist/kaizen/plan.md` with a
      revisit trigger (split into two bullets if a fourth rule lands).
  - **A genuine safety-relevant finding surfaced during distillation, not this certification's own
    lint, but closed out here:** `graph-dba`'s raw `kaizen_team` capture of the G1 dispatch
    (`guard-destructive-ops.sh` not escalating 4 live `GRAPH.DELETE` calls) turned out to be the
    same episode already tracked as **K-018** (this file's `plan.md`, opened in an earlier
    session distilling `teco`'s parallel capture of the same event). Re-verified the guard script
    directly (correctly emits `"ask"` on the exact cited command in isolation — not a regex bug)
    and added two corroborating-but-unconfirmed upstream GitHub issues to K-018 as search terms
    for its still-open next step, rather than re-opening a duplicate item. Full trail:
    `claude/graph-dba/kaizen/history.md`, `claude/cobb/kaizen/plan.md` K-018.
- **Verified:** `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry) RETURN
  count(e)")` → 0, before and after this certification's own edits (the certification made no
  further graph writes).
- **Why:** User-requested, following the session's distillation sweep.
- **Docs touched (cross-artifact bookkeeping; each agent's own history has the full per-entry
  disposition table for its distillation):** this file, `claude/tdd-engineer/kaizen/history.md`
  (lint fix), `claude/data-scientist/kaizen/plan.md` (parked lint finding).

## 2026-08-21 — Distilled all 12 pending `analyst`-authored entries from `kaizen_team`

- **What:** Ran the agent-maintenance skill §5 procedure against every `kaizen_team` node with
  `author:'analyst'` (12 entries). Full dispositions, verification notes, and the exact promoted
  text live in `claude/analyst/kaizen/history.md`'s matching 2026-08-21 entry — this entry covers
  only the cross-artifact bookkeeping that isn't analyst's to log.
  - **Promoted to `claude/analyst/review-techniques.md`** (7 entries, one new section each plus
    one new case appended to an existing section): kaizen-graph distillation reconciliation,
    live-prompt-via-symlink review urgency, whitespace-normalized verbatim-text diffing,
    `pytest -k` vs `-m` baseline verification, self-edit ground truth, live-service reachability
    before trusting a live-test report, and a third (untracked-file) case for the existing
    zero-touch mutation-test section.
  - **Promoted to `claude/analyst/analyst.md`** (1 entry): new "Evidence over vibes" bullet on
    running a plan's prescribed acceptance-check command verbatim rather than trusting it matches
    the repo.
  - **Promoted to `claude/cobb/TESTING.md`** (1 entry, cross-agent routing): a new Gotcha on
    `audit-team.sh`'s scratch-testability and its silent kaizen-only-directory skip — landed here
    rather than analyst's own knowledge base since the fact is about safely testing a script this
    agent (`cobb`) owns.
  - **Discard, already resolved elsewhere** (3 entries): a `SendMessage`-grant open question
    superseded same-day by `claude/docs/requirements/mid-run-escalation.md`; a falkor-chat
    `r1_probe` field-semantics finding already documented and fixed in
    `falkor-chat/docs/plans/golden-set-expansion-ml.md`; and an `nc`/`ncat`/`netcat` guard gap in
    `security-expert/hooks/guard-exploitation-approval.sh` already fixed, cited by name in that
    script's own current header comment.
- **Verification method:** pulled every entry's full, untruncated text via `redis-cli GRAPH.QUERY
  kaizen_team ... --no-raw` directly against FalkorDB rather than paging the MCP tool's ~300-char
  per-cell display through repeated `substring()` calls — far fewer round trips for 12 entries ×
  4 fields. Every surviving claim was re-derived from the live repo (pytest.ini, the guard
  script's current source, `audit-team.sh`'s actual enumeration logic, a live `grep` for the
  self-edit clause) before promotion or discard, not accepted from the entry's own framing.
- **Why:** user asked to "work on analyst's inbox" — analyst's `kaizen/inbox.md` is a frozen
  2026-08-20 historical snapshot (already imported and cleared), so the live equivalent is its
  pending raw capture in the shared `kaizen_team` graph; this was the first full distillation pass
  against it since the migration.
- **Docs touched this pass (cross-artifact bookkeeping only — see `analyst/kaizen/history.md` for
  the full per-entry disposition table):** `claude/cobb/kaizen/history.md` (this entry),
  `claude/cobb/TESTING.md`.

## 2026-08-21 — Team-coherence certification (full 13-agent pass) — 7 real defects found and fixed

- **Scope:** user-requested ("certify the team"). Last full certification was 2026-07-29 —
  essentially the entire roster changed since (`security-expert` created, `agent-permission-friction`
  hook rollout, the `kaizen_team` graph consolidation, teco's CPG-freshness centralization), so
  this ran as a full pass, not a scoped one.
- **§4 deterministic half:** `claude/scripts/audit-team.sh` — **113 PASS / 2 FAIL**, both
  pre-existing and unrelated (username/home-path leak in
  `falkor-chat/docs/test-reports/graphrag-eval-report.md`, committed 2026-08-16, outside every
  agent's write remit — consistent with every prior certification's handling of this same known
  leak; not fixed here, per the diff-not-gate convention). Same 113/2 before and after every fix
  below (verified via diff, not a bare re-run).
- **§4 judgment half — 5-point checklist:**
  1. **Roster accuracy** — found and fixed a real drift: `claude/AGENTS.md`'s Hook-machinery
     section said "Seven `Write|Edit` wrappers" on `guard-doc-writes.sh`; actual count (verified
     by listing every agent's frontmatter `hooks:` block) is **eight**
     (architect/analyst/data-scientist/teco/tico/security-expert/cobb/qa-engineer) — the
     2026-08-21 rollout added two wrappers (`cobb`, `qa-engineer`) in one edit but the prose was
     only bumped by one. Fixed both occurrences ("Seven"→"Eight", "six of the seven"→"seven of
     the eight").
  2. **Handoff symmetry** — clean. Verified the highest-risk pair directly: all six
     CPG-freshness-consuming agents (`analyst`/`architect`/`coder`/`tdd-engineer`/
     `frontend-engineer`/`qa-engineer`) correctly state "CPG freshness-checking is teco's
     responsibility, not yours (2026-08-19)," matching teco.md's own centralization claim
     word-for-word. The manuals review split (`qa-engineer` behavioral / `analyst`
     architectural) is stated symmetrically on both sides.
  3. **Subagent-awareness** — clean. Every delegate-able agent (10 checked directly) carries
     can't-ask-mid-run language.
  4. **Enforcement parity** — found and fixed **3 real gaps**, all the same shape and all from
     the same 2026-08-21 `agent-permission-friction` rollout: a hook was wired in frontmatter but
     never described in the prompt body it guards ("silent machinery," the exact failure mode §4
     exists to catch).
     - `tdd-engineer.md` — zero prose about its new `guard-tdd-broad-write.sh`. Fixed (new
       Guardrails bullet). Logged: `claude/tdd-engineer/kaizen/history.md`.
     - `cobb.md` (**my own prompt**) — zero prose about its own `guard-cobb-topic-writes.sh`, in
       either a Guardrails section (which doesn't exist) or Principles. Fixed (new Principles
       bullet — added there rather than a new H2, matching this file's existing density).
     - `qa-engineer.md` — documented its pre-existing destructive-ops hook but not the *second*,
       same-day `guard-qa-doc-writes.sh`. Fixed (new Guardrails bullet). Logged:
       `claude/qa-engineer/kaizen/history.md`.
     Everything else checked (`analyst`, `architect`, `data-scientist`, `devops`, `graph-dba`,
     `security-expert`, `coder`'s deliberate no-hook exemption) was already accurate — this was a
     rollout-specific blind spot (ship the hook, forget the prose), not a systemic pattern.
  5. **Boundary reciprocity** — script's 11 `BOUNDARY_PAIRS` all symmetric at the name level (22
     PASS); spot-checked semantic complementarity on the newest three pairs
     (`security-expert:analyst/cobb/devops`) — each states "advisory... X weighs it but keeps
     final authority," genuinely reciprocal, not just name-matched.
- **§7 lint fold-in** — every artifact changed since 2026-07-29 (19 files: 15 agent prompts + 4
  skills). Forked the work three ways to keep it out of my own context (findings only came back,
  not full file reads) — two forks (`analyst`/`architect`/`coder`/`data-scientist`/`devops`, and
  `frontend-engineer`/`graph-dba`/`security-expert`/`tico`) returned real, on-target findings; the
  third (assigned the 4 changed skill files) came back off-target — it echoed my own
  already-completed judgment-checklist work instead of linting its assigned files, a context-bleed
  failure mode worth a kaizen note of its own (see below). Did that piece directly instead of
  re-forking it. **Findings, across all 19 files:**
  - **2 persona findings (minor, but confirmed against a real, dated team decision — not a fresh
    opinion):** `data-scientist.md` and `frontend-engineer.md` both still opened "You are a
    senior ___" — the team dropped "senior" framing collection-wide 2026-06-20 (this file,
    2026-06-20 entry, "overconfidence concern; persona-prompting evidence shows role labels are
    weak-to-neutral," applied explicitly to `cobb` itself and stated as harmonizing the whole
    collection). Both files postdate that sweep and were never checked against it. Fixed both
    (dropped the one word each). Re-swept the whole `claude/*/*.md` tree afterward — zero
    remaining hits (the one surviving "senior" match, `cobb/TESTING.md`, names an unrelated
    OpenCode agent, `coding-senior`, as a candidate example — not this team's persona).
  - **2 coverage findings (minor, real intra-file staleness, harmless in practice):**
    `tico.md`'s and `teco.md`'s commit-authority grants each still named a "kaizen inbox
    entry"/"your kaizen inbox" as a committable deliverable — dead since the 2026-08-20 graph
    migration (no agent produces a fresh `kaizen/inbox.md` entry any more; `tico`'s case was
    doubly wrong — its own Write/Edit-guard bullet never named `kaizen/inbox.md` as covered in
    the first place). Fixed both. Logged: `claude/tico/kaizen/history.md`,
    `claude/teco/kaizen/history.md`.
  - **1 finding checked and dismissed (false lead):** a fork flagged `security-expert.md`'s
    "or the session scratchpad" clause as possibly unverified against `claude/AGENTS.md`'s
    shorter hook summary. Read the actual `guard-review-doc-writes.sh` script directly — its own
    escalation message says "outside a `docs/reviews/` directory **or the `/tmp` scratchpad**,"
    confirming the prompt's claim and not `AGENTS.md`'s summary, which simply omitted the detail
    (a summary, not a denial). No fix needed.
  - **1 non-finding worth recording:** `coder.md` has no `tools:` frontmatter field, unlike every
    other file in its lint batch — confirmed intentional, not a gap: `claude/README.md` already
    documents `coder`/`tdd-engineer`/`frontend-engineer`/`qa-engineer` as the four agents that
    declare no `tools:` allowlist specifically so they inherit `mcp__cypher__query` automatically.
  - **1 non-finding worth recording:** `guard-ds-doc-writes.sh` and `guard-plan-doc-writes.sh`
    are both directory-scoped (`docs/plans/*`), not filename-suffix-scoped — `data-scientist` can
    technically write unprompted anywhere under `docs/plans/`, including `architect`'s own plans,
    and vice versa. Symmetric, pre-existing (predates 2026-08-21), and both files' prose already
    describes the breadth accurately (no overclaim) — not filed as a defect, just noted here in
    case it's news to a future reader.
  - The 4 skill files (`agent-maintenance`, `cpg-analysis`, `joern-cpg`, `python-web-quirks`),
    checked directly after the assigned fork came back off-target: no stale `kaizen/inbox.md`
    operative references (the one hit, `python-web-quirks/SKILL.md`'s origin note, is correctly
    past-tense — describes where the content was distilled *from* in 2026-08-09, not a live
    instruction), no leftover pre-rename `mcp__cpg__*` tool naming (clean sweep from the
    cypher-mcp rename), `agent-maintenance/SKILL.md`'s own three `kaizen/inbox.md` mentions are
    its own correct, current documentation of the frozen-inbox convention.
- **Kaizen-worthy harness observation (not filed as a graph entry — resolved in this same run,
  see the Learning-capture exemption):** a `fork` subagent given a narrow, explicit directive (§7
  lint on 4 named files) can still drift into narrating the *parent* session's own
  already-completed work instead of doing its assigned task — plausible mechanism: a fork
  inherits the full parent transcript, and a long transcript with a lot of the parent's own
  recent narrated actions (this session had just fixed 4 real defects immediately before the
  fork launched) can apparently pull a fork's own generation toward continuing that narration
  rather than executing its distinct directive. Mitigation used here: treat an off-target fork
  result as **unverified**, don't retry with the same shape blind — either re-scope the prompt to
  more forcefully exclude parent-session narration, or (what I did) just do the bounded piece of
  work directly instead of re-forking. Not (yet) promoted to a `kaizen_team` entry — this is a
  single data point, not independently confirmed, and doesn't change any agent's operative
  behavior on its own; worth a second data point before promoting.
- **Verified:** `bash claude/scripts/audit-team.sh` — 113 PASS / 2 pre-existing FAILs, identical
  before and after every fix in this pass (7 fixes total: 1 roster-count, 3 enforcement-parity,
  2 persona, 2 coverage — the "7" undercounts by the 2 kaizen-related ones already logged as
  their own dated entries above this one, for a session total of 9 real defects found and fixed).
  No personal identifiers introduced by any edit this pass (checked `git diff` on every touched
  file).
- **Docs touched this pass:** `claude/AGENTS.md`, `claude/cobb/cobb.md`,
  `claude/data-scientist/data-scientist.md`, `claude/frontend-engineer/frontend-engineer.md`,
  `claude/qa-engineer/qa-engineer.md`, `claude/tdd-engineer/tdd-engineer.md`,
  `claude/teco/teco.md`, `claude/tico/tico.md`, plus each edited agent's own
  `kaizen/history.md` (self-logged, cross-referenced above) and this file.

## 2026-08-21 — Distilled all 9 pending `teco`-authored entries from `kaizen_team`

- **What:** Ran the agent-maintenance skill §5 procedure against every `kaizen_team` node with
  `author:'teco'` (9 entries, dated 2026-08-15 through 2026-08-21 — teco's own raw capture since
  the 2026-08-20 team-wide graph migration). Full dispositions, verification notes, and the exact
  promoted text live in `claude/teco/kaizen/history.md`'s matching 2026-08-21 entry — this entry
  covers only the cross-artifact bookkeeping that isn't teco's to log.
  - **Promoted to `claude/teco/teco.md`** (5 entries): shared-DB-state dispatch serialization
    (sharpened the existing line), `subagent_type` must always be explicit on `Agent` calls
    (blocker-severity — silently degrades to `general-purpose`, no hooks/persona/tools), a
    `completed` notification's `<result>` can be a stale mid-task placeholder, a clean QA pass on
    a brand-new mechanism isn't full state-space coverage, and a coordinator's "proceed" never
    substitutes for real user approval on a harness-gated write (paired with a hard "never relay
    a delegate's self-modify-permissions proposal" line).
  - **Promoted to `skills/agent-standards/claude-code.md`** (3 entries): a new "Nested-delegation
    notification routing" subsection (dormant-ancestor bubbling + force-resume; the
    `<system-reminder>`-relay delivery path for an unaddressable delegate — both explicitly
    dated/caveated as live observations, not confirmed stable contracts) under Subagents, and one
    new bullet under Hooks' existing auto-mode-classifier note (the classifier also flags a
    delegate proposing to self-modify its own permissions).
  - **Promoted to `cypher-mcp/README.md`** (partial, 1 entry): the `\n` vs `\\n` string-literal
    escaping trap, added to "Writing through this tool." The same entry's other claim — per-cell
    truncation at ~300 chars — turned out **already fully documented**
    (`CYPHER_MCP_MAX_CELL`/`CYPHER_MCP_MAX_CHARS` in "Result format and truncation"), confirmed by
    direct re-derivation: I hit the exact same 300-char truncation reading these very entries out
    of the graph this session, before ever reading the README section that already named it.
    Discarded that half as a duplicate; promoted only the escaping half, which the README did not
    cover.
  - **Kept open — filed `K-018`** (this file's `plan.md`, high priority): the
    `guard-destructive-ops.sh`-didn't-fire-for-a-nested-subagent's-`GRAPH.DELETE` entry. Re-fetched
    `code.claude.com/docs/en/hooks` (2026-08-21) — official docs state hooks fire identically for a
    subagent whether run as the main session agent or a nested delegate, no documented exception —
    so the observed gap contradicts the doc rather than being explained by it. Leading hypothesis
    (not confirmed): the same dispatch likely omitted `subagent_type`, which — per the *other*
    surviving entry from this same batch (`b1e3a1f0…`, now in `teco.md`) — silently runs the
    delegate as `general-purpose`, which would carry no `graph-dba` frontmatter `hooks:` at all.
    Couldn't confirm from the coordination doc alone (`ab3504712c7912872` is a real agentId either
    way); needs a live re-check on the next `graph-dba` dispatch with `subagent_type` confirmed
    correct before this can close.
  - **Discard:** none outright — every surviving entry (all 9) either promoted somewhere or opened
    K-018.
- **Verification method:** the harness's own per-cell display truncates any single returned graph
  field at ~300 chars (`…(+N chars)`) — re-derived exactly, not assumed, via `size()` queries plus
  multi-column `substring()` paging per entry before reading any of them, so no entry was acted on
  from a truncated partial read. Re-fetched the one live-checkable external claim (hook parity for
  nested subagents) against current official docs rather than trusting the entry's own framing.
- **Why:** user asked to "work on teco's inbox" — teco's `kaizen/inbox.md` is a frozen
  2026-08-20 historical snapshot (already imported and cleared), so the live equivalent is its
  pending raw capture in the shared `kaizen_team` graph; this was the first full distillation pass
  against it since the migration.
- **Docs touched this pass (cross-artifact bookkeeping only — see `teco/kaizen/history.md` for the
  full per-entry disposition table):** `claude/cobb/kaizen/{plan,history}.md` (this file + K-018),
  `skills/agent-standards/claude-code.md`, `cypher-mcp/README.md`.

## 2026-08-21 — Added a topic-bounded `Write|Edit` guard (agent-permission-friction FR-1)

- **What:** `cobb` previously had no custom write guard at all (unrestricted `Write`/`Edit`,
  frontmatter `permissionMode: acceptEdits` only). Added
  `claude/cobb/hooks/guard-cobb-topic-writes.sh` — a new thin wrapper over the shared
  `claude/scripts/guard-doc-writes.sh` core, wired via a new frontmatter `hooks:` block — plus a
  core-script change (that core now emits an explicit `permissionDecision:"allow"` on a glob
  match, instead of a silent `exit 0`, and gained an optional `on_mismatch` `ask|pass` 3rd arg;
  the six pre-existing callers are unaffected, verified byte-identical `ask`-branch behavior).
  Cobb's allowlist is **topic-bounded, not folder-bounded** — any agent's own definition file
  (`claude/*/*.md`), kaizen curation for any agent (`claude/*/kaizen/{history,plan}.md`), the
  team catalog/context files (`claude/README.md`/`AGENTS.md`/`CLAUDE.md`), cobb's own skill
  packages (`skills/agent-{maintenance,standards}/*`, `skills/README.md`), and a small,
  explicitly maintained list of MCP/agent-standards docs living outside `claude/`/`skills/`
  (seeded with `cypher-mcp/README.md`) — every `claude/`/`skills/`-rooted entry doubled (a bare
  form plus a `*/`-prefixed sibling) so it matches whether `tool_input.file_path` arrives
  repo-relative or absolute. `docs/BACKLOG.md` and anything else genuinely outside that topic
  still escalates.
- **Why:** Requirements doc `claude/docs/requirements/agent-permission-friction.md` (FR-1,
  instances 1-3,5,6,9, counter-example C2): the stakeholder was hitting a manual confirmation
  prompt on cobb's own routine, in-remit work (editing other agents' definition files, kaizen
  curation, MCP-standards docs) despite `acceptEdits` since 2026-07-24. Root cause (design doc
  `claude/docs/plans/agent-permission-friction.md` §1, `analyst`-reviewed, verdict approve):
  frontmatter `permissionMode` is a Task-tool-subagent-scoped setting, silently ignored/overridden
  by the parent session's mode in documented cases (including auto mode, the Pro/Max/Team
  default) — an explicit hook `"allow"` is the one mechanism confirmed to suppress the prompt
  regardless of ambient mode. Mutation-tested (deliberately broke the match branch, confirmed the
  guard correctly fell back to `"ask"` on a previously-allowed path, then restored and reconfirmed
  `"allow"`) and regression-checked against the six pre-existing `guard-doc-writes.sh` callers
  (`architect`, `analyst`, `data-scientist`, `teco`, `tico`, `security-expert`'s review guard —
  all byte-identical `ask`-branch text, only their match branch changed to explicit `allow`).
- **Plan items:** —

## 2026-08-20 — Designed and created the `security-expert` agent (K-016)

- **What:** Closed `kaizen/plan.md` **K-016** — designed the new `security-expert` agent from
  `claude/docs/requirements/security-expert.md` (`Status: Ready for design`, `tico` interview
  confirmed 2026-08-17). Full design, files, and rationale: `claude/security-expert/kaizen/
  history.md`'s own "2026-08-20 — Created" entry (the source of truth for the agent's own design
  decisions — not duplicated here). Summary for this agent's own log:
  - New folder `claude/security-expert/` — `security-expert.md`, two hook scripts
    (`hooks/guard-review-doc-writes.sh`, a normal thin wrapper over the shared
    `scripts/guard-doc-writes.sh` core scoped to `docs/reviews/*`; `hooks/
    guard-exploitation-approval.sh`, a **new standalone script**, deliberately not layered on the
    shared `scripts/guard-destructive-ops.sh` core — different hazard class, see its own header
    and the security-expert kaizen entry for the full reasoning), `kaizen/{plan,history}.md` (no
    `inbox.md` — created after the 2026-08-20 shared-graph consolidation, FR-12/AC-9).
  - Deployed: `~/.claude/agents/security-expert` symlinked to `claude/security-expert`.
  - Boundary pairs declared in `claude/scripts/audit-team.sh` (`security-expert:analyst`,
    `security-expert:cobb`, `security-expert:devops`) with reciprocal description clauses added to
    `analyst.md`, `cobb.md` (this agent — a self-edit, see the standing open item below), and
    `devops.md`. `teco.md` gained a routing-table row + handoff-contract line. `claude/README.md`
    gained a catalog row + kaizen-links entry; `claude/AGENTS.md` gained a roster mention and a
    "Hook machinery" paragraph describing the new agent-owned exploitation-approval hook as a
    departure from the two existing shared cores. Root `AGENTS.md` checked — no change needed
    (doesn't enumerate agent names; confirmed via check 5b's own logic).
  - **Verified:** `bash claude/scripts/audit-team.sh` — all checks pass for `security-expert`
    itself and every edited file (kaizen pair, deployment symlink, both hooks exist+executable,
    teco roster mention, both catalogs, all three boundary pairs symmetric, no commit-authority
    claim). The run's one FAIL (`falkor-chat/docs/test-reports/graphrag-eval-report.md` leaking
    the maintainer's home path/username) is **pre-existing and unrelated** — confirmed via `git
    log`/`git status` that the file is untouched by this session and was last committed
    2026-08-16; flagged in the task's final report for separate follow-up, not fixed here (out of
    this task's scope, and not this agent's file to silently rewrite mid-unrelated-task). Also ran
    `bash -n` on both new hook scripts plus 8 manual test cases through the exploitation guard
    (benign commands and local-marked network commands pass silently; named offensive tools, a
    listener setup, and an external-host network command all correctly escalate).
  - **Open item carried forward, not resolved here:** this change includes a self-edit to
    `cobb.md`'s own `description` (the reciprocal boundary clause) — exactly the shape flagged in
    this file's own parking lot ("Extend the independent-review-gate practice to `cobb.md`
    self-edits specifically") as going out unreviewed. Noted in the task's final report as
    something `analyst`'s independent review of this whole change should specifically double-check
    when it runs.
- **Why:** Executing a named, dated backlog item (K-016) per this agent's own maintenance
  duties — the requirements doc had sat at `Ready for design` for three days with the design pass
  not yet run.
- **Plan items:** K-016 closed (moved to the Closed line). No new cobb-side plan items — the
  design's own judgment calls and follow-ups live in `security-expert/kaizen/plan.md`, where they
  belong (agent-specific, not team-maintenance-specific).

## 2026-08-20 — D-1 fix: `cobb.md`'s "Maintenance duties" section still described the pre-M7 convention (inbox-seeding, per-agent `kaizen_<agent>` graph); `docs/BACKLOG.md`'s M7 section flipped 🔵→✅

- **What:** Dispatched directly (two small closeout fixes for M7, not `teco`-coordinated) off
  `Q2`'s closing acceptance report (`docs/test-reports/generic-cypher-mcp2-report.md`, defect
  **D-1**). Two edits, both in-scope for `cobb`'s own self-edit carve-out (§3.7 of
  `docs/plans/generic-cypher-mcp2.md`):
  1. **`claude/cobb/cobb.md` lines 65 and 71** (the "Kaizen" and "Learnings distillation" bullets
     under "Maintenance duties"). Both still described the **pre-M7** state — line 65 said a new
     agent gets an `inbox.md` "seed[ed] on creation" (contradicts FR-12/AC-9 and
     `skills/agent-maintenance/SKILL.md:62`); line 71 said raw capture goes into "its own
     working-memory FalkorDB graph, `kaizen_<agent>`" and described curator-clearing against "the
     agent's own `kaizen_<agent>` graph" (describes the **interim** per-agent-graph shape,
     `ccf9c8b`, already superseded — 10 of those 12 keys are retired by `G1`). Both directly
     contradicted the same file's own, already-correct "## Learning capture" section
     (lines 84–98). Rewrote both bullets to state the current convention: no `inbox.md` for a new
     agent (the 12 pre-2026-08-20 agents keep theirs frozen); raw capture → shared `kaizen_team`
     graph, `author`-partitioned, curator-clear scoped by `entryId`. Targeted edit only — did not
     restructure the section, matching the brief's sizing instruction ("similarly to how 'Learning
     capture' already reads").
  2. **`docs/BACKLOG.md`'s M7 milestone-map row + `C-701`…`C-721` item bullets**, still 🔵
     (proposed) though `Q2`'s verdict (**PASS with noted open items**) is now in. Flipped the
     milestone row and all individual items to ✅, but **not** as a blanket flip — three genuine
     open nuances kept as inline notes rather than silently absorbed: (a) the inbox-header-retarget
     half of every `C-<agent>` unit was **dropped entirely by stakeholder decision** (not merely
     deferred — see the coordination ledger's `cobb`-batch row); (b) `teco`'s (`e40a95fe-…`) and
     `analyst`'s (`fe2007f5-…`) per-agent migrations each carry one already-known, already-tracked
     data-fidelity defect on a single entry, pending a separate stakeholder approval for the
     `cobb` curator-clear fix; (c) `G1` correspondingly leaves `kaizen_teco`/`kaizen_analyst` live,
     deliberately, pending the same fix. Modeled the "✅ with a caveat clause" phrasing on M4's own
     precedent (DEF-4 folded as a named residual, not a blocker) rather than inventing a new
     convention. Did **not** invent new backlog `C-` numbers for the two pending defects — they're
     already tracked in `docs/plans/generic-cypher-mcp2-coordination.md`'s unit ledger (the
     `C-teco`/`C-analyst` rows), referenced by pointer instead of duplicated.
- **Why:** `Q2`'s own recommended fix, direct dispatch. D-1 is the report's one new, genuinely
  unaddressed finding (High severity, self-contradiction inside `cobb`'s own always-loaded
  prompt); the BACKLOG flip is the report's own housekeeping note ("recommend `teco`/`cobb` flip
  both to ✅ once this report's verdict is accepted") plus FR-13's explicit "incremental delivery
  is valid progress" framing, which is why the two still-open per-agent defects got a note instead
  of blocking the flip.
- **Verified:** re-read both edited `cobb.md` bullets against `skills/agent-maintenance/SKILL.md:62`
  and against `cobb.md`'s own "Learning capture" section — no remaining contradiction. Re-read the
  M7 section of `docs/BACKLOG.md` against `docs/plans/generic-cypher-mcp2-coordination.md`'s unit
  ledger row-by-row (S0–S4, T1, all 12 `C-<agent>` rows, `G1`, Q1/Q2) to confirm every ✅/note
  matches the ledger's own recorded status, not an assumption.
- **Not done in this pass (flagged, not silently skipped):** the report's Feedback #2 recommends an
  independent `analyst`/`qa-engineer` diff review specifically for self-edited files (the gap that
  let D-1 ship unnoticed in the first place) — this fix is itself another `cobb` self-edit with no
  independent review yet. Parking-lot note added below; not resolved here (no reviewer available
  in a direct, non-`teco`-coordinated dispatch) — flag for whoever next coordinates a `cobb.md`
  touch, or route to `teco` on request.
- **Docs touched:** `claude/cobb/cobb.md`, `docs/BACKLOG.md`, this `history.md` entry,
  `kaizen/plan.md` (parking-lot note below).
- **Plan items:** see `plan.md`'s new parking-lot note — extend the independent-review-gate
  practice to `cobb.md` self-edits specifically (this fix included).

## 2026-08-20 — `cypher-mcp/README.md`'s own curator-clear example was wrong (missing space); fixed in-line during the Q2 AC-5 acceptance distillation

- **What:** While re-deriving (not just re-citing) `graph-dba` entry `a3f4e1b2…`'s claim during the
  Q2/AC-5 acceptance distillation (below, `claude/graph-dba/kaizen/history.md`), the harness's
  auto-mode classifier blocked a live re-run of the entry's own DDL repro, so I read
  `cypher-mcp/server.py` directly instead. Found `_CURATOR_CLEAR_RE` (`:265-269`) requires the
  literal substring `entryId: ` (colon **then a space**) before the quote, and the pre-match
  normalization (`" ".join(cypher.split())`, `:366`) only *collapses* existing whitespace runs —
  it never *inserts* a missing one. Consequence: `cypher-mcp/README.md`'s own documented
  curator-clear example, `{entryId:'...'}` (no space), does **not** match the regex and is
  rejected — confirmed live, twice, in this session: the no-space form on the real clear I was
  about to run got "Rejected: this write is neither an author-write ... nor the recognized
  curator-clear shape," and the identical query with a space after the colon succeeded
  (`nodes_deleted=1.0`). Fixed the README's curator-clear code example to include the space and
  added a explanatory note naming the exact mechanism, so the next reader who copy-pastes it
  doesn't hit the same rejection.
- **Why:** A documented example that fails when copy-pasted verbatim is worse than no example —
  it actively teaches the wrong shape. This is squarely a `cypher-mcp` project-docs fix (not a
  `graph-dba`-specific fact), found and fixed in the same pass per the standing Learning-capture
  instruction ("verify and promote... in the same run, in-bounds for you alone as the
  maintainer") rather than filed as a raw `kaizen_team` entry for a later pass.
- **Verified:** read `server.py`'s `_CURATOR_CLEAR_RE` and the `normalized = " ".join(cypher.split())`
  line directly (ground truth, not inference); reproduced the rejection/success pair live against
  `kaizen_team` in this same session (two real tool calls, differing only in that one space).
- **Scope note:** code (`server.py`) itself was **not** touched — only the README's documented
  example and explanation. Whether the regex itself should tolerate the no-space form is a
  `cypher-mcp` implementation question, out of scope for a docs-only maintenance pass; not filed
  as a backlog item since the doc fix alone resolves the practical footgun.
- **Docs touched:** `cypher-mcp/README.md` (curator-clear example + note), this `history.md` entry.
- **Plan items:** none opened — see the scope note above.

## 2026-08-20 — M7 `C-<agent>` units, cobb's half (all 12): prompt retarget to `kaizen_team` delivered; header-retarget half dropped by stakeholder decision
- **What:** Dispatched by `teco`, batched per the coordination doc's own sizing note (same owner —
  `cobb` — disjoint files across 12 units, an efficiency batch, not a mega-dispatch), to execute
  the `cobb`-owned half of all 12 `C-<agent>` units in `docs/plans/generic-cypher-mcp2.md` §4.2
  (Version 4): each agent's own data-migration half is that agent's separate, independently-run job
  (not this unit).
  - **Prompt retarget (delivered, all 12).** Every `claude/<agent>/<agent>.md`'s Learning-capture
    section rewritten to the plan's §3.3 recipe verbatim: target graph `kaizen_team` (`author`-
    partitioned) instead of the agent's own `kaizen_<agent>`, plus the new `sessionId` field
    (`$CLAUDE_CODE_SESSION_ID`, FR-8a) — `analyst`, `data-scientist`, `qa-engineer`, `teco`,
    `graph-dba`, `architect`, `coder`, `devops`, `frontend-engineer`, `tdd-engineer`, `tico`, and
    `cobb.md` itself (the one legitimate self-edit, §3.7 — every other agent's own prompt still
    reads "never edit your own agent definition," `cobb.md`'s alone omits it).
  - **`teco.md`'s extra M3 fix (delivered).** Two stale cross-reference passages outside its
    Learning-capture section, both still describing a file-based `kaizen/inbox.md` capture
    mechanism that no longer matches reality: the "Fencing carve-out" bullet (formerly ~line 72)
    rewritten — raw learnings capture is a graph write (`mcp__cypher__query` against `kaizen_team`),
    not a file write, so a brief excluding a subtree (`claude/`) never blocks it, no carve-out
    needed for that step specifically; the "Learnings ride the handoff" bullet (formerly ~line 89)
    now checks for a dated `:KaizenEntry` in `kaizen_team` rather than an `inbox.md` entry.
  - **C-cobb data-migration (delivered, no-op).** Live-checked `kaizen_cobb` via `mcp__cypher__query`
    — the graph key does not exist ("Graph 'kaizen_cobb' does not exist," listed alongside the
    other live keys), confirming 0 entries, matching the plan's 2026-08-20 snapshot. Nothing to
    migrate; no graph write made.
  - **Header retarget (attempted, then fully dropped — stakeholder decision, mid-run).** The plan's
    §4.2/P3-M3 authorized retargeting each of the 12 `kaizen/inbox.md` headers' *prescriptive*
    clause (the copy-pasteable `mcp__cypher__query(graph='kaizen_<agent>', ...)` pointer) to
    `kaizen_team`, while leaving the 4 agents' (`analyst`/`data-scientist`/`qa-engineer`/`teco`)
    true past-tense provenance clause untouched — reasoning: each header's own immutability promise
    is scoped to *"Content below,"* so the header note itself was never inside that promise.
    Attempted 3 of the 4 scoped files (`analyst`, `data-scientist`, `qa-engineer`) plus `teco`
    (landed first); the permission system **denied all 3 scoped attempts live**, each with the
    reason "this is frozen" / "it is frozen." `teco`'s edit had already landed before the pattern
    was clear. Stopped, reported the conflict rather than continuing to the other 8 or guessing;
    the stakeholder relayed, via `teco`: **Option 2 — stop entirely, revert what already landed.**
    Reverted `claude/teco/kaizen/inbox.md` via `git checkout -- <path>` (not hand-reconstructed) to
    its exact HEAD text; verified via `git diff --stat -- 'claude/*/kaizen/inbox.md'` (empty) and
    `git status --porcelain -- 'claude/*/kaizen/inbox.md'` (empty) that **all 12** `inbox.md` files
    are untouched, byte-identical to HEAD. **This supersedes the plan's own §4.2/P3-M3 header-retarget
    authorization** — a correction for whoever closes the plan/coordination doc to record (not made
    here; `docs/plans/*` isn't this agent's file to edit).
  - **Ambiguity found and flagged, now moot but worth recording:** `claude/graph-dba/kaizen/inbox.md`'s
    header (dated 2026-08-18, pre-dating the general 2026-08-20 migration) actually carries a real
    past-tense provenance clause too ("Its contents… were imported once into the `kaizen_graph_dba`
    FalkorDB graph") — the plan's C-graph-dba row calls it "no entries, so no provenance clause to
    protect," which a direct read doesn't support. Moot now that header edits are off the table
    entirely, but the plan/coordination doc's premise on that row was slightly wrong independent of
    the stakeholder decision.
- **Why:** `docs/plans/generic-cypher-mcp2.md` (analyst-approved, "approve with suggestions") — the
  prompt half is FR-2/FR-7/FR-8a/FR-11's delivery mechanism for all 12 agents; the header half's
  fate was decided live, mid-dispatch, by direct stakeholder call relayed through `teco`, overriding
  the plan's own written authorization for that one narrow piece.
- **Verification commands run:** `mcp__cypher__query(graph='kaizen_cobb', ...)` (graph absent, 0
  entries); `git diff --stat -- 'claude/*/kaizen/inbox.md'` and `git status --porcelain -- 'claude/*/kaizen/inbox.md'`
  (both empty, post-revert); `git status --porcelain -- claude/` (12 modified files, all
  `<agent>/<agent>.md`, matching exactly the prompt-retarget scope, nothing else).
- **Explicitly not touched:** any `claude/<agent>/kaizen/inbox.md` (all 12, including `teco`'s, now
  reverted); any `claude/<agent>/kaizen/plan.md`/`history.md` other than `cobb`'s own; any
  `docs/plans/*` file (the plan's own supersession note is `teco`'s to make, not `cobb`'s); any
  `kaizen_<agent>` graph data for the other 11 agents (each agent's own separately-dispatched job).
- **Plan items:** see `plan.md`'s new note below — inbox.md headers are enforced-frozen in
  practice, not just by written convention; don't plan future work assuming the "Content below"
  scoping argument is actionable without re-confirming live.

## 2026-08-20 — M7 substrate units S4/S3/S2: consolidate per-agent `kaizen_<agent>` graphs onto shared `kaizen_team`, deliver FR-12/AC-9 (no `inbox.md` for a new agent)
- **What:** Dispatched by `teco` to execute 3 of the 21 units in
  `docs/plans/generic-cypher-mcp2.md` (Version 4), in the plan's required order (`S4` before `S3`,
  since `S3` depends on `S4` per P3-m3):
  - **`S4`** — `claude/scripts/audit-team.sh` check 1 narrowed from a three-way `plan.md`+
    `history.md`+`inbox.md` conjunction to a two-way `plan.md`+`history.md` pair; header comment
    (lines 8-9) updated to match. Verified per the plan's own isolated-scratch-copy method (P3-M1):
    a synthetic agent with `<name>.md`+`kaizen/{plan,history}.md` and **no** `inbox.md`, audited
    from a copy of the script at `<scratch>/sub/scripts/audit-team.sh` (so `ROOT` resolves to
    `<scratch>`), produced `PASS <name>: kaizen plan + history present` on check 1 specifically —
    overall run still `FAIL`s on checks 2/4/5/5b, expected (deployment/roster/catalogs fail for any
    synthetic agent by construction). Live re-run of the unmodified 12-agent `claude/` collection:
    all 12 still `PASS` check 1 (their `inbox.md` is present but no longer required).
  - **`S3`** — `skills/agent-maintenance/SKILL.md`: every `kaizen_<agent>`/`kaizen_<name>`/
    `kaizen_{name}` occurrence retargeted to `kaizen_team` with an `author`-filtered pattern (§1
    step 1, §2 step 2, §5's intro/distillation-procedure/read-and-clear calls); §1's "Creating"
    procedure no longer seeds an `inbox.md` for a new agent — points the new agent's
    Learning-capture section straight at the `kaizen_team` recipe instead; §5's "Inbox template"
    block rewritten from "seed on creation" to a historical "Inbox header shape" reference note
    (the target header shape `C-<agent>` units write into the 12 existing frozen files) plus a
    note on the 4 agents whose provenance clause must stay untouched (P3-M3). Verified by
    `grep -n 'kaizen_' skills/agent-maintenance/SKILL.md`: 13 of 15 hits are `kaizen_team`
    (correct, retargeted); the 2 surviving old-pattern hits are both genuinely past-tense history
    (the Inbox-header-shape note describing the real frozen files' provenance clause, and the
    Origin block's account of the 2026-08-20 migration lineage) — no prescriptive pointer to the
    old per-agent-graph convention survives.
  - **`S2`** — `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, `docs/BACKLOG.md` all
    retargeted to describe `kaizen_team`, `author`-partitioned, as the standing convention (each
    file's kaizen-adjacent paragraph, not just the one the plan named — also fixed `claude/AGENTS.md`
    line ~63 and root `AGENTS.md`'s "Working in this repo" bullet, both of which still instructed
    seeding `inbox.md` on agent creation). `claude/README.md`'s Kaizen section now carries §3.1's
    `MATCH`/`RETURN`/`ORDER BY` FR-7 recipe verbatim, full field list, as a copy-pasteable example,
    and states every `kaizen/inbox.md` as a **permanent** frozen snapshot (not "required to
    exist"). `docs/BACKLOG.md` gets a new `## M7` body section (verbatim from the plan's §4.4) and
    a matching Milestone-map row after M6, mirroring M1–M6's format.
- **Why:** `docs/plans/generic-cypher-mcp2.md` (analyst-approved, "approve with suggestions"),
  consolidating the ad hoc per-agent `kaizen_<agent>` graphs (below) onto one shared,
  `author`-partitioned `kaizen_team` graph, and delivering FR-12/AC-9 as literally written.
- **Verification commands run:** the isolated-scratch `audit-team.sh` check-1 test (above); live
  `bash claude/scripts/audit-team.sh` (all 12 agents `PASS` check 1; the run's overall `FAIL` is a
  **pre-existing, unrelated** personal-info leak in `falkor-chat/docs/test-reports/graphrag-eval-report.md`,
  confirmed already committed at `1578af3`, outside this dispatch's scope); `grep -n 'kaizen_'
  skills/agent-maintenance/SKILL.md` (above).
- **Explicitly not touched in this dispatch** (separate `C-<agent>` units, per-agent, dispatched
  separately): any `claude/<agent>/<agent>.md` prompt file or `claude/<agent>/kaizen/inbox.md`
  header note — including `cobb.md`'s own.
- **Plan items:** none opened — this is execution of an already-gated plan, not a new backlog idea.

## 2026-08-20 — Learnings capture redesigned team-wide: file-based inbox → per-agent `kaizen_<agent>` FalkorDB graph, mirroring `graph-dba`
- **What:** User redirected the previous day's file-pointer fix (below) with "I will migrate all
  agents to write their learnings to the graph like graph-dba... we need to rethink the
  solution." Executed the full migration in one pass:
  - **Mechanism.** Every agent's "Learning capture" closing-protocol section (11 agent prompts +
    `cobb.md` itself, 12 total) rewritten to write a `:KaizenEntry` node directly into its own
    `kaizen_<agent>` FalkorDB graph via `mcp__cypher__query`, byte-for-byte mirroring
    `graph-dba`'s pre-existing `kaizen_graph_dba` pattern (same node schema — `entryId`, `date`,
    `fact`, `evidence`, `context`, `suggestedHome`, `author`, `createdAt` — same
    author-write-authorization contract). `graph-dba` itself untouched (already correct).
    `cypher-mcp`'s write authorization needed **no server-side change**: confirmed by reading
    `cypher-mcp/server.py`'s `authorize_write()`/`_author_claims()` that the two authorized write
    shapes (author-matched `CREATE :KaizenEntry`, curator `DETACH DELETE` by `entryId`) were
    already agent-generic, never hardcoded to `graph-dba` — the MCP layer was ready for this
    before today, only the 11 agents' own prompts and `inbox.md` files were on the old convention.
  - **Migration of real content.** 4 of the 11 non-`graph-dba` inboxes had genuine unprocessed
    entries: `analyst` (5), `data-scientist` (4), `qa-engineer` (6), `teco` (5) — 20 total. Wrote
    a Python parser (`/tmp/migrate_inbox.py`, not committed) that splits each `inbox.md` on its
    `## YYYY-MM-DD — <fact>` headers and extracts the Evidence/Context/Suggested-home fields
    programmatically, rather than hand-transcribing — deliberately avoiding the exact
    silent-drop-on-transcription failure mode `analyst`'s own 2026-08-11 kaizen entry warns
    about (reconcile removed-entry counts against claimed counts). Generated a `uuid4` `entryId`
    and a shared `createdAt` per entry, Cypher-escaped every field (backslash-escaped single
    quotes, matching `cypher-mcp`'s own backslash-aware string-literal scanner), and live-wrote
    each agent's batch as one multi-`CREATE` `mcp__cypher__query` call (`agent='<agent>'`) —
    each `CREATE` clause needed its own bound variable name (`k0`, `k1`, ...) after the first
    attempt hit "The bound variable 'k' can't be redeclared in a CREATE clause" reusing `k`
    across clauses in one statement. Verified via `MATCH (e:KaizenEntry) RETURN count(e)` against
    each of the 4 graphs post-write: 5/4/6/5, exact match. The other 7 agents' inboxes (`architect`,
    `coder`, `devops`, `frontend-engineer`, `tdd-engineer`, `tico`, `cobb`) had nothing beyond the
    seeded header template — no migration needed.
  - **Frozen inboxes.** All 11 `kaizen/inbox.md` files (the 4 with migrated content plus the 7
    empty ones) got a `**FROZEN — 2026-08-20.**` header block, mirroring `graph-dba`'s own frozen
    `inbox.md` note exactly: states the file is a historical snapshot, names the entry count
    migrated, gives the live-read `mcp__cypher__query` recipe, and says the agent no longer
    appends here.
  - **Tool grants.** `data-scientist` and `tico` had no MCP tool access at all before this —
    added `mcp__cypher__query` to both frontmatters (checked all 12 agents' tool lists first;
    `architect`/`analyst`/`teco` already had it explicitly, `coder`/`devops`/`frontend-engineer`/
    `qa-engineer`/`tdd-engineer`/`cobb` already had it implicitly via `tools: All tools`).
  - **Docs.** `cobb.md`'s own Learning-capture section and its "Learnings distillation"
    maintenance-duties paragraph rewritten for the universal graph pattern.
    `skills/agent-maintenance/SKILL.md` §5 rewritten end-to-end: header renamed "Learnings
    inboxes" → "Learnings graphs", the `graph-dba`-is-the-exception framing replaced with
    graph-is-the-default framing generalized to `<agent>`/`kaizen_<agent>` throughout (steps
    1-4, the append-before-clear sub-procedure, the dedup-check wording), the seeded "Inbox
    template" replaced with a frozen-stub variant for future agent creation, plus the skill's own
    frontmatter `description` and three other stray "inbox" references (§1 bullet list, the
    creation-procedure step, the order-of-operations step, the §7 fold-in cross-reference) swept
    for consistency. `claude/AGENTS.md` and root `AGENTS.md`'s kaizen-convention paragraphs
    rewritten to state graph-based capture as the norm (no more "except graph-dba" framing).
  - **Bookkeeping.** Dated `kaizen/history.md` entries added to all 10 other touched agents
    (this entry is `cobb`'s own, the 11th) plus `cobb/kaizen/plan.md`'s parking-lot item updated
    to record the reversal of yesterday's item-1 fix and flag the open distillation follow-up.
- **Why:** Direct user instruction, given only after item 1 of the verbosity diagnosis (below)
  had already shipped — an explicit "this is not correct... we need to rethink the solution,"
  not a request I inferred. `graph-dba`'s graph-based pattern was already live, already reviewed
  (`docs/reviews/graph-dba-kaizen-distillation.md`), and the MCP server's authorization was
  already generic across agents — so generalizing it team-wide closes a design inconsistency
  (one agent on a fundamentally different capture mechanism than the other eleven) rather than
  introducing a new one.
- **Verified:** `bash claude/scripts/audit-team.sh` — same two pre-existing, unrelated FAILs only
  (username/home-path leak in `falkor-chat/docs/test-reports/graphrag-eval-report.md`, predates
  this session). All 4 populated graphs' entry counts confirmed live via `MATCH (e:KaizenEntry)
  RETURN count(e)`. `grep -rn "kaizen/inbox\|graph-dba" skills/agent-maintenance/SKILL.md`
  reviewed line-by-line — remaining hits are correctly-retained historical/precedent references
  (origin notes, the pilot citation), not stale operative instructions.
- **Docs touched:** all 11 non-`graph-dba` agent `.md` files (Learning-capture section) + their
  `kaizen/inbox.md` (frozen) + their `kaizen/history.md` (this migration's entry); `cobb.md`
  (Learning-capture + distillation paragraph); `data-scientist.md`/`tico.md` frontmatter
  (`tools:`); `skills/agent-maintenance/SKILL.md`; `claude/AGENTS.md`; root `AGENTS.md`;
  `cobb/kaizen/plan.md`.
- **Plan items:** open follow-up noted in `plan.md` — a live §5 distillation pass against the 4
  populated graphs (20 entries total) hasn't run yet; scoped as ordinary future distillation
  work, not blocking.

## 2026-08-19 — Verbosity-diagnosis item 1 executed: Learning-capture de-dup via existing inbox headers (11-agent sweep, own prompt included)
- **What:** Investigated before executing the originally-proposed fix ("extract into a skill
  pointer"). Found the extraction target already exists: every `kaizen/inbox.md`'s header already
  carries the exact entry-format/promotion-model text (agent-maintenance skill §5's seeded
  template) that each agent's "Learning capture" paragraph was separately restating — genuine
  duplication, not just similar-looking boilerplate, and a stronger target than a new skill file
  would have been (the `agent-maintenance` skill isn't loaded by the other 11 agents; the inbox
  file is something every one of them already opens to append). Trimmed the redundant clause
  — "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture —
  the team maintainer verifies and promotes..." — out of `analyst`, `architect`, `coder`,
  `data-scientist`, `devops`, `frontend-engineer`, `qa-engineer`, `tdd-engineer`, `teco`, `tico`,
  and `cobb.md` itself (all but `graph-dba`, whose graph-based capture mechanism has no comparable
  file header — never part of the "near-identical" set to begin with). Kept in every file: the
  discipline-specific fact-kind clause (real, non-boilerplate customization), the inbox path,
  "skip task-specific details," "never edit your own agent definition," and the write-guard
  clause where the agent is doc-scope-guarded.
- **Why:** Continuation of the 2026-08-19 verbosity diagnosis — item 1 was the largest of the four
  parked items (13 files vs. 1) and the one flagged as needing the most design judgment (where
  should the shared text live). Investigating first rather than defaulting to the original "new
  skill" proposal avoided creating an indirection (a file most agents don't load) when a better
  target (a file they already touch) was sitting there.
- **Verified:** `bash claude/scripts/audit-team.sh` clean (no new FAILs; same two pre-existing,
  unrelated FAILs in `falkor-chat/docs/test-reports/graphrag-eval-report.md`). Spot-checked word
  counts: ~22 words saved per file across the 10 non-`cobb` agents (analyst 2285→2263, architect
  1427→1405, coder 1079→1057, tdd-engineer 1830→1808, qa-engineer 1845→1823,
  frontend-engineer 1444→1422, teco 4074→4051), plus `data-scientist`/`devops`/`tico`/`cobb`
  (not previously measured, edited identically).
- **Docs touched:** the 11 agent `.md` files above, their `kaizen/history.md` files, and this
  agent's own `kaizen/plan.md` (item 1 marked done, item 4 now the sole remaining item).
- **Plan items:** parking-lot verbosity item updated — items 1, 2, and 3 done; item 4
  (hedge-pruning) remains the only open slice.

## 2026-08-19 — Verbosity-diagnosis items 2 & 3 executed: incident-narrative excision (teco), run-on-sentence-to-sub-list (analyst)
- **What:** User picked two of the four parked verbosity-diagnosis items (own history entry
  below) to execute now. (2) `teco.md`'s step-table sizing bullet kept its operative rule but
  dropped the inline K-042 incident narrative (458k tokens/222 tool calls, dropped test files,
  stakeholder quote) in favor of a dated pointer to `teco/kaizen/history.md`'s 2026-08-11 entry,
  which already carries the full story verbatim — nothing lost, just de-duplicated between prompt
  and change log. (3) `analyst.md`'s "Evidence over vibes" Guardrails bullet — a single run-on
  sentence carrying 5 sub-rules after four separate clause extensions, flagged 2026-08-09 and
  never fixed — restructured into a lead sentence plus a 5-item sub-list under the same bullet;
  content unchanged, no new top-level Guardrails bullet.
- **Why:** Continuation of the 2026-08-19 verbosity diagnosis below; these two were the
  lowest-risk, single-file, no-content-loss slices (unlike item 1, which touches all 13 files, or
  item 4, which needs a case-by-case judgment call on which hedges are actually backed
  structurally).
- **Verified:** `bash claude/scripts/audit-team.sh` clean (no new FAILs; the two pre-existing
  unrelated FAILs in `falkor-chat/docs/test-reports/graphrag-eval-report.md` untouched). Word
  counts: `teco.md` 4137 → 4074 (−63); `analyst.md` 2271 → 2285 (+14, sub-list markdown overhead
  — the goal was scannability, not further shrinkage).
- **Docs touched:** `claude/teco/teco.md`, `claude/analyst/analyst.md`,
  `claude/teco/kaizen/history.md`, `claude/analyst/kaizen/history.md`, `claude/cobb/kaizen/plan.md`.
- **Plan items:** parking-lot verbosity item updated — items 2 and 3 marked done, items 1 (Learning
  capture dedup) and 4 (hedge-pruning) remain open.

## 2026-08-19 — Team verbosity diagnosis + CPG-freshness centralization on teco (7-agent sweep)
- **What:** Two-part session. (1) **Diagnosis** (no edits): user asked why the team's prompts are
  verbose. Measured evidence: the "Learning capture" section (~90–190 words, near-identical) in
  all 13 agent files (~1,500 words repo-wide); the CPG freshness-check paragraph duplicated
  verbatim across 6 files (~780 words); `teco`'s step-table sizing rule carrying a ~100-word
  incident narrative inline; a parking-lot precedent (`analyst`'s "Evidence over vibes" bullet,
  flagged 2026-08-09) showing kaizen distillation as an additive-only ratchet. Recommended:
  extract duplicated boilerplate into skill pointers, move incident narratives to `history.md`,
  convert nested prose caveats into tables, prune hedges once a rule has structural backup. (2)
  **Executed one concrete slice** the user chose to act on immediately: moved CPG
  freshness-checking off `analyst`/`architect`/`coder`/`tdd-engineer`/`frontend-engineer`/
  `qa-engineer` (each lost the ~130-word freshness paragraph) and centralized it on `teco`
  (`mcp__cypher__query` added to its `tools:`; new §3 dispatch-time freshness bullet). Fully
  centralized per explicit user choice — a specialist run standalone no longer checks freshness
  at all, an accepted capability loss. Wrote `docs/plans/cpg-agent-adoption2.md` (Extends the
  archived `cpg-agent-adoption.md` — AC-2's `CPG:` line and the six-agent CPG-orientation contract
  are untouched); added the `Extended by:` header pointer on the archived original. Updated
  `skills/cpg-analysis/SKILL.md` §4 and `references/freshness.md`'s consumer line. Logged a dated
  `history.md` entry in each of the 7 touched agents; flagged the `mcp__cypher__query` grant as
  **not yet live-verified** (teco has a known Grep/Glob declared-but-absent precedent, 2026-08-10)
  in `teco.md`'s Guardrails and a `teco/kaizen/plan.md` parking-lot item.
- **Why:** User-directed — a direct design conversation, not a `teco`-coordinated unit, so no
  independent-review gate was invoked; the design fork (coordinated-only vs. fully-centralized;
  teco-only vs. teco+tico) was resolved via `AskUserQuestion` rather than guessed. Recorded here
  because it's a real behavior change to a shipped, AC-gated feature (M4) plus a same-run
  demonstration of the diagnosis's own prescription (boilerplate removed, no incident narrative
  added to the operative prompts — the origin lives in this entry and the plan doc instead).
- **Plan items:** the broader diagnosis (Learning-capture dedup, teco's step-table narrative
  excision, `analyst`'s run-on-sentence table conversion) is **not yet executed** — the user asked
  only for the CPG-freshness slice this session. Parking below.

## 2026-08-18 — Own-inbox promotion: strengthened `agent-maintenance` §5 step 2 ("verify") after a real graph-dba distillation pass surfaced a citation that was real but wrong
- **What:** While running the real `kaizen_graph_dba` distillation pass (see `claude/graph-dba/
  kaizen/history.md`'s matching 2026-08-18 entry), re-verifying entry `f8c28d75…` against
  `falkor-chat/server/falkorchat/repository.py` found the entry's own quoted "Evidence" had
  omitted a real `CREATE (sr)-[:RAN]->(cur)` edge present in the exact function it cited —
  and, via `git log -S`, present since five weeks before the entry was even written. Logged this
  as a learning in `cobb/kaizen/inbox.md`, then promoted it same-run: `skills/agent-maintenance/
  SKILL.md` §5 step 2 now reads "**Re-derive the fact yourself; don't just confirm the entry's
  cited evidence still exists at that path/line** — a citation can be real and still misdescribe
  what's there," with this origin note attached.
- **Why:** Durable process fact about the distillation procedure itself, not about any one
  agent's domain — belongs in the skill (every future distillation pass pays for reading it,
  and every future pass benefits from not repeating the miss). High enough value to promote
  same-run rather than let it sit; cobb is the one agent whose maintainer role puts full §1/§2
  bookkeeping for its own inbox in-bounds mid-run.
- **Verified:** re-read the edited skill section after the change — reads correctly, doesn't
  contradict the surrounding step-2/step-3 text, and the origin note's facts (edge name, commit
  `3921f87`, 2026-07-12 date) match what was independently confirmed during the graph-dba pass.
- **Docs touched:** `skills/agent-maintenance/SKILL.md` §5 (knowledge-base edit), `claude/cobb/
  kaizen/inbox.md` (entry cleared after this write), this `history.md` entry.
- **Plan items:** none — not on an active K-item; a direct procedural fix.

## 2026-08-18 — generic-cypher-mcp U6 (steps 4a+4b): repo-wide catalog + both agents' operative prompts + `agent-maintenance` §5 retargeted from `inbox.md` to `graph-dba`'s working-memory graph

- **What:** `teco`-coordinated delivery of `docs/plans/generic-cypher-mcp.md` §7 steps 4a+4b (unit
  U6), following U5's real migration (`graph-dba/kaizen/inbox.md` frozen 2026-08-18, `kaizen_graph_dba`
  live with 6 `:KaizenEntry` nodes, `entryId` index + uniqueness constraint both `OPERATIONAL`).
  Seven files updated to describe `graph-dba`'s actual post-migration behavior — raw kaizen
  capture now targets the graph, not `inbox.md` — while leaving the convention correct and
  unchanged for every other agent, which still append to their own file-based `kaizen/inbox.md`:
  1. **`claude/AGENTS.md`** (line 3) — the directory-level convention sentence gained the
     `graph-dba` carve-out (writes `:KaizenEntry` nodes into `kaizen_graph_dba` via
     `mcp__cypher__query`, `inbox.md` now frozen) alongside the unchanged generic statement for
     everyone else.
  2. **`claude/README.md`** (Kaizen section) — same carve-out, phrased for the human catalog;
     distillation clause now states clear-by-inbox-edit vs. clear-by-curator-`DETACH DELETE`
     (`agent='cobb'`) as the two dispositions.
  3. **`docs/BACKLOG.md`** — added the `## M5 — Generic Cypher MCP` milestone-map row and section
     (items `C-501`…`C-506`, one per §7 step 1/2/3/4a/4b/5), per plan §6 verbatim, inserted after
     M4's own Follow-ups block and before the legacy `## Follow-ups (post-M2)` tail section.
     **Corrected same day, post code re-gate:** `analyst`'s diff-scoped review (`docs/reviews/
     generic-cypher-mcp.md`, "Code re-gate (U6, diff-scoped)") caught that the first draft marked
     all six items 🔵 proposed, contradicting the M4 precedent (commit `50f9aaa`) where individual
     items flip to ✅ as their steps land and the milestone-map row itself carries 🟡 with prose
     naming what's still queued. Fixed to match: `C-501`…`C-505` (steps 1/2/3/4a/4b — U4/U4-fix,
     U5, and this U6 are all delivered and gated) → ✅; `C-506` (step 5, `qa-engineer`'s acceptance
     pass) stays 🔵 proposed, genuinely still queued; the milestone-map row → 🟡 with "Implementation
     (C-501…C-505) complete; step 5 (`qa-engineer` acceptance pass against AC-1…AC-8) still queued",
     mirroring M4's exact phrasing shape.
  4. **`claude/graph-dba/graph-dba.md`** — "Learning capture" section's inbox-append instruction
     replaced with the graph-write instruction (concrete `CREATE (...:KaizenEntry {...})` Cypher +
     `mcp__cypher__query(..., agent='graph-dba')` call, matching the live-verified schema/pattern from
     `docs/plans/generic-cypher-mcp-graph.md` §1 and U5's actual migration call). The
     `falkordb-quirks.md` direct-home carve-out is untouched. Own `kaizen/history.md` entry added
     (2026-08-18, this date).
  5. **`claude/cobb/cobb.md`** — the "Learnings distillation" bullet's blanket "every agent (you
     included) appends... to its `kaizen/inbox.md`" claim corrected: `graph-dba`'s raw capture is
     now graph-based (cited: `:KaizenEntry` via `mcp__cypher__query`, attributed to itself), every
     other agent's is still file-based, exactly as before. The clear-step clause now names both
     dispositions (curator `DETACH DELETE` with `agent='cobb'` for `graph-dba`, an inbox edit for
     everyone else) instead of only "clear the inbox."
  6. **`skills/agent-maintenance/SKILL.md` §5** — the largest edit. The section intro now states
     the `graph-dba` exception and its schema/attribution shape; step 1 ("Read every inbox") gained
     the live graph-read query; step 4 ("Log & clear") now branches — a file-based agent's entry is
     removed from `inbox.md` directly, while `graph-dba`'s entries follow a **non-negotiable**
     four-step sequence (read the raw entry → verify → `Edit history.md` and confirm the write
     succeeded → **only then** curator-clear via `mcp__cypher__query(..., cypher="MATCH (e:KaizenEntry
     {entryId:'<id>'}) DETACH DELETE e", agent='cobb')`) — this is the **one and only place** the
     append-before-delete ordering constraint is documented, per `docs/plans/generic-cypher-mcp.md`
     §3.5's explicit resolution of the plan-gate's open question (deliberately **not** duplicated
     into `graph-dba.md`, since `graph-dba` never runs the delete half). Also widened the inbox
     template's lead-in to flag it as the file-based schema (`graph-dba`'s schema instead lives in
     `docs/plans/generic-cypher-mcp-graph.md` §1), and lightly reworded the frontmatter
     `description`'s "verify → route → log → clear each agent's kaizen/inbox.md" clause to "…each
     agent's raw capture — kaizen/inbox.md, or graph-dba's working-memory graph" (889 → 940 chars,
     still under the 1024-char cap) so the skill's own routing description doesn't itself overclaim.
  7. **Root `AGENTS.md`** (line 34, the `claude/` component bullet) — added, per `teco`'s follow-up
     direction after the initial six-file pass: this bullet's own "each with a `kaizen/` plan +
     history + learnings inbox the agent appends to during runs" claim was the same stale-universal
     shape FR-11/AC-7 targets, just one file the plan's step table hadn't named. Carved out
     `graph-dba` with the same "except `graph-dba`, whose raw capture writes directly into the
     `kaizen_graph_dba` FalkorDB graph instead" phrasing already used in `claude/AGENTS.md` and
     `claude/README.md`, so all three carve-outs read consistently — a minimal, one-clause fix, not
     a rewrite of the bullet.
- **Close-out grep sweep (the plan's actual acceptance test, not a fixed file list):**
  `grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/ skills/agent-maintenance/SKILL.md` — **36
  files, both before and after** (identical file list; no file gained or lost a hit). Triaged all
  36: the six files above got the intended edit (confirmed by re-grepping each for its own
  `graph-dba`-carve-out sentence, present in every case); the remaining 30 are legitimately
  non-`graph-dba`-specific and correctly left untouched — every other agent's own
  `Learning capture` closing-protocol sentence and its `kaizen/history.md`'s dated "Added
  `kaizen/inbox.md`..." creation entries (11 agents × 1–3 hits each), the five doc-scoped write
  guards' `.sh` scripts (their allowlisted-path comments, unrelated to *which* agent),
  `audit-team.sh`'s kaizen-triple existence check (still correct — every agent, `graph-dba`
  included, still carries an `inbox.md` file on disk, just a frozen one), `claude/docs/requirements/
  security-expert.md` (a hypothetical future agent's requirements doc, out of scope), and
  `claude/graph-dba/kaizen/history.md`'s own dated entries narrating *past* inbox-based work
  (correctly left as an unedited historical record, not a live claim). Zero new false claims found,
  zero files touched outside the intended six (plus each edited agent's own `kaizen/history.md`).
  Root `AGENTS.md` was outside this sweep's path scope by design (the plan's grep command covers
  `claude/` + `skills/agent-maintenance/SKILL.md` only) — its own stale claim (item 7 above) was
  caught separately, by `teco`'s direct read, not by the sweep; re-running the sweep after fixing it
  confirmed the file count was unaffected (still 36), as expected.
- **Why:** `docs/requirements/generic-cypher-mcp.md` FR-11/AC-7 (every doc describing the standing
  kaizen-inbox convention updated to describe `graph-dba`'s actual behavior, starting with
  `claude/AGENTS.md`); `docs/plans/generic-cypher-mcp.md` §7's step table (4a/4b, split from a
  single step 4 at the plan-gate's B1 finding specifically because the two agents' own *operative*
  prompts were originally omitted) and its "Close-out done-condition for 4a+4b jointly" paragraph
  (a grep sweep, not a fixed list, is the real acceptance test).
- **Scope discipline:** the six files named in the original brief were edited, plus root
  `AGENTS.md` (item 7 above — `teco`-approved same-day expansion, not a unilateral scope
  widening: it named the plan's own author as not having foreseen that file, same spirit as the
  plan-gate's B1 finding, and asked for the fix folded into this unit), plus the two edited agents'
  own `kaizen/history.md` (this repo's standing "prompt edit → dated history entry" rule, confirmed
  by precedent in both agents' existing history files before writing these entries). `inbox.md`/
  `history.md` content from U5 (already done), `cypher-mcp/` (U4), and the requirements/plan/review/
  coordination docs for this feature were not touched.
- **Independent review:** `analyst`'s diff-scoped code re-gate (`docs/reviews/
  generic-cypher-mcp.md`, "Code re-gate (U6, diff-scoped) — 2026-08-18") — **approve with
  suggestions**, no blocker. One Major (M-B, the `docs/BACKLOG.md` status-marker convention gap,
  fixed above) and one minor (m-B, this entry's missing mention of the root `AGENTS.md` edit —
  fixed by this same update) closed same day, in place, no new review cycle.
- **Verified:** re-read each of the seven edited files after the edit; `docs/BACKLOG.md`'s M5
  table row and item list checked against the plan's §6 verbatim text and item mapping, then
  against the M4 precedent's (`50f9aaa`) status-marker convention for the M-B fix;
  `claude/scripts/audit-team.sh` clean before and after (96 PASS, same 2 pre-existing FAILs in an
  unrelated `falkor-chat` file, no new failures).
- **Docs touched:** `claude/AGENTS.md` · root `AGENTS.md` · `claude/README.md` · `docs/BACKLOG.md` ·
  `claude/graph-dba/graph-dba.md` (+ its own `kaizen/history.md`) · `claude/cobb/cobb.md` ·
  `skills/agent-maintenance/SKILL.md`.
- **Plan items:** none opened or closed — not on the active K-list.

## 2026-08-16 — Cross-session peer-addressing near-miss: same-run promotion into `agent-standards/claude-code.md` + `teco.md`

- **What:** in this session, resuming a paused K-026 GraphRAG-eval coordination, I called
  `ListAgents`, saw one row named `teco`, and sent it a full multi-paragraph resume brief on the
  assumption — from stale prior-session summary context — that it was the K-026 coordinator. It
  was actually a different, independently-launched `teco` session mid-coordination on an
  unrelated task (`cpg-agent-adoption`). That session's own human caught the mismatch and it
  stepped back cleanly, having done only a small amount of safe read-only work (a test re-run, a
  state-restoring reseed) first. No K-026 harm done, but a wasted round-trip and a real near-miss:
  bare-name `SendMessage` addressing is ambiguous whenever more than one independently-launched
  session shares an agent name, and `ListAgents` showing one row is not proof there's only one.
- **Why same-run promotion (not just an inbox entry):** this is a durable, non-obvious
  Claude-Code-mechanism fact (SendMessage/ListAgents cross-session peer addressing), squarely in
  scope for the `agent-standards` skill I already maintain, and the fix needed enforcing
  immediately (I was about to repeat the same pattern to actually dispatch K-026's next steps).
- **Where it landed:**
  1. **`skills/agent-standards/claude-code.md`** — new "Cross-session peer addressing (`SendMessage`
     + `ListAgents`)" subsection, verified 2026-08-16: documents that `SendMessage`/`ListAgents` now
     reach independently-launched peer sessions (not just Agent-Teams teammates, which is what the
     skill's existing, older Agent-Teams section describes — flagged as not yet reconciled, not
     overwritten), the bare-name ambiguity, this incident as evidence, and the practice (identity
     probe before a substantive brief to an unverified peer; prefer a fresh subagent reading
     persistent state when in doubt).
  2. **`claude/teco/teco.md`** step 4 — one new bullet next to the existing "incoming
     resume/pause message" rule: a message describing a task absent from the session's own ledger
     is a *misrouting* signal, not just a staleness one — pause and confirm identity with the user
     before doing *anything*, even read-only checks, rather than only re-verifying the claimed
     facts. (This teco session's own good behavior — decline + minimal safe work + explicit
     transparency — is exactly what this bullet now asks for explicitly, rather than leaving it to
     the receiving human to catch.)
- **Verified:** `bash claude/scripts/audit-team.sh` clean before and after (98+ PASS, 0 FAIL). No
  personal identifiers introduced. Practice fix applied immediately in this same session: the
  actual K-026 resume was then re-sent only after independently confirming (via the correcting
  peer's own message) which session was *not* it, and no further bare-name dispatch was attempted
  without that confirmation.
- **Plan items:** none opened in `cobb`'s own plan — this was a direct fix, not a backlog item.
  Counterpart: `teco/kaizen/history.md` gets its own entry for the `teco.md` edit.

## 2026-08-15 — Distillation redirect: teco's 2026-08-12 credit-crash-recovery entry lands here as a parking-lot idea, not a prompt change

- **What:** processing `teco/kaizen/inbox.md`'s sole entry (agent-maintenance §5) found its
  suggested home (`teco.md` step 2/3) didn't fit — the coordination-ledger rule it asked for
  already existed and predated the incident, because the incident's original unit (a 39-file
  team-wide kaizen sweep I ran directly, gated "needs changes" by `analyst`) was never routed
  through `teco` at all. Discarded from `teco`'s inbox; the underlying observation — a
  directly-invoked review-gated sweep survived a mid-run credit crash only because the review
  doc was self-sufficient (baseline commit + explicit scope) — is about *my own* operating
  pattern, not teco's, so it's logged here instead: `plan.md` parking lot, no prompt change (one
  data point, no repeat, and the safety net that saved it is already `analyst`'s standing
  practice).
- **Why:** no directly-invoked agent should let a durable-but-unpromoted observation about its
  own risk pattern evaporate, even when it doesn't clear the bar for a prompt change.
- **Verified:** `docs/plans/kaizen-inbox-distillation2-coordination.md` and
  `docs/reviews/kaizen-inbox-distillation2.md` cross-checked against the inbox entry's claims —
  matched exactly (K-041 cross-reference, no-backlog-id header, `agentId (prior session, not
  resumable)` row).
- **Plan items:** none promoted; parking-lot idea added (see `plan.md`).

## 2026-08-12 — Fixed every Blocker/Major/Minor/Nit from `analyst`'s gate on the 2026-08-11 distillation

- **What:** worked `docs/reviews/kaizen-distillation-2026-08.md`'s full findings list (verdict:
  needs changes — B-1, M-1..M-5, m-1..m-4, n-1/n-2) directly in the still-uncommitted working
  tree from the 2026-08-11 distillation, without opening a new diff.
- **B-1 (coder inbox entries dropped with no disposition):** found `coder/{coder.md,kaizen/
  history.md}` already carrying a corrected version (both learnings promoted to `coder.md` step 5
  — attributed-delta reporting and the skip-count clause mirroring `tdd-engineer.md:42` — and the
  history entry's promoted list already scoped to real `coder`-inbox entries). Fixed the one thing
  still wrong: the header arithmetic ("6 promoted, 1 discarded, 1 promoted late" → **"5 promoted,
  1 discarded, 2 promoted late"**, since both B-1 gap entries were late promotions, not one).
- **M-1 (5 more unlogged dispositions, 4 miscounted history headers):** `graph-dba` — added
  discard dispositions for the two 2026-07-19 CPG-topology entries ("already covered in
  `skills/joern-cpg/references/cpg-model.md`"), fixed the header 5→7. `qa-engineer` — added
  discards for the MCP `send_message` asymmetry (already in `DESIGN.md` §14.7, already tracked as
  **K-041**) and the Bash-tool-backgrounding entry (already in `skills/agent-standards/
  claude-code.md`); header was already 15 but two dispositions were missing from the prose.
  `devops` — fixed header 9→13 (all 13 were already described). `tico` — fixed header 3→4, added
  the missing 2026-07-31 `version`/`defVersion` discard ("already tracked as **K-040**"). Also
  logged, per the review's note: `teco`'s inbox held 6 headed entries + one headless continuation
  block correctly folded into the preceding entry — flagged in `teco/kaizen/history.md` so a
  future pass doesn't re-count it.
- **M-2 (`python-web-quirks` description didn't cover 5 of its 8 entries):** extended the
  frontmatter `description` to name all 8 topics; kept the char count under the 1024-char cap
  (`skills/README.md`'s own stated limit). **Judgment call:** widened the skill's stated scope
  ("mostly web/async, plus two general pytest/import-timing traps") rather than splitting the two
  non-web entries (`monkeypatch.setenv` timing, function-local import binding) into a new skill —
  both surfaced in the same Python web codebase, the consumer roster is identical
  (`coder`/`tdd-engineer`/`architect`/`analyst`), and a 2-entry skill for a narrow pytest-timing
  niche isn't worth the new-`SKILL.md`/catalog-row overhead yet. Updated `skills/README.md`'s
  catalog row to match.
- **M-3 (3 new agent KBs not annotated in `claude/AGENTS.md`):** added the parenthetical KB
  annotation (pattern already used for `graph-dba`) to `devops`, `qa-engineer`,
  `data-scientist`, and `analyst`'s pre-existing `review-techniques.md` (never annotated before).
- **M-4 (verify-only):** confirmed `teco`'s direct fix to `cypher-mcp/server.py`'s module docstring
  is present and reads correctly (`RESULTSET_SIZE` framing, matching the other 3 corrected sites).
  Did not re-edit `server.py`. Updated `graph-dba/kaizen/history.md`'s entry to record the
  `server.py` fix alongside the `README.md` one it previously listed alone.
- **M-5 (sizing rule placement):** added a one-sentence forward pointer in `teco.md` step 2
  ("Decompose & sequence") to the full rule already stated in step 3, per the review's suggested
  wording.
- **m-1 (wrong test-file count):** fixed "3 of ~15 plan-named test files" → "3 of the 11 test
  files the plan names — 3 of the 5 rewired consumer bindings" in `teco.md`, `teco/kaizen/
  history.md`, and this file's own 2026-08-11 entry (three sites, same source typo).
- **m-2 (`QUERIES.md` §11.2 duplicated the callout below it):** collapsed the new prose to one
  forward-pointing sentence and dropped the "the schema guarantees" overstatement in the same
  edit (the rewrite no longer states the claim at all, so it can't overstate it).
- **m-3 (`DESIGN.md` §14.7 re-parented the K-042 bullet into the new QA list):** moved the new
  5-bullet block to *after* the K-042 bullet, restoring its original adjacency to the "Verifying a
  claimed test count safely" paragraph. Fixed `qa-engineer/kaizen/history.md`'s "four new …
  bullets" → "five," naming the MCP `send_message` one as the fifth (done together with M-1's
  `qa-engineer` fix, same entry).
- **m-4 (~180 words of K-042 forensics in `teco.md`, take-or-leave):** **judgment call: kept as
  is.** After the m-1 wording fix the sizing bullet is tighter than the review measured, and the
  review's own counter-argument — the stakeholder's verbatim quote is what keeps the rule from
  eroding — is the one I find more persuasive for an always-loaded orchestration rule tied to a
  standing user directive. Not a hill worth re-litigating if a future pass disagrees.
- **n-1/n-2 (double-logged promotions):** removed the no-string-repetition promotion and the
  `db.indexes()` discard from `graph-dba/kaizen/history.md` (both actually `coder`'s and `teco`'s
  own inbox entries respectively, each already correctly logged in *those* agents' histories).
- **Open question 2 (MCP `send_message` wants a backlog item) — moot, not acted on:** already
  filed as **K-041**, delivered 2026-08-01 (`falkor-chat/docs/BACKLOG.md:1242`); noted as
  "already tracked" in the `qa-engineer` disposition instead of filing anything new.
- **Not touched, per instructions:** open question 1 (the review document's own rename/collision
  fix — `analyst`'s call) and open question 3 (`coder.md`/`tdd-engineer.md` convergence — left
  open for the stakeholder).
- **Incidental fix, not in the findings list:** `bash claude/scripts/audit-team.sh` initially
  **FAILed** check 7 (personal-info leak) — the review document itself
  (`docs/reviews/kaizen-distillation-2026-08.md:7`) and a new `analyst`-inbox entry it prompted
  (`claude/analyst/kaizen/inbox.md:26`, appended by `analyst` *during* the review, dated
  2026-08-11, not yet distilled) both quoted the live absolute repo path verbatim. Genericized
  both to `/home/<user>/prg/graphmind-ai-lab`. Left `analyst`'s two new 2026-08-11 inbox entries
  otherwise unprocessed — they postdate the distillation this review gated and are legitimate
  material for the *next* §5 pass, not this corrective one.
- **Verified:** `bash claude/scripts/audit-team.sh` → `RESULT: PASS` (deterministic checks clean)
  after all fixes above, including the incidental PII leak.
- **Docs touched:** `claude/coder/kaizen/history.md` · `claude/graph-dba/kaizen/history.md` ·
  `claude/qa-engineer/kaizen/history.md` · `claude/devops/kaizen/history.md` ·
  `claude/tico/kaizen/history.md` · `claude/teco/{teco.md,kaizen/history.md}` ·
  `claude/analyst/kaizen/inbox.md` · `claude/AGENTS.md` ·
  `skills/python-web-quirks/SKILL.md` · `skills/README.md` ·
  `falkor-chat/docs/{QUERIES.md,DESIGN.md}` · `docs/reviews/kaizen-distillation-2026-08.md`.

## 2026-08-11 — Full-team inbox distillation triggered by a 400k-token context-blowout report; diagnosed orchestration (not verbosity) as the cause

- **What:** stakeholder reported several recent sessions (teco included) blowing past 400k tokens
  of context, floating two hypotheses ("too verbose" vs. "orchestration not good"). Diagnosed
  using evidence already in the team's kaizen files rather than assuming, then ran a full §5
  distillation sweep across every agent inbox with unprocessed entries (teco, coder, analyst,
  architect, cobb's own, data-scientist, devops, graph-dba, qa-engineer, tico —
  frontend-engineer/tdd-engineer were already empty).
- **Diagnosis:** prompt bodies are 42–274 lines, already through two team-wide slimming passes
  (2026-07-11/2026-07-24); `coder.md` ≈1.4k tokens, `teco.md` ≈5.7k — small, roughly-constant
  additions to context, not something resent 222 times in a way that explains a six-figure total.
  The cited incident (K-042 Landing 1: one `coder` dispatch covering 6 plan steps/~10 files,
  458k tokens/222 tool calls/~45 min per `/context`) is context growth from the sheer volume of
  file reads/diffs/test-run output accumulated across one unbroken multi-step session — an
  orchestration/dispatch-sizing problem, not a prompt-size problem. The same oversized unit's
  `analyst` gate also found 3 of the 11 test files the plan names — 3 of the 5 rewired consumer
  bindings — silently dropped from its own stated scope — a correctness cost from the same cause,
  not just a token cost. **Verdict: orchestration,
  not verbosity.** No verbosity contributor found worth a further pass.
- **Delivered:** promoted a dispatch-sizing standing rule into `teco.md` (tied explicitly to the
  stakeholder's own "please never again create a landing so big" directive, quoted in the prompt
  so it can't silently erode) plus 4 smaller promotions from the same inbox; swept the other 9
  non-empty inboxes per §5 — 3 new on-demand knowledge bases created (`claude/devops/
  ops-quirks.md`, `claude/qa-engineer/qa-testing-techniques.md`,
  `claude/data-scientist/lm-studio-model-notes.md`), ~20 entries folded into existing knowledge
  bases (`skills/python-web-quirks/SKILL.md`, `claude/graph-dba/falkordb-quirks.md`,
  `skills/cpg-analysis/SKILL.md`, `claude/cobb/TESTING.md`), several project-doc corrections
  (including fixing two now-incorrect "the reported total is always exact" claims in
  `cypher-mcp/README.md` and `skills/cpg-analysis/SKILL.md`, and an incorrect invariant claim in
  `falkor-chat/docs/QUERIES.md` §11.2), and small prompt additions to `analyst.md`, `tico.md`, and
  `data-scientist.md`. My own 3 inbox entries (subagent tool-set narrower than frontmatter,
  agent-definition edits needing a fresh session to verify, AutoMem index-only-to-subagents) went
  into `skills/agent-standards/claude-code.md`.
- **Left unresolved, flagged to the stakeholder (not guessed at):** `tico`'s inbox carried a
  2026-07-31 entry recording the stakeholder pushing back twice on tico's Agent/write-scope
  guardrails and asking to relax them — two shapes proposed, neither self-evidently right. Moved
  to `claude/tico/kaizen/plan.md` as an open item pending a stakeholder decision rather than
  promoted or discarded.
- **Caught one stale finding before promoting it:** `graph-dba`'s inbox reported (2026-07-30) that
  `pipeline.sh --reset` bypasses the destructive-ops guard. Cross-checking `claude/AGENTS.md`'s
  hook-machinery section before writing this up showed the gap was already closed 2026-08-08
  (C-311) — the guard now pattern-matches that wrapper directly. Rewrote the knowledge-base entry
  to state the fix instead of re-filing an already-closed gap as new work.
- **Verified:** `bash claude/scripts/audit-team.sh` clean before and after. No personal
  identifiers introduced across ~25 edited/created files.
- **Docs touched:** see each agent's own 2026-08-11 `history.md` entry for its file list; this
  entry is the team-wide summary.

## 2026-08-10 — Reviewed and reworked `teco`'s coordination/tracking/continuation machinery (§7 lint + §5 distillation)

- **Scope:** user asked for a review of how `teco` coordinates, keeps track of tasks, and routes
  between *running* and *fresh* agents. Read `teco.md`, all three kaizen files, the five real
  `*-coordination.md` ledgers on disk, and ~20 transcripts; then implemented the full rework the
  user approved (all three scoping choices taken at the recommended option).
- **What the review found (evidence, not impression):** `SendMessage` — the continuation
  mechanism K-007 shipped on 2026-07-29 — appears **42×** across this box's transcripts and **0×**
  in any confirmed teco run. Delegate identity lived only in teco's context window, which
  compaction destroys on exactly the long coordinations where continuation matters. None of the
  five real ledgers records a running delegate's id or an in-flight state, and each invents its
  own table shape. The prompt had **no in-flight model at all** despite `Agent` defaulting to
  background. Two independent instances of a `model:"haiku"` doc-closeout fabricating numbers sat
  unpromoted in the inbox.
- **Delivered:** ledger schema mandatory at 3+ units or any gated unit; resume-from-ledger path;
  a new "Track what's in flight" step; `agentId` recorded at dispatch and used to address
  `SendMessage`; the haiku-fabrication rule paired with mandatory numeric re-verification; five
  standing practices promoted out of the user's AutoMem file into the committed prompt; steps 3–5
  and the Guardrails commit bullet split into sub-bullets (§7 dimension 4 — three parking-lot
  deferrals closed). Full detail in `claude/teco/kaizen/history.md`.
- **§7 lint findings this pass** (on `teco.md`): *cognitive load* — **major**, two ~350-word
  blocks holding ~11 sub-rules, fixed by splitting before adding; *coverage* — **major**, no
  in-flight/abandon/self-resume paths, all three added; *ambiguity* — **minor**, "large or
  long-running work" had no operational test (now a unit-count threshold), "note the id when a
  follow-up seems likely" required predicting the future (now always), "have the reviewer
  re-check" didn't say fresh-or-resumed (now the same reviewer by id); *composition* — **major**,
  five practices lived only in user-scoped memory that reaches a subagent as an index without
  bodies; *contradiction* and *persona* — clean.
- **Three harness facts discovered by live probe, filed to my own inbox** (not yet promoted):
  a subagent's runtime tool set can be **narrower than its frontmatter** (`Grep`/`Glob` declared
  and absent, silently); custom agent definitions load at **parent-session start**, so an
  agent-definition edit cannot be verified from the session that made it; AutoMem reaches a
  subagent as an **index only**, never entry bodies. The first two cost a wasted probe before the
  confound was spotted — the corollary (verify agent edits from a fresh session) is the durable
  lesson.
- **Verified:** `claude/scripts/audit-team.sh` 98 PASS / 0 FAIL before **and** after (diff, not a
  bare gate, per `agent-maintenance` §4). No personal identifiers in any edited file.
- **Docs touched:** `claude/teco/teco.md` · `claude/teco/kaizen/{history,plan,inbox}.md` ·
  `claude/README.md` (teco row) · root `AGENTS.md` (flip-table authority) ·
  `falkor-chat/AGENTS.md` (`node` on PATH) · `claude/cobb/kaizen/{history,inbox}.md`.

## 2026-08-09 — Diagnosed tico's Portuguese-greeting bug; removed its `initialPrompt` + language-mirror rule; renamed "first-order" → "interactive" team-wide
- **What:** User asked why tico's opening line kept defaulting to Portuguese, and whether spending
  tokens on a self-introduction was worth it at all. Diagnosed live: tico's `initialPrompt`
  auto-submits as the first *user* turn in main-session mode, but that turn carries no real
  linguistic evidence (nobody has actually written anything yet), so the "mirror if they write in
  it" half of tico's language rule was vacuously false and the "English by default" half should
  have governed — instead the model most likely leaned on other in-session context (the operator's
  git identity, a Portuguese-signaling name) to guess a language, overriding
  the stated default. Fixed at the source rather than by strengthening the instruction: removed
  `initialPrompt` and the language-mirror line from `claude/tico/tico.md` entirely, so there's no
  ungrounded first line left to guess from — mode selection is now explicitly inferred from the
  stakeholder's real opening message (full detail in `claude/tico/kaizen/history.md`, same date).
  Separately, the user corrected my own inline claim that tico was "the only first-order agent" —
  teco is also designed to converse with and pause for the human; the word "first-order" (my own
  coinage, not a Claude Code term) was the flawed part, not the roster. Asked the user to pick a
  replacement via `AskUserQuestion` (interactive / foreground / human-facing) — **interactive**
  won — and renamed it everywhere it labeled this quality: `claude/tico/tico.md` (description +
  body), `claude/AGENTS.md`, `claude/README.md` (two spots), `claude/teco/teco.md`'s reference to
  tico, and `skills/agent-standards/claude-code.md`'s generic "MAIN session agent" section header
  (which is where I'd originally coined "first-order, conversational agent" as the category label
  for that Claude Code mechanism). Also promoted the underlying gotcha — an `initialPrompt`
  greeting plus a "default language" rule don't reliably compose, because the model has other
  context to lean on besides the literal absence of user text — into that same skill section,
  tagged observed/not-doc-sourced (its own `Verified:` stamp block gained a matching line) so the
  next agent author wiring an `initialPrompt` greeting doesn't rediscover this the same way.
- **Why:** same-run promotion is in-bounds for me alone (per my own "Learning capture" rule) —
  this was a durable, non-obvious fact about how main-session `initialPrompt` interacts with
  language defaults, worth fixing in the skill immediately rather than parking it in an inbox.
  The terminology question (what to call "runs interactive with a human, not a delegation target
  for background work") was a genuine naming call, not mine to make unilaterally, hence
  `AskUserQuestion` rather than picking on my own judgment.
- **Verification:** re-read `claude/tico/tico.md` after edits — no dangling reference to
  `initialPrompt` or the removed language line remained; grepped the repo for stray "first-order"
  occurrences afterward and confirmed the only remaining hits are historical (dated kaizen
  `history.md` entries elsewhere, correctly left untouched as a dated record) or in an unrelated
  module (`mcp-monitor/docs/plans/mcp-monitor-coordination.md`, out of scope for this pass).
- **Plan items:** none of cobb's own opened — the follow-up (a live e2e check of tico's new
  opening) is tico's own K-007, not cobb's to hold.

## 2026-08-09 — Independent review of U1/U2/U6 (C-308, C-312, C-319 skill/doc units), including self-review
- **What:** Reviewed three parallel Wave-1 deliverables from `docs/plans/cpg-followups-coordination.md`
  against skill-authoring conventions: U1 (`graph-dba`, C-308, Q4 transitive-upward-call-closure
  recipe in `skills/cpg-analysis/references/impact-analysis.md`) and U2 (`graph-dba`, C-312,
  `--verify-prefix` on `skills/joern-cpg/scripts/pipeline.sh` + matching `SKILL.md` Gotchas) both
  approved clean — query logic traced by hand, `WITH`-splitting idiom cross-checked against
  `test-gap.md`, shell repeatable-flag parsing and no-short-circuit reporting verified line-by-line,
  and the "red herring" root-cause claim corroborated against pre-existing `docs/HISTORY.md`/
  `docs/BACKLOG.md` text (not fabricated). U6 (my own earlier same-session C-319 promotion into
  `skills/agent-standards/claude-code.md` §MCP) got a self-review per the brief's explicit ask not to
  rubber-stamp it: the distillation bookkeeping (verify/route/log/clear) checked out, but I found the
  promoted bullet's "distinct from the discovery mechanism, which stays uniform via
  `$CLAUDE_PROJECT_DIR`" clause asserts an unverified causal link — WebFetch of
  `code.claude.com/docs/en/mcp` confirms `CLAUDE_PROJECT_DIR` is documented only as a spawned-server
  env var / path-expansion mechanism, not as how `.mcp.json` file discovery works, and neither the
  original inbox entry nor the C-319 backlog filing (which states the two facts as parallel, not
  causal) supports the "via" framing. Flagged as a Major finding with a concrete rewrite. Filed the
  review at `docs/reviews/cpg-followups-skills-impl.md` after a same-topic-path collision with
  `analyst`'s parallel U3–U5 review (both units wrote `docs/reviews/cpg-followups-impl.md`
  concurrently; the later write won) — split per each unit's brief, with a pointer note added to
  both files' headers.
- **Why:** Wave-2 review gate for this round, `cobb`'s domain per the C-303/C-307 precedent
  (skill/standards-content review). The self-review finding is exactly the kind of drift this
  agent's own machinery exists to catch — an inference added during promotion that outran its
  evidence, in a doc other agents will cite as fact.
- **Plan items:** none new; not on the active K-list. Overall verdict recorded in the review:
  approve with suggestions (U1/U2 clean, U6 needs one clause reworded before Wave 3 closeout).

## 2026-08-09 — C-319 follow-up: applied the self-review's suggested rewrite to `claude-code.md` §MCP
- **What:** On `teco`'s direction (trivial, docs-only, factual-accuracy fix, no design stakes —
  the "genuinely trivial" exception to independent review), applied the fix from my own review
  finding (`docs/reviews/cpg-followups-skills-impl.md`, U6): replaced the unsupported "distinct
  from the discovery mechanism, which stays uniform via `$CLAUDE_PROJECT_DIR` (see below)" clause
  in `skills/agent-standards/claude-code.md` §MCP → "Scopes, precedence, and the approval gate"
  with wording that states discovery's cwd-independence and `${CLAUDE_PROJECT_DIR}`'s
  cwd-independence as two parallel, separately-caused facts — matching how `docs/BACKLOG.md`'s
  C-319 filing already phrased it, per the finding. Re-read the edited paragraph after the change
  to confirm it reads cleanly.
- **Why:** Closes the Major finding without a second review loop, per `teco`'s explicit low-risk
  exception call.
- **Plan items:** resolves the specific instance in the 2026-08-09 parking-lot entry below (the
  general lesson about causal-compression during promotion stays open).

## 2026-08-09 — C-319: promoted `.mcp.json` approval-scoping fact into `claude-code.md` §MCP
- **What:** Added a bullet to `skills/agent-standards/claude-code.md` § MCP → "Scopes,
  precedence, and the approval gate": `.mcp.json` **discovery** walks up to the git root and is
  cwd-independent, but project-scope **approval** is keyed to the session's cwd — a session
  started in a subdirectory can see a server the root already approved as still `⏸ Pending
  approval`. Cited the source evidence directly (`claude/devops/kaizen/inbox.md`, 2026-07-25
  entry, C-319): `claude mcp list` from the repo root reported `✔ Connected`, the identical
  command from `falkor-chat/` reported `⏸ Pending approval`, and `~/.claude.json`'s `projects`
  map carried one entry (the root) and none for the subdirectory. Re-checked the `projects` map
  live today — still exactly one entry, same shape — before promoting; did not re-derive the
  `claude mcp list` contrast itself (a quick attempt in this environment hit unrelated infra
  failures — no reachable `docker`/FalkorDB backing the `cypher` server — so it couldn't cleanly
  reproduce either outcome; the inbox's original, cleaner evidence is what's cited). Promoted the
  underlying inbox entry out of `claude/devops/kaizen/inbox.md` into
  `claude/devops/kaizen/history.md` (agent-maintenance skill §5 distillation).
- **Why:** Backlog item C-319 — the fact was already filed in devops's inbox as durable and
  non-obvious; this closes the loop by moving it into the standards doc other agents actually
  read, in the same evidence-based style as the section's other observed-behavior bullets (e.g.
  the Lifecycle section's containerized-stdio-server note).
- **Plan items:** none new; not on the active K-list.

## 2026-08-09 — Review follow-ups: `claude/README.md` catalog completion + `claude-code.md` stamp gap
- **What:** Two minor fixes from independent reviews of the same-day inbox-distillation batch
  (`docs/reviews/{kaizen-inbox-distillation,analyst-inbox-distillation}.md`): (1)
  `claude/README.md`'s catalog rows for `architect`, `coder`, `tdd-engineer`, and `analyst` now
  mention the `python-web-quirks` skill, mirroring the existing `cpg-analysis` mention pattern in
  the same rows — each agent's frontmatter `description` already carried the routing clause
  (logged in each agent's own `kaizen/history.md` on creation), but the human-facing catalog had
  gone stale relative to the repo's own precedent. (2) `skills/agent-standards/claude-code.md`'s
  top `Verified:` stamp block gained a line for the new "Bash tool environment" section
  (shell-shadowed `find`/`grep`), observed 2026-07-26/2026-08-08, not doc-sourced — every other
  section already had a stamp line and this one was silently missing, a discoverability gap for
  anyone skimming just the header.
- **Why:** Both flagged as non-blocking minors by the two reviews; relayed by `teco` for a small
  follow-up fix rather than a new review cycle.
- **Plan items:** none.

## 2026-08-09 — Consolidated Kiro-facts edit: three held inbox entries (`analyst` #28, `architect` #15/#16) landed in `skills/agent-standards/kiro.md`
- **What:** Follow-up to the same-day full-team inbox distillation: three facts from two agents'
  inboxes had been deliberately left "held, not cleared" (rather than applied immediately) because
  all three target the same file and two concurrent distillation sessions writing it at once would
  race — `teco` queued them for one consolidated pass once both source sessions were done. Before
  writing, re-verified all three live against `kiro-cli 2.16.2` (the installed version had moved
  from `2.14.1` at original capture on 2026-08-01 to `2.16.2`) — all three held with no drift:
  1. **`kiro-cli agent create`'s default template is `"resources": []"`, never pre-populated**
     (`analyst` inbox #28). Reproduced: `EDITOR=true kiro-cli agent create <name> -d
     .kiro/agents` in a fresh scratch dir → `"resources": []` verbatim. Landed as an addition to
     the `resources` config-key bullet under "CLI custom agents — JSON," alongside the existing
     note on the separate `chat.disableInheritingDefaultResources` inheritance setting (which
     governs *inheritance*, independent of the template itself always starting empty).
  2. **Local-agent discovery is exact-CWD only, no upward directory walk** (`architect` inbox
     #15). Reproduced: an agent created in `probe/.kiro/agents/` is listed by `kiro-cli agent
     list` from `probe/`, but not from `probe/subdir/` (no local `.kiro/agents/` there, no
     parent fallback). Landed as a new sub-bullet under the CLI agents' `Location` bullet, with
     the practical consequence spelled out (a repo-checked-in agent's "no manual wiring" claim
     still depends on which directory the run command `cd`s into).
  3. **`mcpServers` remote entries carry no `"type"` field** — local vs. remote is discriminated
     by `command` vs. `url` key presence (`architect` inbox #16). Reproduced: `kiro-cli mcp add
     --name X --url <url> --agent <name> --force` wrote `"mcpServers": {"X": {"url": "..."}}`
     with no type key. Landed as a rewrite of the existing `mcpServers` bullet (which previously
     only documented the local `command`-keyed case, phrased as "each needs `command`" — now
     stale/incomplete) to cover both shapes, plus a flag that `falkor-chat/docs/DESIGN.md`
     §15.3's generic MCP-client example (`{"type": "streamable-http", "url": "..."}`) is a
     *different* client's config spelling, not kiro-cli's — don't copy that shape into a
     kiro-cli config.
  Bumped the file's top `Verified:` header block with a new sentence covering all three facts and
  the 2026-08-01 → 2026-08-09 re-verification span. Cleared the corresponding entries from both
  source inboxes (`claude/analyst/kaizen/inbox.md`, `claude/architect/kaizen/inbox.md` — both now
  at the standard empty placeholder) and logged the promotion in each agent's own
  `kaizen/history.md`.
- **Why:** Standing distillation duty (`agent-maintenance` skill §5) — these three facts had
  already cleared verification and disposition during the main distillation pass; only the
  shared-file write itself was deferred to avoid a race between the two source sessions. This
  entry was itself a bookkeeping gap flagged by independent review (`docs/reviews/kaizen-inbox-distillation.md`):
  the `kiro.md` content edit had landed without a matching `cobb` history entry, only the earlier
  review-log entry below existed.
- **Plan items:** none.

## 2026-08-09 — Independent second-opinion review of the `analyst` inbox-distillation pass (self-review-conflict routing)
- **What:** `analyst` would normally review a `cobb` deliverable, but the artifact under review here
  *was* `analyst`'s own prompt/kaizen files (the same-day full inbox distillation logged below and
  in `claude/analyst/kaizen/history.md`) — a real `analyst` session judging its own future prompt
  would be a self-review conflict, so `teco` routed the second opinion to a fresh `cobb` session
  (this one, no shared memory with the session that did the distillation) instead. Reviewed
  `analyst.md`, the new `review-techniques.md`, `analyst`'s kaizen history/inbox, the cross-checked
  `falkordb-quirks.md`/`claude-code.md` knowledge-base additions, and `cobb`'s own kaizen log for
  honesty — independently re-deriving technical claims rather than trusting citations: live-ran
  `GRAPH.PROFILE`/`EXPLAIN`-prefix/`sum(CASE...)` behavior against the running `falkordb-dev`
  container, installed and exercised `mcp` 1.28.1's `FastMCP` `outputSchema`/`structured_output`
  behavior in `cypher-mcp/.venv`, reproduced the pydantic nested-`exclude_unset` drop in
  `falkor-chat/server/.venv`, re-ran `claude/scripts/audit-team.sh` (full PASS), and reproduced the
  `DESIGN.md` SHA-lock re-extraction command byte-for-byte. Verdict: **approve with suggestions** —
  zero blockers/majors, every reproducible claim reproduced, `analyst.md`'s scope matched exactly
  what was pre-approved (1 new Guardrails bullet + clause extensions to one sentence + 2
  routing/pointer additions, nothing more). Two minor, non-blocking polish notes (the "Evidence
  over vibes" bullet reads dense after four clause additions; `claude-code.md`'s top-of-file
  `Verified:` stamp block doesn't list the new "Bash tool environment" section). Written to
  `docs/reviews/analyst-inbox-distillation.md`.
- **Why:** `agent-maintenance` skill §5 fold-in — a distillation pass gets independent review like
  any other significant deliverable, and the routing rule (never let an agent review its own future
  prompt) needed a live instance to route correctly.
- **Plan items:** none closed; the two minor findings above are parking-lot items (below), not
  planned work — they're take-or-leave polish the original stakeholder can act on or not.

## 2026-08-09 — `analyst` inbox distillation: new `python-web-quirks` skill + four `agent-standards/claude-code.md` knowledge-base additions
- **What:** Two pieces of machinery-adjacent work from a full `analyst` learnings-inbox
  distillation pass (see `claude/analyst/kaizen/history.md`, 2026-08-09, for the complete
  disposition record across all 31 entries):
  1. **Created `skills/python-web-quirks/SKILL.md`** — a new reference skill (Read/WebFetch/
     WebSearch only) holding three live-verified, version-pinned Python/web-framework gotchas
     (`asyncio.create_task` fire-and-forget GC-safety, Starlette/FastAPI `BackgroundTasks`'
     bounded-threadpool concurrency vs. an unbounded raw `threading.Thread`, FastAPI/pydantic
     `response_model_exclude_unset` dropping fields on nested models). Registered in
     `skills/README.md` and root `AGENTS.md`'s `skills/` bullet; wired via a routing clause in
     `coder`/`tdd-engineer`/`architect`/`analyst`'s frontmatter `description` (each agent's own
     kaizen carries its edit). **No `kaizen/` dir under the new skill's own folder** — logging
     it here instead, following the precedent set by `agent-maintenance`/`agent-standards`
     (per `skills/README.md`'s Maintenance section) since no skill in this repo actually carries
     a self-contained `kaizen/` despite the agent-maintenance skill's general §1 rule; this is
     partial progress on **K-014** (below) but doesn't close it — the convention still isn't
     written down anywhere, only followed by example.
  2. **Four additions to `skills/agent-standards/claude-code.md`**: a new "Output limits" bullet
     on FastMCP's `structured_output=False` opt-out (a `str`-returning tool otherwise ships its
     payload twice via a spurious `outputSchema`); a new "Lifecycle" bullet on why a
     containerized stdio MCP server's orphan-check ("`docker ps --filter label=…` must be
     empty") is unsatisfiable from inside the very session that's checking it; and a new
     top-level "Bash tool environment" section merging two prior observations (`find`→`bfs`,
     `grep`→`ugrep` — this environment's interactive shell shadows both with wrapper functions
     under a spoofed `ARGV0`, not inherited by a spawned subprocess) into one entry, since they're
     the same underlying phenomenon discovered on different dates.
- **Why:** Stakeholder decisions on the `analyst` inbox distillation proposal: general
  Python/web-framework facts belong in a skill personas consult, not duplicated project docs
  (hence the new skill rather than folding into `falkor-chat/AGENTS.md`, the only current
  FastAPI consumer); the Claude-Code-harness facts (FastMCP behavior, MCP container lifecycle,
  shell-shadowing) belong in cobb's existing perishable reference rather than `analyst`'s own
  always-loaded prompt, since they're general harness/library facts, not review technique.
- **Plan items:** touches **K-014** (see plan.md) — not closed.

## 2026-08-08 (Pass 2) — Fixed a regression my own C-311 tightening introduced; corrected the overclaiming docs; promoted a testing gotcha into TESTING.md
- **What:** `analyst`'s Pass-2 re-review of my C-311 regex tightening (below) downgraded from
  approve to **needs changes**: my single-alternation fix
  (`pipeline\.sh.*--reset|--reset.*pipeline\.sh`) had both boundary groups reaching for the same
  separator character when only one space stood between the tokens, so `--reset pipeline.sh`
  (bare basename, flag before the name) silently stopped matching — a genuine regression against
  the already-approved `6ab4ffe`, confirmed by analyst diffing `6ab4ffe` against the working
  copy and driving both through the real script. It also falsified the "before or after the
  path" claim I'd already written into `docs/BACKLOG.md`/`docs/HISTORY.md`. Rated major, not
  blocker (no realistic single command puts `--reset` before a *bare* `pipeline.sh`, since a
  shell has to name the executable before its flags) — but the written claim was still wrong,
  which is its own defect independent of exploitability.
  **Fix:** decoupled the single alternation into two independent `grep` checks ANDed together
  (`pipeline.sh` present + basename-anchored, AND `--reset` present as its own token) — each
  boundary now consumes its own separator, so match order and adjacency no longer matter.
  Re-verified against the full matrix (including the regression case and the still-required
  `mypipeline.sh --reset` negative) through the actual script, not a standalone `grep -qiE`
  typed at the prompt — analyst separately flagged that this sandbox's interactive `grep` is
  shadowed by a `ugrep`-backed shell function with different ERE semantics than the GNU grep the
  script subprocess runs, so a bare-`grep` sanity check can silently test the wrong thing.
  Confirmed the divergence myself (`type grep` vs. `bash -c 'type grep'`) and corrected both
  `docs/BACKLOG.md` and `docs/HISTORY.md`'s C-311 writeups to describe what the fixed regex
  actually does instead of the falsified claim.
- **Learning promoted, same run:** the grep-shadowing testing gotcha is a durable,
  non-obvious environment fact, not project-specific — verified live and written directly into
  `claude/cobb/TESTING.md` (new "Gotcha" subsection under the two-altitude table, plus a third
  testing-kind row for shell-based `PreToolUse` guards) rather than parked in `kaizen/inbox.md`,
  since I'd already verified it in this same run (system-prompt-authorized same-run promotion for
  the maintainer).
- **Why this matters beyond the one-line fix:** the Pass-1 tightening was reviewed and approved
  once already; a second review still caught a real behavioral regression by *executing* the
  script rather than re-reading the regex — validates why this team's convention is
  execution-based verification for hook/guard changes, not just a read-through, and why an
  approve verdict on a prior pass doesn't retire that discipline on the next edit to the same
  code.
- **Left uncommitted** — routed back to `analyst` for a Pass-3 confirmation before `teco` commits.

## 2026-08-08 (later) — C-311 follow-up: tightened the pipeline.sh match after review; fixed stale C-312 owner
- **What:** `teco` routed a follow-up after committing the C-309/C-311 work below (`6ab4ffe`):
  `analyst`'s independent review (`docs/reviews/safety-net-guard-fixes.md`, approve, no blockers)
  flagged the new `guard-destructive-ops.sh` branch as unanchored on the left — it matched
  `pipeline.sh` as a bare substring, so `mypipeline.sh --reset` also tripped it. Stakeholder asked
  for it tightened. The judgment call: `skills/joern-cpg/SKILL.md`'s own documented usage
  (`scripts/pipeline.sh <source> ...`) is written cwd-relative, so anchoring the fix to the full
  `skills/joern-cpg/scripts/` path — the obvious-looking tightening — would have silently
  reopened C-311 for any invocation issued from inside `skills/joern-cpg/` or
  `skills/joern-cpg/scripts/` (both plausible agent cwds). Chose the narrower fix instead: a left
  token-boundary on the `pipeline.sh` basename alone (start-of-string or non-alnum immediately
  before it), which rejects the reviewer's concrete false positive without narrowing the set of
  real invocation shapes the guard catches. Re-verified with ~12 synthetic PreToolUse payloads
  covering every plausible invocation form (full path, `bash`/`sh` prefix, SKILL.md's documented
  cwd-relative form, bare basename, absolute path, `--reset` on either side) plus the false
  positive and all pre-existing branches — all behaved as intended, documented in the code
  comment left in place. Also fixed `docs/BACKLOG.md`'s C-312 `Owner: joern` (stale — folded into
  `graph-dba`, `cbf26c4`) after confirming `graph-dba` is the right owner given what C-312 asks
  for (a `joern-cpg` pipeline post-load check).
- **Why:** Same rationale as the entry below — `guard-destructive-ops.sh` is shared,
  cobb-maintained hook-core infrastructure other agents' `PreToolUse` hooks depend on live; a
  review-driven correction to it is squarely in-scope, including the judgment call on how far to
  anchor the regex without reopening the gap the fix exists to close.
- **Left uncommitted** per the delegation — `teco` routes the regex change back through `analyst`
  for a quick re-review, then commits both together.

## 2026-08-08 — Closed C-309 and C-311: audit-team.sh untracked-file blindness, guard-destructive-ops.sh wrapped-delete blindness
- **What:** Fixed two confirmed gaps in the team's own safety-net scripts, delegated by `teco`
  with the root cause already diagnosed (this run's job was the fix + verification, not the
  diagnosis). **C-309(a)** — five backlog-flagged PII leaks — turned out already genericized as
  fallout from unrelated work; confirmed clean by direct grep of all five paths plus a green
  `audit-team.sh` run, closed as bookkeeping with no code change.
  **C-309(b)** — `audit-team.sh` check 7 scanned via `git grep`, seeing tracked files only, so a
  brand-new file leaking `$HOME`/username/etc. passed silently until its first commit. Fixed by
  unioning `git ls-files --cached` with `git ls-files --others --exclude-standard` before grepping
  (`claude/scripts/audit-team.sh`, check 7 block + header comment). Verified live: planted an
  untracked file containing `$HOME` under `claude/`, confirmed the gate FAILed on it (both the
  "home path" and "username" labels fired), removed it, confirmed `RESULT: PASS` returned.
  **C-311** — `guard-destructive-ops.sh` matched only the literal Bash command string, so
  `skills/joern-cpg/scripts/pipeline.sh --reset` (which runs `GRAPH.DELETE` *inside* the script)
  deleted a graph with zero human approval. Added a wrapper-match branch (`pipeline.sh` +
  `--reset`, either token order) alongside the existing `docker`/`FLUSHALL`/`GRAPH.DELETE`
  branches, with a code comment scoping it as ad hoc (population of one; re-grepped
  `skills/*/scripts/` and confirmed no second wrapper needs the same treatment) and flagging that
  a second wrapper should trigger a documented wrapper-registry convention instead of more
  one-off patterns. Verified with manual PreToolUse-payload tests: both `--reset` orderings ask;
  `pipeline.sh` without `--reset`, an unrelated benign command, and all pre-existing patterns
  behave unchanged.
- **Also touched:** `claude/AGENTS.md`'s "Hook machinery" section (one line, naming the new
  wrapper pattern) and `docs/BACKLOG.md`/`docs/HISTORY.md` (C-309/C-311 resolution writeup,
  per the module-documentation convention). Both scripts' documented stdin/stdout contract and
  existing match behavior were preserved unchanged, per the delegation's explicit constraint —
  these are live safety nets other agents' `PreToolUse` hooks depend on.
- **Why:** These two scripts are `cobb`'s own maintenance machinery
  (`claude/AGENTS.md`: "`cobb` (team maintainer: ... `scripts/audit-team.sh`...)"; the
  destructive-ops guard core is likewise cobb-maintained shared infrastructure), so the fix and
  its bookkeeping fall in-scope for `cobb` even when delegated in rather than self-initiated.
- **Left uncommitted** per the delegation — `teco` reviews and commits.

## 2026-07-31 — Triaged an "Instruction Poisoning" flag on an `analyst` kaizen-inbox entry; edited another agent's inbox directly (own normal channel, verified rather than assumed)
- **What:** `teco` routed a security-check flag on `analyst/kaizen/inbox.md`'s 2026-07-31 entry (a scratch-copy-and-reverse-patch technique, written up as "here's what to do when the classifier blocks `git stash`"). Judged it a genuine framing concern, not a false positive: the *action* taken was benign and independently confirmed harmless, but the *entry's framing* taught "route around a safety-classifier block" as reusable precedent rather than teaching the safety property (zero working-tree touch) that actually justified the substitute — a materially different shape from this inbox's other classifier-adjacent entry (2026-07-25, `pipeline.sh --reset`), which reports a gap in a repo-owned guard for its maintainer rather than instructing a workaround. Reworded the entry in place and routed the reusable environment fact (auto mode's Bash classifier has no reversible-op carve-out for `git stash`) to `skills/agent-standards/claude-code.md` §Hooks. Full account in `claude/analyst/kaizen/history.md` (2026-07-31 entry).
- **On write access:** `teco` flagged, correctly, that it has no authority to edit another agent's inbox or to adjudicate this itself, and pre-emptively warned `cobb` not to route around its *own* write guard if blocked, given the subject matter. Checked rather than assumed: `analyst`'s `guard-review-doc-writes.sh` is a `PreToolUse` hook wired in `analyst`'s **own** frontmatter (`analyst.md` `hooks:` block) — Claude Code hooks fire per-subagent-session, so it applies only while `analyst` itself is the active agent, not when a different subagent (`cobb`) issues the Write/Edit. `cobb`'s own frontmatter carries no hook. Direct maintainer edits to another agent's kaizen files are exactly `cobb`'s documented normal channel (`agent-maintenance` skill §5: "the maintainer (cobb) distills... verify → route → log → clear"). Confirmed by attempting the edit (it succeeded, no interception) rather than declaring in advance that access did or didn't exist — the point being not to defer uncritically to another agent's claim about my own permissions, in either direction, on a task about exactly that failure mode.
- **Why:** Delegated triage — agent/prompt-safety judgment on a flagged kaizen entry is squarely cobb's lane, and teco has neither the write access nor the adjudication authority.
- **Plan items:** new backlog item added (see `plan.md`) for the fuller distillation pass this inbox still owes.

## 2026-07-30 — Stakeholder decision implemented: formalized teco's commit authority; declined recommendation stays declined
- **What:** Follow-up to the "declined" entry directly below. The stakeholder made the recommendation
  surfaced there an explicit decision, verbatim: **"I dont want the subagents to proliferate
  commits, tico (you) and teco are special and have coordination rights."** Two things this
  settles, not one to evaluate: (1) `cobb`'s own recommendation below — extend narrow
  commit-as-you-go rights to `analyst`/`qa-engineer` for their own doc kinds — is **declined**, not
  left open; no other agent gains commit authority, ever, absent a fresh stakeholder decision.
  (2) `tico` and `teco` **specifically** are confirmed as the two commit-capable agents. Implemented:
  - **`claude/teco/teco.md`** — new Guardrails bullet + a step-4 sentence formalizing `Bash`'s
    integration-commit authority: `git add`/`git commit` a coordinated specialist's
    already-verified deliverable, by explicit path, one unit per commit, never bulk-staged/pushed/
    reset/rebased/amended. **Scoped deliberately differently from `tico`'s grant**, and said so in
    the prompt: `tico`'s commit scope mirrors its own Write/Edit guard exactly (only ever commits
    what it authored); `teco`'s is wider than its Write/Edit guard (which reaches only the
    coordination doc + its inbox) because its role — integrating a whole coordinated unit's
    output — is structurally different. This was the open design question the task asked me to
    resolve with judgment, not copy tico's grant verbatim; the honest answer is that teco's
    write-footprint and its natural commit footprint were never going to be the same shape, so
    forcing the tico framing onto it would have been a fiction. Full text: `claude/teco/teco.md`
    Guardrails, second bullet.
  - **`claude/scripts/audit-team.sh`** — new **check 8**, a deterministic containment backstop:
    `COMMIT_AUTHORS=("tico" "teco")`; fails if any other agent's own `<name>.md` claims
    `git add`/`git commit`, and fails if `tico`/`teco` ever lose their documented grant. No
    `PreToolUse` hook can gate a *prose* capability the way the existing doc-scoped/destructive-ops
    hooks gate Write/Edit paths and Bash command patterns — this grep-based check is the only
    mechanical trip-wire available, and it's exactly the "harness enforcement over hopeful prose"
    philosophy this team already applies everywhere else. Full audit re-run clean after the change
    (all prior checks + the new one, 0 FAIL).
  - **Catalogs**: `claude/README.md` — teco's row gained the integration-commit clause, cross-
    referencing `tico` and the 2026-07-30 stakeholder confirmation; `tico`'s row gained a matching
    one-clause cross-reference for symmetry. `claude/AGENTS.md` — new paragraph in "Hook machinery"
    stating plainly that git-commit authority is prompt-level, not hook-enforced, naming both
    grants' different scoping and pointing at check 8. `audit-team.sh`'s own header comment updated
    (item 8 in the checks list).
  - **`claude/teco/kaizen/history.md`** and **`claude/tico/kaizen/plan.md`** — dated entries; the
    tico parking-lot note left 2026-07-30 (the "recommendation surfaced, not implemented" pointer)
    now carries a `RESOLVED` addendum so nobody re-opens it as a live question.
- **Verified the four commits that prompted this, rather than accepting the stakeholder's framing
  on faith** (the task's own instruction: formalizing authority isn't the same question as whether
  the specific commits were done safely). Read all four directly (`git show --stat`, then full
  diffs): `15d3ad5` (`docs/reviews/cpg-getting-started.md`, analyst's review — 1 file),
  `4fe43a0` (`docs/test-plans/cpg-getting-started.md` + `docs/test-reports/cpg-getting-started-report.md`,
  qa-engineer's plan+report — 2 files, one coherent deliverable), `10f13ae`
  (`claude/analyst/kaizen/inbox.md`, analyst's own learnings entry — 1 file), `38e020d`
  (`claude/qa-engineer/kaizen/inbox.md`, qa-engineer's own learnings entry — 1 file). Every diff
  contains **exactly** the files its subject line names, nothing extra — consistent with
  explicit-path `git add`, not `-A`/`-a`. Four distinct hashes ~10s apart (18:53:26 → 18:53:54):
  sequential `git commit` calls, not one commit amended repeatedly. No `push`/`reset`/`rebase` in
  the sequence. Every committed file was authored by `analyst` or `qa-engineer`, not by `teco`
  itself — exactly the "deliverable from a specialist it's coordinating" shape the new guardrail
  now names, and exactly `tico`'s own established discipline (explicit path, no bulk staging, no
  history rewrites) applied to a different author. **Verdict: safe and disciplined** — nothing
  found here needed flagging as a separate problem. The gap was real but was purely a
  documentation gap (teco's prompt said "never mutating the tree" with no carve-out, while its
  actual, now-sanctioned behavior already had one) — disposition (a) from the task's own framing,
  confirmed rather than assumed.
- **Not done:** did not commit any of this session's own edits — `cobb` carries no standing commit
  authority of its own (only `tico`/`teco` do, per the very decision this entry implements; cobb's
  past commits were one-off review-and-commit judgment calls the user routed explicitly, not a
  standing grant), and no explicit "commit this" instruction was given this time. These edits sit
  in the working tree for the user, or a `teco`-coordinated close, to commit.
- **Why:** explicit stakeholder decision (verbatim quoted above), relayed as a direct implementation
  instruction rather than a design question to evaluate — the design question had already been
  closed by the entry below; this entry is the follow-through.
- **Plan items:** none opened in cobb's own `plan.md` (no unresolved design question remains); the
  one §7 minor from linting the new guardrail bullet was logged to `teco/kaizen/plan.md` (its
  artifact) instead.

## 2026-07-30 — Design review: declined "give tico commit authority over its summoned team"
- **What:** `tico` relayed a stakeholder proposal (not tico's own) — widen tico's `git commit`
  authority from its own two doc kinds to also cover artifacts produced by agents it summons
  under Mode 3 (e.g. `analyst`'s `docs/reviews/*`, `qa-engineer`'s `docs/test-plans/*` /
  `docs/test-reports/*`), i.e. become an orchestrator-with-commit-authority "like teco." Trigger:
  a live session where tico offered `qa-engineer`/`analyst` a verification pass on
  `docs/manuals/cpg-getting-started.md` (per the 2026-07-29 review-gate rollout below); their
  artifacts sat uncommitted because no convention has subagents self-commit, and tico's own
  guardrail has no carve-out for committing anyone else's files. tico correctly declined to
  self-waive it and routed the design question here instead of guessing.
  - **Verdict: declined as proposed** (both the blanket and the session-scoped-summoned-only
    variant tico's message floated as a softer alternative). Reasoning:
    1. **Breaks an invariant every commit-capable agent in the team currently holds without
       exception**: git-commit scope == the agent's own Write/Edit-guard scope (tico's own
       2026-07-23 grant: "only files your Write/Edit guard already allows you to touch"). No
       agent commits anything outside what it itself authored. Widening tico's commit scope
       without widening its write scope creates a first-ever asymmetry between "what I may
       write" and "what I may commit" — exactly the kind of split that's hard to audit later.
    2. **The premise in the routed message was factually wrong and worth correcting**: it
       claimed teco "already has broader authority" to commit. Read: teco's own guardrail says
       Bash is "read-only investigation plus running the project's suites/scripts — never
       mutating the tree" — grep across every agent file (`grep -rl 'git commit' claude/*/[a-z]*.md`)
       confirms **tico is the only agent in the team with any git-commit authority at all**.
       Routing the immediate need to teco would not have resolved it under teco's own
       documented rules.
    3. **tico's original commit grant was justified by a specific property this extension
       doesn't share**: tico is first-order, main-session, stakeholder-watching-in-real-time —
       "commit as you go" is low-risk because the stakeholder saw the file being written,
       turn by turn. A subagent's deliverable, produced in an isolated context tico only sees
       the returned path of, doesn't carry that same live-witnessed property — committing it
       under tico's authorship is a materially different act wearing the same guardrail
       language.
    4. **Second-orchestrator coherence conflict**: `claude/scripts/audit-team.sh` hardcodes
       `ORCHESTRATOR="teco"` (single, singular) for its roster-completeness check; root
       `AGENTS.md`, `claude/AGENTS.md`, and `claude/README.md` all describe teco as *the*
       orchestrator and tico as explicitly **not a delegation target**. Grafting
       orchestrator-style commit authority onto tico without reconciling that framing (and the
       script) everywhere it's asserted is a bigger, riskier change than the stated ask.
    5. **No hook backstops this today, and building one for the scoped variant is a real
       lift, not a copy-paste.** Every existing guard is a stateless `PreToolUse` path-glob
       match (`guard-doc-writes.sh` core). "Only artifacts from agents I summoned *this
       session*" needs session-scoped state (a manifest of what got spawned) that no current
       hook infrastructure tracks — so the scoped variant would ship as pure prompt-level
       self-discipline for a wider blast radius than tico's current exception has, which is
       backwards from how the team has hardened every other guardrail (harness enforcement
       over hopeful prose).
  - **What actually resolved the immediate block**: by the time this review ran, the pending
    artifacts (`docs/reviews/cpg-getting-started.md`, `docs/test-plans/cpg-getting-started.md`,
    `docs/test-reports/cpg-getting-started-report.md`) were already committed (`15d3ad5`,
    `4fe43a0`) — via the same review-and-commit pattern logged in the entry directly below this
    one (route a stuck, uncommitted deliverable to `cobb` for read-then-commit judgment, since
    `cobb` carries full tool access and no doc-kind write-scope restriction). That's the
    existing, precedented mechanism for exactly this gap — no new authority needed.
  - **Recommendation surfaced to the user, not implemented** (crosses beyond what tico asked
    me to evaluate — needs the stakeholder's own sign-off, touches agents' guardrails I wasn't
    asked to change): if uncommitted subagent deliverables become a recurring pain rather than
    a one-off, the architecturally consistent fix is to extend tico's exact existing
    "commit-as-you-go" pattern to `analyst` and `qa-engineer` **for their own doc kind only**
    (mirrors the 2026-07-23 tico precedent verbatim: `git add`/`commit` scoped to exactly what
    each agent's own write-guard already allows, no bulk-staging, no push/reset/rebase/amend).
    That keeps the write-scope==commit-scope invariant intact team-wide, touches nothing about
    tico's role or teco's orchestrator status, and directly closes the actual gap instead of
    routing it through a second orchestrator.
- **Why:** user-relayed design proposal via `tico`; evaluated per this agent's design/coherence
  mandate rather than implemented on request alone, per the user's explicit instruction to form
  independent judgment first.
- **Plan items:** none (declined; the alternative is a recommendation pending stakeholder
  decision, not a plan item cobb owns unilaterally).

## 2026-07-30 — Committed the manuals-review-gate rollout; caught a missed catalog update
- **What:** The 2026-07-29 rollout below (certification #2's resolution) had been sitting **staged in the git index, uncommitted**, for one day — routed to cobb for review-and-commit judgment since `tico`'s own write/commit access doesn't reach these files. Read every staged diff end to end (`tico.md`, `analyst.md`, `qa-engineer.md`, `teco.md`, and all five kaizen `history.md`/`plan.md` files) plus the one unstaged file sitting alongside it (`teco/kaizen/inbox.md`). Confirmed the 10 staged files are one coherent unit — cross-checked the description character counts the entry below claims (563/762) against the actual staged frontmatter and they match exactly, cross-checked `tico/kaizen/plan.md`'s K-005 pointer and `cobb/kaizen/plan.md`'s current (entry-free) parking lot against the "added then resolved same day" story, and re-ran `audit-team.sh` clean. Caught one real gap the "implemented across five files" line below didn't cover: `README.md`'s catalog rows for `teco`, `qa-engineer`, and `analyst` still didn't mention the new manuals-review duty, breaking the "catalog updated in the same change" rule (`claude/AGENTS.md` Maintenance rules). Fixed by adding one clause to each of the three rows (teco's independent-review default, qa-engineer's execution list, analyst's review-target list) and folded that fix into this same commit. The unstaged `teco/kaizen/inbox.md` change turned out to be unrelated — two new learnings entries from the same-day K-036 (falkor-chat) delivery, not part of the manuals rollout — so it was left out of this commit and handled separately.
- **Why:** a stale git index is a real risk (work sitting unreviewed, uncommitted, and un-backed-up); routed here rather than committed unread per the user's explicit instruction not to skip the review step.
- **Plan items:** none.

## 2026-07-29 — Resolved the manuals review-gate open question (from certification #2)
- **What:** Certification #2 (below) surfaced that `docs/manuals/` had no independent-review gate. Asked the user which agent should review and how mandatory it should be, rather than picking unilaterally — a genuine product/process call, not a mechanical fix. Decision: **split by claim type** (`qa-engineer` verifies walkthroughs against the running app — behavioral claims; `analyst` checks architectural/factual claims and clarity) and **mandatory via teco coordination, offered (not forced) in tico's own first-order sessions**. Implemented across five files: `teco.md` (review-gate defaults, routing-table row, handoff contract), `tico.md` (Mode 3 offered-verification bullet + a Guardrails bullet scoping `Agent` tool use), `analyst.md` (new reviewed-artifact category + description), `qa-engineer.md` (new verification-target section + description). Re-ran `audit-team.sh` clean after each file; confirmed description lengths stayed within the team's existing range (analyst 563, qa-engineer 762 chars — below graph-dba's 915).
- **Why:** direct follow-up to certification #2's logged observation; closes it rather than leaving it open indefinitely.
- **Plan items:** removed the parking-lot entry (resolved, not just noted).

## 2026-07-29 — Team-coherence certification #2 (post tico Mode 2/3 addition) — clean
- **Scope:** user-requested certification following the same-day `tico` change (commit `8582c49`):
  tico gained two new modes (didactic project explanation; user-manual maintenance at a new
  `docs/manuals/` doc kind), with matching edits to its write-guard hook (renamed
  `guard-requirements-doc-writes.sh` → `guard-tico-doc-writes.sh`, allowlist extended), root
  `AGENTS.md`'s documentation convention (new kind registered), and `teco.md`'s routing
  table/handoff contracts/doc-impact scan. This certification's baseline is the prior pass
  (below, commit `0bdc9f7`, same day) — the diff between them is exactly these 9 files.
- **Deterministic audit (`audit-team.sh`):** full **PASS** on the first run — all 12 agents × 5
  per-agent checks (kaizen triple, deployment symlink, hook exists+executable, teco roster,
  both catalogs), collection check 5b, all 16 boundary-pair directions, personal-info check.
  Confirms the hook rename didn't break check 3 (frontmatter `command:` re-resolves to the new
  filename and the file is executable) and that renaming didn't silently orphan the old path
  anywhere live (grepped separately — the only surviving `guard-requirements-doc-writes.sh`
  mentions are dated history entries, which is correct: they describe what was true at the time).
- **§4 judgment checklist:**
  - **Roster accuracy ✓** — `teco.md`'s tico rows (routing table + handoff contracts) now name
    the manuals path and the delegable/non-delegable split accurately against tico.md's actual
    Mode 2/3 text; re-read both side by side to confirm.
  - **Handoff symmetry ✓** — manuals ownership is stated on both sides (tico.md Mode 3; teco.md
    routing table + handoff contracts; root `AGENTS.md` owner-by-kind table). No third party
    needs to reference manuals for symmetry — unlike requirements→architect, nothing downstream
    consumes a manual to do further engineering work, so there's no missing counterpart to add.
  - **Subagent-awareness ✓** — tico's "If you are invoked as a subagent anyway" section is now
    split by mode: Mode 1 keeps the one-round-per-invocation degrade; Modes 2/3 explicitly
    support one-pass completion from a self-contained brief (no live back-and-forth needed),
    matching teco's new "delegable to tico for a self-contained manual write/update" claim
    exactly — checked both sides state the same contract, not just that both mention delegation.
  - **Enforcement parity ✓** — read `guard-tico-doc-writes.sh` directly (not just the prompt's
    claim): it allowlists `docs/requirements/*`, `docs/manuals/*`, and the kaizen inbox, matching
    tico.md's Guardrails section prose exactly; re-tested allowed (manuals, requirements) and
    denied (source file) paths through the actual script.
  - **Boundary reciprocity ✓** — tico isn't a `BOUNDARY_PAIRS` party (checked script-side, still
    true); teco's "pause → user for live Q&A, delegable for self-contained manual work" is the
    only new cross-agent claim, and it's reciprocated on tico's side per subagent-awareness above.
- **§7 lint fold-in (every artifact changed since the prior cert):** `tico.md` — full six-dimension
  lint already run at authoring time (see `tico/kaizen/history.md` 2026-07-29 entry and
  `tico/kaizen/plan.md`'s parking lot for the two minors logged there — cognitive load headroom,
  two unaddressed edge-cases). `teco.md`'s new row + handoff-contract sentence + doc-impact-scan
  clause — clean: the "not a delegation target" / "delegable for manuals" split reads as a
  deliberate, clearly-flagged exception on both mentions, not an unintended contradiction; no
  ambiguity in "self-contained brief" (defined inline: states the facts, no conversation needed).
  Root `AGENTS.md`'s new `manuals/` bullets — clean: fixed the one seam that would have been a
  contradiction pre-emptively (softened "all of a module's *engineering* docs" to "*documentation*"
  before committing, since `manuals/` is explicitly the non-engineering kind); the "five `Status:`
  values" count stays accurate (manuals reuse `active`/`superseded`/`archived`, no new value
  needed); the "family rule below" cross-reference resolves correctly.
- **One observation, not a blocker — logged to `cobb/kaizen/plan.md` parking lot:** every other
  doc kind in the taxonomy has an independent-review gate of some form (`Ready for design` needs
  explicit stakeholder confirmation; plans/code get `analyst`; ML notes get `data-scientist`;
  behavior gets `qa-engineer`) — `manuals/` currently has none; tico both writes and self-certifies
  its own manuals' accuracy. Not acted on without a decision (would mean routing tico's manual
  drafts through another agent, which the user didn't ask for and may not want given tico's
  first-order/live-conversation design) — recorded as an open question, not fixed unilaterally.
- **Certificate:** PASS, no fixes required this pass (the deterministic run was clean on the first
  try, unlike the prior cert which needed a PII fix + script relaxation).

## 2026-07-29 — Team-coherence certification (full 12-agent pass) — script fixed, one PII leak found and fixed
- **Scope:** user-requested certification, following the same-day teco kaizen-backlog work
  (K-006/008/009/010/011 closure + inbox distillation). Ran the §4 pass over the whole `claude/`
  collection.
- **Deterministic audit (`audit-team.sh`) — first run:** 2 FAIL classes.
  1. **PII leak** — `claude/teco/kaizen/history.md` embedded the literal flattened
     `~/.claude/projects/-home-<user>-...` transcript path (contains the OS username) in a K-009
     evidence note written earlier the same day. Fixed: genericized to
     `<flattened-repo-path>`; logged in teco's own history.md. The leak had reached one shared
     commit (`e7ec4a3`) — left as-is rather than rewriting history, per the repo's norm against
     rewriting shared commits.
  2. **`coder`/`devops`/`frontend-engineer`/`tdd-engineer` "missing from AGENTS.md"** — root
     cause: commit `70d0981` (2026-07-28, direct user edit, not routed through cobb) deliberately
     deleted the inline 12-agent roster from root `AGENTS.md` as duplication ("already documented
     in each component's own README/AGENTS.md... already pointed to via Structure/Component
     docs") — exactly the DRY principle this skill's §2 teaches. `audit-team.sh` check 5 predated
     that trim and still required every agent's literal name in root `AGENTS.md`; the other 8
     agents only passed incidentally (named in an unrelated doc-lifecycle table). **Asked the
     user** rather than guessing between "relax the script" / "restore the roster" / "accept as a
     known FAIL" — a real architecture call, not mechanical drift. Decision: **relax the script**.
     Implemented: `audit-team.sh` check 5 now runs per-agent against only the two catalog owners
     (`claude/AGENTS.md`, `claude/README.md`); a new collection-wide check 5b verifies root
     `AGENTS.md` still *points at* that catalog (`claude/AGENTS.md` + `claude/README.md`
     substrings present) instead of repeating every name. Script header comment and this skill's
     §4 deterministic-half paragraph both updated to describe the new split (the paragraph's
     `BOUNDARY_PAIRS` list was also stale — 3 pairs documented vs. 8 actual — fixed in the same
     edit since it was the same sentence).
- **Deterministic audit — re-run after fixes:** full **PASS** (all 12 agents × 5 per-agent
  checks, collection check 5b, all 16 boundary-pair directions, personal-info check).
- **§4 judgment checklist:** roster accuracy ✓ (teco's routing table names all 12 agents with
  current contracts); handoff symmetry ✓ (no counterpart-file changes since the 2026-07-28
  certification other than teco's own prompt, so symmetry is unchanged from that certified
  state); subagent-awareness ✓ (teco's step 3 still carries the can't-ask-mid-run reminder);
  enforcement parity ✓ (read `guard-coordination-doc-writes.sh` directly — it does allowlist
  `teco/kaizen/inbox.md` alongside `docs/plans/*`, matching the guardrail prose exactly); boundary
  reciprocity ✓ (teco is the orchestrator, not a `BOUNDARY_PAIRS` party — no gap).
- **§7 prompt-quality lint** (folded in, scoped to what changed since the 2026-07-28
  certification: `claude/teco/teco.md`, `skills/agent-standards/claude-code.md`): one **minor**
  finding on `teco.md` step 3 — the newly-added model-routing sentence carries an inline
  `"verified 2026-07-29 to reach a call made from inside a subagent"` evidence clause that
  duplicates the fuller evidence already recorded in `teco/kaizen/history.md`; the operative
  prompt only needs the instruction, not the dated citation. Not fixed in this pass (cosmetic,
  user didn't ask for a teco.md edit) — noted in teco's `plan.md` parking lot instead, alongside
  the already-flagged step-3 density item from earlier today. `claude-code.md`'s new Lifecycle
  bullet: clean on all six dimensions (reference material, no persona/instruction-following
  stakes).
- **Why:** user-requested certification pass.
- **Plan items:** none opened in cobb's own plan.md — the one lint finding was routed to teco's
  plan.md instead (it's teco's artifact).

## 2026-07-28 — Retired the `joern` agent into `graph-dba`; team-coherence certification
- **What:** Executed the user's decision to retire the standalone `joern` subagent
  (CPG generation is genuinely rare — doesn't justify a dedicated standing
  persona) and fold its capability into `graph-dba` as a small, explicitly
  on-demand addition. Full change set: `claude/joern/` deleted (agent, hooks,
  kaizen) plus its `~/.claude/agents/joern` deployment symlink; `graph-dba.md`
  gained a capability clause in `description` and a short pointer paragraph +
  boundary edit in the body (no restatement — the `joern-cpg` skill already
  carried the procedural detail); `joern`'s inbox distilled before deletion
  (see `graph-dba/kaizen/history.md` 2026-07-28 for the full routing table —
  three genuinely new facts landed in `skills/joern-cpg/SKILL.md` Gotchas/
  Prerequisites, one in `falkordb-quirks.md`, one cross-reference fix in
  `cpg-model.md`; the rest were already fixed/documented, discarded as
  duplicates); `joern:graph-dba` dropped from `audit-team.sh` `BOUNDARY_PAIRS`;
  every "routes to the `joern` agent" / "(the `joern` agent)" reference across
  `skills/joern-cpg/`, `skills/cpg-analysis/` (SKILL.md + all four
  `references/*.md` recipes), `skills/README.md`, `claude/README.md`,
  `claude/AGENTS.md` (roster + hook-machinery four→three guards), root
  `AGENTS.md`, and `claude/teco/teco.md` (routing table + handoff contract)
  updated to `graph-dba`.
- **§7 prompt-quality lint** (folded into this certification) over the changed
  artifacts — `graph-dba.md`, `skills/joern-cpg/SKILL.md`,
  `skills/cpg-analysis/SKILL.md`: **clean** on all six dimensions. Notably no
  contradiction between the new "on-demand, not proactive" capability clause and
  the description's existing "Use proactively for…" list (CPG generation was
  deliberately kept out of it), and the body addition is a pointer to the skill
  rather than a restatement, per the user's explicit "keep it lean" constraint.
- **§4 certification — deterministic half:** `claude/scripts/audit-team.sh` run
  post-change: 12 agents (was 13), **95 PASS**, 0 joern-related FAIL (kaizen
  triples, deployment symlinks, hook existence, teco-roster mentions, all three
  catalogs, and boundary-pair symmetry all green). Two pre-existing FAILs
  (personal home-path/username leaks in `.claude/settings.json`,
  `claude/architect/kaizen/inbox.md`, `claude/devops/kaizen/inbox.md`,
  `docs/plans/m2-cpg-analysis-skill.md`) are **unrelated drift this task didn't
  touch** — none of those files are in this change's diff; flagged as a
  follow-up, not fixed here (out of the requested scope).
- **§4 certification — judgment half:** roster accuracy (graph-dba's roster
  line and catalog entries describe the actual new capability, not just a name
  change); handoff symmetry (teco's routing table + handoff contract updated in
  the same change as the capability move); boundary reciprocity (the
  graph-dba↔joern pair is gone because the border dissolved into one agent, not
  two — correctly removed rather than left dangling); no new subagent-awareness
  or enforcement-parity gaps introduced (graph-dba's existing destructive-ops
  guard already covered `GRAPH.DELETE` for a CPG reload, so no new hook needed).
  Repo-wide grep swept for remaining `joern`-agent references after all edits:
  none outside historical kaizen-history prose (correctly left untouched — kaizen
  history is a dated record, not live routing) and one dangling citation path in
  `claude/architect/kaizen/inbox.md:111` (an unprocessed inbox entry citing
  `claude/joern/kaizen/inbox.md:19`, now deleted) — left alone as an
  append-only raw-capture file outside this task's scope; noted for the next
  general inbox distillation pass.
- **Why:** User decision after a short design discussion (see conversation).
- **Plan items:** —

## 2026-07-27 — `agent-standards`: `model` frontmatter field re-verified
- **What:** Updated `skills/agent-standards/claude-code.md` — the `model` field now records the full accepted value set (`opus` | `sonnet` | `haiku` | `fable` | a full model ID | `inherit`) and, the fact that mattered here, that it **defaults to `inherit`** when omitted. Added a dated line to the file's `Verified:` stamp block.
- **Why:** Needed the authoritative default before unpinning the 13 agents from `model: opus` — "omit the field" is only equivalent to "use the system default" because the default is `inherit`. The cached snapshot listed neither `fable`, full model IDs, nor the default. Verified against `code.claude.com/docs/en/sub-agents` (frontmatter table).
- **Plan items:** —

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-25 — M3 / CPG query access: skill surface, agent wiring, and MCP knowledge capture (C-303/C-304/C-307)
- **What:** Implemented steps S4, S5 and S7 of `docs/plans/cpg-query-access.md` (re-gated
  "approve with suggestions", 0 blockers).
  - **S4 (C-303):** `skills/cpg-analysis/SKILL.md` re-pointed at the `mcp__cypher__query` MCP tool —
    `description`, `allowed-tools: mcp__cypher__query, Bash, Read`, §1 rewritten (two parameters, no
    shell; read-only `GRAPH.RO_QUERY`; graph discovery without a `list_graphs` tool; `EXPLAIN`-only
    with the `PROFILE` refusal and its reason; display-only truncation; a labelled `redis-cli`
    fallback block), §3's parameter note generalised to "neither path binds Cypher parameters".
    Preamble of `references/impact-analysis.md` moved to the tool; `skills/README.md` row updated.
  - **S5 (C-304):** `mcp__cypher__query` added to the `tools:` allowlists of `analyst` and `architect`
    (`qa-engineer` inherits), `claude/README.md` rows 9/16/17, root `AGENTS.md` (`cpg/` structure
    entry, component-docs row, `cpg-analysis` bullet, smoke command), the three agents' kaizen
    histories, and a status annotation closing out `claude/tico/kaizen/inbox.md:19`.
  - **S7 (C-307):** `skills/agent-standards/claude-code.md`'s three-line `## MCP` stub replaced by a
    verified reference section; a new **MCP servers** section plus re-verified **Skills** specifics
    added to `opencode.md`; the cross-tool "MCP wiring does not port" rule added to the skill's
    `SKILL.md`.
- **Why:** The plan's own §2.4 collected the most perishable and most reusable content it had, and
  archiving the plan would have taken it with it (review finding M-5). The `tools:`-allowlist fact
  is the load-bearing one: `analyst` and `architect` declare allowlists, so without S5 the feature
  would have been silently inert for two of its three consumers.
- **Verified live against `code.claude.com/docs/en/mcp` and `opencode.ai/docs/{skills,mcp-servers}`
  (2026-07-25)** rather than copied from the plan — which caught material the plan predates:
  **tool search is on by default** (MCP tools are deferred; a `ToolSearch` event is expected in a
  transcript, and *server* `instructions` — not just the tool description — are the routing signal,
  truncated at 2 KB), **`alwaysLoad`**, and the **output-limit behaviour** (warn >10k tokens, cap
  25k via `MAX_MCP_OUTPUT_TOKENS`; above the threshold a result is **persisted to disk and replaced
  by a file reference**, so a trailing truncation notice is the first thing lost — the per-tool
  escape is `_meta["anthropic/maxResultSizeChars"]`, ceiling 500k chars). Also corrected the plan's
  "no per-server tool filtering" claim into its precise form: whole-server toggles exist
  (`disabledMcpServers`/`enabledMcpServers`), per-*tool* subsetting does not.
- **Portability spot-check (review finding m-3), resolved:** `opencode debug skill` — a cheap,
  offline way to test `SKILL.md` portability — shows OpenCode parsing `cpg-analysis` with the
  Claude-only `mcp__cypher__query` in `allowed-tools`, description intact, no warning; its docs
  confirm unknown frontmatter is ignored. **New quirk found and promoted, not inboxed:** repeated
  runs return *different subsets* of the same 9 skills with no error — recorded in
  `agent-standards/opencode.md` and `skills/README.md`, and the root `AGENTS.md` "all 9 skills
  visible to each" claim qualified. An actual OpenCode *invocation* remains unexercised (C-310).
- **Also fixed:** root `AGENTS.md` ended with a committed `</content>\n</invoke>` XML trailer from
  an earlier tool-assisted edit — removed. It sat in an always-loaded context file (imported by the
  root `CLAUDE.md`), i.e. in every session's prompt.
- **Not done, deliberately:** the server (`cypher-mcp/`, S2 · coder), `.mcp.json` + settings (S3 ·
  devops) and `docs/requirements/` (S6 · coder) — other owners; and the `instructions=` /
  `alwaysLoad` *implementation* those two steps now have the facts for.
- **Plan items:** advances K-001 (Claude Code MCP moves off the 2026-05-31 baseline; Skills/Memory/
  Hooks/SDK remain).

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 519 → 448 chars (-13%): tightened phrasing, dropped restated detail. `cobb` has no boundary pairs in `claude/scripts/audit-team.sh`; full audit re-verified green regardless. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, completing the same-day team-wide pass (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`, `teco`, `tico`, `data-scientist`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** `cobb` carries no write-guard hook (unlike the doc-scoped and destructive-ops agents), so this is a plain, unconditional friction reduction — nothing to reconcile it against.
- **Plan items:** none.

## 2026-07-17 — joern K-001 closed: live FalkorDB load verified; 2 transformer bugs fixed + learnings routed
- **Scope:** ran the live `--load` verification the creation pass had deferred (FalkorDB was down).
  Started `falkordb:v4.18.11`, ran the joern pipeline end-to-end into an isolated `cpg_smoke` graph
  (pre-existing `reference`/`ws:live`/`ws:acme` untouched; test graph removed after).
- **Result:** real CPG round-tripped — 107 nodes / 462 edges loaded 30/30 stmts, index present,
  full CPG edge-layer set, call graph traverses correctly. The `CREATE INDEX FOR (n:CpgNode) ON
  (n.id)` DDL is accepted by this build.
- **Two bugs the test caught (fixed in `skills/joern-cpg/scripts/cpg-to-falkordb.py`):**
  `graph_nonempty` regex-scanned the whole redis-cli reply → the exec-time stat line made every
  graph read "non-empty" (loader refused all loads); and `:boolean` columns stored as strings →
  boolean predicates never matched. Fixed via `GRAPH.RO_QUERY`+integer-line parse and a `bool`
  kind emitting Cypher `true`/`false`. Details in `claude/joern/kaizen/history.md`.
- **Learnings routed same-run (§5, cobb acting as maintainer):** UPPER_CASE property-name query
  gotcha → `skills/joern-cpg/references/cpg-model.md`; two FalkorDB engine quirks (`GRAPH.QUERY`
  read materializes an empty key; `RO_QUERY` errors + creates nothing on a missing graph) →
  `claude/graph-dba/falkordb-quirks.md` (+ graph-dba history). No inbox residue.
- **Scope note:** no roster/boundary/frontmatter change → no re-certification needed; the 2026-07-17
  creation certificate below still stands. `joern.md` unchanged, so no §7 re-lint.

## 2026-07-17 — Created the `joern` agent + `joern-cpg` skill; team re-certified
- **Scope:** built a new CPG specialist (`claude/joern/` + `skills/joern-cpg/`) end to end — see
  `claude/joern/kaizen/history.md` for the full artifact manifest. Then ran the §4 certification
  (roster-changing edit) folding in the §7 single-artifact lint.
- **Deterministic audit (`audit-team.sh`):** all 13 agents; **every joern check PASS** — kaizen
  triple, deployment symlink, hook executable, present in teco's roster, cataloged in all three
  docs, and boundary reciprocity **both** ways (`joern`↔`graph-dba` after adding the pair to
  BOUNDARY_PAIRS + the reciprocal clause in graph-dba's description).
- **Pre-existing FAIL (not from this work):** check 7 flags `.claude/settings.json` (committed
  `ef8b2d7`) hardcoding the maintainer's absolute `/home/<user>/…/audit-team.sh` path in a
  `permissions.allow` Bash matcher. Genericizing it needs care — `${CLAUDE_PROJECT_DIR}` expansion
  in *permission matchers* (vs. hooks) is unverified — so it's surfaced to the user, not silently
  rewritten. Tracked as a follow-up, separate from joern.
- **§4 judgment checklist:** roster accuracy ✓ (teco routing row + handoff note added), handoff
  symmetry ✓ (joern's non-doc artifact convention stated on joern + teco), subagent-awareness ✓
  (can't-ask-mid-run → return the question; destructive request → return to caller),
  enforcement parity ✓ (destructive-ops guard wired + described on both sides), boundary
  reciprocity ✓.
- **§7 lint of `joern.md`:** clean — no blocker/major. The "owns the mechanical load" vs. "defer
  the FalkorDB model to graph-dba" split is stated explicitly (no contradiction); persona and
  altitude consistent; composition with the repo `AGENTS.md` (FalkorDB = OpenCypher, no APOC/GDS)
  agrees with the skill.
- **Verification of the artifact itself:** full pipeline ran green on a Python sample (build →
  export → transform → 29 Cypher stmts, exit 0); live FalkorDB `--load` not exercised (server
  down) — logged as joern K-001.

## 2026-07-16 — §7 refined from the first-run smoke test (K-013)
- **What:** Two one-sentence additions to `skills/agent-maintenance/SKILL.md` §7's output-form paragraph, from the same-day teco.md smoke test. (1) A **prompt-severity rubric**: blocker = wrong behavior in most sessions; major = a real gap that bites in some sessions; minor = polish. (2) **Cross-cutting attribution**: when one issue spans dimensions, report it once under the most informative dimension (note the others in a clause) rather than filing it several times. No renumbering, no other §7 change; the section stayed lean.
- **Why:** The teco lint applied `blocker/major/minor` with ad-hoc calibration and hit a finding that landed on three dimensions at once (contradiction + composition + cognitive load) — the checklist gave labels but no rubric and no attribution rule. Both gaps were one-sentence fixes; deferred from the review-only smoke-test run per its brief, then approved.
- **Plan items:** K-013 done (moved to the Closed line in plan.md).

## 2026-07-16 — Single-artifact prompt-quality lint (§7) added to the review machinery (K-012)
- **What:** Added a new **§7 "Prompt quality review (single-artifact lint)"** to `skills/agent-maintenance/SKILL.md` — a semantic, intra-artifact judgment pass over six dimensions (contradiction, semantic ambiguity, persona consistency, cognitive load, semantic coverage, composition conflict), each with 2–3 concrete probes and a `finding — severity — suggested rewrite` output form; the composition-conflict dimension resolves the artifact's full load-set (CLAUDE.md/AGENTS.md chain, `@`-imports, wired skills, reaching-`inclusion` steering — verify per tool via `agent-standards`) then re-runs the contradiction/persona/coverage probes over the combined context. Testing stayed §6, prompt-lint appended as §7 (no renumbering; the approved design named it §7). Wired at two trigger points: (1) a lean **Prompt-quality lint** bullet in `cobb.md` Maintenance duties (checklist stays in the skill); (2) a **§7 fold-in** paragraph in the skill's §4 certification pass — lint every artifact changed since the last certification, roll findings into the certificate (mirrors the §5 inbox-distillation fold-in). Also updated the skill frontmatter `description` (routable) + its "When this applies" list. Catalogs touched only where they enumerate the skill's procedures and would otherwise misdescribe by omission: `skills/README.md` (agent-maintenance row + when-to-use), `claude/README.md` (agent-maintenance skill bullet), root `AGENTS.md` (skills procedure list). The `cobb` agent catalog rows were left unchanged — cobb's role description didn't change.
- **Why:** The user asked whether cobb's machinery covered six LLM-judgment prompt dimensions; assessment found it covered only structural (§3) and inter-agent (§4) drift — the six dimensions are semantic and single-artifact, essentially uncovered. User approved the design (new §7 + two trigger points; no `audit-team.sh` changes — these are judgment, not greppable). The optional composition load-set enumerator script was skipped as not cheap enough (noted in plan.md parking lot).
- **Plan items:** K-012 done (promoted from the dormant "self-review checklist" parking-lot idea, re-flagged 2026-06-07); moved to the Closed line in plan.md.

## 2026-07-12 — Team-wide learning-capture loop designed and installed
- **What:** Generalized graph-dba's quirks-file pattern into a team self-improvement loop: (1) every agent gained an append-only `kaizen/inbox.md` (12 files, seeded from a shared template) plus a "Learning capture" closing-protocol section in its prompt — durable, non-obvious environment facts get captured as dated, evidence-backed entries during runs; agents never promote their own entries. (2) The five doc-scoped write guards (architect, analyst, data-scientist, teco, tico) allowlist exactly the agent's own inbox path. (3) The agent-maintenance skill gained §5 "Learnings inboxes — capture & distillation" (verify → route to prompt / on-demand knowledge base / project docs / discard → log in history.md → clear; folded into every certification pass), with Testing renumbered to §6 and the frontmatter description updated. (4) `audit-team.sh` check 1 now requires the kaizen triple (plan/history/inbox). (5) cobb's prompt gained the distillation duty + its own capture section (with same-run promotion in-bounds for the maintainer alone); teco's integration step now confirms delegates filed reported learnings. (6) Catalogs updated: `claude/README.md` (Kaizen section + conventions), `claude/AGENTS.md`, root `AGENTS.md`, `skills/README.md`.
- **Why:** The user asked how the agents could self-improve from what they learn exploring their areas. Subagents are stateless, so self-improvement = capture into files during runs + curated fold-back into persistent artifacts. Cheap unreviewed capture with reviewed promotion keeps prompts lean (protects the 2026-07-11 token-cost pass) and keeps project facts in project docs instead of private agent files.
- **Plan items:** none.

## 2026-07-11 — Certification findings implemented (K-010, K-011 + handoff fixes)
- **What:** All four same-day certification findings fixed. (1) graph-dba gained the `docs/plans/<slug>-graph.md` design-note handoff contract, mirrored in teco's contracts list (graph-dba K-004). (2) `cobb.md` gained the subagent-awareness clause — isolated context, can't ask mid-run, questions/approvals return as the deliverable (K-010). (3) Destructive-ops guard parity: `devops`'s standalone guard refactored into a shared core `claude/scripts/guard-destructive-ops.sh` (agent name as arg) with thin wrappers wired into `devops`, `graph-dba`, and `qa-engineer` frontmatter, each described in its prompt; hit/pass-through verified by piping PreToolUse JSON through the wrappers (K-011). (4) tdd-engineer now names the inbound analyst-RCA path and routes acceptance passes to qa-engineer in its description; qa-engineer's description reciprocates, and `tdd-engineer:qa-engineer` joined `BOUNDARY_PAIRS` in `audit-team.sh`. Catalogs updated in the same change: `claude/README.md` (graph-dba + qa-engineer rows, deployment hooks section) and `claude/AGENTS.md` (hook machinery — two shared cores now). Post-change `audit-team.sh`: full PASS, including the new eighth boundary pair.
- **Why:** User approved implementing the certification findings ("go ahead").
- **Plan items:** K-010 done, K-011 done (moved from plan.md); graph-dba K-004 and the tdd-engineer parking-lot item closed in their own kaizens.

## 2026-07-11 — Team-coherence certification (full 12-agent pass)
- **What:** Ran the §4 certification on the whole `claude/` collection. Deterministic half: `scripts/audit-team.sh` **PASS** (all checks — kaizen pairs, deployment symlinks, hooks executable, rosters, three catalogs, all seven boundary-pair directions, repo-wide personal-info check). Judgment half found four gaps, recorded as plan items rather than fixed: (1) `graph-dba` is the only design-producing specialist without a written-deliverable path contract, so its designs get paraphrased in teco handoffs — graph-dba K-004; (2) `cobb` is the only delegate-able agent missing the subagent-awareness (can't-ask-mid-run) clause — cobb K-010; (3) enforcement gap: the destructive-ops guard protects the shared live FalkorDB only when `devops` acts, while `graph-dba`/`qa-engineer` run the same shapes unguarded — cobb K-011; (4) tdd-engineer's inbound RCA handoff and its qa-engineer altitude boundary are stated only on the counterpart sides — tdd-engineer parking lot. Everything else clean: roster accuracy, tico-not-a-delegate handling, plan/review/test-doc path symmetry, hook↔prompt enforcement parity for the six wired guards.
- **Why:** User asked for an analysis of the team's handoffs ("are they optimal? where can we improve?").
- **Plan items:** opened graph-dba K-004, cobb K-010, cobb K-011; findings only — no agent sources changed.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 728 to 515 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-10 — Check 7 widened to the whole repo
- **What:** `audit-team.sh` check 7 now greps **all tracked files** for the five runtime-derived personal identifiers, not just `claude/` and `skills/` (the `-- claude skills` pathspec dropped; FAIL messages relabeled `repo:`). `skills/agent-maintenance/SKILL.md` §2 rule and §4 invariant list updated to the repo-wide scope. Follows the same-day falkor-chat docs cleanup (three absolute home-path references and an Owner email genericized), which the narrower scope had left unpoliced.
- **Why:** User asked for repo-wide coverage — the leak class isn't specific to agent artifacts.
- **Plan items:** none.

## 2026-07-10 — Check 7 broadened: any personal identifier, not just the home path
- **What:** `audit-team.sh` check 7 now greps tracked files under `claude/` and `skills/` for **five runtime-derived personal identifiers** — home path (`$HOME`), OS username (`id -un`), git `user.name`, git `user.email`, and hostname (`hostname`) — case-insensitively, word-bounded for the short bare tokens (username, hostname) to avoid substring noise. Patterns are never hardcoded in the script (that would itself be the leak), so the check guards whoever runs it. Both outcomes verified: clean run PASS; a planted email+hostname in a tracked kaizen file produced two labeled FAILs, exit 1, then reverted. `cobb.md`'s principle retitled **"No personal information in committed artifacts"** and `skills/agent-maintenance/SKILL.md` §2 rule + §4 invariant list broadened to match (prose genericization guidance included: `/home/<user>/…`, "the maintainer").
- **Why:** User asked the guardrail to be inclusive — any personal information, not only the home dir.
- **Plan items:** none.

## 2026-07-10 — Home-path leak: portable-path rule + audit-team.sh check 7
- **What:** Three guardrails from the same incident, plus the fix itself. (1) Six agents' frontmatter hook commands (teco, tico, architect, analyst, data-scientist, devops) were committed with the maintainer's absolute `/home/<user>/prg/…` path — each rewired to `$HOME/.claude/agents/<name>/hooks/<script>.sh`, resolving through the user-scope deployment symlink; shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands (verified 2026-07-10 against `code.claude.com/docs/en/hooks`; `${CLAUDE_PROJECT_DIR}` rejected — these agents guard in any project). Logged in each agent's own kaizen. (2) `cobb.md` Principles gained **"No machine paths in committed artifacts"**. (3) `skills/agent-maintenance/SKILL.md` §2 gained the **"Machine-portable paths (rule)"** subsection, and §4's deterministic-half paragraph now lists the leak check. (4) `claude/scripts/audit-team.sh`: check 3 now mirrors the shell-form `$HOME`/`~` expansion before `test -x`, and new collection-wide **check 7** fails if any tracked file under `claude/` or `skills/` contains the machine's literal home path. Post-change audit run: PASS.
- **Why:** User flagged that committed hooks leaked his personal home path into the repo. The leak class is silent (everything works on the author's machine) — exactly the kind of invariant the deterministic audit half exists to catch.
- **Plan items:** none.

## 2026-07-09 — agent-maintenance: audit-team.sh check 6 — boundary-pair description symmetry
- **What:** `claude/scripts/audit-team.sh` gained a collection-wide check 6: for each declared boundary pair (`BOUNDARY_PAIRS`: coder↔tdd-engineer, analyst↔qa-engineer, graph-dba↔devops), each agent's frontmatter `description` must name the other — mechanizing the name-level half of the certification's "boundary reciprocity" judgment item. `skills/agent-maintenance/SKILL.md` §4 updated on both halves: the deterministic-half paragraph lists the new check, and judgment item 5 notes what stays judgment (whether the claimed scopes actually complement) plus the rule to grow `BOUNDARY_PAIRS` when a new adjacent specialist joins. Driven by the same-day description-symmetry fixes (analyst↔qa-engineer, graph-dba↔devops; coder↔tdd-engineer was already symmetric). Post-change audit run: PASS, all six directions.
- **Why:** Frontmatter descriptions are the routing contract every router sees (the main session and teco's `Agent` listing); the asymmetry class — A defers X to B, B never names A — drifts silently, the same failure family as the teco roster drift that spawned the script.
- **Plan items:** none.

## 2026-07-09 — agent-standards: main-session (`--agent`) mode added to the Claude Code cache
- **What:** `skills/agent-standards/claude-code.md` gained a "Running a definition as the MAIN session agent" section (+ stamp bump): `claude --agent <name>` / the `agent` setting make the main thread take on a definition's prompt/tools/model; `initialPrompt` auto-submits as the first user turn; frontmatter hooks fire in main-session mode alongside `settings.json` hooks; the withheld-tools list (e.g. `AskUserQuestion`) applies to subagents only; `Agent(agent_type)` allowlist syntax works only in main-thread mode.
- **Why:** Drift found while building `tico` as a first-order conversational agent — the cache (verified 2026-06-20) predated/omitted the whole main-session mode. Reconciled against the live `code.claude.com/docs/en/sub-agents` page per the skill's update procedure.
- **Plan items:** none.

## 2026-07-09 — TESTING.md: dropped personal-preference rationale
- **What:** In the two-altitude table, the pytest row's rationale "matches the user's TDD preference and the `tdd-engineer` agent" → "matches the `tdd-engineer` agent's discipline".
- **Why:** User ruling (same-day coder/tdd-engineer routing fix): remove personal-preference framing from agent artifacts — standing preferences are quality and efficiency, encoded as objective rules. The pytest guidance stands on the discipline itself.
- **Plan items:** none (out-of-band).

## 2026-07-09 — Team-coherence certification: skill §4 + `audit-team.sh` + cobb mandate
- **What:** Made inter-agent drift certification an explicit cobb duty, in three pieces:
  1. **`agent-maintenance` skill** — new **§4 "Team coherence certification"** (testing renumbered §5): a two-half pass — the deterministic script first, then a five-point judgment checklist (roster accuracy, handoff symmetry, subagent-awareness, enforcement parity, boundary reciprocity), with the certificate logged as a dated entry in the maintainer's kaizen history. Also added a per-edit rule to §2's order of operations: adding/renaming/removing an agent means updating every prompt that *enumerates the team* (an orchestrator's roster, the collection-count line) in the same change. Frontmatter `description` extended so "certify/audit the team" routes to the skill.
  2. **New `claude/scripts/audit-team.sh`** — read-only deterministic half: per agent it checks the kaizen pair exists, the `~/.claude/agents/` symlink resolves, frontmatter hook commands exist + are executable, the agent is named in teco's prompt, and it's cataloged in `claude/AGENTS.md` / `claude/README.md` / root `AGENTS.md`. Exit 1 on any FAIL. Verified both ways: PASS on the post-fix collection (8 agents, all green); FAILs correctly when pointed at an empty deploy dir (`CLAUDE_AGENTS_DIR` override).
  3. **`cobb.md`** — description now names auditing/certifying a team as a trigger; Maintenance duties gained the certification bullet (script → checklist → kaizen log) and the per-edit roster rule folded into "In-scope vs. cross-scope".
  Catalogs synced: `skills/README.md` (agent-maintenance row), `claude/AGENTS.md` (cobb bullet), `claude/README.md` (skills section + script pointer), root `AGENTS.md` (skills line).
- **Why:** The 2026-07-09 teco review exposed a drift class cobb's machinery couldn't see: per-edit duties and the §3 doc audit both check *catalogs vs. disk*, but **other agents' prompts are also consumers of the roster** — qa-engineer and devops existed for days with perfect catalog entries while teco still enumerated a five-agent team, and several delegates carried "ask" phrasing that assumes an interactive session. User asked whether certifying is cobb's role (yes — he's the team's maintainer) and to build the improvement.
- **Certification note:** the 2026-07-09 teco review + fixes effectively constitute the first certification run — script PASS (8/8 agents: hooks, symlinks, roster, catalogs) and the judgment checklist applied (rosters completed, handoff symmetry restored on tdd-engineer, subagent-awareness added to coder/tdd-engineer/qa-engineer/graph-dba, enforcement parity closed with teco's guard hook).
- **Same-day follow-up (user feedback):** the initial version also guarded a collection-count line ("All eight use `model: opus`") in `claude/AGENTS.md`. User flagged it as token waste — right call: the sentence duplicated frontmatter ground truth in an always-loaded context file and rotted on both agent-count and model changes. **Deleted the fact instead of guarding it** — removed the line from `claude/AGENTS.md` and the equivalent from root `AGENTS.md`, dropped script check #6, and rewrote the skill's §2 rule to say so: don't create enumerated summary facts in always-loaded context docs; the per-agent `Model` column in the human `README.md` catalog is where that info lives. Lesson folded into the skill (delete brittle duplicated facts rather than build machinery to babysit them). Second round of the same feedback: the `Model` column in the `claude/README.md` catalog went too — the collection is model-agnostic and model is a *deployment choice* living in frontmatter; the skill's Audience-1 entry spec no longer lists "model" and now says not to advertise per-agent deployment choices (model, tool lists) in catalogs at all.
- **Plan items:** none closed; complements K-005/K-009 (deterministic drift guards — this one covers the agent collection itself).

## 2026-06-21 — Documented agent-to-agent messaging: Claude `SendMessage`/Agent Teams + Kiro (no equivalent)
- **What:** User asked whether `SendMessage` is a tool and whether cobb "should have it," then asked to verify for Claude Code **and** Kiro. Researched live (`code.claude.com/docs/en/agent-teams`, `code.claude.com/docs/en/sub-agents`; `kiro.dev/docs/chat/subagents`, `kiro.dev/docs/cli/experimental/delegate`). **Findings:** (1) **Claude Code** has two comms models — plain **subagents = report-back only** (no inter-agent tool) vs **Agent Teams = mailbox + `SendMessage`** (teammates message by name; `SendMessage`+task tools always available even under a restrictive `tools`). **`SendMessage` exists ONLY in Agent Teams, which is EXPERIMENTAL + OFF by default**, gated by env var **`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`** — so it's not exposed in a normal session and **can't be added via frontmatter** (the `tools:` allowlist filters from what the runtime exposes; it can't conjure an absent tool). A subagent *definition* can be reused as a teammate (honors its `tools`/`model`, body appended) and then gets `SendMessage` automatically; `skills`/`mcpServers` frontmatter is ignored for teammates. Limits: one team/session, no nested teams, fixed lead, no `/resume` of in-process teammates. (2) **Kiro** has **only report-back** subagents (communicate with main agent only, never each other; main waits for all) — **no `SendMessage`/mailbox equivalent**; the experimental `delegate` CLI feature is results-based parallel delegation, being deprecated in favor of subagents. **Edits:** rewrote the multi-agent-primitives section of `skills/agent-standards/claude-code.md` (named `SendMessage`, the env-var gate, the "frontmatter can't conjure it" point, teammate-from-definition, team limits) and added a "Communication model — report-back only" subsection to `kiro.md`; bumped both `Verified:` stamps to 2026-06-21.
- **Why:** Real gap — both per-tool files described agent teams/subagents but neither named `SendMessage`, the experimental env-var gate, or stated Kiro's lack of a messaging primitive. The user's "should cobb have it?" hinges on the runtime-gate-vs-frontmatter distinction, now captured. Practical impact: continuing a spawned subagent in this harness needs context re-injection (no `SendMessage` exposed), which this session confirmed empirically (ToolSearch found none).
- **Plan items:** advances K-001 (keep per-tool standards current).

## 2026-06-20 — Added Kiro **Knowledge** (CLI, experimental) to `agent-standards/kiro.md`
- **What:** Researched Kiro's Knowledge feature live (`kiro.dev/docs/cli/experimental/knowledge-management` + the CLI custom-agents config reference; WebSearch to locate the page) and added a new "Knowledge (CLI, **experimental**)" section to `skills/agent-standards/kiro.md`. Captured: enable flag (`kiro-cli settings chat.enableKnowledge true`), full `/knowledge` subcommand set (`add/show/update/remove/clear/cancel` with exact `add` flag syntax), Fast (BM25) vs Best (`all-minilm-l6-v2` semantic) index types, per-agent isolation + KB-sync on session init/agent swap, the `resources` `knowledgeBase` JSON schema (`type/source/name`+`description/indexType/autoUpdate`) and the `knowledge` built-in tool, default storage paths (Linux/macOS/Windows), `knowledge.*` settings tunables, supported file types + caveats (binaries skipped, large-file chunking, no auto-cleanup, irreversible `clear`), and two gotchas: **`--index-type Fast|Best` vs JSON `indexType` `fast|best` casing mismatch** and **not portable** (no Claude/OpenCode 1:1). Bumped the file's `Verified:` stamp note to cover Knowledge @ 2026-06-20.
- **Why:** User asked cobb to research the Kiro Knowledge feature; it was a genuine gap in `kiro.md` (which previously mentioned `knowledge` only as a built-in tool name). Folded the findings into the perishable skill so the next port/debug has it cold.
- **Plan items:** advances K-001 (keep Kiro standards current).

## 2026-06-20 — Repo-wide `AGENTS.md`/`CLAUDE.md` drift audit + DRY reconcile
- **What:** User reported context-file deviations across the repo. Ran the `agent-maintenance` §3 audit (`git ls-files` + filesystem enumeration; read all 7 context files). **Findings:** (a) the two co-located pairs that physically exist (root, `falkor-chat`) had **zero** content drift — both `CLAUDE.md`s were already `@AGENTS.md` stubs; the real deviations were *missing components* and *tool-convention mismatches*. (b) `opencode/agents/severino/` is an **OpenCode** project but its context file was named `CLAUDE.md` — the tool that runs Severino (OpenCode reads `AGENTS.md`, **not** `CLAUDE.md`) wasn't loading its own context. (c) `claude/` carried its agent catalog only in `CLAUDE.md` (no `AGENTS.md`) — inconsistent with the repo's own DRY rule. (d) `salesperson/` had only `AGENTS.md` and no `CLAUDE.md` → **Claude Code was not loading salesperson context at all** (verified live: code.claude.com/docs/en/memory states "Claude Code reads `CLAUDE.md`, not `AGENTS.md`"). (e) untracked stray empty dir `joern/`; (f) new untracked `kiro/` (with `kiro/DESIGN.md`) absent from root `AGENTS.md`. **Fixes (per user's choices):** `git mv` Severino + `claude/` `CLAUDE.md`→`AGENTS.md`, added `CLAUDE.md` = `@AGENTS.md` stubs to both + `salesperson/`; now **all 5 content components follow content-in-`AGENTS.md` + `CLAUDE.md`-stub** uniformly. Updated Severino `AGENTS.md` heading/intro to be tool-neutral. Repointed live refs: root `AGENTS.md` (4× `claude/CLAUDE.md`, 1× severino), `claude/README.md` maintenance rule, `skills/agent-standards/opencode.md` severino pointer. **Deferred by user:** documenting `kiro/` (untracked draft — wait until committed); left `joern/` (empty stray) untouched.
- **Why:** Cross-tool portability is a core cobb mandate; an OpenCode project documented in a Claude-only file and a Streamlit app whose context Claude Code never loaded are real, not cosmetic, defects. Verified the governing fact (Claude Code ≠ AGENTS.md reader) against the live doc before acting.
- **Plan items:** exercised the K-004 audit/reconcile method (now skill §3). Surfaced K-009 (see plan): the repo lacks a guard that every `AGENTS.md` has a `CLAUDE.md` stub — candidate for the K-005 drift job.

## 2026-06-20 — Refreshed `agent-standards` / opencode.md (agents, permissions, rules)
- **What:** Reconciled `skills/agent-standards/opencode.md` against live docs (`opencode.ai/docs/agents`, `/permissions`, `/rules`). Key changes: (1) **`tools` field is now DEPRECATED** — gate via `permission` instead (the old file listed `tools` as current). (2) Named **built-in agents**: primary `build` (all tools, default) / `plan` (edit+bash→ask); subagents `general` (full) / `explore` (read-only code) / `scout` (read-only external docs). (3) **Nesting confirmed: subagents CAN invoke subagents via the Task tool, gated by `permission.task`** (parallels the Claude finding; contrast Kiro where nesting is undocumented). (4) **Notable divergence flagged:** docs indicate OpenCode subagents *receive the parent session's conversation + file context* — opposite of Claude's isolated subagent — marked verify-on-version since the phrasing is loose and consequential; AGENTS.md→subagent propagation unspecified. (5) Permission **defaults** (`allow` except `doom_loop`/`external_directory`=`ask`, `.env` read=`deny`), which keys support glob control, `color` theme values, `instructions` supporting remote URLs (5 s timeout). Bumped stamp to 2026-06-20.
- **Why:** User stood up the OpenCode refresh after Claude + Kiro; opencode.md was the last per-tool file still on the 2026-06-07 baseline. Caught a real deprecation (`tools`) and a cross-tool behavioral divergence worth knowing before porting.
- **Plan items:** advances K-001 — **all three per-tool subagent/agent surfaces now re-verified 2026-06-20.** Remaining stale: Claude Code Skills/Memory/Hooks/MCP/SDK (2026-05-31).

## 2026-06-20 — Refreshed `agent-standards` / kiro.md (agents/subagents + steering/specs/hooks)
- **What:** Major reconcile of `skills/agent-standards/kiro.md` against live Kiro docs (`kiro.dev/docs/chat/subagents`, `/cli/custom-agents/configuration-reference`, `/steering`, specs & hooks pages; URL discovery via WebSearch since `kiro.dev/docs/llms.txt` 404s). Key additions: (1) **Two surfaces** — IDE custom agents are **Markdown+YAML** in `.kiro/agents/` (`name`/`description`/`tools`/`model`/`includeMcpJson`/`includePowers`); CLI agents are **JSON** in `.kiro/agents/*.json` with a rich schema (`prompt` file:// URIs, `tools`/`allowedTools`/`toolAliases`/`toolsSettings` incl. `toolsSettings.subagent` + `write.allowedPaths`, `resources` file/skill/knowledgeBase, `mcpServers`, `hooks` keyed `agentSpawn|userPromptSubmit|preToolUse|postToolUse|stop`, `keyboardShortcut`, `welcomeMessage`). (2) **Re-read the docs on the old "disputed" subagent caveat** — docs now affirm **steering + MCP reach subagents; Specs and Hooks do NOT**. ⚠️ Honesty correction: this does **not** *resolve* the original dispute (which was docs-say-X vs. field-reports-say-not-X). Re-fetching docs only confirms the documentation side; runtime behavior still needs a live test on the user's install. kiro.md + the porting note were softened from "verified/resolved" to "docs say X; field-disputed, verify on your install." (3) Concurrency (reported max 4, Ctrl+G monitor) + permission gating via `toolsSettings.subagent`; nesting still undocumented. (4) Steering: added Team/MDM scope + fileMatch array syntax. (5) Specs: `requirements.md`/`design.md`/`tasks.md` (+`bugfix.md`), dependency-graph **wave** concurrency. (6) Hooks: full trigger set. (7) Added a Claude-Code↔Kiro porting note. Bumped stamp to 2026-06-20.
- **Why:** User is standing up a Kiro ecosystem in the repo and asked for the same depth of research I did for Claude Code, to port agents over. The prior kiro.md left "what reaches a subagent" explicitly unverified — now resolved.
- **Plan items:** advances K-001 (re-verify standards) — Kiro now current.

## 2026-06-20 — Refreshed `agent-standards` / claude-code.md (subagent specifics)
- **What:** Re-verified the Claude Code subagent section against `code.claude.com/docs/en/sub-agents` (live) while building the `teco` coordinator, and updated `skills/agent-standards/claude-code.md`. Changes: (1) **tool inheritance + withheld-tools list** — `AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode` (unless `permissionMode: plan`), `ScheduleWakeup`, `WaitForMcpServers` are withheld even if listed in `tools`; **the `Agent`/Task tool is NOT withheld → subagents can delegate to subagents** (supersedes the old "no nesting" lore). (2) **Expanded frontmatter** — added `mcpServers`, `hooks`, `maxTurns`, `background`, `color`, `prompt`/`initialPrompt` (CLI), full `permissionMode` value set (`default|acceptEdits|auto|dontAsk|bypassPermissions|plan`), and the `skills` preload nuance (full content injected). (3) **Discovery/scopes** — managed/project/user/plugin/CLI scopes + priority; walk-up discovery with v2.1.178 nearest-wins; `--add-dir` scanning. (4) **New multi-agent primitives** — *agent teams* and *background agents*. (5) Subagent receives only its system prompt + cwd (not the full CC system prompt); `cd` doesn't persist. Bumped the file's `Verified:` stamp to 2026-06-20.
- **Why:** Drift-resistance mandate + direct need: the `teco` design hinged on whether subagents can spawn subagents. The 2026-06-07 snapshot lacked the withheld-tools list and the newer fields/primitives.
- **Plan items:** advances K-001 (re-verify standards) — subagents portion now current; Skills/Memory/Hooks/MCP/SDK still on the 2026-05-31 baseline.

## 2026-06-20 — Dropped "senior" framing (collection harmonization)
- **What:** Body opener "a senior practitioner of agentic software development" → "a practitioner of agentic software development". Catalog row in `claude/README.md` "Senior practitioner" → "Practitioner". (cobb's frontmatter `description` already led with "Expert…", no "senior".)
- **Why:** Collection-wide harmonization after the new `architect`/`coder` agents dropped "senior" entirely (overconfidence concern; persona-prompting evidence shows role labels are weak-to-neutral for correctness). Brings cobb in line so the whole Claude collection is consistent. Supersedes the 2026-06-05 stance that kept "Senior" as an altitude signal.
- **Plan items:** —

## 2026-06-16 — Unified skills into a repo-root `skills/` home (moved cobb's machinery skills)
- **What:** Per user request, merged `claude/skills/` and `opencode/skills/` into a new repo-root **`skills/`** sibling of `claude/` and `opencode/` — `git mv` of all 7 skill folders (cobb's `agent-maintenance`, `agent-standards` + OpenCode's `comparison-driver`, `python-coding`, `skill-builder`, `user-preferences`, `write-tutorial`; no name collisions, history preserved). Created `skills/README.md` (unified human catalog, deployment notes, portability caveat). Repointed all live references: `opencode/agents/rpg.md` storage path, root `AGENTS.md` (Structure, component-docs table, "Claude Code subagents" section, new "Skills" section, user-preferences path, working-in-repo rules), `claude/CLAUDE.md` (skills-moved note + maintenance rule), `claude/README.md` (Skills section now points at `../skills/`, Deployment section, agent-standards links). Left generic/canonical-path mentions (`skill-builder` deploy examples, `agent-standards/claude-code.md` canonical `.claude/skills/` path, graph-dba hypothetical) and immutable kaizen log entries untouched.
- **Why:** User wants one canonical skills home enabling single-source deployment to multiple tools (Claude Code/OpenCode/Kiro all read the open `SKILL.md` standard). Explicitly deferred the symlink repointing ("don't worry about symlink yet") — flagged in 3 docs that `~/.claude/skills → claude/skills` now dangles and must be repointed to `skills/` before cobb's skills resolve at runtime.
- **Plan items:** structural/no plan item; supports the cross-tool-portability theme (K-001 adjacent). Flagged follow-up: repoint deployment symlinks; decide deliberate per-tool exposure (cobb machinery vs OpenCode skills) when wiring deployment.

## 2026-06-16 — Deployed unified `skills/` to all three harnesses (whole-dir symlinks)
- **What:** Created whole-dir symlinks from each tool's global config to the repo-root `skills/` home: `~/.claude/skills`, `~/.config/opencode/skills` (both replaced the now-dangling links that targeted the moved `claude/skills`/`opencode/skills`), and `~/.kiro/skills` (new — Kiro freshly installed today, no prior skills dir). All three resolve to the 7 skills; Claude Code picked it up live this session (skill list now shows all 7, OpenCode-authored ones included). Updated the deployment docs to past-tense/“deployed via” in `skills/README.md`, `claude/README.md`, and root `AGENTS.md`.
- **Why:** User chose **all 7 to every harness** ("I will try all agents on all harnesses") over selective per-skill scoping — simplest, and progressive disclosure makes unused skills ~free. Resolved the exposure decision flagged in the prior entry. Kept the per-skill-symlink escape hatch documented for future scoping.
- **Plan items:** closes the "repoint deployment symlinks / decide exposure" follow-up from the unification entry above. Symlinks are personal/not in-repo → README/AGENTS carry the recreate-on-new-machine note.

## 2026-06-16 — Refreshed `agent-standards/kiro.md`: added Kiro Agent Skills (K-001)
- **What:** While answering a user question on cross-tool skill portability, caught drift in `agent-standards/kiro.md` (stamped 2026-06-07, predated Kiro Skills). Verified live against `kiro.dev/docs/skills` + `kiro.dev/docs/cli/skills` and added a new **Agent Skills** section: open `agentskills.io` standard (added 2026-02-05), location `.kiro/skills/<name>/SKILL.md` (workspace) / `~/.kiro/skills/` (global, workspace wins), required frontmatter `name` (=folder name, ≤64 chars, lowercase/numbers/hyphens) + `description` (≤1024 chars, the routing signal), optional `license`/`compatibility`/`metadata`, **no documented tool-restriction field** (so `allowed-tools` sandboxing does NOT port — re-audit on port), and explicit-only reference-file loading. Updated the building-blocks line (Steering/Specs/Hooks → + Skills) and bumped the `Verified:` stamp to 2026-06-16 for the Skills section.
- **Why:** Drift-resistance — the question exercised the exact lookup that was stale; agent-standards' own rule says reconcile + bump stamp + log on discovering drift. Surfaced the portability answer to the user too (format ports across all three; tool-gating + activation behavior do not).
- **Plan items:** advances K-001 (Kiro re-verified 2026-06-16; Kiro Skills now covered). Reinforces K-005 (skill remains the single doc-drift patch target).

## 2026-06-07 — Allowlisted official doc domains for unprompted WebFetch; documented as deployment
- **What:** Added a `permissions.allow` block to `~/.claude/settings.json` (user scope) allowing `WebFetch` to `code.claude.com`, `platform.claude.com`, `docs.anthropic.com`, `kiro.dev`, `opencode.ai`, plus `WebSearch` — so cobb's live-doc verification (and the `agent-standards` freshness re-checks) runs without a confirmation prompt. Documented it as a **Deployment** section in `claude/README.md` (symlink layout + the allowlist JSON + scope/redirect/persistence caveats).
- **Why:** User wanted cobb's doc fetches unprompted. Considered a cobb-only `PreToolUse` hook (truly per-agent, harness-enforced) vs. a user-scope allowlist (session-wide, trivial); user chose the simple user-scope allowlist — acceptable blast radius since these are read-only GETs to five official doc hosts. Recorded the hook alternative for posterity. settings.json is personal/not in-repo, hence the README note to re-add on a new machine.
- **Plan items:** supports K-001/K-005 (makes live re-verification frictionless).

## 2026-06-07 — Extracted standards into the `agent-standards` skill (K-007); re-verified Kiro + OpenCode (K-001)
- **What:** Built the **`agent-standards`** skill at `claude/skills/agent-standards/` — `SKILL.md` (frontmatter `name`/`description`/`allowed-tools: Read, WebFetch, WebSearch`; body = navigation table + stable cross-tool standards + canonical URLs + a "this is a cache, check the `Verified:` stamp and WebFetch before asserting" rule + the drift-update procedure) plus three per-tool reference files (`claude-code.md`, `kiro.md`, `opencode.md`), each with its own `Verified:` stamp. Replaced cobb.md's ~24-line "Standards you know cold" block with a compact resident version: stable mental models per tool + cross-tool standards + canonical URLs + an instruction to load the skill for exact field/path specifics. **Re-verified Kiro and OpenCode live** (kiro.dev/docs/steering; opencode.ai/docs/agents + /rules) — Kiro confirmed as-was; OpenCode had drifted: `mode` has a third value `all` (the default), new fields `disable`/`color`/`top_p`/`steps`, the `permission` set is far more granular than "edit/bash" (read/edit/glob/grep/list/bash/task/external_directory/todowrite/webfetch/websearch/lsp/skill/question/doom_loop), and AGENTS.md precedence rules (local AGENTS.md > CLAUDE.md, first match wins). Updated catalogs: `claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md` (two spots).
- **Why:** K-007 — the perishable specifics violated cobb's own Lean-context + Drift-resistance principles by sitting in the always-on prompt. **Decision recorded: skill, NOT RAG** — corpus is small (a few KB), cleanly structured by tool × artifact, and demands exact retrieval; RAG's approximate top-k attacks the "never fabricate a frontmatter key" guarantee and a local embedded snapshot reintroduces the staleness we're fixing. Three-layer architecture: resident prompt (mental models + URLs + dates) → `agent-standards` skill (curated, freshness-stamped; the K-005 patch target) → live WebFetch as the real-time layer. Revisit RAG only at K-003 scale, and even then hybrid. The build doubled as a K-001 refresh and immediately paid for itself by catching the OpenCode drift.
- **Plan items:** K-007 (✅ done — moved to history); advances K-001 (Kiro/OpenCode re-verified 2026-06-07; remaining gap: Claude Code Skills/Memory/Hooks/MCP/SDK still on 2026-05-31). Reinforces K-005 (skill is now the concrete update target).

## 2026-06-07 — Self-review pass (no prompt change); logged K-007, K-008
- **What:** User asked cobb to analyze itself. Review-only — `cobb.md` left unchanged. Findings: (A) the "Standards you know cold" block is the most perishable content but sits in the always-on prompt, against cobb's own Lean-context + Drift-resistance principles → logged **K-007** (extract to a progressively-disclosed `agent-standards` reference skill, K-006 pattern); (B) cobb under-uses the frontmatter it teaches → logged **K-008** (evaluate `memory: project`; keep agent-maintenance on-demand, do not pin via `skills:`); (C) Kiro/OpenCode stamps a week stale, reinforces K-001/K-005; (D) re-flagged the parking-lot self-review checklist for promotion. Noted strengths to preserve: the routing-grade `description` and the kaizen discipline itself.
- **Why:** Kaizen discipline requires a review pass to keep `plan.md` current and leave a trail even when no artifact line changes.
- **Plan items:** added K-007, K-008; re-flagged the self-review-checklist idea; reinforced K-001/K-005.

## 2026-06-07 — Reconcile pass: added `claude/skills/` to the repo-root `AGENTS.md`
- **What:** Cross-scope follow-up to the K-006 work. Ran the skill's audit/reconcile method (§3) against the root `AGENTS.md`: enumerated ground truth, confirmed the only drift was the new `claude/skills/` home, and updated three spots — the `claude/` Structure bullet (added the `skills/` sub-bullet), the renamed "Claude Code subagents **& skills** (`claude/`)" section (noted `agent-maintenance`), and the "Working in this repo" rule (now "subagent / skill tasks"). Rest of the doc verified current (falkor-chat, graph-dba, severino all present).
- **Why:** Demonstrates the in-scope/cross-scope split the same session just encoded: the per-edit duty (claude/README.md + claude/CLAUDE.md) happened inline with the skill creation; the repo-root catalog was deferred to this explicit reconcile pass rather than bolted onto the edit. User said "you can reconcile already."
- **Plan items:** exercises K-004 (the now-documented audit/reconcile method).

## 2026-06-07 — Slimmed the prompt: extracted the maintenance machinery into the `agent-maintenance` skill (K-006)
- **What:** Replaced the two `OBLIGATORY` sections (~95 lines — kaizen templates + file-location decision tree + dual-audience documentation table + order-of-operations) with a tight ~8-line **"Maintenance duties"** block that states the obligations and points at a new progressively-disclosed **`agent-maintenance` skill**. Authored the skill at `~/.claude/skills/agent-maintenance/SKILL.md` (frontmatter `name`/`description`/`allowed-tools`), porting the kaizen procedure + plan/history templates, the dual-audience documentation method (incl. the DRY `CLAUDE.md → @AGENTS.md` import rule), the file-location rules, and — promoted from idea to documented method — the **K-004 audit/reconcile** drift procedure (`git ls-files` vs. the doc's claims). Split the doc duty into **in-scope** (per-edit: update the artifact's own kaizen + catalog entry, stays resident) vs. **cross-scope** (repo-root catalog reflecting all components → on-demand reconcile pass, not bolted onto every edit). Folded `TESTING.md` in by reference (skill §4 points at `claude/cobb/TESTING.md`, which stays tracked in-repo and referenced by `claude/CLAUDE.md`). Added per-section `*Verified: DATE*` stamps to the three "Standards you know cold" subsections so staleness is visible.
- **Why:** ~95 of ~160 prompt lines were reference *manual* sitting in the always-loaded prompt and firing on every turn (including pure Q&A) — a direct violation of Cobb's own "lean context / push detail into progressively-disclosed skills" principle. Moving the manual to a skill costs nothing on the compliance axis (inline `OBLIGATORY` text was never enforcement, only hopeful prompt text) while reclaiming context weight; the *mandate* stays resident, the *procedure* loads on demand. Also gave the orphaned `TESTING.md` a reachable pointer (the prompt previously had none).
- **Deployment note:** authored via `~/.claude/skills/`; the user then created the `~/.claude/skills` → `claude/skills` symlink, so the file actually lives in-repo at **`claude/skills/agent-maintenance/SKILL.md`** (version-controlled, parallels the `~/.claude/agents` → `claude/<name>` agent symlinks). `claude/skills/` is now the collection-level skills home.
- **Plan items:** K-006 (✅ done — moved to history); absorbs K-004 (the audit/reconcile method is now documented in the skill) and K-002's skill-bundle intent.

## 2026-06-07 — Added "Drift-resistance" principle + cross-tool subagent comparison; opened K-005 (automated drift check)
- **What:** Added a "Drift-resistance" bullet to the prompt's Principles: keep stable mental models + canonical URLs in the always-on prompt, treat field lists / "who-loads-what" tables / feature availability as perishable (stamp `verified DATE against <url>`, prefer live-verify or an updatable skill), and don't assume one tool's behavior transfers. Verified the cross-tool picture for subagent context-loading: **Claude Code** custom subagents auto-load the `CLAUDE.md` hierarchy; **OpenCode** docs are silent on whether subagents get `AGENTS.md` (subagents inherit the invoking primary's model); **Kiro** has subagents since 0.9 and its docs claim `inclusion: always` steering reaches them, but open issues (#7131, #7758) dispute that in practice — Specs/Hooks definitely don't reach Kiro subagents. Opened **K-005** to automate doc-drift detection (scheduled doc-diff → kaizen item).
- **Why:** User asked whether the subagent CLAUDE.md behavior is Claude-specific or portable, and "how can we ensure the info will not drift?" The answer is that it's tool-specific and in flux — which makes drift-resistance a first-class principle, not just K-001's manual re-verify.
- **Plan items:** K-005 (added); reinforces K-001.

## 2026-06-07 — Sharpened subagent context-loading knowledge (verified against official docs)
- **What:** Expanded the Claude Code "Subagents" bullet in the prompt with facts I verified live at code.claude.com/docs/en/sub-agents: (1) custom subagents **do** auto-load the full `CLAUDE.md`/memory hierarchy via message flow even though the body replaces the default system prompt, and `@`-imports expand into them — built-in **Explore/Plan** are the only ones that skip `CLAUDE.md`+git (not configurable); a **fork** inherits the whole parent conversation instead; (2) subagents don't see parent conversation history/tool results/skills — must be passed in the delegation prompt; (3) added the `memory:` frontmatter field (persistent `agent-memory/<name>/` store, distinct from `CLAUDE.md`) and noted other current frontmatter fields (`disallowedTools`, `permissionMode`, `skills`, `isolation`, `effort`, `inherit`) with a verify-against-docs caveat.
- **Why:** A user question ("do agents like you autoload CLAUDE.md?") exposed a real gap in an area the prompt claims to know cold: it described subagent context only as "isolated," implying CLAUDE.md might not load, and omitted the `memory:` field entirely. Fixed because broken/stale Claude Code specifics make me produce wrong artifacts — the core risk K-001 guards against.
- **Plan items:** advances K-001 (re-verified subagent docs; baseline date refreshed to 2026-06-07).

## 2026-06-07 — Learned the DRY import pattern + drift-audit method (from syncing graphmind-ai-lab's AGENTS.md)
- **What:** While bringing the repo's stale root `AGENTS.md` back in sync (it was missing the entire `falkor-chat/` component, the `graph-dba` agent, and the `severino` OpenCode agent), recommended and created a root `CLAUDE.md` containing just `@AGENTS.md`. Folded that single-source-of-truth rule into the prompt's "Documentation" section ("Don't duplicate the same catalog into two files"). Opened K-004 to capture the standalone audit-&-reconcile method (drift detection via `git ls-files` vs. the doc's claims), flagged as skill material to keep the prompt lean.
- **Why:** Two durable learnings surfaced from the session: (1) when `CLAUDE.md` and `AGENTS.md` would carry the same catalog, importing avoids divergence and keeps `AGENTS.md` as the broadest-reach source; (2) reconciling an already-drifted context doc is a recurring task distinct from the "sync on my own edits" duty already in the prompt.
- **Plan items:** K-004 (added).

## 2026-08-24 — Revised `write-guard-classifier-gap.md` (v2) addressing `analyst`'s review

- **What:** `analyst` reviewed the design (`claude/docs/reviews/write-guard-classifier-gap.md`,
  `374a350`, verdict "needs changes"). Independently re-read the live wrapper scripts to confirm
  each finding before revising rather than trusting the review's summary: (1) **Blocker**
  confirmed — `guard-cobb-topic-writes.sh`'s allowlist bundles low-stakes kaizen/catalog docs with
  prompt-governing files (`claude/*/*.md` = every agent's own definition/hooks/tool-grants,
  `skills/agent-maintenance/*`, `skills/agent-standards/*`), none of which the protected-path
  carve-out catches (it's dot-prefix-only; this repo's agent defs live at plain
  `claude/<name>/<name>.md`). Split §5.3 in two: kaizen/catalog docs stay a candidate, the
  agent-definition/skill-package globs are excluded outright — a standing `Edit(claude/**/*.md)`
  rule would silently reopen the exact self-modification hazard already caught and stopped
  2026-08-20. (2) Added §5.2's per-glob risk table — `docs/reviews/**`/`docs/test-reports/**`
  flagged as self-approval risk (the agent under review/test could silently edit its own verdict),
  `docs/requirements/**` as moving the acceptance-criteria goalposts unnoticed, `docs/plans/**` as
  moderate (backstopped by a companion review), test-plans/manuals as fine — folded into a
  revised §8 as an explicit per-glob sign-off rather than one blanket trade judgment. (3) Re-`grep`'d
  `claude/teco/hooks/guard-coordination-doc-writes.sh` and `claude/data-scientist/hooks/guard-ds-doc-writes.sh`
  myself and confirmed the review's finding: both prior-table descriptions ("`-coordination.md`
  suffix," "ML-scoped") were wrong — the live globs are byte-identical to `architect`'s and the
  union of `architect`'s+`analyst`'s respectively. Rewrote §5.1 to state the real shape (four
  distinct glob surfaces claimed by multiple agents, not eight independent decisions). (4) Rewrote
  §6 to show, not just assert, why an `ask`-only (no blanket allow) translation doesn't rescue
  `tdd-engineer`'s deny-list shape — it never touches the actual friction source, since it only
  fires on paths that were already correctly escalating. Also folded in the review's "Note" finding
  (§2.1 caveat: two live hypotheses on whether rules bypass the classifier for subagent writes at
  all, not just one) so whoever runs the live test in §7 knows to distinguish them. Bumped
  `Version: 2`, added a `Reviews:` header pointer, left `Status: active` (in-place revision — not
  yet approved/executed, no ordinal successor needed per the repo's doc-lifecycle convention).
  Judgment call on a second full review pass: **recommended against** — every change traces
  directly to one of the review's four numbered findings and its own suggested fix, re-verified
  independently against the live scripts rather than taken on faith, with no new judgment calls
  introduced beyond what the review proposed; a second full pass would be re-checking corrections
  against their own source rather than surfacing anything new.
- **Why:** `analyst`'s review (routed through `teco`) found the design's reasoning framework sound
  but flagged one genuine blocker (a self-modification-hazard exposure the original §5 didn't
  differentiate from lower-stakes doc-kind globs) plus two description-accuracy errors that would
  have misrepresented the design's actual incremental exposure to the review gate. All four
  findings held up under independent re-verification, so all four got folded in as specified rather
  than adjudicated.
- **Plan items:** — (unchanged: empirical validation via a human's live interactive session is still
  the gate before any implementation, per the document's own §7.)

## 2026-08-24 — Wrote `write-guard-classifier-gap.md`: `permissions.allow`-rule design for the classifier gap (dispatched by `teco`)

- **What:** Follow-up to the same-day RCA (below). Re-read `code.claude.com/docs/en/permissions`
  specifically for the rule mechanism (not hooks) and found: (1) the classifier's own documented
  decision order resolves a matching settings.json `allow`/`ask`/`deny` **rule** at step 1, before
  the classifier is ever invoked — unlike a `PreToolUse` hook's `"allow"`, which is never named as
  an input to that decision order; (2) rules only key off `Edit(path)`/`Read(path)`, never
  `Write(path)` — a naive dual-rule translation would silently no-op half of it. Attempted a live
  empirical test (isolated git worktree, ephemeral `--settings` CLI flag, never persisted) to
  confirm a rule actually suppresses the prompt for a Task-delegated write; the attempt was blocked
  outright by my own session's auto-mode classifier before it could run ("spawning a nested
  `claude -p` process" read as exactly the kind of action it's built to catch) — cleaned up fully
  (worktree removed, `git status` clean), did not attempt a workaround per the denial's own
  instruction, reported this as still-open rather than guessing. Wrote up a split-verdict design in
  `claude/docs/plans/write-guard-classifier-gap.md`: the narrow allow-list guards (`analyst`,
  `architect`, `data-scientist`, `teco`, `tico`, `cobb`, `qa-engineer`, `security-expert`'s review
  guard) are reasonable rule-supplement candidates, explicitly trading a per-agent escalation
  guarantee for a per-path one (flagged, not silently accepted); `tdd-engineer`'s deny-list shape is
  explicitly excluded — rules aren't agent-scoped, so the only literal translation would blanket-open
  the entire repo to every session, not just that one agent. Design only, not implemented; routed to
  `teco` for the `analyst` review gate.
- **Why:** The coordinator pushed back on "proceed as-is, defaultMode is the only lever" and asked
  specifically whether a declarative permission *rule* (a documented mechanism separate from hooks)
  closes the gap the RCA found, before the `defaultMode` change becomes the only option on the
  table. The scoping tradeoff (per-agent vs. per-path) is the crux the design had to make explicit
  rather than paper over — a mechanical glob-string copy from hook to rule would look like a clean
  fix while quietly widening who gets auto-approved.
- **Plan items:** — (the document itself carries the open items: empirical validation before any
  implementation, and the three questions flagged for `analyst`'s gate in its §8.)

## 2026-08-24 — Root-caused the `PreToolUse` "allow" write-prompt regression (dispatched by `teco`, `agent-permission-friction2.md` open question 3)

- **What:** Investigated why two live, stakeholder-confirmed instances (`analyst` → `docs/reviews/document-ingestion-impl.md`, `tdd-engineer` → `cypher-mcp/tests/test_server.py`, both 2026-08-23) still triggered a manual confirmation prompt despite each path statically matching its shipped write-guard's allowlist/deny-list and each guard core confirmed (by direct read, both scripts unchanged since `93c3a39`/2026-08-21) to emit an explicit `permissionDecision:"allow"` on that match. Ruled out, with direct evidence: stale `$HOME/.claude/agents/{analyst,tdd-engineer}` symlinks (both live, correct targets), a guard-script regression (git log clean since 2026-08-21), a single-CLI-version bug (instances reproduced on both v2.1.240 and v2.1.241), and a nested-git-repo trust gap (`falkor-chat`/`cypher-mcp` are plain subdirectories, not nested repos; project trust `hasTrustDialogAccepted` confirmed `true`). Found the actual mechanism by reading the two failing tool calls' own session transcripts (`~/.claude/projects/<project-slug>/a668e215-.../subagents/agent-*.jsonl`): both showed a genuine multi-minute-to-multi-hour gap between `tool_use` and its `toolUseResult` (14 min; 4h38m) — a live human-decision gap, not a fabricated report — and the parent (`teco`) top-level session's own transcript carries an explicit `"type":"permission-mode"` record showing the session stayed in **`auto`** mode continuously across both incidents (this account's Pro/Max/Team default, confirmed also in `~/.claude/settings.json`'s `defaultMode:"auto"`). Fetched `code.claude.com/docs/en/permissions` and `.../permission-modes` fresh (2026-08-24, not from phase 1's cached reading) and found: (1) phase 1's §1.3 "hook allow suppresses the prompt... every time" already over-claimed its own cited quote, which carries the caveat "a matching ask rule still prompts even when the hook returned allow" — no such settings.json ask/deny rule exists here, so this alone doesn't explain the instances; (2) auto mode's own documented decision order says a non-protected-path working-directory file edit is auto-approved at step 2, with **zero classifier involvement and zero human prompt, regardless of hooks** — which the two instances directly contradict, since neither target is a protected path; (3) nothing in the docs states whether a `PreToolUse` hook's `"allow"` exempts a **subagent-delegated** action from the auto-mode classifier's own review (the "How auto mode handles subagents" section says subagent actions go through the classifier "with the same rules as the parent session" but is silent on the hook interaction). Conclusion: the friction is a real, live-reproduced gap in the auto-mode-classifier-vs-`PreToolUse`-hook interaction for **Task/Agent-tool-delegated writes specifically** — outside my remit to fix (it isn't a guard-script or settings.json bug; changing the account's or project's `defaultMode` away from `auto` to route around it is a broad, non-hook-engineering call I flagged rather than made unilaterally, per my own standing "stop and ask" instruction for costly-to-reverse scope changes). Promoted the verified finding into `skills/agent-standards/claude-code.md`'s Hooks section (dated 2026-08-24, alongside the existing Bash-classifier note) so a future phase-2-style write-guard design doesn't re-derive phase 1's now-falsified "hook allow is unconditional" premise. Reported back to `teco`: root cause identified and documented; not fixed (no lever in my remit); recommended phase-2 `coder` design proceed, since blocking on this would also block on something that already affects the five shipped agents equally, plus a concrete next test (a fresh, non-concurrent, top-level `--agent analyst` write, mode-bar watched live) that only the stakeholder's own interactive terminal can run.
- **Why:** Dispatched as a blocker check ahead of phase-2 (`coder`) write-guard design, per `agent-permission-friction2.md` open question 3 ("probably needs resolving before phase-2 design proceeds"). The live evidence directly falsified phase 1's root-cause finding, so the fix had to be traced to its actual mechanism (or ruled un-fixable from this repo) before phase 2 could safely reuse the same design pattern.
- **Plan items:** — (candidate open item: once the stakeholder runs the recommended live isolation test, come back and either narrow the `skills/agent-standards/claude-code.md` note to a confirmed single cause, or extend the write-guard design with whatever the test reveals.)

## 2026-05-31 — Redacted absolute paths from eval reports (privacy)
- **What:** Fixed a leak where `baseline/01-explain-python.md` contained the full `/home/<user>/...` path (the model echoed the absolute path it was given via `-f`). In `run.sh`: attach fixtures as paths relative to `PROJECT_DIR` (we already `cd` there), plus a `sed` scrub safety-net on the captured body (`PROJECT_DIR/` → relative, remaining `$HOME/` → `~/`). Re-ran and re-blessed all three cases; verified `baseline/` is now free of absolute home paths. Documented as a design invariant in `cobb/TESTING.md`.
- **Why:** User flagged that a committed baseline exposed the username via an absolute path; reports must use relative paths so baselines are safe to commit to a shared repo.
- **Plan items:** —

## 2026-08-01 — Documented `acceptEdits` scope boundary (working dir + `additionalDirectories` only)

- **What:** Answered a user question about why `tdd-engineer` (`permissionMode: acceptEdits`) still prompts for approval when writing to its own kaizen learnings inbox at `$HOME/.claude/agents/tdd-engineer/kaizen/inbox.md`. WebFetched `code.claude.com/docs/en/permissions` and confirmed: `acceptEdits` "automatically accepts file edits … for paths in the working directory or `additionalDirectories`" — it is **not** a global auto-accept. The inbox path sits outside the project repo the session is launched in and isn't an `additionalDirectory`, so every write to it still prompts, regardless of `permissionMode`. Promoted this into `skills/agent-standards/claude-code.md` inline next to the `permissionMode` field list, stamped `verified 2026-08-01`.
- **Why:** Perishable, non-obvious fact that affects every agent whose learning-capture instruction writes outside the repo (all of them, via the `$HOME/.claude/agents/<name>/kaizen/inbox.md` convention). Same-run promotion is in-bounds for cobb per its own maintenance mandate — verified via live docs, so no inbox detour needed.
- **Plan items:** —

## 2026-05-31 — Added deterministic assertions (#2) to the agent eval harness
- **What:** Implemented item #2 from the promotion roadmap in `opencode/agents/severino/tests/run.sh`: an optional per-case `expect.md` with `require:`/`reject:` literal-substring directives, checked against the **response body only** (so a prompt quoting forbidden text can't trip a `reject:`). Prints per-assertion `PASS`/`FAIL`, a suite tally, and **gates the exit code** (any failure → exit 1; cases without `expect.md` stay advisory, exit 0). Excluded `expect.md` from attached fixtures; hardened `--help` to print only the leading comment block via awk. Seeded the 3 starter cases with assertions and verified live (clean: 4 passed/exit 0; forced fail: 1 passed 1 failed/exit 1). Updated `tests/README.md` and `cobb/TESTING.md` (assertions section + roadmap: #2 moved to done).
- **Why:** User chose to build #2 next. It's the highest-leverage maturity step — diffs are noisy on a stochastic local LLM, so deterministic substring assertions are what make the harness reword-proof and partially CI-gateable.
- **Plan items:** —

## 2026-05-31 — Authored TESTING.md + hardened the Severino eval harness
- **What:** (1) Created `cobb/TESTING.md` documenting the project's two-altitude testing standard (pytest for deterministic code; eval/bless harness for agent behavior) and the reusable agent-eval-harness pattern. (2) In `opencode/agents/severino/tests/run.sh`, did items #1 and #3 from the promotion review: **#3 decoupled from "severino"** — agent name now auto-derives from the parent dir (`AGENT=${AGENT:-$(basename "$PROJECT_DIR")}`), per-run stderr uses `mktemp` instead of a fixed `/tmp/severino-run.stderr`, and banner/help/comments are agent-agnostic; **#1 proven green** — ran all 3 cases, blessed `baseline/`, and confirmed the diff loop works (a re-run correctly flagged `changed`). Updated `tests/README.md` to document the agent-agnostic `AGENT` override.
- **Why:** User wants to promote the Severino harness into a reusable pattern; agreed to harden it first (prove green + decouple) before extracting a shared template. TESTING.md is cobb's living reference for that pattern. Deferred to roadmap: #2 lightweight `expect.md` assertions (highest leverage — case 02 diagnosed a bug correctly but emitted a broken fix, which a pure diff can't catch), temperature pinning, N-sample runs, strict/CI mode.
- **Plan items:** —

## 2026-05-31 — Restructured to per-agent subdirectories
- **What:** Moved each agent into its own folder (`cobb/cobb.md`, `tdd-engineer/…`, `medicina-alternativa/…`) and moved cobb's kaizen from `kaizen/cobb/` to `cobb/kaizen/`. Simplified the kaizen location rule in `cobb.md`: an artifact with its own folder uses `<folder>/kaizen/{plan,history}.md` (no `<name>` nesting); only a lone file sharing a directory namespaces by `<name>`. Made the README rule collection-level (root catalog). Updated `README.md` and `CLAUDE.md` paths.
- **Why:** User opted for a self-contained folder per agent. Verified Claude Code discovers `.claude/agents/` recursively and identifies agents by the `name:` frontmatter field, not the path — so per-agent subdirectories work natively (names must stay unique across the tree).
- **Plan items:** —

## 2026-05-31 — Added dual-audience documentation responsibility
- **What:** Added the "Documentation — keep both audiences informed" section to `cobb.md`: maintain a human-facing `README.md` catalog and update the project's agent-context convention (`CLAUDE.md` / `AGENTS.md` / `.kiro/steering`) on every create/edit/rename/remove. Bootstrapped `README.md` and `CLAUDE.md` for the `agents/` directory.
- **Why:** User wants agents created/edited by Cobb to always be documented for both the user and for other agents working on them.
- **Plan items:** —

## 2026-05-31 — Added kaizen maintenance responsibility
- **What:** Added the "Kaizen — maintain each agent's improvement plan & history" section to `cobb.md`, defining the `<dev-dir>/kaizen/<name>/{plan,history}.md` convention and templates. Bootstrapped Cobb's own `plan.md` and `history.md`.
- **Why:** User wants Cobb to maintain a living improvement plan and change history for every agent/skill it works on.
- **Plan items:** —

## 2026-05-31 — Agent created
- **What:** Initial authoring of the `cobb` agent — expert in agentic development across Claude Code / Claude Agent SDK, Kiro, and OpenCode, with a mandate to web-search official docs when specifics are version-sensitive. Frontmatter `name: cobb`, `model: opus`, routing-oriented `description`.
- **Why:** User requested a new agent specialized in agentic-development standards used by Claude, Kiro, and OpenCode.
- **Plan items:** seeded K-001 (re-verify docs), K-002 (porting example), K-003 (broaden tool coverage).
