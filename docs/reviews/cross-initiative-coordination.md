# Review — cross-initiative-coordination requirements

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/requirements/cross-initiative-coordination.md`

## Pass 1

**Scope.** `docs/requirements/cross-initiative-coordination.md` Version 3, `Status: Interviewing` (uncommitted working-tree edit; baseline HEAD = v2). Checked against root `AGENTS.md`, root `REVIEW.md`, `docs/requirements/document-size-discipline.md`, `claude/teco/teco.md`, `cypher-mcp/README.md`, and the repo's existing coordination ledgers and requirements. No code reviewed.

**Verdict: needs changes** (one blocker, which is a stakeholder decision rather than a text fix; a requirements flip to Ready for design must wait on it).

CPG: not applicable — a requirements-document review with no code-level component.

### Findings

**B1 (blocker, verified) — Overlaps an existing Ready-for-design requirement and active plan, and cites neither.**
`claude/docs/requirements/coordination-graph.md` (Ready for design, 2026-10-21) has a plan, `claude/docs/plans/coordination-graph.md` (active, v2), and a graph note, `claude/docs/plans/coordination-graph-graph.md`. Together they already specify:
- a graph mirror of coordination ledgers, rebuildable from the files alone (its FR-2 and AC around lines 76 and 166);
- a "what's in flight" view across every component (its FR-6);
- sync-time and drift stamps, which this document's FR-4 and AC-10 ask for;
- the ledger table shape and its 362 free-text status values (plan §1 inventory, `coordination-graph.md:52`);
- dependency and shared-resource collision handling: its requirements lines 118-133 and ~199 carry "dependencies respected" and "G and H both rebuild the same graph key".

The new document's grep for `coordination-graph` returns nothing. This is the same class of overlap the document exists to catch, and it is not escalated. Delivery state of that plan is unverified: there are no `HISTORY.md`/`BACKLOG.md` hits under `claude/docs/` or root `docs/`.
*Fix:* stakeholder decision first. Either (a) this document is a delta (units gain `needs`/`resources`, ready-now view, FR-11) extending that requirement (`Extends:` header pointer, and state only the change), or (b) it supersedes it, or (c) the two merge. Until then, FR-1, FR-2, FR-4 and AC-6 are restated facts (growth-stops, FR-6 of the size-discipline requirements).

**M1 (major, verified) — FR-4/FR-10/AC-6 contradict each other on legacy ledgers.**
FR-10 says only post-ship ledgers carry the block and existing ledgers need not migrate. FR-4 says a ledger that "cannot be parsed is rejected loudly". AC-6 says the graph can be "fully rebuilt from the ledger files". Roughly 100 ledgers have no block (`ls */docs/plans/*-coordination.md`; 74 counted in the plan). Under a literal reading, every rebuild or ingest rejects each legacy ledger loudly, or silently skips it, and "rebuilt from the ledger files" is false for them.
*Fix:* state the intended behavior for a ledger with no block. Is it out of the record, or present as an initiative without units? And is that "not adopted" a different state from "malformed"?

**M2 (major, verified) — The first adopter cannot be made queryable by anything in scope.**
AC-9 requires the `connected-code-graph` ledger's units, status and needs to be queryable. That ledger is a 345-line prose ledger; the `teco` ledger schema (`claude/teco/teco.md:95-100`) has no `Needs` or `Resources` column. Editing `teco`'s definition is out of scope (routed through `cobb`), and nothing says who authors the block into the existing ledger or who derives `needs` and `resources` from the plan's step table for new ledgers.
*Fix:* name the owner of the one-off adoption and of the new per-unit duty. Otherwise AC-9 has no deliverer, and FR-3 depends on an unsequenced `cobb` change. At minimum, say FR-3 is blocked on it.

**M3 (major, verified) — Preference mischaracterizes the `cypher` MCP.**
Preferences say the tool is "read-only except for a few narrow write shapes, so a single pipeline writer fits it better". `cypher-mcp/README.md:137-141` shows all 6 shapes are scoped to `:KaizenEntry`/`:Agent` (the retired `kaizen_team`); every other write, including `Initiative`/`Unit` nodes, is rejected. A pipeline writer cannot use this tool at all.
*Fix:* drop the sentence, or say the writer needs its own write path (a design point). The "named earlier ... is retired" clause also restates history; just say the home is open.

**M4 (major, inferred from the text) — AC/FR coverage gaps and mismatches.**
- FR-3 (the block's fields) has no AC.
- FR-9's "what the initiative is" and "path" have no AC; only the version is touched, in AC-10.
- AC-8 says "two units sharing a resource" with no statement that one is already running. FR-6 flags sharing with a unit *already running*, so as written AC-8 can pass on a pair that are both not-started.
- AC-3 ("discoverable from that check alone") tests no FR-7 obligation (the consult) and cannot fail on whether an agent consulted.
- AC-1 ("without a manual step") is unobservable for the actual "present when `teco` opens" trigger, given FR-4's ingest is the only mechanism.
*Fix:* add an AC per uncovered FR clause, and make AC-8 name one running unit, one ready unit sharing its resource, and one ready unit with an unmet need.

**M5 (major) — Terms the stakeholder, not the architect, must define.**
- "Needs met" in FR-6 and AC-8: which statuses count? `delivered`, `gated` or `accepted`? (`teco` uses `queued · in-flight · delivered · gated · accepted · abandoned · paused`.) Does a need on a unit in a different ledger count? Does `abandoned` or `paused` count as not met?
- "Not started" is not defined (`queued` only?).
- "Running" is not defined (`in-flight` only?).
- "Resources it touches" has no source of truth and no naming rule. "Share a resource" is an exact-string match, which needs a naming convention, or a flag that needs one is meaningless.
- "In flight / coordinated work is done" (FR-5, AC-2) has no criterion. Is it the ledger's `Status:` header, all units terminal, or the stakeholder's say? Archived ledgers and an `accepted` unit set differ. Rebuild-from-files (AC-6) reinstates every ledger, so the derivation rule matters.
- "Loudly" (FR-4, AC-7): to whom, and what is the graph's state afterwards? Prior state kept and flagged stale, or removed?
- Which file version is ingested, working tree or committed? `teco` edits ledgers for long stretches before committing, and FR-9 allows "commit or content hash".

**m1 (minor) — Threshold restated incompletely.**
FR-5 and the 2026-08-23 log entry give "3-units-or-gated". `claude/teco/teco.md:93` says three or more units, any unit with a review gate, or any stop-and-ask escalation. A reactive third-trigger ledger would fall outside the stated threshold. `tico` also opens ledgers for docs-only chains (`teco.md:50`), while FR-5 says "`teco` opens". Cite the owner (`claude/teco/teco.md`) for the threshold instead of restating it.

**m2 (minor) — Growth-stops (root `REVIEW.md`).**
v2 → v3 is 1,995 → 2,146 words. The growth is mostly new scope: FR-3, FR-6, FR-11, ACs 6-10 and the new log entries. Rework residue:
- The FR-11/FR-9/AC-10 "copy carries its version" fact is stated three times (FR-9, FR-11, AC-10), and the "no manual step" fact is repeated in FR-4, FR-5, AC-1 and AC-7.
- The 2026-11-02 log entry 1 carries a rationale ("Reason: keeps git history and review and makes the graph rebuildable"), against size-discipline FR-5 (no rationale; one question, one answer).
- The "Pointer only" entry also adds a "Refines FR-3" clause, and "(former Q1)" labels refer to answered open questions that no longer exist in the document.
- The log's earlier entries are untouched, which is correct: `git diff` shows only additions.
- The reversing entries do name what they overturn (the date plus a quoted answer; acceptable). But "Does a separate small registry survive? → No" reverses v1's registry decisions (2026-08-23 "Detection mechanism" and "Registry scope" entries) without naming them.
- `Tracks: — (M<n> TBD)` leaves a template placeholder.

**m3 (minor) — Altitude.**
This is mostly at requirements level (the Out-of-scope delegation to `architect` is good). Two leaks. FR-2 and FR-4 fix a mechanism ("incremental ingest" as an interface), but this is the stakeholder's stated preference, so acceptable. The Preferences bullet on a "single pipeline writer" and "falkor-chat MCP" is design; keep it non-binding or move it to the plan.

**n1 (nit)** — Open questions say "None for the stakeholder", but B1 and M5 are stakeholder questions. Update once decided.

### Facts verified
- `kaizen_team` retired and deleted: root `AGENTS.md` (`claude/` bullet). ✓
- Ledger path `<component>/docs/plans/<slug>-coordination.md` (also exists at repo-root `docs/plans/`): `teco.md:93`. ✓
- Unit dependencies absent from the ledger schema: `teco.md:95-100`. ✓
- `document-ingestion` exists: `falkor-chat/docs/requirements/document-ingestion.md`. ✓
- Threshold: partly wrong (m1). `cypher` MCP: wrong as stated (M3).

### What's solid
The source-of-truth/derived split (FR-2) is clear, testable (AC-6) and well-reasoned. The out-of-scope list is crisp, including the acknowledged gap on concurrent interviews. FR-8 / AC-4 (escalate, never resolve) is unambiguous. The decision log is append-only with the reversals recorded.

### Open questions for the stakeholder
1. B1: merge, extend or supersede `coordination-graph`?
2. M5: the readiness, running and done definitions, and the resource naming rule.
3. M2: who authors the blocks, and is the `cobb` change a prerequisite?

## Pass 2

**Scope.** `docs/requirements/cross-initiative-coordination.md` Version 4 (`Status: Interviewing`, uncommitted; 1,872 words vs. 2,146 in v3), against `claude/docs/requirements/coordination-graph.md` (FR numbers re-read in full) and its plan `claude/docs/plans/coordination-graph.md`. Static review, no code.

**Verdict: approve with suggestions.** No blocker. Two majors (cross-ledger need resolution against archived ledgers; AC-4 under-specified) and a few minors; none blocks the Ready-for-design flip once folded in. Growth-stops check: passed (shrunk; a fact has one home except the FR-1 path item below).

CPG: not applicable — a requirements-document review with no code-level component.

### Pass 1 dispositions
- **B1** fixed. Header `Extends:` present, and `coordination-graph.md` carries `Extended by:` (git diff of that file: one header line only). Stakeholder answer (a) is logged.
- **M1** fixed. FR-4/FR-7 define a unit without facts (visible, excluded from FR-5, listed with what is missing); the "rejected loudly" and "fully rebuilt" wording is gone. Residue: see N2.
- **M2** fixed. FR-7 names `teco` as the adopter, starting with `connected-code-graph`. The `cobb` dependency is stated in Out of scope, but only for `teco`'s definition (see m1 below).
- **M3** fixed. The Preferences section is gone; the writer and path are left to `architect`.
- **M4** partly fixed. Every FR has an AC (FR-2/3 AC-2, FR-4 AC-3, FR-5 AC-4, FR-6 AC-5, FR-7 AC-6; FR-1 AC-1). AC-4 now names R/A/B/C, but see N1.
- **M5** fixed, with two residuals. Need met = `accepted` (cross-ledger counts), running = `in-flight`/`delivered`/`gated`, "not started" = `queued`, and malformed facts and "done" (leaves on `archived`, `coordination-graph.md` FR-9) all defined. "Resource" is replaced by touched areas by reference to its FR-13/14. Residuals: N3, N4.
- **m1** fixed. FR-2 cites `claude/teco/teco.md` for the ledger-opening conditions instead of restating a threshold.
- **m2** mostly fixed. Rechecked: the "(former Q1/Q2/Q3)" labels and the "Reason: keeps git history…" clause (log lines 129-138) are still in the log. These are uncommitted v3 entries, so they could be trimmed in place; as is they are tolerable. The 2026-11-02 entries' FR/AC refs use v3 numbering, and the new entry states that v4's FR/AC replace v3's. The log is intact against HEAD (no deleted lines). The reversing entries name what they overturn (but see m3). `Tracks:` is now `—`.
- **m3** fixed (design leakage removed; mechanism left to `architect`).
- **n1** fixed. "Open questions: None" is now true, since the three stakeholder questions were answered.

### Stakeholder answers folded since Pass 1 (one line each)
- Merge, extend or supersede? -> extend (`Extends:`; FR-1..7 / AC-1..6 replace v3's).
- Need met? -> unit `accepted`; a unit in another ledger counts (FR-5).
- Running? -> `in-flight`, `delivered`, `gated` (FR-5).

### Findings

**N1 (major, verified by reading AC-4/FR-5) — AC-4 cannot fail on the cases that matter.**
AC-4 (lines 72-74) says "A sharing an area with R", "B with all needs accepted", "C with an unmet need", and expects "lists A and B, flags A against R, does not list C". It does not say A's needs are met (so "lists A" could pass for the wrong reason), and does not say B shares no area with R, so a flag on B passes unnoticed. The two stakeholder-decided rules are untested: no AC has a running unit in each of `delivered` and `gated` (versus `queued`/`paused`), or a need met via a unit in another ledger.
*Fix:* restate AC-4 as a small matrix. R `in-flight`; A needs met and shares R's area; B needs met and shares no area; C has one need not `accepted` (one case each for `delivered`, and for a need in another ledger). Expect A listed and flagged, B listed and unflagged, C absent. Add a variant where R is `delivered`/`gated`, A still flagged, and one where R is `accepted`, A unflagged.

**N2 (major, inferred from `coordination-graph.md` FR-9 and plan line 336) — needs on archived ledgers cannot resolve.**
FR-5 says a need on a unit in another ledger counts, but `coordination-graph.md` FR-9 drops a ledger from the in-flight view once `archived`, and the plan mirrors `:Unit` only for non-archived ledgers. A need on a unit of an archived ledger (accepted, by definition) therefore has nothing to resolve against, and the unit is wrongly unmet or unparseable. A unit of a `unknown`-normalized status (plan: `statusToken` of `unknown`) has the same problem.
*Fix:* state in FR-5 that a need's status is read from the mirror regardless of the view's archive filter (or that a need on a unit that cannot be found is listed as unresolvable, not silently unmet), and say what an `unknown` status counts as. Add it to AC-3/AC-4.

**N3 (minor) — "touched areas" is borrowed from backlog items, with an unstated gap.**
Citations are correct (checked FR-2..5, FR-6, FR-9, FR-13, FR-14 against the text: mirror/freshness, in-flight view, leaving the view, overlap definition, recorded planning facts). But FR-13/14 define areas and overlap for **backlog items** (predicted, `changes`/`relies` fact modes in the plan, lines 478-479, `file:` globs). FR-5's "shares a touched area" for running units should say overlap follows FR-13(b) (two `relies` on one fact do not collide; glob overlap counts), else the flag is an exact-string match. Also say that an explicit "none" is a recorded fact (FR-4 otherwise reads an empty needs list as "missing").

**m1 (minor) — Out of scope names only `teco`'s definition.** FR-2 also requires `tico` and `architect` to change their behavior; those edits need `cobb` too. Generalise: "editing the agent definitions that carry FR-2 and FR-4 duties".

**m2 (minor) — FR-1 and FR-6 partly restate `coordination-graph.md`.** Its AC-5 already requires each in-flight entry to carry its source file path, so only the one-line description in FR-1 is new (and its source is unstated: ledger headers have no such field). FR-6 permits "points to the file" as sufficient, which the path already gives, so AC-5 passes with no new behavior. Either cut the path from FR-1, say where the description is recorded, and give FR-6 a real obligation (e.g. the doc-impact scan list and closing report are addressable as sections), or drop FR-6.

**m3 (minor) — Reversals not named.** The 2026-08-23 "Registration ownership: `teco`, automatically" and "Entry lifecycle: removed on completion" entries are also superseded (file-sync does registration; leaving is `archived`, not completion). The reversing entry names only "Detection mechanism" and "Registry scope". Add them to the next reversing entry (append, do not edit).

**n2 (nit / question)** — `paused` is not "running", so a paused unit releases its areas. Confirm that is intended; it was not asked in the status answer.

### Consistency with `claude/docs/plans/coordination-graph.md` (notes for the architect, not defects here)
- The plan's ledger table parse yields owner/status/deliverable only; `needs` and touched areas require a per-unit block or column. The backlog grammar (`Depends on`, `Touches`, plan lines 478-479) is the obvious pattern to reuse.
- The in-flight view and `:Unit` mirror filter on non-archived ledgers (plan line 336): N2 above forces a lookup path for cross-ledger needs.
- Plan §6 and the `unit` / `inflight` subcommands would gain a ready-now subcommand and a one-line description field on `:Initiative`; both are additive. The plan's Q4 (decision log 2026-10-21) already expects this document to be re-scoped against the plan, so no plan change is forced.
- Everything beyond that (block format, write path, file-reading service and its `security-expert` path-scoping) is already routed to `architect`.

### What's solid
The extension framing is accurate: restatement is now limited to the intro summary. The FR citations to `coordination-graph.md` are correct. Out of scope is crisp, and the FR-3 stop-and-raise rule and FR-4's "never silently dropped" give testable edges. The log is append-only and the new reversals name their targets.
