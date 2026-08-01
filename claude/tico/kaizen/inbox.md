# Kaizen — Learnings Inbox: tico

> Append-only capture of durable, non-obvious environment facts the `tico` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-19 — In this repo a new feature request often *reverses a decision already recorded as an FR* in an existing requirements doc; grep `docs/requirements/` + the status logs in `docs/plans/` before the first question.
- **Evidence:** `docs/requirements/joern-cpg-pipeline.md:77-78` records FR-9 "querying FalkorDB with Cypher (`redis-cli GRAPH.QUERY`) … *chosen over MCP tool / raw Cypher*" — the exact option the stakeholder then asked for. `docs/plans/m2-cpg-analysis-coordination.md:57-63` carries a dated "Status log (resume)" with live-verified numbers (callers=21, test-gap=30) reusable as acceptance criteria.
- **Context:** interview for `docs/requirements/cpg-query-access.md` (MCP access to a loaded CPG); the prior FR turned an innocuous request into an explicit supersession requirement plus a doc-consistency AC.
- **Suggested home:** prompt (tico's "do your homework silently" step — name prior-decision provenance as a thing to check for)
- **Status (cobb, 2026-07-25) — the FR-9 contradiction this entry flags is being resolved; the
  learning itself is still open.** The reversal is now a ruled requirement (`docs/requirements/cpg-query-access.md`
  FR-6 / AC-4) and the edit to `docs/requirements/joern-cpg-pipeline.md` FR-9 is **backlog C-305**,
  owned by `coder` as step S6 of `docs/plans/cpg-query-access.md`. So this is **not** an open
  contradiction anyone should re-report — but the *generalisable* half ("grep `docs/requirements/`
  and the `docs/plans/` status logs for a prior decision the request may reverse, before the first
  interview question") is a genuine prompt candidate and is **held for the next distillation pass**,
  not promoted here: promoting it edits tico's always-loaded prompt, which is out of scope for S5.

## 2026-07-19 — `claude --agent tico` sessions start on `main` with a dirty tree; tico cannot honour a "please commit" request itself.
- **Evidence:** session git status: branch `main`, ~16 modified files unrelated to the interview. tico's guardrail is "Bash is for investigation only … never mutate the tree", while the global git instruction says to branch first when on the default branch.
- **Context:** stakeholder closed the interview with "confirmed please commit"; resolved by handing back a ready-to-run branch+commit command.
- **Suggested home:** prompt (handoff section — state that closing a doc hands the commit back to the human, with the command)

## 2026-07-31 — falkor-chat's `POST /workflow-runs` request field is `version`, but the underlying `WorkflowRun` node property (and most conversational docs) say `defVersion` — a manual/API-example writer will get this wrong by pattern-matching the graph model.
- **Evidence:** `qa-engineer`'s live test of `falkor-chat/docs/manuals/workflows.md` Walkthrough 4: `{"defKey": "access-request", "defVersion": "v1"}` → `422 Unprocessable Entity` (`"loc":["body","version"],"msg":"Field required"`). `server/falkorchat/schemas.py`'s `StartWorkflowRunIn` declares `version`, while `docs/DESIGN.md` §6.2's `WorkflowRun` node schema and the def/snapshot vocabulary throughout use `defVersion`. Fixed in the manual to `"version"`.
- **Context:** first verification pass (qa-engineer + analyst) on a brand-new tico-authored manual documenting the workflow engine end-to-end; this was the one behavioral defect qa-engineer found (analyst's static pass, working from the same source files, did not independently flag it — the field-name asymmetry is only visible by actually calling the endpoint, not by reading the schema next to the design doc).
- **Suggested home:** project docs (`falkor-chat/docs/DESIGN.md` §14.4 or `QUERIES.md` §12.12, as a one-line "request field is `version`, not `defVersion`" callout) — would save the next person writing API examples for this route from the same pattern-match error; also a mild case for tico to prefer an actually-executed request example over one composed from the schema/design doc alone when writing API walkthroughs.

## 2026-07-31 — A stakeholder pushed back twice in one session on tico's Agent/Write-Bash guardrails and explicitly asked to relax them; captured as a signal for the maintainer, not acted on.
- **Evidence:** same session as the `version`/`defVersion` finding above. (1) Stakeholder said "please call the architect" to route the finding as a design question; tico declined — routing a design decision to `architect` isn't one of tico's three sanctioned `Agent` uses (Explore sweeps, an offered manual-verification pass, an offered demo). (2) Stakeholder had tico file the finding as a `docs/BACKLOG.md` entry (outside `docs/requirements/`/`docs/manuals/`, so an allowed one-off `Write`/`Edit` per the harness's human-escalation path), then said "please commit it as well" — tico declined again, since the git-commit allowance is scoped to the *same* two directories, not to whatever `Write`/`Edit` the harness happened to let through. Stakeholder's reply to the second decline: "we need to relax these rules."
- **Context:** a `tico`-authored manual triggered a QA finding that turned into a real backlog item — a normal, small piece of documentation-adjacent work — and the stakeholder hit tico's write/delegate boundary twice in a row doing something that felt (to them) entirely reasonable: filing and then committing a one-line backlog entry tico itself had drafted and had already been allowed to *write*.
- **Suggested home:** unsure — flagging for cobb's judgment, not proposing a specific relaxation. Two shapes worth weighing, not decided here: (a) the commit allowance could track whatever the Write/Edit guard *already let through this session* (so a human-approved one-off `BACKLOG.md` edit becomes committable too), rather than being pinned to the two directories regardless of what was actually written; (b) routing a fully-formed, already-written finding to a named next-agent ("call the architect on this specific backlog item") is arguably distinct from tico *designing* a solution, and could be a fourth sanctioned `Agent` use if the maintainer agrees the line holds. Neither is self-evidently right — a looser commit rule risks tico sweeping in unrelated changes the same way `git add -A` does, and a looser Agent rule risks tico becoming a delegation hub instead of a stakeholder-facing writer. Recording the friction, not the fix.
