# Kaizen — Change History: analyst

> Dated log of actual changes to the `analyst` agent. Most recent first.

## 2026-08-16 — U7 fix round: freshness-check sequencing hardened, `CPG:` line anchored (DEF-1/DEF-2/DEF-3)
- **What:** Two wording tightenings per `docs/plans/cpg-agent-adoption-coordination.md` unit U7,
  following U6's `qa-engineer` live-dispatch acceptance pass
  (`docs/test-reports/cpg-agent-adoption-report.md`). (1) The freshness-check sentence now reads
  "query the freshness check … in that same tool call/step, before deciding whether the result
  needs further cross-verification — this is not a separate, optional judgment call" (previously
  "also run the freshness check … as part of that same step") — closes DEF-2 (`architect`
  reasoned its way past the check with a grep/CPG-agreement substitute that doesn't rule out
  "stale but coincidentally consistent"). (2) The `CPG:` line instruction now reads "written
  verbatim and required in all three cases including when the CPG isn't relevant — not
  paraphrased, not dropped" — closes DEF-1 (`coder`, loose prose instead of the literal line) and
  DEF-3 (`tdd-engineer`, dropped entirely on the not-relevant branch). `analyst` itself was not
  live-tested in U6, but carries the same near-verbatim wiring pattern (per the M4 design's own
  "verbatim-identical wording" precedent, U4b-2), so the fix is applied identically here rather
  than assumed unnecessary — the report explicitly flags that untested agents' compliance
  shouldn't be assumed by extension.
- **Why:** U6's acceptance pass found the M4 wiring (U4b/U4b-2) was correctly worded but didn't
  survive contact with a real dispatched agent's own judgment calls — all three live-tested
  dispatches failed a different way (format, skip, silence). Design intent
  (`docs/plans/cpg-agent-adoption.md` §2.3, §3) unchanged: still agent judgment on staleness
  threshold, still no self-triggered rebuild, still a suggestion not a hard rule about *when*
  something counts as stale — only the sequencing and the anchoring got tightened.
- **Plan items:** none new; closes U7.
- **Same-day addendum (U8 diff-gate follow-up):** `analyst` (this agent, in its U8 diff-gate
  role) reviewed this same U7 wording (`docs/reviews/cpg-agent-adoption.md`, Pass 3 — approve
  with suggestions, zero blockers) and flagged two minors and a nit against the freshness
  sentence in all six files: (a) `frontend-engineer.md` was missing the "tool call/" qualifier
  the other five carried, undercutting the U7 ledger row's and commit message's "identically"
  claim; (b) the trailing "this is not a separate, optional judgment call" had an ambiguous
  pronoun referent — a literal reading could bind "this" to the cross-verification *decision*
  rather than the freshness *query itself*, exactly the room DEF-2's `architect` dispatch used
  to reason past a softer version of this sentence; (c) nit — "query the freshness check"
  mismatched a reference-doc noun with a query verb, when the actual queried object is the
  `:CpgBuildInfo` marker (the report's own recommendation said "marker"). Fixed all three: the
  sentence now reads "…query the freshness marker (per
  `skills/cpg-analysis/references/freshness.md`) in that same tool call/step, before you decide
  whether the CPG's answer needs further cross-verification — running the freshness check itself
  is not optional, and skipping it in favor of a substitute check (e.g. grep agreement) doesn't
  satisfy this." — byte-identical across all six files now. The `CPG:`-line wording from the
  original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — M4 cpg-agent-adoption: discovery wording defaulted, freshness-check bundled, evidence-trail line added
- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§3 (U4b). (1) Frontmatter
  `description` reworded from "With a loaded Joern CPG, uses the `cpg-analysis` skill instead of
  reading files" to "Checks whether a relevant CPG exists as part of its normal orientation and,
  when one does, uses the `cpg-analysis` skill instead of reading files" — conditional →
  default-orientation framing. (2) "How you work" step 2 ("Read the real thing") gained a
  sentence: check whether a relevant CPG exists (first guess `cpg_<component>`, per
  `skills/cpg-analysis/SKILL.md` §1), and when one is found and used, also run the freshness
  check (`skills/cpg-analysis/references/freshness.md`) as part of the same step, noting what it
  says in the findings and surfacing a refresh suggestion — not a silent rebuild — if it looks
  stale. (3) The review skeleton's item 1 ("Scope & verdict") gained the one-line `CPG:`
  evidence-trail convention (`CPG: used <graph> — <clause>` / `CPG: considered, not relevant —
  <clause>` / `CPG: not applicable — <clause>`).
- **Why:** M4 (`cpg-agent-adoption`) widens CPG discovery from a conditional check to a default
  orientation step across the three already-wired consumers (`analyst`, `architect`,
  `qa-engineer`), bundles the freshness recipe into that same step (FR-6's surfacing half), and
  adds a spot-checkable `CPG:` evidence trail (AC-2). Per `docs/plans/cpg-agent-adoption.md`
  §2.1-2.3, §3, §6 step 2.
- **Plan items:** none.

## 2026-08-11 — Inbox distillation: 3 entries — 1 prompt addition, 2 to `python-web-quirks`/project docs

- **What:** `cobb` processed all 3 entries in `analyst/kaizen/inbox.md` (§5).
- **Promoted:**
  - The "held-note staleness" finding (a sibling document's "held pending X" claim can go stale
    when X lands via a *different* agent's files) → new clause on the Guardrails "Evidence over
    vibes" bullet: cross-check sibling kaizen history before trusting a holding document's own
    claim about a pending follow-up.
  - `urllib` timeout taxonomy → `skills/python-web-quirks/SKILL.md`.
  - LM Studio `/v1` 200-envelope quirk — merged with `architect`'s duplicate finding of the same
    fact; the falkor-chat-specific half was **already** fully documented in
    `falkor-chat/docs/DESIGN.md` §14.8 ("The `/v1` normalization rule"), so only the general,
    reusable half went to `python-web-quirks.md`.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/analyst/{analyst.md,kaizen/{history,inbox}.md}` ·
  `skills/python-web-quirks/SKILL.md`.

## 2026-08-09 — Independent safety recheck cleared; `review-techniques.md` marker removed
- **What:** Removed the "⚠️ Pending an independent analyst safety recheck before first use"
  callout from technique (b) (scratch-copy + reverse-patch) in `claude/analyst/review-techniques.md`.
  A separate, narrowly-scoped analyst session ran the independent safety recheck and returned a
  clean verdict — no embedded-instruction/manipulation concern, scope-limiting language intact.
  Also folded in the recheck's one optional suggestion: appended "the block was doing its job;
  this substitute earns the exception on its own zero-touch merits" to the existing "not a general
  license" paragraph, affirming the classifier's original block was itself legitimate scrutiny.
- **Why:** The marker existed because this exact technique had triggered an instruction-poisoning
  flag on 2026-07-31 (see that entry below) before being reframed and promoted into this file on
  2026-08-09. The recheck was the condition for treating it as a routine technique rather than
  "informational only." Requested by `teco`, relaying two independent reviews'
  (`docs/reviews/{kaizen-inbox-distillation,analyst-inbox-distillation}.md`) findings plus the
  separate recheck's result.
- **Plan items:** none.

## 2026-08-09 — Held entry 28 promoted: consolidated Kiro-facts edit landed
`cobb` closed out the "(H) Held, not cleared — entry 28" note from the same-day distillation
entry below: the race window against `architect`'s two held entries (both targeting the same
file) is over, so the `kiro-cli agent create` default `"resources": []"` fact was re-verified
(now against `kiro-cli 2.16.2`, up from `2.14.1` — held) and written into
`skills/agent-standards/kiro.md`'s CLI custom-agents `resources` config-key bullet. `inbox.md`
entry 28 cleared; `inbox.md` is back to the standard empty placeholder.

## 2026-08-09 — Full inbox distillation pass: 30 of 31 entries processed (entry 28 held for a coordinated follow-up)

`cobb` ran the agent-maintenance skill §5 distillation over the full inbox (31 entries spanning
2026-07-19 → 2026-08-08), preceded by a read-only proposal pass that verified each entry against
current repo state, then a second re-verification immediately before applying edits (repo state
was unchanged — `git status` clean both times, aside from an unrelated concurrent edit to
`claude/tdd-engineer/tdd-engineer.md`'s step 5 discovered mid-pass, which this pass's own
description-clause edit landed cleanly alongside). Grouped by disposition; every entry not listed
under "held" is now cleared from `inbox.md`.

**(A) Discard — already resolved/stale, cleared with no promotion (entries 1, 12, 15, 16, 18, 20,
25).** Re-verification found each condition no longer holds: entry 1 (falkor-chat pytest
self-skip) is already documented at `falkor-chat/docs/DESIGN.md` §14.7; entry 12
(`audit-team.sh` failing check 7) — a live run now shows full `PASS`, including check 7, the five
cited leaks having been cleaned up since; entry 15 (`pipeline.sh --reset` bypassing the
destructive-ops guard) — `claude/scripts/guard-destructive-ops.sh` now carries a dedicated C-311
branch (dated 2026-08-08) matching exactly what the entry asked for; entry 16 (Claude Code MCP
25k-token output cap) and entry 18 (`CLAUDE_PROJECT_DIR` expansion) are both already in
`skills/agent-standards/claude-code.md`; entry 20 (MCP startup-timeout doc disagreement) is
already resolved there too (`MCP_TIMEOUT` vs `MCP_TOOL_TIMEOUT` disambiguated); entry 25 (stale
"the joern agent's job" error text in `cpg/mcp/server.py`) — the live string now reads "the
graph-dba agent's job", no trace of "joern agent" left in the file.

**(B) Promoted to project docs — `falkor-chat/AGENTS.md` + `docs/DESIGN.md` (entries 4, 5, 6, 7,
10).** Consolidated into one new "Probing shared graph state without mutating it" subsection in
`AGENTS.md` (entries 5 + 10: the `publish_def`/`materialize_snapshot` graph-seam asymmetry, and
`test_services.py` as the review-safe pytest subset) plus a note on the `bootstrap_schema.sh`
Key-scripts row (entry 6, and entry 7's misfiled second "Suggested home" line, which was about
`bootstrap_schema.sh` despite being appended under the line-number-invariance entry — routed to
its actual topic here). Entry 4 (`pytest --collect-only -q` as the non-mutating way to check a
claimed test count) went to `docs/DESIGN.md` §14.7, next to the existing pytest-hazard bullets.
**Line numbers re-verified and corrected**, not copied from the inbox — the original entries cited
`repository.py:132-134`/`:937`/`:1483` etc.; current `HEAD` has the same functions at
`:156-158`/`:992`/`:1669` (drift from other commits landing between 2026-07-24 and now). Entry
7's own topic — the line-number-invariance re-gate technique — went to (F) below, not here.
Also logged in `falkor-chat/docs/HISTORY.md` (2026-08-09).

**(C) Promoted to knowledge base — `claude/graph-dba/falkordb-quirks.md` (entries 14, 17, 27).**
Entries 14 + 17 bundled into one "Ops, config & tooling" bullet (`GRAPH.PROFILE` executes writes
for real despite suppressing `RETURN` output; neither `RO_QUERY` nor an `EXPLAIN`/`PROFILE`
prefix — even after a Cypher comment — is honored as a planning directive under either query
command). Entry 27 (`sum(CASE...)` returns float `0.0`, never `NULL`, on zero-row aggregation,
and stays `float` not `int` on non-empty input) added under "Cypher dialect & query behavior",
next to the existing aggregation-pitfalls bullets. Edited directly per the established
maintainer-edits-another-agent's-knowledge-base-file channel (precedent: 2026-07-31 entry below);
no `graph-dba`-side kaizen entry needed.

**(D) Promoted to knowledge base — `skills/agent-standards/claude-code.md` (entries 13, 19, 21,
31).** Entry 13 (FastMCP `structured_output=False` — otherwise a `str`-returning tool ships its
payload twice via a spurious `outputSchema`) added to § Output limits. Entry 19 (a containerized
stdio MCP server's own labelled container is legitimately `Up` for the whole session that's
probing it, so a "docker ps --filter label=… must be empty" orphan-check is unsatisfiable from
inside an open session) added to § Lifecycle, framed as a liveness-aware-check rule rather than a
bare reviewer habit. Entries 21 + 31 (this environment's Bash tool shell-shadows `find`→`bfs` and
`grep`→`ugrep` via wrapper functions with a spoofed `ARGV0`, not inherited by a spawned
subprocess) merged into one new § Bash tool environment section, since they're the same
phenomenon discovered on two different dates. Per `skills/README.md`'s Maintenance section
("changes to `agent-maintenance`/`agent-standards` are logged in `claude/cobb/kaizen/history.md`"
— see that file, 2026-08-09).

**(E) New skill — `skills/python-web-quirks/SKILL.md` (entries 2, 11, 29).** Stakeholder decision:
Python/web-framework stack knowledge belongs in a skill consulted by the relevant personas, not
duplicated into one project's docs. Entries 2 + 29 merged into one background-task-GC/threading
note (`asyncio.create_task` fire-and-forget GC-safety warned-but-not-reproduced-under-stress,
paired with Starlette/FastAPI `BackgroundTasks`' bounded-threadpool concurrency vs. an unbounded
raw `threading.Thread` — both are about async-dispatch mechanics an implementer might get wrong
in the same code path). Entry 11 (FastAPI/pydantic `response_model_exclude_unset` silently
dropping defaulted fields on **nested**, not just top-level, models) as its own section. Wired via
a routing clause in `coder`, `tdd-engineer`, `architect`, and this agent's own frontmatter
`description` (mirroring how `cpg-analysis` is wired into `analyst`/`architect`). Registered in
`skills/README.md` and root `AGENTS.md`'s `skills/` bullet. No dedicated `kaizen/` for the new
skill — no existing skill in this repo actually carries one despite the agent-maintenance skill's
general rule (`agent-standards`/`agent-maintenance` are logged in `cobb`'s kaizen per
`skills/README.md`; `joern-cpg`/`cpg-analysis` changes are logged in `graph-dba`'s kaizen instead,
per that file) — followed the established precedent (log in the creating/maintaining agent's own
kaizen, here `cobb`'s) over the written-but-unpracticed rule; logged in `claude/cobb/kaizen/history.md`
(2026-08-09).

**(F) New on-demand file — `claude/analyst/review-techniques.md` (entries 3, 7, 8, 26).**
Stakeholder decision: specialized review techniques go on-demand (mirrors
`graph-dba/falkordb-quirks.md`), not always-loaded prompt body. Holds: the AST line-range
byte-identity hash technique (3), the line-number-invariance re-gate technique (7's actual
topic), the stub-package HEAD-vs-working-tree import technique (8), and the scratch-copy +
reverse-patch technique (26) — written in using the **already-reframed** text that survived the
2026-07-31 security review (see that entry below), not any earlier draft, and carrying an
explicit "⚠️ Pending an independent analyst safety recheck before first use" marker per the
stakeholder's instruction; a separate narrowly-scoped analyst session is checking it. `analyst.md`
gained a one-line pointer to this file (in "How you work" step 3) rather than inlining the
content.

**(G) Core prompt — `analyst.md` (entries 9, 22, 23, 24, 30).** Entry 9 (a deliverable already
sitting at the target path when a run starts — e.g. resuming after an interruption — may have
executed/side-effecting claims narrated in past tense before the command actually ran; re-verify
against the live system before inheriting them) added as its **own** Guardrails bullet — stakeholder
judged it high-severity (a resumed analyst could hand `teco` false confidence about live system
state), not folded into an existing one. Entries 22 (a `git grep`/`git ls-files` count is a bound,
not a fact, when the artifact under review or a sibling deliverable is itself untracked), 23 (a
suggested regex/glob/pattern fix is a claim, run it before writing it into a review — the specific
extglob bug that motivated this is already fully documented in
`docs/plans/doc-reference-convention.md`, no further action needed there), 24 (cross-check a named
agent's `PreToolUse` guard globs when a plan assigns it doc-write ownership), and 30 (`shellcheck`
isn't installed in this environment — `bash -n` + live execution is the substitute) folded as
**clause-level extensions to the existing "Evidence over vibes" guardrail sentence**, per
stakeholder instruction to avoid four new standalone bullets.

**(H) Held, not cleared — entry 28** (`kiro-cli agent create` default `resources: []`).
Disposition decided (promote to `skills/agent-standards/kiro.md`) but left in `inbox.md` with a
one-line "queued for consolidated follow-up" note: `teco` is coordinating a combined edit to that
shared file alongside two related facts from `architect`'s inbox, to avoid two sessions racing on
the same file.

**Inbox-authoring defects found and corrected while applying:** entry 6 had no "Suggested home"
line of its own (its dispositioning text had been appended, in error, under entry 7 — see (B));
entry 1 likewise had no "Suggested home" line (moot — turned out to already be stale). No entries
were found to conflict with each other.

## 2026-07-31 — Inbox entry reframed after an "Instruction Poisoning" flag; classifier-gap fact partially distilled
- **What:** A security check flagged the 2026-07-31 inbox entry ("Auto mode's Bash classifier blocks `git stash`…") as instruction-poisoning-shaped: a persistent, forward-looking "here's how to route around a safety classifier block" write-up, regardless of how benign the originating use was. `teco` (no write access to this inbox, no adjudication authority) routed the triage to `cobb`. Verdict: **reframe needed, not a false positive and not a full policy violation** — the underlying technique (scratch-copy + reverse-patch, zero working-tree touch) is sound and consistent with this inbox's established isolation discipline (the 2026-07-24 stub-package and review-safe-pytest-subset entries), but the entry's *framing* ("here's the workaround now that git stash is off-limits") taught evasion-shaped reasoning rather than the safety property that actually makes the substitute acceptable. Contrast: the 2026-07-25 `pipeline.sh --reset` entry is safe precisely because it reports a gap in a guard **this repo owns** (`claude/scripts/guard-destructive-ops.sh`) for that guard's maintainer to close — it never tells an agent to use the gap. The 2026-07-31 entry, by naming a *product-level* auto-mode classifier (not a repo hook) and framing the substitute as "the answer to being blocked," was the wrong shape even though the action taken was benign and verified harmless (`git status` clean before/after, independently spot-checked by `teco`).
  - Entry rewritten in place (`kaizen/inbox.md`): kept the technique and its evidence, added an explicit scope note disclaiming the "route around any classifier block" generalization, and stated plainly that the classifier itself is not a repo mechanism there's anything here to harden.
  - The classifier-gap fact (no reversible/read-only-verification carve-out) partially distilled: routed to `skills/agent-standards/claude-code.md` §Hooks as an observed (not doc-verified) harness quirk, since it's a Claude-Code-product fact of the kind that knowledge base is for, not project-specific to this repo.
  - **Not done in this pass:** full promotion of the technique itself into this prompt, and the broader backlog of other still-unprocessed "suggested home: prompt" entries in this inbox (stub-package HEAD-vs-working-tree import, review-safe pytest subset, isolatable snapshot side, byte-identity AST hash, line-number-invariance re-gate technique, etc.). This was a narrow security triage, not a full §5 distillation pass — see `cobb/kaizen/plan.md` for the follow-up item.
- **Why:** `cobb` owns inbox distillation (agent-maintenance skill §5) and, per the same skill, edits another agent's kaizen files directly as its normal channel — no hook restricts this (the `guard-review-doc-writes.sh` PreToolUse hook is wired in `analyst`'s own frontmatter and fires only during `analyst`'s own tool calls, not another agent's). `teco`'s message raised the possibility that `cobb` might lack write access here and asked it to say so plainly rather than route around its own guard if that were true; `cobb` verified empirically (an actual Edit call, which succeeded with no hook interception) rather than accepting or rejecting the claim unverified, and is recording that check here since it bears directly on the pass's own subject matter.
- **Plan items:** none pre-existing; see `cobb/kaizen/plan.md` for the new follow-up.

## 2026-07-29 — New review target: a tico-authored user manual's factual/architectural claims
- **What:** `tico` gained a new doc kind, user manuals (`<component>/docs/manuals/<slug>.md`), and the team certification pass flagged that manuals were the only doc kind with no independent-review gate. User decision: split the review — `qa-engineer` verifies the walkthroughs by driving the running app (behavioral claims), `analyst` checks everything else. Added a fourth reviewed-artifact category ("What you review") between source code and RCA: a manual's factual/architectural claims against the real code/config (same grounding discipline as a plan review), plus clarity for a non-technical end-user audience specifically — explicitly *not* the walkthroughs, which stay `qa-engineer`'s to avoid duplicating that check. Frontmatter `description` updated to name the new target and its qa-engineer/analyst split.
- **Why:** user ruling following the 2026-07-29 team certification's open observation (logged in `cobb/kaizen/plan.md`, now resolved). Routed through `teco`'s existing "independent review" default (its own kaizen carries the matching entry) rather than analyst self-selecting when to review a manual.
- **Plan items:** none — no prior plan item covered this; not adding one since it's already implemented.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — `-impl` review role documented; header block required on review docs (step 1 of `docs/plans/doc-reference-convention.md`)
- **What:** Two body edits, no frontmatter change. (1) The deliverable paragraph now names the `-impl` role: a review of an **implementation** is `<component>/docs/reviews/<slug>-impl.md`, the bare slug being the review of the **plan**. (2) One line added between the review skeleton and the RCA skeleton: *"Open the document with the header block from root `AGENTS.md`."*
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.4 found `-impl` **used 4× and documented nowhere** — the only member of the closed role set (`(none)` · `-coordination` · `-ml` · `-graph` · `-rca` · `-impl` · `-report`) missing from the prompt that produces it, and the absence had already broken a document family. The header line is the canonical M9 sentence, byte-identical across the prompts that get it, and is a pointer rather than an inlined template because root `AGENTS.md` reaches every agent through the root `CLAUDE.md` `@AGENTS.md` import. `claude/README.md` row 17 re-checked — it cites the review write paths, not the naming rule, so no catalog edit was needed.
- **Plan items:** none. (K-001's remaining RCA half is untouched; `-rca` was already documented.)

## 2026-07-25 — `tools:` allowlist gains `mcp__cpg__query` (M3 / C-304)
- **What:** Frontmatter `tools:` now ends `…, Agent, mcp__cpg__query`. `claude/README.md` row 17 updated to say the `cpg-analysis` skill reaches the graph through that MCP tool and why the allowlist entry is required. No body or `description` change — the CPG routing clause added on 2026-07-19 stays accurate, and the skill is progressively disclosed.
- **Why:** M3 replaces the CPG read path with a single MCP tool, `mcp__cpg__query(graph, cypher)` (`docs/plans/cpg-query-access.md` S5). **`tools:` is an allowlist, not a hint** — an agent that declares one does not see MCP tools absent from it, so without this line the feature would have been silently inert for `analyst` (and `architect`); `qa-engineer` and `graph-dba` declare no allowlist and inherit it. `redis-cli GRAPH.QUERY` remains the documented fallback and is the only path under OpenCode/Kiro.
- **Verification note:** this is the *edit*; the live proof (a cold `analyst` actually calling the tool) needs the server wired in S3 and is verified in S9, per the plan's m-4 split.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 707 → 469 chars (-33%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (analyst↔qa-engineer, analyst↔data-scientist) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `analyst`'s `guard-review-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed review-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-19 — CPG capability wired into the routing description (M2 / C-207)
- **What:** Frontmatter `description` gained one clause: when a Joern CPG is loaded in FalkorDB, the analyst uses the `cpg-analysis` skill (graph-dba-owned) for impact-analysis, RCA data-flow, and code-review taint queries instead of reading files. `claude/README.md` catalog entry updated to match. No body change — the skill is progressively disclosed and self-describes; the description clause is the routing signal.
- **Why:** M2 delivered the `cpg-analysis` skill (`analyst` is a named consumer for impact/RCA/code-review recipes per FR-10/11/12). C-207 makes the consumer agents' routing contract advertise the capability. cobb wired it as part of Gate-2b (skill also passed the standards vet).
- **Plan items:** none.

## 2026-07-12 — K-001: code-review half of the shakedown proven (K-022 impl review) — RCA remains
- **What:** The **code-review** half of the first-run shakedown ran for real. On falkor-chat
  **K-022 Landing 1** (executor implementation, committed `3921f87`) the analyst reviewed the
  delivered diff and produced `falkor-chat/docs/reviews/m3-executor-impl.md`: verdict
  **approve-with-suggestions, 0 blockers / 1 major (M-1) / 3 minor / 3 nit**, doc landed at the
  right path with the write-guard hook silent. This was the designated vehicle named in K-001 and
  the counterpart to teco K-003 (the team's first fully-gated run). Verdict calibration read as
  healthy — a real major surfaced, not a nitpick flood, and the two deferred seams were ruled
  acceptable-for-Landing-1 rather than inflated to blockers.
- **Why:** teco K-003 closed 2026-07-12 with the gated run committed; that same run is the
  evidence for analyst K-001's code-review half. Recording it here so the shakedown's remaining
  scope is honest.
- **Plan items:** **K-001 narrowed** (not closed): plan-review ✅ (2026-07-11) + code-review ✅
  (this entry); the **RCA** mode is still unexercised — K-001 now tracks that remainder only.
  No prompt change: no verdict-calibration weakness surfaced across the two review runs.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1449 to 525 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-review-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/reviews/*|*/docs/reviews/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/analyst/hooks/guard-review-doc-writes.sh`) to `$HOME/.claude/agents/analyst/hooks/guard-review-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/analyst` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — data-scientist route-away clause (boundary symmetry)
- **What:** Frontmatter `description` and the findings-routing guardrail now route the AI/ML/data-science **methodology** dimension of a plan or change — model/embedding choice, evaluation design, metric validity, statistical claims — to the new `data-scientist` agent, whose methodology review (`docs/reviews/<slug>-ml.md`, same verdict scale) complements the analyst's general static review. Pair `analyst:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09 to work alongside the analyst at review time; "review this ML-heavy change" plausibly matched both, so the boundary must live in both descriptions.
- **Plan items:** none.

## 2026-07-09 — Description gained the qa-engineer route-away clause (boundary symmetry)
- **What:** Frontmatter `description` now states the verification boundary explicitly: analyst judges statically — reading, reasoning, and running what already exists — and planning/executing *new* black-box/acceptance testing of the running system routes to `qa-engineer`. The prompt body already carried this split (findings-routing guardrail); the description — the routing contract every router sees — didn't. Counterpart clause added to qa-engineer in the same change; the pair is now mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): analyst↔qa-engineer was asymmetric at the description level (analyst never named qa-engineer), leaving "test this" work plausibly matching both.
- **Plan items:** none.

## 2026-07-09 — Added root cause analysis (RCA) mode
- **What:** Extended the reviewer into a reviewer-and-diagnostician: a third artifact class ("Defects and failures — RCA") with its own method (reproduce when possible, trace the actual code path, read git history; distinguish root cause vs trigger vs contributing factors; five-whys stops at the deepest cause actionable in the codebase; record ruled-out hypotheses) and its own deliverable skeleton at `docs/reviews/<slug>-rca.md` (symptom & impact → reproduction/evidence → causal chain → root cause with confirmed/inferred confidence → suggested fix + prevention). Frontmatter description updated; guardrail clarified (diagnoses only — the fix routes to the implementer, typically `tdd-engineer` with a reproduction test first, briefed by the RCA path). No hook change needed (`docs/reviews/` already covers the RCA doc). Rosters/catalogs synced (teco, claude/AGENTS.md, claude/README.md, root AGENTS.md).
- **Why:** User: "analyst is also good with RCA" — the team had no owner for cause-unknown defects; tdd-engineer starts from a known bug, qa-engineer finds and reports defects, but nobody's job was tracing a symptom to its root cause.
- **Plan items:** none (K-001 shakedown should now cover an RCA run too).

## 2026-07-09 — Created
- **What:** Initial version of the `analyst` subagent — a systematic, experienced developer acting as a pure reviewer: reviews architect plans (grounding, completeness, soundness, proportionality, test strategy) and source code (correctness → tests → fit → clarity → security/perf, in priority order), plus plan↔code conformance when given both. Deliverable is a severity-ranked (blocker/major/minor/nit), evidence-backed review with a verdict (approve / approve with suggestions / needs changes), written to `<component>/docs/reviews/<slug>.md` by default and handed off by path. Review-only contract is harness-enforced: `hooks/guard-review-doc-writes.sh` (PreToolUse, matcher `Write|Edit`, same pattern as architect's guard) escalates any Write/Edit outside `docs/reviews/` (or `/tmp`) to the human. Subagent-aware (questions/blockers return as the deliverable). Model `opus`, tools `Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent` — mirrors architect. Deployed via `~/.claude/agents/analyst` symlink.
- **Why:** The team had no review gate between handoffs — architect plans went straight to implementation and implementer code straight to QA, with nobody judging design soundness or code quality statically. User requested a systematic reviewer covering both plans and source code.
- **Plan items:** — (K-001, K-002 seeded)
