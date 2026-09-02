# Kaizen — Change History: analyst

> Dated log of actual changes to the `analyst` agent. Most recent first.


## 2026-09-02 — one-token example refresh in the placeholder-vs-expanded-key trap (U6 of `salesperson-ui`)

- **What:** `analyst.md` `:84`'s illustration of the pasted-doc-template-placeholder trap now cites
  `cpg_falkorchat` instead of `cpg_salesperson` as the repo's expanded graph key. The lesson and
  wording are otherwise untouched.
- **Why:** `cpg_salesperson` is a graph of the retired Streamlit app under a name the incoming
  `salesperson/` component will make misleading, and its fate (rename or drop) is undecided —
  `cpg_falkorchat` is unambiguously live and serves the example identically. Done by `cobb` in the
  cross-agent sweep of `salesperson-ui` unit U6.

## 2026-08-25 — `kaizen_team` distillation (unit U1 of the team-wide pass): 20 pending entries processed

`cobb` ran the agent-maintenance skill §5 procedure against every `analyst`-linked `kaizen_team`
node — 12 legacy (`author:'analyst'`, dated 2026-08-21/22) plus 8 current-shape
(`(:Agent{agentId:'analyst'})-[:PRODUCED]->`, dated 2026-08-22 through 2026-08-25; zero `MENTIONS`
edges found). Full untruncated text fetched via `redis-cli --no-raw` (the MCP tool's per-cell
truncation would have lost most of these at ~300 chars). Every surviving claim was re-derived
against the live repo/system, not trusted from the entry's own framing — grep, live FalkorDB
probes on a disposable graph, and direct reads of the current source. No entry was found to be
about a different agent; no `MENTIONS` tag added. Zero kept-open — every entry resolved to a clean
promotion or a discard this pass.

**Promoted to `review-techniques.md`** (7 entries, consolidated into 6 new sections plus one new
sub-technique (d) on the existing uncommitted-diff section, to avoid near-duplicate entries):
  - `b4f8e21a…` — mutation-testing via in-process module substitution under the real dotted name
    (new sub-technique (d), sibling to existing (a)–(c)).
  - `c7d21f4a…` + `e91a4d7c…` — merged into "Re-gating a state-machine guard/invariant fix": two
    checks (foreclosed-pattern check, call-site tracing) a "does the mechanism work" read misses.
    The falkor-chat-specific instance (K-028 v2's unconditional-fallback defect) is fully resolved
    and already documented in the now-archived `workflow-timers.md` v3 itself (lines 1015–1024) —
    only the generalized review lesson was promoted, not a project-docs claim about current system
    behavior (v3 replaced the unconditional fallback with a conditional one; the described bug no
    longer reproduces).
  - `b6f3a1d2…` + `e3a1f6b2…` — merged into "A 'this already exists' claim is a grep away from
    confirmation." Both source facts are already fully resolved/documented in their own review
    docs (`docs/reviews/mid-run-escalation.md` §"grounded in..."; `document-ingestion.md` Pass-2)
    — only the generalized technique was promoted.
  - `9f3d2b1a…` — "An untracked plan/review doc has no re-verification baseline," verbatim
    disposition per its own `suggestedHome: review-techniques` tag.
  - `9e2a9f2e…` — "A truncate → append → truncate-again pipeline can silently discard its own
    repair pass." The falkor-chat instance (extraction.py) is already fixed and documented in the
    code's own comments, citing this exact review pass by name (Pass 3 MAJOR 1) — only the general
    lesson was promoted.
  - `8b1e4f2a…` + `2f3a9b1c…` — merged into "Two checks for a multi-shape authorization/
    security-gate function" (keyword-set completeness; early-match short-circuit smuggling). Both
    findings are already fixed in the live `cypher-mcp/server.py` (`_FOREIGN_TRIGGER_RE` now covers
    all four keywords; `authorize_write()` now calls `_has_foreign_trigger_outside_strings()` after
    an author-claim match) — verified by reading the current source, not the entries' own claims.
    Only the generalized technique was promoted, project-docs facts already fully covered by the
    (already-processed) `docs/reviews/kaizen-agent-ontology.md` review doc itself.

**Promoted to `skills/agent-standards/claude-code.md`** (2 entries):
  - `b1e6f3a2…` — the protected-path carve-out is keyed on the literal dot-prefixed directory name
    (`.claude`, `.git`, etc. — the exact list re-verified live, 2026-08-25, against
    `code.claude.com/docs/en/permission-modes` § Protected paths), not on any project's own
    conventionally-named directory (this repo's dotless `claude/`) — added alongside the existing
    self-modification/classifier callout in § Hooks.
  - `b6f1e6b0…` — FastMCP's `Tool.run` wraps any tool-function exception in a generic `except
    Exception`/`ToolError`, so a per-item try/except inside a bulk tool must catch every failure
    mode, not just its own domain exception — added to § MCP Output limits. The falkor-chat
    instance (`Services.ingest_documents`) is already fixed and documented in the code's own
    docstring; only the general FastMCP fact was promoted here.

**Promoted to `skills/python-web-quirks/SKILL.md`** (1 entry):
  - `e6f2b1a4…` — pydantic `Field(min_length=1)` doesn't reject whitespace-only strings (`" "` has
    len 1); MCP callers with no schema layer are completely unguarded, not just under-guarded. The
    falkor-chat instance (`services.ingest_document`) is already fixed (`text.strip()` check,
    documented in the function's own docstring) — only the general pydantic fact was promoted.
    Skill frontmatter `description` updated to name the new gotcha.

**Promoted to `claude/graph-dba/falkordb-quirks.md`** (2 entries, edited directly per the
established maintainer-edits-another-agent's-knowledge-base-file channel):
  - `a1e6c2f8…` — `db.labels()`/`db.relationshipTypes()` asymmetry for zero-data schema elements.
    Re-verified empirically on a fresh disposable graph (2026-08-25), independent of falkor-chat's
    current data state — the original entry's `ws:acme` evidence is now stale (K-050 has since
    shipped real `HAS_CHUNK`/`ABOUT`/`RELATES_TO`/`SAME_AS` edges on that workspace), but the
    underlying FalkorDB engine behavior it described is a durable, build-specific fact, re-proven
    from scratch.
  - `e3b6f2a4…` — `db.idx.fulltext.queryNodes()` against a label with no fulltext index created at
    all silently returns zero rows, no error. Re-verified empirically (2026-08-25, disposable
    graph).

**Promoted to `falkor-chat/docs/SERVER.md` §1.7 (Testing hazards)** (1 entry):
  - `a3f5e8c2…` — `Services`/`WorkflowExecutor` each default to their own separately-defined
    `_default_clock`, unwired even in production `app.py`; a test injecting `Services(clock=...)`
    alone doesn't control `StepRun.startedAt`. Re-verified live: still true today (no wiring exists
    in current `app.py`).

**Promoted to `claude/AGENTS.md`** (1 entry):
  - `a3f1c9e2…` — `git add` then `git commit` is not atomic against a concurrent process sharing
    the working tree; a staged file can get swept into an unrelated commit during the window before
    the commit runs, even when the two processes' own files never overlap (the race is on the
    index, not on any one path). Added to the Git-commit-authority section as a check-before-commit
    discipline (`git status`/`git diff --cached --name-only` immediately before `git commit`) —
    this is a fact about the whole team's multi-agent git workflow, not analyst-specific, so it
    landed in the shared context file rather than this agent's own prompt/knowledge base.

**Discarded — already resolved/fixed and already documented, no promotable residue** (4 entries):
  - `a3f1c2e4-6b8d…` (`CYPHER` preamble literal-binding syntax) — the underlying mechanism is
    already documented in `falkordb-quirks.md` (the existing "CYPHER preamble needs Cypher
    literals" bullet); the specific K-049 repro it describes is already fully written up at
    `falkor-chat/docs/reviews/unique-constraint-oversized-value-crash-rca.md`.
  - `a3f1c2e4-7b6d…` (sequential-mutation patch/restore/sha256sum technique) — no material
    information beyond `review-techniques.md`'s existing (b)/(c) mutate-restore-verify-hash
    discipline; same zero-touch pattern, different file-tracking status than either sub-case
    already covers.
  - `b3e2c1d4…` (falkor-chat background work is fire-and-forget) — already fully documented
    (`falkor-chat/docs/HISTORY.md`, `QUERIES.md`) and the specific race it was used to flag is
    already fixed at the database layer, per the resulting plan-gate decision in
    `document-ingestion.md` (§"Concurrency note").
  - `b7e1c9a4…` (subagent `permissionMode` inheritance — the two named exceptions are exhaustive)
    — already present, more completely, in `skills/agent-standards/claude-code.md`'s own
    2026-08-24 "parent-session mode inheritance" resolution callout (§ Hooks), which the entry's
    own review target (`permission-default-mode.md`) fed into; nothing this entry adds isn't
    already there.

**Verified:** `bash claude/scripts/audit-team.sh` — PASS, all checks including the personal-
identifier sweep; git-commit-authority check unaffected by the `claude/AGENTS.md` addition (not an
agent-file edit, not scanned by that check). All 20 graph entries cleared (12 legacy
`DETACH DELETE`; 8 current-shape resolved via the `PRODUCED`-edge-delete path, all with
`otherRemaining == 0` since none carried a `MENTIONS` edge — full-node clear in every case).
- **Plan items:** none opened — every surviving entry either promoted cleanly or was already
  resolved elsewhere.

## 2026-08-25 — Output discipline: the finding budget (prompt-waste plan, Stage D — an *addition*)

- **What:** 2,325 → 2,375 w (**+50**). Stage D is the only stage of
  `claude/docs/plans/prompt-waste-reduction.md` that adds; the unit's joint budget with
  `architect.md` is ≤~120 net words and it landed at +92. Two additions here:
  - **The finding budget** (deliverable item 2, appended after the "This is fragile" example):
    *"Keep a finding's inline body to **≤~15 lines**; a longer trace, table, or log excerpt goes to
    a trailing `## Appendix` section, cited from the finding. Never drop evidence to fit the cap —
    it bounds the body, not the review's rigor."*
  - **The `-impl` prohibition**, folded into the *existing* sentence that already named the suffix
    rather than added as a new one (+6 w): "…the bare slug is the review of the **plan**, **and
    implementation findings never grow the plan review**." The naming convention was already
    stated; only the prohibition was missing.
- **Why:** The cap's motivating incident is a **2,455-line review file**. The prohibition and the
  cap attack it from two sides — one bounds a single finding, the other stops one file absorbing a
  second review. The never-drop guard is deliberately welded to the cap in the same breath, because
  a budget stated alone invites meeting it by deleting evidence, which is the one failure this
  agent must never have.
- **Gate (a) — rule inventory, addition-shaped.** Nothing removed; every existing class-1/2 clause
  checked for contradiction by the additions. Three pairs cleared: the cap vs. item 2's *"specific
  enough that the owner can act without re-deriving your analysis"* (15 lines is ample for
  evidence+why+fix, and the appendix escape plus the never-drop guard close the gap); the cap vs.
  step 4's *"Prune ruthlessly"* (**different objects** — step 4 caps the *number* of findings, the
  new rule caps *one finding's body*); and the cap vs. the **RCA skeleton**, which it must not
  reach — an RCA's "Causal chain" and "Reproduction & evidence" legitimately carry long traces.
  Placing the cap inside the *review* deliverable rather than in Guardrails is what scopes it
  correctly; that placement is load-bearing, not incidental.
- **Gate (b):** not applicable — nothing removed. **Gates (c)/(d):** `cobb` §7 lint; `audit-team.sh`
  PASS.
- **One lint correction:** I first wrote "goes to **an appendix**" — but this file's own
  *"A complete review contains:"* enumerates exactly four sections and defines no appendix, so the
  agent was told to write into a container the file's spec doesn't declare. → "a trailing
  `## Appendix` section" (+4 w), which names it in place rather than paying ~10 w for a fifth
  skeleton item.
- **One addition the plan mandated and `cobb`'s lint sent back** — *"a later pass records a closed
  finding as one disposition line, full prose only for new findings."* I had shipped it, arguing it
  was safely split from Stage E's `## Pass N` convention amendment (I owned *findings*, Stage E
  owned *section mechanics*). The split does not hold, on three grounds: Stage E's own bullet
  **already contained "one-line disposition per prior finding" verbatim**, so both halves were
  already written and Stage D was one gate away from shipping one rule into two always-loaded files;
  the rule binds **three** reviewing agents (`analyst`, `data-scientist` via `reviews/<slug>-ml.md`,
  `security-expert`), not one; and this file's established idiom for a root-owned document
  convention is a **pointer**, not an inline restatement (`:66`, "Open the document with the header
  block from root `AGENTS.md`"). **Reverted here, relocated to Stage E's rule-5 amendment**, which
  now carries the disposition tokens so nothing is lost.
- **Plan items:** none. Feeds the plan's **finding 15**.
- **Watch (observation window open):** the verdict scale, severity ranking, the `CPG:` line, the
  evidence-traps list and the `-impl` split are unchanged. What to watch is the cap misfiring —
  a finding truncated to fit 15 lines with its evidence dropped rather than appendixed. That is
  the exact failure the never-drop guard exists to prevent, so it is also the test of whether
  pairing the guard with the cap in one breath actually works.

## 2026-08-24 — Prompt-waste compression, Stage C3 (analyst-specific pass) — file measured at its editorial floor

- **What:** Five edits, one pass (per the Stage C one-pass-by-default rule), 2,510 → 2,473 w (−37, 1.5%).
  (1) § "How you work" step 3 blockquote: dropped the tail "it is not part of this always-loaded
  prompt" (meta-commentary about prompt structure, no behavioral content); the
  `review-techniques.md` pointer and its three technique examples kept — the examples *are* the
  trigger, since "specialized verification techniques" is not a condition the agent can check.
  (2) § Guardrails "Evidence over vibes" lead: "Specific traps that have bitten a review before:"
  → "Specific traps:". (3) Placeholder-token trap: the frequency generalization ("Plans routinely
  paste…") replaced with a direct imperative ("Watch for a pasted doc-template placeholder"); the
  placeholder-vs-expanded-key mechanism deliberately kept, since it is what makes the trap
  recognizable. (4) "held, pending until X lands" trap: wording compressed, both halves intact.
  (5) "A deliverable that already exists at your target path" bullet: dropped "e.g. resuming after
  an interruption", `Offline/static` → `Static` (binds to this prompt's own "static reviewer"
  vocabulary), both branches intact.
- **Rule inventory (gate a), edited regions — all preserved:** consult `review-techniques.md` on
  demand + its three trigger examples (1); evidence-vs-inference distinction, the two hard nevers
  (suite-green, traced-path), all six traps with their mechanisms and consequences (2–4);
  not-authoritative half + grep-siblings half (4); side-effecting-claims-unverified branch and
  static-claims-inheritable branch (5). Verbatim `CPG:` three-form sentence, audit-check-8 commit-grant
  tokens, frontmatter, hooks, learning-capture block and RCA skeleton untouched.
- **Removed class-5/6 material, recorded where:** the traps' incident provenance → this file's
  2026-08-23 entry (line 90, the placeholder-token trap) and 2026-08-19 "Evidence over vibes"
  entry (which names all five original traps individually).
- **Two dedup candidates considered and rejected**, both the shape that caused the C1 pass-2
  regression: the intro's "the artifact under review stays untouched" against Guardrail 1 (kept —
  the intro is the *scope contract* stating what the agent's output **is**, the guardrail is a
  *prohibition* with a hook behind it; different speech acts, and deleting the intro sentence
  opens the prompt describing a job without saying what it produces), and step 3's "did you check,
  or does it just look wrong?" against "Evidence over vibes" (kept — step 3 is a per-finding
  *sufficiency gate* that pressures toward going and checking; the guardrail explicitly *permits*
  inferring provided you label it. Two different behaviors at two decision points).
- **Correction to that second rejection, from the lint:** the reason originally given was
  gather-time vs. reporting-time, which does not hold — step 3's tail "— say which" genuinely is
  a reporting-time labeling instruction duplicating the guardrail. What earns the keep is step 3's
  *first half*, not the split. Recorded so a later pass doesn't re-derive the wrong rationale.
- **Fixed while in there (composition conflict, pre-existing):** the placeholder-token trap cited
  `kaizen_analyst` as an example of "the repo's expanded keys" — but that graph key was **retired**
  (this file's 2026-08-21 entry, `G1`'s last two retirements). The prompt was asserting as current
  a key its own auto-loaded context says no longer exists. Now `cpg_salesperson`, live-verified
  against the FalkorDB instance's loaded-graph list.
- **Verified:** `audit-team.sh` PASS (all 13 agents, including the personal-identifier and
  check-8 sweeps); `cobb` §7 lint on the result — **0 blockers, 0 majors**, 3 minors + 2 nits, all
  four actionable ones applied (the retired-key fix, +3 w restoring the unanchored comparative's
  referent, +5 w restoring "side-effecting ones are not" as anti-trigger fencing on the
  static-claims permission, +2 w restoring the distributive "any of its"). First unit in this plan
  with no MAJOR.
- **Finding for the plan — this file is at its editorial floor.** Post-edit residual class-6/7
  inventory across the *whole* file is **under 25 words**. C3's plan target of ~1,900 w (a 26% cut)
  was unreachable without deleting rules; the measured floor is ~2,473 w. The ~350 w evidence-traps
  block is distilled class-3/4 lesson payload, not narrative — no further prose editing reaches it.
- **Plan items:** opened **K-003** (progressive disclosure — move the traps' mechanisms to
  `review-techniques.md`, keep trigger stubs), the structural lever this floor leaves.

## 2026-08-23 — Freshness-clause grammar fix (Stage B wave 2 micro-shape)
- **What:** "a `teco`-issued brief that states the graph's freshness, take it as given" → "when a `teco`-issued brief states the graph's freshness, take it as given" — closing the hanging-topic construction cobb's wave-1 lint flagged as minor; applied uniformly across all files carrying the clause. No rule change; both branches intact.

## 2026-08-23 — Prompt-waste compression, Stage B wave 1 (boilerplate sweep)
- **What:** Applied the three pilot-validated boilerplate compressions from
  `claude/docs/plans/prompt-waste-reduction.md` (§3 doctrine, Stage B), same shapes as the
  `architect.md` pilot; the full analyst-specific compression (plan Stage C3) is a separate,
  later unit. (1) CPG-freshness clause (§ "How you work" step 2): dropped the "(2026-08-19)" date
  and the redundant "without re-deriving staleness yourself" tail. (2) Interactive-commit-grant
  passage (§ Guardrails, Bash bullet): dropped the provenance sentence "Stakeholder decision,
  2026-08-21 — see `kaizen/history.md`." and ", same as before"; the "(spawned via
  `Agent`/`Task`)" clarifier moved from the interactive-definition parenthetical to the carve-out
  sentence (was stated in both). (3) Learning capture: intro dropped "directly" and "identified by
  a real `:Agent` node it's `PRODUCED`-linked to," (the Cypher template below shows the MERGE +
  PRODUCED edge); tail dropped the inbox-replacement history sentence and "exactly like the old
  inbox was".
- **Rule inventory (gate a), edited regions — all preserved:** freshness is `teco`'s / brief taken
  as given / standalone = current (block 1); interactive-mode definition, explicit-path grant,
  full never-list, delegated-subagent carve-out, deliverable left for `teco` post-verification
  (block 2); capture trigger + graph + Cypher template, skip-known-facts, raw-capture/`cobb`
  promotes, never edit own definition (block 3). Verbatim `CPG:` three-form sentence and
  audit-check-8 tokens untouched.
- **Removed class-5/6 material, recorded where:** inbox-replacement history → this file's
  2026-08-21 "kaizen/inbox.md deleted" entry; commit-grant provenance → this file's 2026-08-21
  grant entry + `claude/AGENTS.md` § Hook machinery; freshness centralization date → this file's
  2026-08-19 "Freshness-check clause removed" entry.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint pass on the result.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** The Bash guardrail's "investigation only" bullet now also grants: when running
  interactively (`claude --agent analyst`, a human present turn-by-turn — not a delegated
  subagent), may `git add`/`git commit` its own review document from the session, by explicit
  path, never bulk-staged/pushed/reset/rebased/amended; the grant does not apply when spawned as
  a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Your deliverable"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered). It had been frozen — never written to — since the 2026-08-20 graph migration (see that date's entry below, which already confirms this file's own pre-migration content was imported into the graph verbatim at the time).
- **Why:** user-directed team-wide cleanup, "no point keeping a file already in git history." Before deleting any of the 12 agents' frozen inboxes, `cobb` live-confirmed `kaizen_team` — the single shared graph every agent's raw capture has routed through since the 2026-08-20 consolidation — holds **zero** entries for any agent: every raw capture any agent ever wrote there (including this agent's own 12-entry pass, immediately above) has since been fully distilled and cleared. Combined with the migration-time import guarantee above, nothing in this file was ever a live, undistilled input to anything — it was a pure redundant backup copy. Same session also completed `G1`'s last 2 of 12 `kaizen_<agent>` graph-key retirements (`kaizen_analyst`/`kaizen_teco`, executed by `graph-dba`), closing `docs/plans/generic-cypher-mcp2-coordination.md`'s one remaining open item.
- **Verified:** live `mcp__cypher__query` count against `kaizen_team` (0 entries) before any deletion; every entryId this file's own 2026-08-21 distillation entry (above) lists was cross-checked against this file's pre-deletion contents — all present, none missing.
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: all 12 pending `analyst`-authored entries processed

- **What:** `cobb` ran the agent-maintenance skill §5 procedure against every `kaizen_team` node
  with `author:'analyst'` (12 entries, dated 2026-08-11 through 2026-08-21 — analyst's raw
  capture since the 2026-08-20 team-wide graph migration; none of these overlap the 3 entries the
  2026-08-11 inbox.md distillation already processed, which was a distinct, file-based pass from
  before the migration). Full per-entry disposition:
  - **Promoted to `review-techniques.md`** (7 entries):
    - Reconciling a kaizen-graph distillation's claimed dispositions against ground truth
      (`481f29ed…`) — adapted from its original file-diff framing (`grep -c '^-## '` on an
      inbox diff) to the graph-based mechanism, since the inbox-diffing technique itself is now
      moot (inboxes are frozen); the underlying discipline — verify an aggregate "N processed"
      claim against the itemized ground truth — carries over unchanged, and is exactly what this
      pass's own bookkeeping had to do.
    - An uncommitted agent-prompt edit under review is already live, via the deployment symlink
      (`854e701d…`).
    - Verifying a "copied verbatim" text-block claim needs a programmatic whitespace-normalized
      diff, not a read-through (`fe2007f5…`).
    - `pytest -k` is not a substitute for the project's own `-m` marker filter when verifying a
      cited baseline (`eea48dac…`) — re-verified live: `cypher-mcp/pytest.ini`'s
      `addopts = -m "not live"` still matches the fact as described.
    - Ground truth for "may an agent edit its own definition?" — the literal "never edit your own
      agent definition" clause (`b9ed574b…`) — re-verified live: `grep -rln` today returns all 12
      non-cobb prompts (grew from 11 at entry-creation time — `security-expert` now also carries
      it), `cobb.md` still the sole exception.
    - Check live-service reachability before trusting a live-test report (`b3a1f2e4…`).
    - A brand-new untracked file has no `HEAD` baseline for the existing zero-touch mutation-test
      methods — added as case (c) alongside existing (a)/(b) (`e7f3a1b2…`).
  - **Promoted to `claude/analyst/analyst.md`** (1 entry): a new "Evidence over vibes" bullet —
    run a plan's prescribed acceptance-check command verbatim before approving; the doc-template
    placeholder token (`kaizen_<agent>`) silently matches nothing against the repo's expanded key
    (`8b881e50…`).
  - **Promoted to `claude/cobb/TESTING.md`** (1 entry): a new Gotcha — `audit-team.sh`'s `ROOT`
    resolution makes it scratch-testable, and a kaizen-files-only directory is silently skipped
    from agent enumeration rather than failing check 1 (`0b11bf16…`) — routed to cobb's testing
    doc rather than this agent's own knowledge base since the fact is about safely testing a
    script `cobb` owns, not a review technique this agent performs.
  - **Discarded, already resolved elsewhere** (3 entries):
    - The "no `SendMessage` means a nested review-gate result misroutes" open question
      (`4a4a031a…`) — fully resolved the same day this pass ran:
      `claude/docs/requirements/mid-run-escalation.md` (Status: Ready for design, 2026-08-21)
      settles it directly — `teco` performs the `SendMessage` resume, not the delegate, so
      review-gate reporters never need the grant. A stronger, more current answer than anything
      this entry could be promoted into.
    - The `r1_probe` field-semantics finding on falkor-chat's `golden_guards.jsonl`
      (`a1e6f3d2…`) — already fully documented and the underlying mismarking already fixed:
      `falkor-chat/docs/plans/golden-set-expansion-ml.md` §"r1_probe semantics (2026-08-20
      addition, analyst review)" states the exact rule, and `cs-10`/`cs-13` are confirmed flipped
      to `r1_probe: false` in that file's finalized golden set.
    - The `nc`/`ncat`/`netcat` local-marker exemption gap in `security-expert`'s
      `guard-exploitation-approval.sh` (`eadd7a90…`) — already fixed: the live script's branch
      (c) and its header comment cite this exact finding ("analyst review 2026-08-20") as the fix
      that gave `nc`/`ncat`/`netcat`-with-a-shell-flag its own unconditional always-ask branch,
      re-verified by reading the current script.
- **Verification method:** fetched every field's exact, untruncated text via `redis-cli
  GRAPH.QUERY kaizen_team ... --no-raw` (the `mcp__cypher__query` tool's own per-cell display
  truncates around 300 chars, `…(+N chars)`, and paging every field via `substring()` for 12
  entries × 4 fields would have been far more round trips than one raw redis dump). Re-derived
  each surviving claim against the live repo rather than trusting the entry's own framing —
  `pytest.ini`, `audit-team.sh`'s actual `ROOT`/enumeration logic, the `grep` for the self-edit
  clause, and the two "already fixed/documented" discards were all re-read from source, not
  assumed from the entry text.
- **Why:** user asked to "work on analyst's inbox" — `analyst/kaizen/inbox.md` is a frozen
  2026-08-20 historical snapshot (already imported and cleared, 3 entries), so the live
  equivalent is analyst's pending raw capture in the shared `kaizen_team` graph; this is the
  first full distillation pass against it since the migration.
- **Plan items:** none opened — every surviving entry either promoted cleanly or was already
  resolved elsewhere; nothing was kept open pending further verification.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_analyst`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_analyst` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 5 pre-existing
  entries were parsed out programmatically and imported into the graph verbatim (entryId assigned,
  `author: 'analyst'`), preserving every field; its own header explains the freeze and gives the
  live-read query. The trailing "Your write guard allows exactly this inbox path" clause was
  dropped — the write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP tool, so it no
  longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — "Evidence over vibes" Guardrails bullet converted to a sub-list
- **What:** The Guardrails bullet had grown into a single run-on sentence carrying 5 sub-rules (the untracked-baseline trap, the unrun-regex trap, the guard-glob cross-check, the no-shellcheck note, the "held pending" note) after four separate clause-extension edits. Restructured as one lead sentence plus a 5-item sub-list under the same bullet — no new top-level Guardrails bullet added, content unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 3 of the parked diagnosis (`cobb/kaizen/plan.md`) — flagged 2026-08-09 during the inbox-distillation review as hurting scannability, never fixed until now.
- **Plan items:** —

## 2026-08-19 — Freshness-check clause removed (centralized on teco)
- **What:** Dropped the CPG freshness-check paragraph from the CPG-orientation step — still checks whether a relevant CPG exists and uses it via `cpg-analysis`, but no longer queries the `:CpgBuildInfo` freshness marker itself. That responsibility is now `teco`'s alone (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`); running standalone (no `teco`-issued brief), staleness is simply not checked.
- **Why:** User-directed prompt-verbosity reduction: the freshness paragraph was ~130 words, byte-identical across six agent files. Stakeholder chose full centralization over a per-agent dedup, accepting the standalone-run capability loss.
- **Plan items:** —

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
"the joern agent's job" error text in `cypher-mcp/server.py`) — the live string now reads "the
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

## 2026-07-25 — `tools:` allowlist gains `mcp__cypher__query` (M3 / C-304)
- **What:** Frontmatter `tools:` now ends `…, Agent, mcp__cypher__query`. `claude/README.md` row 17 updated to say the `cpg-analysis` skill reaches the graph through that MCP tool and why the allowlist entry is required. No body or `description` change — the CPG routing clause added on 2026-07-19 stays accurate, and the skill is progressively disclosed.
- **Why:** M3 replaces the CPG read path with a single MCP tool, `mcp__cypher__query(graph, cypher)` (`docs/plans/cpg-query-access.md` S5). **`tools:` is an allowlist, not a hint** — an agent that declares one does not see MCP tools absent from it, so without this line the feature would have been silently inert for `analyst` (and `architect`); `qa-engineer` and `graph-dba` declare no allowlist and inherit it. `redis-cli GRAPH.QUERY` remains the documented fallback and is the only path under OpenCode/Kiro.
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
