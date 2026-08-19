# Kaizen distillation (2026-08-11) — cross-agent diff review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (no backlog id; stakeholder-triggered `cobb` sweep) · **Extends:** `docs/reviews/kaizen-inbox-distillation.md`

## Scope & verdict

**Reviewed:** the uncommitted working-tree diff at `/home/<user>/prg/graphmind-ai-lab`
(`git diff` + the three untracked files), 39 files / +615 −507 — `cobb`'s full-team kaizen-inbox
distillation triggered by the stakeholder's 400k-token context-blowout report. Baseline is `HEAD`
(`2c13e35`); nothing is committed.

**Not reviewed:** the *substantive truth* of individual environment facts I could not re-probe
(no FalkorDB, LM Studio or Docker reachable from this run) — those are judged on internal
consistency and corroboration across sibling documents, and I say which is which per finding.

**Verdict: needs changes.** One blocker (captured learnings deleted from an inbox with neither a
promotion nor a logged discard, behind a history entry that reads complete), five majors. The
headline deliverable — `teco.md`'s dispatch-sizing rule — is sound, accurate to its evidence, and
enforceable; the defects are in the distillation's *bookkeeping* and in three routing/catalog
surfaces that weren't updated with the content.

**Deployment note, load-bearing for urgency:** `~/.claude/agents/<name>` are symlinks into this
working tree (`ls -la ~/.claude/agents`), and `~/.claude/skills → <repo>/skills`. Every prompt and
skill edit in this diff is **already live** for the running team while uncommitted — this review's
own system prompt carries the new `analyst.md` clause. Defects here ship immediately; they are not
waiting on a commit.

---

## Findings

### Blocker — B-1. Two `coder` inbox entries were cleared with no promotion and no logged discard, and the history entry hides it by crediting two other agents' entries to `coder`

**Evidence.** `git diff claude/coder/kaizen/inbox.md` removes **8** entries and replaces the body
with `*(empty — no unprocessed learnings)*`. `claude/coder/kaizen/history.md`'s new entry claims
"8 entries routed (6 to … , 1 discarded)" — arithmetic that already doesn't close (6 + 1 = 7) —
and its promoted list names *urllib timeout taxonomy* and *LM Studio `/v1` 200-envelope*, neither
of which was ever in `coder`'s inbox (both are in the removed text of
`claude/analyst/kaizen/inbox.md` / `claude/architect/kaizen/inbox.md`, and both agents' history
entries claim them). Mapping the 8 real entries to a disposition leaves **two with none at all**:

| Removed `coder` entry | Suggested home (its own text) | Disposition found |
|---|---|---|
| 2026-07-24 — "a `pytest --collect-only` baseline can move under you mid-run; report the *attributed* delta" | prompt | **none — not logged, not promoted** |
| 2026-07-24 — "a green pytest exit code is not evidence an integration suite ran — read the skip count" | prompt | **none — not logged, not promoted** |

**Why it matters.** The inbox is the *only* capture channel (root `AGENTS.md`; `agent-maintenance`
SKILL.md §5), and it is now empty. The second entry is a live gap, not a redundant one: I checked
`claude/coder/coder.md:22` — step 5, "Verify and report" — and it says only "Run the full suite at
the end. Report what you actually ran and saw." No skip-count clause. `tdd-engineer.md:42` *does*
carry exactly this rule ("a suite … can exit 0 while a chunk of it silently never ran, so report
the `passed`/`skipped`/`deselected` counts"), which is what makes the asymmetry visible: the agent
that filed the learning is the one still missing it. Once this diff is committed, the durable
record says "8 entries routed" and nobody re-derives the two that vanished.

**Suggested fix.** (1) Restore both entries to `claude/coder/kaizen/inbox.md` (recoverable now via
`git diff`; not recoverable after commit), or route them explicitly — the skip-count one is a
one-clause addition to `coder.md:22` mirroring `tdd-engineer.md:42`, the attributed-delta one is a
plausible discard *if* logged as one. (2) Correct the history entry's promoted list to contain only
entries that were in this inbox, and make its count reconcile.

---

### Major — M-2. `python-web-quirks` grew 3 → 8 entries, but the only thing that triggers it — the `description:` frontmatter — still names the original three

**Evidence.** `grep -c '^## ' skills/python-web-quirks/SKILL.md` → **8**; five were added here
(urllib error taxonomy, OpenAI-compat 200-envelope, fence-fragile LLM-judge `json.loads`,
`monkeypatch.setenv` vs. an import-time-frozen constant, function-local import binding). The
frontmatter `description:` (lines 3–11) still enumerates only *asyncio fire-and-forget GC-safety*,
*BackgroundTasks concurrency*, and *`response_model_exclude_unset` on nested models*. The repo's own
standards doc is explicit that this is dispositive: `skills/agent-standards/claude-code.md:203` —
"**Progressive disclosure:** only the `description` is loaded at startup". `skills/README.md:23`
likewise still enumerates the old three.

**Why it matters.** Two of the five new entries (`monkeypatch` timing, import binding) are not
web/async at all, so an agent hitting "why is my autouse fixture's `setenv` a no-op" has no trigger
to load the skill. The promotion is written but effectively unroutable — the distillation's stated
purpose (get the fact where the next agent will see it) fails at the last hop. Root `AGENTS.md`
also requires the catalog (`skills/README.md`) to be updated in the same change.

**Suggested fix.** Extend the `description:` with the new topic triggers — stdlib `urllib` error
taxonomy / OpenAI-compatible-server response shapes / pytest `monkeypatch` + import-binding timing —
and either widen the skill's stated scope beyond "web/async" or move the two pure-Python entries to
a better-fitting home. Update `skills/README.md:23`'s enumeration to match.

---

### Major — M-3. Three new agent knowledge bases, but `claude/AGENTS.md` wasn't touched

**Evidence.** `claude/data-scientist/lm-studio-model-notes.md`, `claude/devops/ops-quirks.md`,
`claude/qa-engineer/qa-testing-techniques.md` are new (untracked). `claude/README.md` was correctly
updated with a linked clause in each agent's row. `claude/AGENTS.md` is **not in the diff**, yet its
roster (lines 12–19) is the file that already carries this exact annotation for the only agent that
previously had one: "`graph-dba` (carries two on-demand knowledge bases: `falkordb-quirks.md` …
and `falkordb-reference.md` …)".

**Why it matters.** Root `AGENTS.md` states the rule literally: editing an agent means updating
"its source, its `kaizen/{plan,history,inbox}.md`, the relevant catalog (`claude/README.md` for
agents …), **and `claude/AGENTS.md` in the same change**." Left as-is, the roster reads as if
`graph-dba` is still the only KB-carrying agent, which is now wrong for four of them. This is a
**recurrence** of the class flagged as a Minor in the previous pass's review
(`docs/reviews/kaizen-inbox-distillation.md`, "catalog rows weren't updated") — the file moved, the
failure mode didn't.

**Suggested fix.** Add the parenthetical KB annotation to the `devops`, `qa-engineer` and
`data-scientist` roster entries in `claude/AGENTS.md` (and, while there, `analyst`'s pre-existing
`review-techniques.md`, which the roster has never mentioned).

---

### Major — M-4. The corrected `rows=` claim now contradicts `cypher-mcp/server.py`'s own module docstring, and no follow-up was filed

**Evidence.** The correction itself is **well-founded** — I cross-checked it and it is not a bare
assertion: `docs/test-reports/cpg-getting-started-report.md:33` (TP-008/DEF-001, 2026-07-30, live:
`rows=10000` vs. a true `count(n)` of **110048**, `GRAPH.CONFIG GET RESULTSET_SIZE` → `10000`),
`claude/graph-dba/falkordb-quirks.md:263-269` ("Verified 2026-07-30, v4.18.11"),
`docs/HISTORY.md:138`, and `docs/manuals/cpg-getting-started.md:150-158` (DEF-001 already folded in
on the same date). Four independent sites, consistent numbers.

But `cypher-mcp/server.py:20-22` — the module docstring of the very server the README documents —
still reads:

```
* **Display-only truncation.** The full result set is materialised before
  formatting; the caps below shape the *rendering*, so the reported row count is
  always exact and memory/latency are bounded by the query, not by the caps.
```

**Why it matters.** This is the same false claim the diff corrects in two places, left standing in
the third and most authoritative one. A reader of `server.py` (or a future agent grounding a change
in it) gets the pre-correction answer. Correctly, `cobb` didn't edit source — but no history entry
files the follow-up, so it exists nowhere.

**Suggested fix.** Route a one-line docstring fix to `coder`/`devops` ("…exact below FalkorDB's
`RESULTSET_SIZE` (default 10000), at or above which it is itself a cap") and record the handoff in
`claude/graph-dba/kaizen/history.md`'s entry, which currently lists `cypher-mcp/README.md` as fully
handled.

---

### Major — M-5. The dispatch-sizing rule lives at step 3 (Delegate) but governs step 2 (Decompose & sequence), where unit granularity is actually fixed

**Evidence.** `claude/teco/teco.md:69` places the rule under "**3. Delegate with complete briefs**".
Step 2 (`:54-58`) is where teco "Decompose & sequence" into "Ordered units, each with an owner,
inputs, a done-condition" and where the coordination-doc ledger is opened and populated. Step 2's
text is unchanged by this diff.

**Why it matters.** By the time teco is at step 3, the ledger already says `U4 — coder — Landing 1`.
The rule then asks it to split a unit it has already recorded as one, which is exactly the friction
that produced the original decision ("the plan already sequenced it, so one coder should execute the
sequence" — `teco/kaizen/inbox.md`, removed text). A rule that fires one step after the decision it
governs is the weakest position it can occupy. This is a placement issue, not a soundness one —
teco reads the whole prompt each run, so the rule will still be *seen*.

**Suggested fix.** One sentence in step 2 pointing forward: "Unit size is not free — apply the
step-table sizing rule (§3, *Size each dispatch…*) **when you draw the units**, not when you
dispatch them." Keeps the rule stated once.

---

### Major — M-1. Five more inbox entries cleared without a logged disposition, and four of the ten history entries carry a count that doesn't match their own diff

**Evidence.** Same failure class as B-1 but without content loss, so ranked lower. Counting removed
`## ` entries per inbox against each history entry's own header:

| Agent | Entries removed | History header says | Unlogged dispositions |
|---|---|---|---|
| `graph-dba` | 7 | "5 entries" | the two 2026-07-19 CPG-topology entries |
| `qa-engineer` | 15 | "15 entries" (sub-counts don't reconcile: 3+8+3+1 with `doctor` double-counted) | MCP `send_message` asymmetry; Bash-tool backgrounding |
| `devops` | 13 | "9 entries" | none — all 13 are described in the prose |
| `tico` | 4 | "3 entries" | 2026-07-31 `version` vs. `defVersion` |
| `teco`, `analyst`, `architect`, `cobb`, `coder`, `data-scientist` | 6·, 3, 1, 3, 8, 4 | matches (except `coder` — see B-1) | — |

I checked each unlogged one for real content loss and found **none**, which is why this isn't a
second blocker: `graph-dba`'s two CPG entries are already in
`skills/joern-cpg/references/cpg-model.md` §"Consumer-query facts" (verified: `CALL.NAME` caller
matching at `:116`/`:121`, `REACHING_DEF` intraprocedural at `:140`); `tico`'s `version`/`defVersion`
is already tracked as **K-040** in `falkor-chat/docs/BACKLOG.md:1210`; and both `qa-engineer` ones
actually landed — MCP `send_message` in `falkor-chat/docs/DESIGN.md` §14.7, the Bash-tool one in
`skills/agent-standards/claude-code.md` — they just aren't named in `qa-engineer`'s history.

**Why it matters.** The history entry is the durable record (§5 step 4: "the history entry is the
durable record"). A count that overstates or a disposition that's silent makes the next
distillation — or the next reviewer — unable to tell a considered discard from an accident, which
is precisely how B-1 slipped through.

**Suggested fix.** Add a one-line disposition for each of the five (three are "already covered at
`<path>` — discarded", two are "promoted to `<path>`") and fix the four header counts to match the
diff. `·teco`'s inbox is 6 headed entries plus one *headless* continuation block (a stray
`- **Evidence:**` with no `## ` heading, the 458k/222-call one) — cobb's treatment of it as part of
the preceding Landing-1 entry is correct; worth a note so the next pass doesn't re-count it.

---

### Minor — m-1. `teco.md`'s "3 of ~15 plan-named test files" isn't what the plan or the gate say

**Evidence.** `falkor-chat/docs/plans/llm-provider-config.md` §5 names **11** test files (2 new:
`test_transport.py`, `test_modelconfig.py`; 9 extended). The gate's own account is
`falkor-chat/docs/plans/llm-provider-config-coordination.md:283-286`: "**three of five rewired
consumer bindings** (`test_executor_agent.py`, `test_responder.py`, `test_tools.py`) untouched
despite being named in plan §5's file list". The "~15" appears to be a transcription of the source
inbox entry's "3 of the plan's ~15 named **files**" (§5 has ~16 file rows) with "test" inserted.
The wording is repeated verbatim in `claude/teco/kaizen/history.md` and `claude/cobb/kaizen/history.md`.

Everything else in the rule checks out against the evidence: 6 steps (`L1-1..L1-6`), ~10 files,
**458k tokens / 222 tool calls / ~45 min** (`llm-provider-config-coordination.md:218`), and the
stakeholder quote is reproduced exactly as the inbox recorded it.

**Suggested fix.** "…silently dropped 3 of the 11 test files the plan names — 3 of the 5 rewired
consumer bindings — from its own stated scope."

---

### Minor — m-2. The new `QUERIES.md` §11.2 prose duplicates the ⚠️ CONDITIONAL callout eight lines below it, and overstates the invariant's source

**Evidence.** `falkor-chat/docs/QUERIES.md:963-971` (new) and `:975-985` (pre-existing) now both
state: the one-row collapse is a premise not an engine property; two `START` edges ⇒ two rows;
each row carries the full `steps` collection; `result_set[0]` silently picks an arbitrary one. The
callout says it with more precision and better provenance — "**Verified live on
falkordb/falkordb:v4.18.11 (K-031 V-1, snapshot side, throwaway `ws:k031probe`)**" — and already
routes the second-edge question onward ("how a root acquires a second `START` edge … is K-034's").
The new prose instead cites `claude/graph-dba/falkordb-quirks.md`, an agent-private KB whose entry
is created in this same diff.

Two smaller accuracy points: (a) "**the schema guarantees** exactly one `START` edge per def"
overstates — there is no graph-level constraint; the guarantee is a service-layer gate
(`services._check_no_structural_conflict`, 409, per the §11 preamble), which is exactly the
distinction the callout is careful to preserve; (b) the callout attributes the invariant to K-031
and the *mechanism* to K-034 — the new prose attributes both to K-034. K-034 is genuinely delivered
(`falkor-chat/docs/BACKLOG.md:790`, ✅ 2026-08-01), so the claim is true, just differently framed
from the paragraph beneath it.

**Suggested fix.** Collapse the new prose to one forward-pointing sentence — "`start.key` is a
grouping key, not an engine-level constant: the one-row collapse is a cardinality **premise**, see
the callout below" — and let the callout keep the evidence. Change "the schema guarantees" to
"K-034's service-layer topology gate keeps".

### Minor — m-3. The `DESIGN.md` §14.7 insertion re-parents an existing bullet

**Evidence.** The new "QA/acceptance-testing gotchas" list (5 bullets) is inserted directly above
the pre-existing `- **A wired agent now requires two config files (K-042).**` bullet. Separated
only by a blank line at the same list level, that bullet now renders as a **sixth** member of the
QA list, though it is a pytest/config hazard, not a black-box observation. Related: `qa-engineer`'s
history calls it "four new … bullets" — there are five (the MCP `send_message` one is unlisted, see
M-1).

**Suggested fix.** Move the new block to *after* the K-042 bullet, or give the K-042 bullet its own
lead-in line. Fix "four" → "five" and name the fifth.

### Minor — m-4. The sizing bullet adds ~180 words of incident forensics to the always-loaded prompt of the agent whose context was the reported problem

**Evidence.** The diff adds 491 words to `teco.md` (now 122 lines / 3807 words / ~25.6 KB ≈ 6.4k
tokens). Roughly half the sizing bullet is the K-042 narrative (458k / 222 calls / ~45 min / the
second dispatch), reproduced **verbatim** in `claude/teco/kaizen/history.md`'s 2026-08-11 entry.

This is not a contradiction of cobb's own diagnosis — the diagnosis (prompt size is a small,
roughly-constant fraction; the blowup is accumulated tool output) is sound and I agree with it. It's
a proportionality point, and there's a real counter-argument cobb states explicitly: quoting the
stakeholder verbatim is what keeps the rule from eroding.

**Suggested fix (take or leave).** Keep the rule, the thresholds and the stakeholder quote; reduce
the forensics to "(origin and full incident numbers: `claude/teco/kaizen/history.md`, 2026-08-11)".

### Nit — n-1/n-2. Two promotions are double-logged across agents

`claude/graph-dba/kaizen/history.md` lists the *no-string-repetition* promotion as its own; that
entry came from `coder`'s inbox (and `coder`'s history logs it too). It also lists the `db.indexes()`
entry as its discard; that entry lived in **`teco`**'s inbox (`teco`'s history also claims it).
Harmless in itself — the edits landed once — but it's part of why the "5 entries" header doesn't
reconcile (M-1).

---

## What's solid

- **The core deliverable is accurate and enforceable.** The rule's thresholds (">~3 steps or >~5
  files"), its evidence (6 steps / ~10 files / 458k / 222 calls / ~45 min), its correctness cost
  (the dropped test files), and the verbatim stakeholder directive all trace to real artifacts I
  checked (`llm-provider-config.md` §5/§6, `llm-provider-config-coordination.md:218,283-286`). It
  is a **decidable** rule — a step table either exceeds the thresholds or it doesn't — not advice.
  It does **not** contradict the neighbouring "Dispatch" bullet: that bullet says serialize
  same-file units, and the new one produces exactly same-file dependent units and says so. The
  "standing user directive, not a ceremony trade-off" clause correctly pre-empts the "match ceremony
  to the task" line at `:50`.
- **The `RESULTSET_SIZE` correction is genuinely corroborated**, not asserted — four independent
  sites with consistent live numbers (see M-4). Both edits (`cypher-mcp/README.md`,
  `skills/cpg-analysis/SKILL.md` ×2) are internally consistent with each other and with the
  end-user manual's already-corrected wording, and the "below 10k exact / at-or-above a floor"
  framing is the right precision.
- **K-008 is genuinely held, not quietly decided.** `claude/tico/kaizen/plan.md`'s new entry records
  both proposed shapes *with their risks*, names why the 2026-07-30 commit-authority decision
  doesn't settle it, and picks neither. Cross-checked the counterfactual: `claude/tico/tico.md`'s
  diff is **one line** (the homework bullet) — the Guardrails / write-scope / `Agent`-use sections
  are untouched. The claim holds.
- **A stale finding was caught before promotion.** The `pipeline.sh --reset` guard-bypass entry
  (2026-07-30) was re-checked and found already closed (C-311, 2026-08-08); the KB entry was
  rewritten to state the fix plus the generalizable lesson instead of re-filing closed work. This is
  §5 step 2 done properly.
- **`bash claude/scripts/audit-team.sh` → `RESULT: PASS`**, run by me against the working tree.
  cobb's "clean" claim is verified, not taken on trust. (Notably it now passes outright, where the
  removed `coder` inbox entry recorded a standing 2 FAIL personal-info baseline.)
- **The `data-scientist` κ addition is methodologically sound.** For a judge whose construction pins
  specificity near ceiling, κ is dominated by sensitivity and carries essentially no information
  about the false-advance class; κ is additionally prevalence-dependent (the well-known kappa
  paradox), which the source entry demonstrates with a 20k-trial simulation (E[κ] 0.70 at 11/10 vs.
  0.55 at 18/3). Gating on class-conditional rates and demoting κ to a reported diagnostic *with
  marginals* is the right prescription, and it's scoped to the case that warrants it rather than
  stated as a blanket rule.
- **The other prompt edits are correctly scoped.** `analyst.md`'s clause lands on the right bullet
  (Guardrails → Evidence over vibes) and states a checkable procedure. `tico.md`'s prior-decision
  check sits inside "Do your homework silently", before the first interview question, which is where
  a supersession has to be discovered to change the interview — it fits the existing flow rather
  than bolting a new step onto it. `devops.md` and `qa-engineer.md` get pointer blockquotes, not
  inlined facts, which is the documented KB pattern. All three new KB files exist, are reachable
  from both the prompt (backticked repo-root path) and `claude/README.md` (relative link, per the
  citation convention), and carry perishability headers.

---

## Open questions

1. **Document slug — resolved in Pass 2.** I wrote to the path the brief specified. Under root
   `AGENTS.md` collision rule 5 this was the same kind+topic as the existing
   `docs/reviews/kaizen-inbox-distillation.md` (`Status: active`, the 2026-08-09 batch), which had
   been executed against — so the convention prescribed a successor with the ordinal on the slug.
   Done in Pass 2: this document renamed (plain filesystem move; it was untracked, nothing to
   `git mv`) from `kaizen-distillation-2026-08.md` to `kaizen-inbox-distillation2.md`, gained
   `Extends: docs/reviews/kaizen-inbox-distillation.md` in its own header, and the earlier document
   gained `Extended by: docs/reviews/kaizen-inbox-distillation2.md`. Note for whoever integrates
   this: as of this rename, roughly nine other files in the working tree
   (`claude/{analyst,cobb,coder,devops,graph-dba,qa-engineer,teco,tico}/kaizen/{history,inbox}.md`,
   `claude/cobb/kaizen/plan.md`, and this coordination's own `docs/plans/
   kaizen-inbox-distillation2-coordination.md:10`) still cite the pre-rename filename
   `kaizen-distillation-2026-08.md` verbatim — checked via `git grep -n
   'kaizen-distillation-2026-08'`. The `history.md`/`inbox.md` citations are dated log entries
   describing the document as it was named at the time and don't need retroactive correction; the
   coordination doc's own line ("path pending an analyst-owned rename... see U2") is the one
   forward-looking reference that will read as stale once this lands and is worth a one-line fix
   by `teco` (out of this document's write scope — `docs/plans/*` isn't mine to edit).
2. **The MCP `send_message` asymmetry wants a backlog item.** Its inbox entry asked for both a
   DESIGN note *and* "a new K-item once filed" in `falkor-chat/docs/BACKLOG.md`. The DESIGN half
   landed; no K-item exists and `BACKLOG.md` isn't in this diff. A front door that silently never
   triggers the responder is a product gap, not just a testing gotcha. *Recommend routing to `teco`
   for a K-item* — out of `cobb`'s remit, but it shouldn't evaporate with the inbox entry.
3. **Is `coder.md` supposed to converge with `tdd-engineer.md` on suite-reporting discipline?**
   B-1's fix is a one-clause addition, but the underlying question — whether the two implementer
   prompts should state the same verification rules or deliberately diverge — is a `cobb`/stakeholder
   call, not mine.

---

## Pass 2 (2026-08-12) — re-review of `cobb`'s fix pass

**Scope.** Re-read the whole current working-tree diff (`git diff` + the still-untracked new
files: `claude/data-scientist/lm-studio-model-notes.md`, `claude/devops/ops-quirks.md`,
`claude/qa-engineer/qa-testing-techniques.md`, `docs/plans/
kaizen-inbox-distillation2-coordination.md`, this document) fresh — not diffed against Pass 1's
snapshot, the whole tree against `HEAD` (`2c13e35`, unchanged) — per U3's brief
(`docs/plans/kaizen-inbox-distillation2-coordination.md`). 44 tracked files carry changes from
`cobb`'s fix pass (up from 39 at Pass 1 — the fix pass touched additional history/catalog files;
this document's own rename-header edit is a 45th, mine, not `cobb`'s), still nothing committed.
Verified every Pass-1 finding against the actual diff text and, where relevant, ran commands
myself rather than trusting `cobb`'s history-entry narration.

**Verdict: approve.** Every blocker, major, minor and nit from Pass 1 is closed. One informational
note (below) for whoever integrates next, and the same three open questions, all now correctly
scoped and two of them explicitly deferred to the stakeholder rather than guessed at.

### B-1 — closed, verified

`claude/coder/kaizen/history.md`'s 2026-08-11 entry header now reads "8 entries routed (5
promoted, 1 discarded as redundant, 2 promoted late after an `analyst` review caught them
missing)" — arithmetic closes (5+1+2=8). The promoted list is now the 8 real entries removed from
`coder/kaizen/inbox.md` (confirmed: `git diff claude/coder/kaizen/inbox.md | grep -c '^-## '` → 8,
matching); the two previously mis-credited entries (`urllib` timeout taxonomy, LM Studio `/v1`
200-envelope) are gone from the list, with an explicit note that this entry "replaces a first
version that mis-credited two entries to `coder`." Both previously-dropped entries now have real
dispositions: the `pytest --collect-only` attributed-delta one is promoted into `coder.md` step 5,
and the skip-count one is promoted into the same step, mirroring `tdd-engineer.md:42` almost
verbatim (`git diff claude/coder/coder.md` — one line, both clauses added: "a suite can exit 0
while a chunk of it silently never ran, so report the `passed`/`skipped`/`deselected` counts" plus
the attributed-delta clause). `tdd-engineer.md` itself is untouched (`git status --short --
claude/tdd-engineer/` → empty) — confirms the fix stayed inside the narrow B-1 promotion (see the
scoping-call check below).

### M-1 — closed, verified

Removed-entry counts vs. corrected headers, re-counted independently (`git diff <inbox> | grep -c
'^-## '`):

| Agent | Removed (re-verified) | Header now says | Dispositions for the previously-unlogged ones |
|---|---|---|---|
| `graph-dba` | 7 | "7 entries" | both 2026-07-19 CPG-topology entries now discarded, "already covered in `skills/joern-cpg/references/cpg-model.md`" |
| `qa-engineer` | 15 | "15 entries" | MCP `send_message` (discarded, already in `DESIGN.md` §14.7 + K-041) and Bash-tool backgrounding (discarded, already in `claude-code.md`) both now named |
| `devops` | 13 | "13 entries" | all 13 already described in prose, header corrected |
| `tico` | 4 | "4 entries" | 2026-07-31 `version`/`defVersion` entry now discarded, "already tracked as K-040" |

All four headers reconcile with the diff. `teco`'s headless-continuation-block note is preserved
verbatim in `teco/kaizen/history.md`.

### M-2 — closed, verified; judgment call on scope-widening reviewed and accepted

`skills/python-web-quirks/SKILL.md`'s `description:` frontmatter now names all 8 topics (asyncio
`create_task`, `BackgroundTasks`, `exclude_unset`, `urllib` taxonomy, OpenAI-compat 200-envelope,
fence-fragile `json.loads`, `monkeypatch.setenv`, function-local import binding) — read the full
file, confirmed the body has exactly these 8 `## ` entries and the description covers each.
`skills/README.md`'s catalog row was updated to match (diffed side by side, consistent).

**On the scope-widening call (M-2 open item (a) in the brief):** `cobb` widened the skill's stated
scope ("mostly web/async, plus two general pytest/import-timing traps") instead of splitting the
two non-web entries into a new skill. I find this defensible, not a defect: the two entries
(`monkeypatch.setenv` timing, function-local import binding) surfaced in the same Python-web
codebases the skill already targets, the consumer roster (`coder`/`tdd-engineer`/`architect`/
`analyst`) doesn't change, and a 2-entry skill for a narrow pytest-timing niche would add a whole
new `SKILL.md`/catalog-row/frontmatter-description overhead for content two paragraphs long. The
skill's name (`python-web-quirks`) is now a step ahead of its full scope — a reader skimming just
the name, not the description, could still miss that it covers general pytest timing traps — but
the description is the load-bearing surface per `skills/agent-standards/claude-code.md:203`
("only the `description` is loaded at startup"), and that surface is now accurate. Worth a rename
to something like `python-quirks` if the skill picks up a third non-web entry, not before.

### M-3 — closed, verified

`claude/AGENTS.md`'s roster now carries the KB parenthetical for `devops` (`ops-quirks.md`),
`qa-engineer` (`qa-testing-techniques.md`), `data-scientist` (`lm-studio-model-notes.md`), and
`analyst` (`review-techniques.md`, previously never annotated despite pre-existing). Read the full
diff — the `graph-dba` two-KB pattern this cited as precedent is unchanged and the new entries
follow its shape.

### M-4 — closed, verified (both halves)

`cypher-mcp/server.py:20-22`'s docstring now reads "the reported row count is exact below FalkorDB's
`RESULTSET_SIZE` (default 10000), at or above which it is itself a cap" — consistent with the
already-corrected `cypher-mcp/README.md` and `skills/cpg-analysis/SKILL.md` wording. `claude/graph-dba/
kaizen/history.md`'s entry now has an explicit "M-4 follow-up, verified closed" paragraph
confirming the `server.py` fix and listing all three now-consistent sites; the "Docs touched" line
lists `server.py` alongside `README.md`, correcting the previous omission.

### M-5 — closed, verified

`claude/teco/teco.md` step 2 ("Decompose & sequence") now carries: "**Unit size is not free — apply
the step-table sizing rule (§3, *Size each dispatch…*) when you draw the units, not when you
dispatch them:** by step 3 the ledger already records a unit at whatever size it was drawn here, so
re-splitting there is exactly the friction this rule exists to avoid." This is the suggested fix
almost verbatim. The full rule still lives once, at step 3 — confirmed no duplication.

### m-1 through m-4, n-1/n-2 — all closed

- **m-1:** `teco.md`, `teco/kaizen/history.md`, and `cobb/kaizen/history.md`'s 2026-08-11 entry all
  now read "3 of the 11 test files the plan names — 3 of the 5 rewired consumer bindings" (three
  sites, same source typo, all three fixed together per `cobb`'s own history entry).
- **m-2:** `falkor-chat/docs/QUERIES.md:963-965`'s new prose collapsed to one sentence — "`start.key`
  is a grouping key, not an engine-level constant: the one-row collapse is a cardinality
  **premise**, see the callout below" — and the "the schema guarantees" overstatement is gone
  (the rewrite doesn't restate the claim at all, so it can't overstate it — a cleaner fix than my
  own suggested wording, which would have kept a version of the claim).
- **m-3:** `falkor-chat/docs/DESIGN.md` — the new 5-bullet "QA/acceptance-testing gotchas" block now
  sits at line 1022, *after* the pre-existing K-042 bullet at line 1011 (confirmed via `grep -n`).
  `claude/qa-engineer/kaizen/history.md` now says "five" and names the MCP `send_message` bullet as
  the fifth.
- **m-4 (take or leave):** `cobb` made an explicit judgment call to keep the ~180 words of K-042
  forensics as-is in `teco.md`, reasoning that the stakeholder's verbatim quote is what keeps the
  rule from eroding — this was flagged as take-or-leave in Pass 1 and citing the same
  counter-argument I raised there. Not re-litigating.
- **n-1/n-2:** `claude/graph-dba/kaizen/history.md` no longer claims the no-string-repetition
  promotion (now solely `coder`'s) or the `db.indexes()` discard (now solely `teco`'s) — both
  removed with a note explaining the reattribution, matching the nit's suggested fix.

### PII leak — closed, verified independently

`git grep -n '/home/[a-z]'` (case-sensitive lowercase to catch a literal path, not the placeholder)
against `claude/analyst/kaizen/inbox.md` and this document returns nothing; both now read
`/home/<user>/...`. `bash claude/scripts/audit-team.sh`, run by me fresh: `RESULT: PASS`, including
check 7 ("no personal identifiers... in any tracked or untracked (non-ignored) file"). `cobb`'s
history entry frames this as an incidental fix outside the findings list (it was — my own review
document and a same-session `analyst`-inbox entry leaked the path, not something Pass 1 flagged as
a finding) and correctly leaves `analyst`'s two new 2026-08-11 inbox entries otherwise unprocessed,
since they postdate the distillation this review gated.

### Scoping-call check — coder/tdd-engineer convergence

`claude/coder/kaizen/history.md`'s entry states it "made the call to converge on suite-reporting
discipline specifically" between `coder` and `tdd-engineer`, framing it as resolving Pass 1's open
question 3. Checked the actual diff, not just the prose: `coder.md`'s only change is the one-line
step-5 edit (skip-count + attributed-delta clauses); `tdd-engineer.md` has zero diff. So the
*substance* of the change is exactly the narrow B-1 promotion the brief authorized — a single
clause mirroring a rule `tdd-engineer.md` already had — not a merge of the two prompts' broader
disciplines (TDD-cycle narration, etc.), which remains genuinely untouched. The *framing* in
`coder`'s history entry ("made the call... to converge") reads more sweeping than the one-line diff
it describes, and could mislead a future reader into thinking a broader convergence decision was
made here. Not a defect worth reopening — `teco`'s own U2-verification note in the coordination doc
already caught and corrected this exact framing gap ("the brief only authorized the narrow B-1
promotion... `cobb`'s own history entry confirms it explicitly did **not** merge the two prompts'
broader disciplines"), and the substance is right. Recording it here so it doesn't quietly become
the account of record without the caveat.

### What's solid (Pass 2 additions)

- **The fix pass is unusually well self-documented.** `claude/cobb/kaizen/history.md`'s 2026-08-12
  entry maps every Pass-1 finding (B-1 through n-2) to a concrete disposition, states two judgment
  calls explicitly with reasoning (M-2 scope-widening, m-4 keep-as-is) rather than silently picking
  one, and separately logs the incidental PII fix and what it deliberately left untouched (open
  questions 1 and 3). This is the same discipline the original distillation's history entries
  should have had from the start (the root cause of B-1/M-1) — good that it shows up here in full.
- **No scope creep.** Diffed `cobb`'s "Docs touched" list against `git status` — every touched file
  traces to a Pass-1 finding, `teco`'s own M-4 pre-fix, or the incidental PII discovery. Nothing
  unexplained.
- **The coordination doc's own U2 spot-check (`teco`, 2026-08-12) independently corroborates
  several of the same findings I re-checked here** (M-3's four KB annotations, B-1's corrected
  count, the m-2/m-3 doc fixes, the PII re-check) — two independent passes landing on the same
  "closed" conclusion for the same evidence is stronger than either alone.

### Open questions (Pass 2)

Unchanged from Pass 1's items 2 and 3 — both correctly left open for `teco`/the stakeholder rather
than acted on in this fix pass; nothing new surfaced in Pass 2 beyond the rename follow-up already
noted in item 1 above (the coordination doc's own stale forward-reference at
`kaizen-inbox-distillation2-coordination.md:10`, out of this document's write scope to fix).
