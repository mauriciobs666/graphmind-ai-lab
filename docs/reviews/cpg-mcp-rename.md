# CPG MCP server/tool rename — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (`docs/requirements/cpg-mcp-rename.md`)

## Scope & verdict

Plan-gate review of `docs/plans/cpg-mcp-rename.md` (architect, U1 of
`docs/plans/cpg-mcp-rename-coordination.md`) against `docs/requirements/cpg-mcp-rename.md`
(Status: Ready for design, FR-1…FR-7 / AC-1…AC-6). Static review only — no file under review was
changed. Verified by: reading both documents in full; reading `cpg/mcp/server.py`'s configuration
and fixed-string blocks, `cpg/mcp/image-tag.sh` in full, `cpg/mcp/README.md`'s relative-link
lines, `.mcp.json`/`.claude/settings.json`, `claude/analyst/analyst.md` and
`claude/architect/architect.md`'s `tools:` lines, `skills/cpg-analysis/SKILL.md`'s frontmatter,
`docs/reviews/cpg-mcp-joern-agent-string-fix.md` in full, `docs/BACKLOG.md`/`docs/HISTORY.md`
excerpts, and `docs/plans/generic-cypher-mcp.md`'s cited precedent lines; running the plan's own
stated discovery `git grep` pattern against the live tree (both as literally specified and against
hand-picked candidate misses); running `cpg/mcp`'s offline pytest suite and `build.sh
--verify-inputs` from the current, pre-rename location to confirm the plan's cited baselines.

**Verdict: needs changes.** One blocker: the plan's own discovery/verification mechanism — the
piece it explicitly presents as the fix for the M5 "B1" fixed-list failure — has a reproducible
blind spot for bare, unquoted `cpg` tool-identity references, including inside the MCP server's
own self-description text presented to every connecting agent. One major: the self-referential
exemption list omits this very review document, which (per this delivery's own timeline) will
exist on disk before the docs sweep runs and would otherwise get incorrectly rewritten. Both are
narrow, mechanically fixable gaps in an otherwise well-grounded, well-reasoned plan — not a
redesign.

**CPG:** considered, not relevant — this is a text/config/build-script rename across the monorepo's
own docs, prompts, and MCP plumbing, not a question about application code semantics, matching the
plan's own CPG line. Independently confirmed live via `mcp__cpg__query` (`GRAPH.LIST` probe): the
only loaded graphs are `ws:test, cpg_falkorchat, reference, ws:qa-tico-workflows-manual, ws:acme,
cpg_salesperson, ws:eval, kaizen_graph_dba` — none represents this repo's own docs/scripts/prompts.
All verification in this review was direct file reads, `git grep`, and running the existing test
suite/build script — no call-graph or data-flow question was in scope.

---

## Findings

### Blocker

**B1 — The §3.2 discovery/verification `git grep` pattern misses bare, unquoted `cpg`
tool-identity references, including the running server's own self-description.** The plan's
pattern is:

```
mcp__cpg__query|cpg/mcp|"cpg"|CPG_MCP_|cpg-mcp|cpg_mcp_
```

I ran this exact pattern (with `-E`, matching the plan's own invocation) against isolated lines
that are plainly tool-identity prose — a bare backtick-quoted `` `cpg` ``, not adjacent to `/mcp`,
an underscore, a hyphen, or a double quote — and confirmed no match (exit 1, i.e. "no hit"):

- `cpg/mcp/server.py:2` — the module's own docstring: `"""The \`cpg\` MCP server — one tool,
  \`query\`, over a named FalkorDB graph.` This exact line is the plan's own §2 worked example of
  "same docstring, two categories of cpg" — but the discovery grep never surfaces it as a hit in
  the first place, so rule 5's disambiguation is never even applied to it.
- `cpg/mcp/server.py:131` — worse: this is inside `SERVER_INSTRUCTIONS`, the literal string passed
  to `FastMCP(name="cpg", instructions=SERVER_INSTRUCTIONS)` (line 680) and therefore the text
  Claude Code shows an agent inspecting the connected server (`/mcp`, or the harness's own
  server-instructions surface). Its opening sentence reads `"The \`cpg\` server exposes a single
  tool, \`query\`: ..."` — after the rename this server is literally named `cypher`, so this string
  would keep telling every connecting agent the server is called `cpg`. This is the single most
  load-bearing miss: it directly perpetuates the exact "name doesn't match identity" confusion this
  whole delivery exists to fix (per the requirements doc's own Problem statement and first user
  story), and it is not just stale prose — it is functionally what the tool tells the model about
  itself.
- `docs/BACKLOG.md:240` — `` - **C-310 — OpenCode + Kiro MCP wiring for the `cpg` server.** ``
  (still-open backlog item, 🔵) — no match.
- `docs/BACKLOG.md:376` — `` - **C-320 — Containerize the `cpg` MCP server.** ✅ **Delivered
  2026-07-26.** `` — a living-log entry the plan's own §3.2 rule 4 explicitly puts in scope for
  surgical substitution — no match.
- `docs/HISTORY.md:465` — `` ## 2026-07-26 — The `cpg` MCP server is containerized (C-320) ✅ `` —
  same living-log category, same miss.
- `claude/graph-dba/falkordb-quirks.md:277` — `` v4.18.11, via the `cpg` MCP tool vs. raw
  `GRAPH.RO_QUERY`.) `` — an `active` agent-knowledge document not even named anywhere in the
  plan's §2 inventory or §4 step-table file lists, so nothing about the plan's design would draw
  an implementer's attention to it independent of the grep.

I also verified the *shape* of the miss systematically: diffing every file containing a
word-boundary `cpg` token against every file the plan's own pattern matches turns up ~45 files with
no pattern hit at all in the whole file (not just the specific line) — most are correctly
CPG-domain-only (`cpg-analysis`/`joern-cpg` mentions, `cpg_<component>` graph names, which are
FR-7-protected and must *not* change), but the six instances above are confirmed tool-identity and
confirmed missed.

**Why this is a blocker, not a minor gap.** §3.2 and §5 both state the *same* pattern is the final
proof gate: *"re-run the same `git grep` after every step lands... any other surviving hit is a
defect"* (§3.2), and AC-1's check is literally this grep (§5). A verification mechanism that shares
its blind spot with the discovery mechanism it's meant to audit will report a clean pass while
these (and structurally similar) stale self-descriptions survive — the exact "fixed-artifact
silently under-covers" failure mode (M5's B1) this design explicitly set out to not repeat, now
reproduced one layer down, in the pattern itself rather than in a transcribed file list.

**Suggested fix.** Add a bare-word alternative to the discovery pattern — e.g. append `` |\bcpg\b ``
(case-insensitive, so it also matches `CPG`) to the existing alternation — then run the *existing*
§3.2 rule-5 disambiguation (tool-identity vs. CPG-domain) against the wider hit set. Rule 5 itself
is sound wherever it actually gets applied (confirmed against `server.py`'s and
`cpg-getting-started.md`'s worked examples below); the fix is exclusively about making sure every
tool-identity hit reaches it. Re-verify the `--verify-inputs`/offline-suite/AC checks are otherwise
unaffected — they are, since this only changes what counts as "in scope for review," not the
tool's mechanics.

### Major

**M1 — The self-referential exemption list omits this delivery's own review document (and, by the
same logic, its test-plan/test-report), which is a real gap given the actual production order.**
§2/§3.2 step 2 name exactly three exempt documents: `docs/requirements/cpg-mcp-rename.md`,
`docs/plans/cpg-mcp-rename.md`, `docs/plans/cpg-mcp-rename-coordination.md`. This review,
`docs/reviews/cpg-mcp-rename.md`, is produced *before* step 3b's docs sweep runs (the plan-gate
review is U2, dispatched alongside/ahead of implementation per the coordination doc's own unit
ledger), will carry `Status: active`, and is — by design — full of literal `mcp__cpg__query` /
`"cpg"` / `cpg/mcp` quotations describing what is being renamed *from*, exactly the same
"renamed-from-X" narration the plan protects the plan/requirements/coordination trio for. Under
the plan's own step-3 classification (*"Anything else (`active`...) → in scope, go to step 5"*),
`cobb`'s step-3b sweep would rewrite this review's own findings — including, ironically, this very
finding's quoted evidence — corrupting it exactly as editing the plan document itself would. This
repo's own family convention (root `AGENTS.md`: *"the same slug across several kinds is the
family... required, not merely tolerated"*) already implies the whole
`docs/{requirements,plans,reviews,test-plans,test-reports}/cpg-mcp-rename*.md` family shares the
same "this document's subject is the rename itself" status; the plan's exemption list should say so
explicitly rather than naming three specific paths. (`qa-engineer`'s step-4 test-plan/test-report
are produced *after* step 3b per the dependency graph, so they're not at the same immediate risk
this run — but naming the whole family closes the gap for good rather than by lucky timing.)

**Suggested fix.** Widen §3.2 step 2's exemption to: *"any document under
`docs/{requirements,plans,reviews,test-plans,test-reports}/` whose basename starts with
`cpg-mcp-rename`"* — a one-line rule change, no new judgment call for the implementer.

### Minor

**m1 — §3.2 step 3's classification glob names a `docs/plans-coordination/` directory that does
not exist.** The rule reads: *"Does the hit's own document carry a `Status:` header
(`docs/{requirements,plans,plans-coordination,reviews,test-plans,test-reports}/*.md`)?"* — but this
delivery's own coordination doc lives at `docs/plans/cpg-mcp-rename-coordination.md`, inside
`docs/plans/`, not a sibling `docs/plans-coordination/` tree (confirmed: no such directory exists
anywhere in the repo). Harmless in practice, since actual discovery is the repo-wide `git grep`,
not an iteration over this glob — but worth a one-line correction so the classification rule
doesn't describe a path that isn't real.

**m2 — §6's fourth open item overstates the risk in `docs/reviews/cpg-mcp-joern-agent-string-fix.md`
and can be resolved outright rather than left open.** I read the file in full and confirmed all
four of its pattern-matching hits (lines 1, 7, 8, 60) are plain path citations —
`` `cpg/mcp/server.py` ``, `` `cpg/mcp/tests/test_server.py` `` — naming *where the fixed code
lives*, not quoted identity-string literals under review. The document's actual quoted evidence
concerns a `"joern agent"` → `"graph-dba agent"` phrasing fix, unrelated to this rename. Updating a
path citation to reflect the file's new location is the same, unambiguous class of edit as any
other doc's path reference elsewhere in this sweep — not "rewriting what a specific historical diff
said." Recommend closing this open item as "approve the substitution, no reviewer override needed"
rather than carrying it forward as a standing question.

---

## §6 open items — reviewer's view (as requested)

- **M6→M7 renumbering bump, folded into step 3b.** Sound. `teco`'s coordination doc already
  establishes the ordering (this rename lands before `generic-cypher-mcp2` design starts
  specifically so M6-proper is designed against the final name), so the numbering collision is real
  and the resolution — bump the sibling doc's header — is the only one that avoids a guaranteed
  `docs/BACKLOG.md` defect. Confirmed no existing `C-6xx` entries collide (current ceiling is
  `C-507`). One suggestion, not a blocker: since `generic-cypher-mcp2.md` is owned by `tico` per
  root `AGENTS.md`'s owner table, step 3b's done-condition should note the edit is informational to
  `tico` (a one-line header-metadata bump, not a `Status:` flip, so it doesn't require `tico`'s
  hand under the letter of the convention — but a heads-up avoids surprise).
- **Surgical substitution inside living logs.** Sound, and correctly grounded: AGENTS.md's
  archived/active split has no dated-vs-undated carve-out, and `BACKLOG.md`/`HISTORY.md` never
  carry the `Status:` header that would exempt them. Agree with the plan's rejection of the
  alternative ("never touch a dated entry") — a living `BACKLOG.md` that still names a tool that no
  longer exists is exactly the confusion this rename exists to prevent for a reader using it as a
  current reference, as the plan argues.
- **`CPG_MCP_*`→`CYPHER_MCP_*` env vars and `cpg-mcp`→`cypher-mcp` Docker image/label rename.**
  Sound. These are "cpg" spelled into the tool's own implementation surface (`docker ps`, `docker
  image ls`, an operator's `.env` override), not CPG-domain vocabulary — leaving them stale would
  produce exactly the half-renamed state FR-6 exists to prevent, just in a place FR-1…FR-3's literal
  text doesn't individually name. Agree with the plan's reading of the decision log's "Full
  identity, everywhere."
- **Rewriting quoted historical content in `docs/reviews/cpg-mcp-joern-agent-string-fix.md`.**
  See m2 above — resolved, not merely opinable: the actual hits are path citations, not quoted
  string content, so there's nothing here to second-guess.

---

## What's solid

- **`image-tag.sh`'s location-independence claim is correct.** Read the file in full:
  `CPG_MCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` and every hashed path
  (`cpg_mcp_input_files`/`cpg_mcp_input_dirs`) is relative to that variable, never to the repo root
  or an absolute path. A directory move changes nothing about *how* the hash is computed, only the
  renamed *bytes* change its *value* — exactly the intended one-time rebuild, not a broken gate.
- **The §5 AC↔check table is complete on paper** — all seven FRs map into the design (§3.1–§3.6)
  and all six ACs get a concrete static-or-live check; no FR or AC is silently dropped.
- **Rule 5's tool-identity/CPG-domain disambiguation is sound wherever it actually gets applied.**
  Spot-checked both worked examples the plan cites — `server.py`'s `TOOL_DESCRIPTION`/
  `SERVER_INSTRUCTIONS` blocks and `docs/manuals/cpg-getting-started.md:65-67` — and confirmed the
  rule correctly separates the two categories in both. (The manual's example is, in practice, safe
  from B1's blind spot: its bare `` `cpg` `` mentions on lines 65-66 sit in the same paragraph as
  `` `cpg/mcp/build.sh` `` on line 67, which *does* match the pattern, so the paragraph gets pulled
  into review anyway — unlike the isolated instances in B1, which have no such neighbor.)
- **Baselines are accurate, not invented.** Ran `cpg/mcp`'s offline suite from its current,
  pre-rename location: `84 passed, 7 deselected` — matches §2/§5's cited figure exactly (note:
  this requires the default `pytest.ini` `addopts = -m "not live"`; a plain `-k "not live"` gives a
  different, wrong split). Ran `cpg/mcp/build.sh --verify-inputs`: passes today, consistent with
  the plan's claim that it should keep passing post-rename since file existence doesn't change,
  only content. Also verified the `generic-cypher-mcp.md` step-4a/4b precedent this plan cites for
  its own step-sizing justification: confirmed 3-5 named files per step against "on the order of 30
  hits," the exact figures quoted in §4.
- **Step-sizing reasoning (steps 1/3a/3b exceeding the ≤5-file guideline) is well-argued and
  precedented**, contingent on B1's fix landing first — a large step's done-condition is only as
  safe as the grep it's gated on.
- **FR-7 preservation design is careful.** The plan correctly separates tool-identity from
  CPG-domain vocabulary in its mapping table (§3.3) and names a dedicated reverse-direction proof
  gate (unchanged count/content for the FR-7 preservation grep) — a good check *in addition to* the
  forward gate, not a substitute for fixing B1.

## Open questions

None beyond what's already captured in the findings above and the reviewer's-view section — B1 and
M1 are both narrow, mechanically specified fixes; nothing here requires stakeholder input to
resolve.

---

## Pass 2 — 2026-08-19 (re-gate of Version 1.1)

Re-read `docs/plans/cpg-mcp-rename.md` (now `Version: 1.1`, §7 revision note) in full. Re-derived
every claimed fix myself — live regex runs against the current tree, content spot-checks, and one
empirical test of `git grep`'s tracked/untracked behavior — rather than accepting the architect's
self-report.

**Verdict: approve with suggestions.** The blocker (B1) and the major (M1) are both genuinely
closed, verified by direct reproduction, not just by reading the revision note. The two minors are
also closed. Two new, non-blocking observations surfaced during re-verification — neither
undermines B1/M1's fixes; both are cheap, optional hardening.

### B1 (blocker) — verified closed

Ran both variants of the widened pattern live against the current tree:

- Case-insensitive `` |\bcpg\b ``: **141 files** (plan claims 141 — exact match).
- Case-sensitive `` |\bcpg\b `` (as shipped): **135 files** (plan claims 135 — exact match).
- Pre-fix pattern (no bare-word alternative): **95 files** in my environment (plan/Pass-1 baseline:
  94) — the +1 is my own Pass-1 kaizen-inbox entry (`claude/analyst/kaizen/inbox.md`, a *tracked*
  file I edited between passes, which now itself contains `cpg/mcp` path citations) — not a plan
  defect; confirmed by diffing the hit-list against my saved Pass-1 snapshot.

**All 6 of Pass 1's confirmed misses now match** the case-sensitive pattern (re-ran the pattern
against each exact line): `cpg/mcp/server.py:2`, `cpg/mcp/server.py:131` (`SERVER_INSTRUCTIONS`),
`docs/BACKLOG.md:240`, `docs/BACKLOG.md:376`, `docs/HISTORY.md:465`,
`claude/graph-dba/falkordb-quirks.md:277` — all six now hit, zero remain silent misses.

**The case-sensitivity claim is correct, not just plausible.** Diffed the case-insensitive (141)
and case-sensitive (135) hit-lists: exactly 6 files differ, and I read every one —
`falkor-chat/docs/BACKLOG.md`, `falkor-chat/docs/plans/must-post-engine-contract.md`,
`falkor-chat/docs/requirements/workflow-dependence-overlay.md`,
`falkor-chat/docs/reviews/must-post-engine-contract{,-impl}.md`,
`kiro/docs/plans/kiro-demo-agent-coordination.md`. Every one of the 6 is confirmed noise — either
the uppercase `` `CPG:` `` deliverable evidence-trail convention line or "Code Property Graph
(CPG)"/"CPG-style" acronym-definition prose, never a lowercase tool-identity mention. The
case-sensitive choice is empirically the better trade, exactly as the revision claims.

**Rule 5's new bare-token sub-bullet and its 2 new exclusions disambiguate correctly on real
hits.** Diffed the case-sensitive-widened list (135) against the pre-fix list (95) to get the ~40
newly-surfaced files and read a representative sample beyond what Pass 1 already checked:
`cpg/.gitignore` (its `cpg.bin` mention — explicitly named by the revised rule's own text, correctly
domain), `docs/requirements/cpg-query-access.md` (`Status: archived`, moot regardless of
domain/identity classification), `mcp-monitor/docs/requirements/mcp-monitor.md` (a real
tool-identity hit — *"any change to falkor-chat's, cpg's, or any other existing MCP server"* — but
**`Status: archived`**, so correctly out of scope via rule 3 before rule 5 is ever reached),
`falkor-chat/docs/reviews/guard-judge-calibration.md` (`Status: active`, but its one case-sensitive
hit is `/tmp/cpg-src/falkor-chat-server` — a Joern build tmp-path, genuinely domain, correctly left
alone), and `skills/agent-standards/opencode.md:163-164` (*"the `cpg` server is wired for Claude
Code only"* — genuine tool-identity, correctly in scope, will be swept by step 3a's
`skills/{...}/**` target). No misclassification found in the sample.

### M1 (major) — verified closed

The basename-prefix rule (`docs/{requirements,plans,reviews,test-plans,test-reports}/` +
basename starts with `cpg-mcp-rename`) does cover `docs/reviews/cpg-mcp-rename.md` (the gap Pass 1
found) — confirmed by construction, the file's basename literally starts with `cpg-mcp-rename`.
Checked for over-exemption: `find . -iname 'cpg-mcp-rename*'` repo-wide returns exactly the 4
expected family members (`requirements/`, `plans/`, `plans/…-coordination.md`, `reviews/`) and
nothing else — no unrelated document accidentally swept under the wider prefix rule.

### Minors — both verified closed

- **m1** (nonexistent `docs/plans-coordination/` directory): §3.2 step 3 now reads "there is no
  separate `docs/plans-coordination/` directory; coordination docs live inside `docs/plans/`
  itself" — correct, matches the actual repo layout.
- **m2** (`cpg-mcp-joern-agent-string-fix.md` open item): §6 now closes it outright ("Resolved, not
  open: approve the substitution in step 3b, no reviewer override needed"), reflecting Pass 1's
  finding verbatim rather than leaving it as a standing question.

### §6 confirmations — wording checked against what I actually concluded

Both are accurate. The M6→M7 confirmation states the C-6xx-ceiling and ordering reasoning
correctly and folds in the suggested `tico`-notification note as I proposed, without overstating it
into a plan-text change ("no plan text change needed beyond noting it here" — accurate, that's what
I suggested). The env-var/Docker-namespace confirmation restates my own reasoning (implementation
surface vs. domain vocabulary, FR-6 half-renamed-state risk) without adding anything I didn't say.

### New, non-blocking observations (Pass 2)

**Minor — step 3a's Files column glob (`claude/*/*.md`) doesn't reach `claude/docs/requirements/
security-expert.md`, a real live miss, but the step's done-condition saves it anyway.** This
two-levels-deep file (`claude/docs/requirements/`, not `claude/<agent>/`) is the repo's only file
under `claude/docs/`, carries `Status: Ready for design` (in scope, not archived), and line 124
reads *"use the project's existing Code Property Graph (the `` `cpg` `` MCP tool /
`cpg-analysis` skill pattern..."* — a genuine, confirmed tool-identity hit requiring the rename.
Neither step 3a's Files column (`claude/*/*.md` only globs one level deep) nor step 3b's
(scoped to root `docs/`) names it. It survives anyway because step 3a's done-condition is worded
generically — *"every remaining hit **under `claude/`** or `skills/`"* — not restricted to the
literal Files-column enumeration, so a `cobb` run that actually re-runs the grep as instructed
would still be forced to resolve this hit. (I also checked the analogous risk for `claude/*/kaizen/
plan.md` — omitted from the `{history,inbox}.md` glob the same way — and for the mcp-monitor/
falkor-chat files step 3b names only a fixed subset of: both turned out to have no live tool-identity
content, so no actual miss there, just a narrower-than-ideal Files column with no consequence.)
Since the done-condition already covers it, this doesn't warrant reopening the blocker — but the
Files column and done-condition scope disagreeing is exactly the inconsistency the plan's own
stated philosophy ("the grep is the truth, not the file list") argues against tolerating. Suggested
fix: reword the glob to `claude/**/*.md` (or note explicitly that it recurses into `claude/docs/`)
so the two don't have to be reconciled by an implementer's diligence.

**Suggestion — the discovery `git grep` is tracked-files-only by default, which is currently
invisible-but-harmless for this delivery's own documents, and cheap to close.** Empirically
verified: `git grep '<marker>'` finds nothing in a brand-new file until it is `git add`ed (even
`git add` alone, no commit, is enough — confirmed by staging a throwaway test file, immediately
resetting it after). `docs/plans/cpg-mcp-rename.md`, its coordination doc, and this review are all
currently untracked (`git status --porcelain` shows `??` for all three), so the discovery command as
literally written never sees them — which happens to be harmless today only because rule 2's family
exemption protects exactly these documents regardless of whether the grep would ever surface them.
This is a coincidence of scope, not a designed safety net: a future contributor who creates a
genuinely in-scope new file and leaves it unstaged at gate-check time would be silently skipped by
this exact mechanism. Suggested fix: add `--untracked` to the discovery command (§3.2, §5's AC-1
row) — zero cost, and it removes "was this new file staged yet" as a variable in what the plan
otherwise designs as a fully mechanical, non-interpretive gate.

### Pass 2 conclusion

Both gating findings from Pass 1 are genuinely fixed, not just narrated as fixed — reproduced the
regex behavior, the file classification, and the exemption-rule coverage independently rather than
trusting §7's revision note. The two new observations are real but both have a working fallback
today (the done-condition's broader wording; rule 2's independent protection), so neither rises to
blocker/major. Recommend folding both into the next revision if one happens anyway, but they should
not hold up dispatch.
