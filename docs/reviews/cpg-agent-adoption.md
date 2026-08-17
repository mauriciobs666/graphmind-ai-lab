# Review — CPG agent adoption (plan gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** cpg-agent-adoption (M4)

## Pass 1 — Plan gate

**Artifacts reviewed:**
- [`../requirements/cpg-agent-adoption.md`](../requirements/cpg-agent-adoption.md) — baseline (FR-1…FR-9, AC-1…AC-6).
- [`../plans/cpg-agent-adoption-graph.md`](../plans/cpg-agent-adoption-graph.md) — `graph-dba`, FR-5/FR-6(mechanical)/FR-7/FR-8, the `:CpgBuildInfo` freshness marker.
- [`../plans/cpg-agent-adoption.md`](../plans/cpg-agent-adoption.md) — `cobb`, FR-1/FR-2/FR-3/FR-4/FR-6(surfacing)/FR-9 and AC-1/AC-2/AC-5/AC-6, the six-agent roster and discovery/evidence-trail design.
- Cross-checked against [`../plans/m2-cpg-analysis-skill.md`](../plans/m2-cpg-analysis-skill.md), [`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md), the delivered `skills/cpg-analysis/SKILL.md`, `skills/joern-cpg/references/cpg-model.md`, `skills/joern-cpg/scripts/pipeline.sh`, all six affected agent files (`claude/{analyst,architect,qa-engineer,coder,tdd-engineer,frontend-engineer}/*.md`), `claude/README.md`, `skills/README.md`, root `AGENTS.md`, `skills/agent-standards/claude-code.md`, `docs/BACKLOG.md`, and [`../plans/cpg-agent-adoption-coordination.md`](../plans/cpg-agent-adoption-coordination.md).
- Live-verified (read-only `mcp__cpg__query`): `cpg_falkorchat` METHOD count (2,904, matches graph-dba's grounding note), `cpg_salesperson` METHOD count (359, matches), absence of `:CpgBuildInfo` on both live graphs (matches the "no backfill" rollout claim).

This is a **static, pre-implementation review** — no code exists for this feature yet. Scope is the two design documents together, judged against the requirements doc and against the two prior milestones' consumer-scope boundary. No files were mutated; no destructive queries were run.

**Verdict: approve with suggestions.** Zero blockers. One major (a real but non-blocking traceability gap in the backlog proposal), two minors, one nit. The pair of documents composes coherently, is well-grounded in the live system, and the reconciliation with M2/M3 genuinely holds — not just as an assertion, but as verified fact.

---

## Findings

### Major

**MAJ-1 — `cobb`'s §7 "Requirements coverage" claim overclaims: FR-4 and AC-3/AC-4/AC-5 are never tagged on any C-4xx backlog item, contradicting "none are left uncovered."**

Evidence — the proposed `docs/BACKLOG.md` M4 section (`docs/plans/cpg-agent-adoption.md` §7) tags each item with the FR/AC it covers:

| Item | Tagged FR/AC |
|---|---|
| C-401 | FR-5/FR-6(mechanical)/FR-7/FR-8 |
| C-402 | FR-1 (skill-side) |
| C-403 | FR-2, FR-6 (surfacing) / AC-1 |
| C-404 | FR-1, FR-2, FR-3 / AC-1 |
| C-405 | FR-1, FR-2, FR-3 / AC-1 |
| C-406 | AC-2 |
| C-407 | *(none)* |

**FR-4, AC-3, AC-4, and AC-5 appear on no item at all**, yet the section's closing line reads: *"All of FR-1…FR-9 / AC-1…AC-6 map onto C-401…C-407 above; none are left uncovered."* That sentence is false as written — a direct grep of the seven item descriptions confirms no `FR-4`, `AC-3`, `AC-4`, or `AC-5` token anywhere in §7. Contrast with FR-9/AC-6, which the same section *does* explicitly carve out with a stated reason ("AC-6 (reconciliation) is satisfied by `plans/cpg-agent-adoption.md` §4, not by a separate backlog item") — FR-4/AC-3/AC-4/AC-5 get no such carve-out; they're just silently absent.

This isn't a design gap — the substantive content exists elsewhere in the same document: §2.2 ("one `mcp__cpg__query` call... satisfies AC-5") covers FR-4/AC-5, and §2.3 ("Bundling the freshness check into the same default step") covers AC-3/AC-4. It's purely a backlog-traceability defect. It matters because the coordination ledger routes `docs/BACKLOG.md`'s M4 section to `qa-engineer` at U6 as the acceptance-pass input (`cpg-agent-adoption-coordination.md`'s unit ledger), and this repo's own convention treats a "Requirements coverage" block as the thing a downstream reader traces ACs through — an AC with no backlog anchor is exactly the kind of silent gap this whole feature exists to prevent happening to *agents*, now happening to the *backlog* that governs them.

*Suggested improvement (owner: `cobb`, before U4b executes the write):* add `FR-4 / AC-5` to C-402's tag (the discovery-mechanic item is where "no noise on a miss" is actually delivered), add `AC-3, AC-4` to C-403's tag (surfacing integration lands there), and correct the closing sentence to name the FR-9/AC-6 carve-out explicitly rather than claim blanket coverage.

### Minor

**m-2 — `SKILL.md` §4's Navigation table "Consumer" column isn't in `cobb`'s task list, and will read stale immediately after M4 lands.**

Evidence — the live table (`skills/cpg-analysis/SKILL.md:243-248`) names consumers per recipe: `analyst, architect` for impact-analysis, `analyst` for rca/code-review, `qa-engineer` for test-gap. `cobb`'s own roster reasoning (§1 of the primary plan) explicitly ties `coder`/`tdd-engineer` to the impact-analysis use case ("'what calls this / what would break' … is exactly the question `coder` should be asking") and `frontend-engineer` to the same recipe via `cpg_salesperson`/`chatbot.py`. But the task list (§6 step 1) scopes the `SKILL.md` edit to only the frontmatter `description` and the §1 discovery-mechanic paragraph — it never mentions the §4 table. Left as-is, the skill's own internal consumer listing will still say "analyst, architect" the day after six agents are wired, which is precisely the "one doc says three, another says six" drift the feature is meant to close.

*Suggested improvement:* add one bullet to §6 step 1 (or fold into step 2's "reword existing wiring" pass): update the Navigation table's Consumer column per recipe to reflect the actual post-M4 roster (impact-analysis gains `coder`, `tdd-engineer`, `frontend-engineer`; leave rca/code-review/test-gap as-is unless a stated reason widens them too).

**m-3 — `cobb`'s §4 AC-6 reconciliation ("This plan adds zero new recipes and zero new skills") is scoped to `cobb`'s own slice and could mislead a reader skimming only that document.**

Evidence — `docs/plans/cpg-agent-adoption.md` §4 states plainly: *"M2's shape is untouched… This plan adds zero new recipes and zero new skills — the same four recipes simply gain three more callers."* True for `cobb`'s slice (U2) in isolation. But the sibling plan `graph-dba` delivers in the same milestone explicitly adds a new file, `skills/cpg-analysis/references/freshness.md`, and frames it in its own §2 as *"recommended as a fifth recipe file"* alongside the existing four. AC-6 is about the **consumer-scope boundary** (who can query, not the skill's file count), so this doesn't violate AC-6's literal requirement — and `graph-dba`'s own doc correctly scopes itself ("Nothing here changes who is wired… that's `cobb`'s slice"). But a reader who reads only `cobb`'s §4 (the section written specifically to satisfy AC-6's "states explicitly" requirement) could reasonably conclude the whole M4 feature leaves the skill's file surface untouched, which is not accurate for the combined feature.

*Suggested improvement:* add one clause to §4 naming the fifth reference file `graph-dba` adds, and stating explicitly that it's additive/orthogonal to the consumer-scope question AC-6 governs (a sentence, not a rewrite — the substance is already correct, just under-stated).

### Nit

**n-4 — the `cpg_<component>` naming guess is an observed operator convention, not `pipeline.sh`'s actual default.**

Evidence — `pipeline.sh:61` derives its own default graph name from `basename "$SRC"` (`cpg_$(basename "$SRC" | tr -cs 'A-Za-z0-9_' '_')`), not from the repo-root component directory name. The one recorded precedent for `cpg_falkorchat` (`cpg-query-access.md` §S8) built from a staged copy of `{falkorchat, tests}` and passed `--graph cpg_falkorchat` **explicitly** — the tool's own default would have produced something else (e.g. `cpg_server`, depending on the staged root's basename) had `--graph` been omitted. So `cpg_<component>` is a convention two operators happened to choose, not something the pipeline enforces. Low stakes: both the proposed `SKILL.md` paragraph (`cpg-agent-adoption.md` §2.2, "Guess that name first…") and `cobb`'s own §8 already hedge this correctly ("an observed pattern from two data points, not a stated contract anywhere else in the repo") and the fallback (not-found-error graph enumeration) genuinely covers a wrong guess. Recorded for completeness, not because it needs a design change.

---

## Grounding & soundness — what I verified

- **Live facts check out.** `cpg_falkorchat` = 2,904 methods, `cpg_salesperson` = 359 methods, neither graph carries a `:CpgBuildInfo` node today — all three match `graph-dba`'s design-time claims exactly, re-verified independently this session.
- **`META_DATA` absence and the shared `:CpgNode` label rationale are both accurate** against `skills/joern-cpg/references/cpg-model.md` — the "you'll see most" hedge (line 24) is real, not a contradiction as `graph-dba`'s design claims, and the label-per-index rationale (lines 54–67) for keeping `:CpgBuildInfo` off `:CpgNode` is the documented reason, correctly applied (the marker has no Joern id and no edges).
- **The `pipeline.sh` insertion point is exactly where the design says.** The `--load` block (lines 93–126) ends with the `--verify-prefix` loop, which `exit 1`s on failure *before* the block's closing `fi` — so a structurally-failed load genuinely never reaches the proposed stamp. `MERGE (b:CpgBuildInfo)` with no property in the match pattern is correctly idempotent across a `--reset` (graph deleted first, so `MERGE` creates fresh) and a same-graph re-load without `--reset` (finds and overwrites the existing marker) — I traced both code paths and the claim holds.
- **The `git` binary soft-degrade and the staged-source-loses-`SOURCE_COMMIT` limitation (graph-dba's own §6 flags, explicitly asking the reviewer to weigh in): both judgments are sound.** Hard-failing an entire CPG load over a missing `git` binary, for a metadata-only convenience field, would be disproportionate to the harm (the fallback — raw `BUILT_AT` age — still satisfies FR-5's "some indication of currency," just a weaker one). The staged-source gap is honestly documented with its real precedent (`cpg-query-access.md` §S8) and degrades to the same weaker-but-real fallback rather than silently lying. Neither needs a design change before implementation.
- **Six-consumer roster exclusions are individually defensible and internally consistent.** `devops` (Joern doesn't parse infra-as-code — true, Joern's frontends target application source languages), `cobb`/`teco`/`tico` (all match their own catalog descriptions in `claude/README.md` verbatim — "coordinates, doesn't design solutions," "product altitude only… never HOW"), `data-scientist` (routes general-correctness code reading to `analyst`, which is already CPG-wired — a defensible judgment call, not an oversight; worth a light open question below).
- **§2.1's rejection of a root-`AGENTS.md` discovery line is grounded in a verified fact, not asserted.** `skills/agent-standards/claude-code.md:161-163` ("What loads into a subagent," verified 2026-06-20 against `code.claude.com/docs/en/sub-agents`) confirms: *"The full `CLAUDE.md`/memory hierarchy still auto-loads via the normal message flow."* So the claim that a root-`AGENTS.md` bullet would reach every agent regardless of `tools:`/`description` wiring — including the five ruled out in §1 — is correct, and the alternative chosen (per-agent `description`/body wiring) is the one that actually respects the roster decision made in the same document.
- **The file/line-anchor table (`cpg-agent-adoption.md` §2.4) is accurate as of this pass.** I independently re-read all six agent files and matched every cited line number and anchor text: `analyst.md:3` (description), `:45` ("Read the real thing"), `:61` ("Scope & verdict"); `architect.md:3`, `:36` ("Investigate the codebase first"), `:25` ("Context & findings"); `qa-engineer.md:3`, `:28` ("Read the sources of truth"), `:50` ("Summary"); `coder.md:18` ("Orient"), `:22` ("Verify and report"); `tdd-engineer.md:35` ("Understand first"), `:42` ("Verify honestly"); `frontend-engineer.md:11-19` (Orient-first list), `:57` ("Verify in the running UI"). Every anchor lands exactly where claimed — this is genuinely implementation-ready, not hand-waved.
- **`coder`/`tdd-engineer`/`frontend-engineer` all currently omit `tools:` frontmatter**, confirmed by direct read of all three files — the "no `tools:` change needed, they already inherit `mcp__cpg__query`" claim (§8) is correct, cross-checked against `qa-engineer` (also no `tools:`, already CPG-wired) as the stated precedent.
- **`claude/README.md` row numbers cited in the task list (§6 step 5) are exact** — architect=9, coder=10, tdd-engineer=13, frontend-engineer=14, qa-engineer=15, analyst=16, verified against the live file.
- **`docs/BACKLOG.md` numbering is genuinely free.** Highest existing item is C-323; no C-4xx exists; "M4" appears nowhere in `BACKLOG.md` or `HISTORY.md` today. The proposed section doesn't collide with anything.
- **FR-8/M3 read-path claim holds by simple absence**, not just assertion — neither plan proposes a single edit under `cpg/mcp/**`, `.mcp.json`, or the tool's contract; both designs are entirely additive (a new label + recipe file, six agent-prompt edits, catalog rows).
- **Scope discipline holds.** Neither document proposes an MCP tool change, an auto-rebuild trigger, a proactive build-out, or a usage-tracking mechanism. `graph-dba`'s explicit refusal to backfill or pre-emptively rebuild either live graph (§5) is the correct, reasoned application of the "no proactive build-out" out-of-scope line — a shorter, more convenient path (rebuild both graphs now so the marker exists sooner) was available and was correctly declined for exactly that reason.
- **The stale method-count drift `graph-dba` flagged (2,037 → 2,904, undocumented rebuild) doesn't mislead either plan's own numeric claims.** Both plans cite the corrected 2,904/359 figures consistently (I re-verified live); the requirements doc's 2,037 figure is left as historical context, not re-cited as current anywhere in either design.

## What's solid

- **AC-6's literal wording is satisfied, not just gestured at.** `cobb`'s §4 states, near-verbatim, exactly what AC-6 demands ("extends — not silently overrides"), and the claim is true on inspection: nothing either document proposes removes or narrows what `analyst`/`architect`/`qa-engineer` could already do, and the MCP tool's contract is untouched.
- **The freshness-check-doubles-as-existence-probe design (graph-dba §2, cobb §2.3) is genuinely elegant and I confirmed it works today.** A live `MATCH (n:CpgBuildInfo) RETURN n` against `cpg_falkorchat` returns a clean zero-row success — distinguishable from the tool's not-found error for a truly absent graph — so the single query correctly serves double duty (existence + freshness) without ambiguity, exactly as both documents claim.
- **The roster decision explicitly guards against its own obsolescence** — §1's framing ("a roster call that would flip the moment a third CPG existed would be the wrong kind of call to make") judges agents by the shape of their work, not today's two-graph snapshot, which is the right invariant for a decision meant to outlive the current coverage.
- **Sequencing and file-collision awareness are handled correctly.** `cobb`'s task list explicitly defers U4b until after U4a lands specifically because both units touch `skills/cpg-analysis/SKILL.md` (different sections — §1 vs. §4 — but the same file), and instructs re-reading fresh rather than trusting this design pass's line numbers. That's the right call and it's stated, not assumed.
- **The two open risks `graph-dba` flagged for the reviewer are both judged sound** (see Grounding above) — this review is the explicit confirmation both docs asked for.

## Open questions

1. **`data-scientist`'s exclusion (cobb §1) is a defensible judgment call, not an obvious miss — but worth a stakeholder sanity-check if the discipline's task shape ever grows deep pipeline-code correctness reads** (e.g. verifying an eval harness's chunking/retrieval logic structurally, not just methodologically). Today's routing — general-correctness code reading goes to `analyst`, which is CPG-wired — is sound as designed; flagging only because the brief asked whether any plausible in-scope agent was excluded without good reason, and this is the one exclusion that's a genuine judgment call rather than a structural fact (unlike `devops`'s Joern-can't-parse-infra reasoning, which is closer to a fact).
2. **MAJ-1's fix is mechanical** (add four missing tags, correct one sentence) and doesn't require a design change — recommend it land before U4b writes `docs/BACKLOG.md`, so the section that ships is the one `qa-engineer` will actually trace AC-3/AC-4/AC-5/FR-4 against at U6.

---

## Pass 2 — Diff-scoped code gate (U5)

**Scope.** The delivered diff for `cpg-agent-adoption` (M4), commits `35b108f` (U1–U4a: design
docs + freshness-marker mechanics) and `50f9aaa` (U4b-1..5: six-agent CPG-consumer wiring +
catalog/doc sync), reviewed against the already-approved plans
([`../plans/cpg-agent-adoption.md`](../plans/cpg-agent-adoption.md) §2.4/§3/§4/§6,
[`../plans/cpg-agent-adoption-graph.md`](../plans/cpg-agent-adoption-graph.md)) and the
requirements baseline ([`../requirements/cpg-agent-adoption.md`](../requirements/cpg-agent-adoption.md),
FR-1…FR-9/AC-1…AC-6). This is a **conformance check** — does the diff do what the gated plan
said, not a re-litigation of the design itself (Pass 1 already gated that, approve with
suggestions, all three findings fixed in place per U2-fix). Read both commits directly via
`git show`, not a paraphrase; read all six touched agent files and their `kaizen/history.md`
entries in full (not just the diff hunks) for in-context coherence; cross-checked
`docs/BACKLOG.md`, `docs/HISTORY.md`, `claude/README.md`, `skills/README.md`,
`skills/cpg-analysis/SKILL.md`, root `AGENTS.md`, and the coordination ledger. Ran
`bash claude/scripts/audit-team.sh` myself (not trusting the ledger's earlier claim of a clean
run) and `bash -n` on `pipeline.sh`. Live-verified one freshness query against `cpg_falkorchat`.
No files were mutated.

**Verdict: approve.** Zero blockers, zero majors, zero minors. One nit (all in-scope YAML
frontmatter quirk was found pre-existing, not diff-introduced — noted for completeness, not
actionable here). The diff is a faithful, complete, well-scoped implementation of the gated
plan — every one of the six agent edits verbatim-matches what §2.4 specified, the evidence-trail
convention landed exactly where §3 said, FR-8 holds by simple absence, and the milestone
bookkeeping is honest about what's actually done versus still queued.

`CPG: not applicable — this diff is agent/skill-prompt markdown and a shell script; no source
tree with a loaded CPG is under review here (the freshness-recipe correctness was verified
directly against the live `cpg_falkorchat` graph instead, see Grounding below).`

### Findings

None at blocker, major, or minor severity.

**n-1 (nit) — `tdd-engineer.md` and `frontend-engineer.md`'s YAML frontmatter fails a strict
`yaml.safe_load` parse (a bare colon inside an unquoted `description:` plain scalar), but this is
pre-existing, not introduced by this diff.** Evidence: `python3 -c "import yaml; yaml.safe_load(...)"`
against both files' frontmatter raises `mapping values are not allowed here` at the first
`: ` inside a clause (e.g. "the efficient path: a bug fix" in `tdd-engineer.md`). Re-ran the same
parse against `git show 50f9aaa^:claude/tdd-engineer/tdd-engineer.md` (the pre-diff version) —
identical failure, so the diff neither caused nor worsened it. `claude/scripts/audit-team.sh`
(which is grep-based, not a YAML parser) doesn't catch this shape, and Claude Code's own
frontmatter reader evidently tolerates it in practice (both agents were already deployed and
working before this diff). Recorded for completeness, not as an action item on this diff; if it
matters, it's a `cobb`/`agent-standards` question about the harness's actual frontmatter grammar,
not something U4b introduced or should have caught.

### Point-by-point

1. **Fidelity to the plan (§2.4/§3/§6).** Verified file-by-file against the plan's task table.
   Every one of the six agents got exactly the three edits specified — description reword,
   orientation-step addition (with the freshness-check bundling from §2.3), and the `CPG:`
   evidence-trail line in the deliverable skeleton — with the wording lifted near-verbatim from
   §2.2/§2.3/§3's own suggested phrasing. `skills/cpg-analysis/SKILL.md` got the frontmatter
   `description` widened to six consumers (983 chars, confirmed by direct count — under the 1024
   budget the plan flagged), the §1 `cpg_<component>` discovery paragraph, and the §4 nav-table
   impact-analysis row gaining `coder`/`tdd-engineer`/`frontend-engineer` (m-2's Pass-1 fix,
   correctly carried through — the other three recipe rows were correctly left untouched, per the
   plan's own "no stated reason to widen them" call). No drift found anywhere.
2. **Consistency across the six agents.** Five of six open with the verbatim-identical clause
   "Checks whether a relevant CPG exists as part of its normal orientation and, when one does,
   uses…" — confirmed by direct read of all six files, not just the diff hunks.
   `frontend-engineer`'s "Checks for a relevant Joern CPG (`cpg_salesperson` today) as part of
   that orientation and uses…" is the one flagged variance. I agree with the ledger's own
   assessment: same default-orientation semantics (explicitly *not* the old conditional "With a
   loaded Joern CPG…" framing that `coder`/`tdd-engineer` briefly shipped in and were corrected
   out of, per their kaizen addenda), and the plan's own §2.1/§2.4 mandates the *reframing*, not
   verbatim-identical prose — naming `cpg_salesperson` concretely is arguably clearer given
   `frontend-engineer`'s single-graph reality today. Non-blocking, confirmed.
3. **AC-1…AC-6, mapped against the diff.**
   - **AC-1** (default-orientation discovery) — satisfied structurally: all six agents' body
     prompts now carry an unconditional discovery step, not a reminder-gated one.
   - **AC-2** (spot-checkable evidence) — satisfied: the `CPG:` line convention landed in all six
     deliverable skeletons, verbatim to §3's three allowed shapes.
   - **AC-3/AC-4** (freshness signal + stale-surfacing) — satisfied at the prompt level: every
     orientation-step edit bundles the freshness check and instructs "note what it says… surface
     a refresh suggestion — not a silent rebuild — if it looks stale," matching §2.3's mandated
     wording almost word-for-word across all six files.
   - **AC-5** (no-noise-on-a-miss) — satisfied by construction: the discovery mechanic added to
     `SKILL.md` §1 is one cheap query with a defined miss path (fallback enumeration, then stop);
     nothing in the six agent edits adds ceremony beyond that.
   - **AC-6** (explicit extends-not-overrides statement) — re-confirmed independently. Plan §4
     states, in its own words: *"This plan **extends** — it does not override, narrow, or
     silently diverge from — the consumer-scope boundary `m2-cpg-analysis-skill.md` (M2) and
     `cpg-query-access.md` (M3) drew,"* with four concrete bullets backing the claim (M2's recipe
     count untouched, M3's read path untouched, only the consumer list/default-ness widens, both
     prior docs remain historically accurate as-written). This satisfies AC-6's literal wording,
     as Pass 1 already found — re-verified here because AC-6 is specifically a claim *about the
     plan document*, and the plan document delivered is the one gated (no drift between the
     gated §4 text and what's on disk).
   AC-1 through AC-5 are all prompt-level commitments, not runtime-observable from a static diff
   read — genuine behavioral confirmation (does an agent actually discover and use the CPG on a
   real task, does the evidence trail actually show up in a real deliverable) is U6's job, not
   this gate's. That's a scope boundary, not a gap in this review.
4. **FR-8 (no read-path change).** Confirmed independently: `git diff --stat` and `git log` for
   both commits against `cpg/mcp/server.py` and `.mcp.json` are empty — neither file is touched,
   named, or referenced for modification anywhere in either commit.
5. **No scope creep.** Both commits' own file lists (`git show --stat`) match the plan's §6 task
   list and the coordination ledger's per-unit deliverable list exactly. `35b108f` additionally
   touches `claude/graph-dba/kaizen/inbox.md` (two learnings-capture entries from the design/
   implementation work) — legitimate process artifact of U1/U4a, already in scope of the U4a spot
   check, not new scope for this U5 gate. `50f9aaa` touches exactly: `claude/README.md`, the six
   agent files + their `kaizen/history.md`, `docs/BACKLOG.md`, `docs/HISTORY.md`,
   `docs/plans/cpg-agent-adoption-coordination.md`, `skills/README.md`,
   `skills/cpg-analysis/SKILL.md` — nothing beyond that.
6. **`docs/BACKLOG.md`/`docs/HISTORY.md` accuracy.** Both documents are honest about gate state.
   `BACKLOG.md`'s milestone-map row reads 🟡 (not ✅) and says explicitly "Implementation (C-401…
   C-407) complete; U5 (`analyst` re-gate) and U6 (`qa-engineer` acceptance pass) still queued."
   `HISTORY.md`'s entry has its own "**Not yet closed as of this entry**" paragraph naming both
   remaining gates by unit number, and a closing note confirming no agent-prompt/`SKILL.md`/
   `pipeline.sh` file was touched in U4b-5's own scope (correct — those are U4a's/U4b-1..4's
   work, cataloged not re-edited). The Requirements-coverage block's FR/AC tagging matches
   Pass 1's U2-fix exactly (FR-4/AC-3/AC-4/AC-5 now tagged, MAJ-1 closed). `C-401`…`C-407` numbers
   are free of collision (highest prior item C-323; verified by grep).
7. **Prompt-quality regressions.** Ran `bash claude/scripts/audit-team.sh` directly. All
   agent-specific checks for the six touched agents PASS — kaizen triple present, deployed,
   catalog entries, boundary-pair symmetry (`coder`↔`tdd-engineer`, `coder`↔`frontend-engineer`,
   `analyst`↔`qa-engineer`, `tdd-engineer`↔`qa-engineer` all still route correctly post-edit), no
   commit-authority leakage. The overall script exit is `FAIL`, but the two failing checks
   (username/home-path leak) point at `falkor-chat/docs/test-reports/graphrag-eval-report.md` — a
   file neither `35b108f` nor `50f9aaa` touches; it belongs to the separate, in-flight K-026
   coordination the ledger's own Notes section already flagged as "not touched by this
   coordination." Out of scope for this gate; noted so the verdict isn't misread as resting on a
   clean tool run when the tool actually reported FAIL for an unrelated reason. Direct full-file
   reads of `coder.md`, `tdd-engineer.md`, `frontend-engineer.md`, and `qa-engineer.md` (not just
   diff hunks) found no orphaned sentences, no frontmatter/body contradictions, and no awkward
   insertion points — each new sentence lands as a natural continuation of its surrounding
   paragraph.

### Grounding & soundness — what I verified

- **Live-verified**: `MATCH (b:CpgBuildInfo) RETURN …` against `cpg_falkorchat` returns 0 rows
  today, consistent with the "no backfill of the two live graphs" design decision and with U4a's
  own live-verification claim — nothing in U4b silently changed that.
- **`skills/cpg-analysis/SKILL.md` frontmatter `description` is 983 characters**, independently
  counted (not trusting the ledger's figure), confirming the 1024-char budget held.
- **All six agents' YAML frontmatter parses under Claude Code's actual loader** (inferred from
  deployment + `audit-team.sh`'s catalog/deployment checks passing) even though two files
  (`tdd-engineer.md`, `frontend-engineer.md`) fail a strict `pyyaml.safe_load` — and that gap
  pre-dates this diff (see n-1).
- **`pipeline.sh` passes `bash -n`** — no syntax regression from U4a's stamping-step insertion.
- **`git diff`/`git log` confirm zero touch to `cpg/mcp/server.py` and `.mcp.json`** across both
  commits, independently re-verified rather than taken from the commit message's own claim.
- **Root `AGENTS.md`'s `cpg-analysis` bullet was independently re-read**: "`cpg-analysis` (the
  consumer side)" — already consumer-agnostic, confirming U4b-5's "checked, correctly left
  unchanged" claim rather than an unverified assertion.

### What's solid

- **The two-commit split (design+U4a, then U4b) is legible and each commit is independently
  reviewable** — `35b108f`'s stat and `50f9aaa`'s stat both match their stated scope with no
  surprises.
- **The kaizen `history.md` entries are unusually good evidence, not decoration** — each one
  states the exact before/after wording of the frontmatter clause and body sentence, which made
  cross-checking the actual file against the stated change fast and unambiguous; the two
  same-day addenda (`coder`/`tdd-engineer`'s conditional-framing correction) are honest about a
  real first-pass miss rather than silently smoothed over.
- **The coordination ledger's own self-reported friction (session-limit failures, the
  conditional-framing miss, the frontend-engineer wording variance) all independently checked out
  exactly as narrated** — nothing in the ledger's account of what happened needed correcting.
- **The `CPG:` evidence-trail convention is genuinely uniform** — all six agents use the identical
  three-shape line (`used <graph> — <clause>` / `considered, not relevant — <clause>` / `not
  applicable — <clause>`), placed in each agent's natural existing deliverable section rather than
  as a bolted-on new one, exactly as §3 specified.

### Open questions

None. Nothing in this diff needed the caller's input to resolve.

---

## Pass 3 — U7 fix-round diff gate

**Scope.** `cobb`'s U7 fix round: a narrow wording-only patch to the same six wired agent
files (`claude/{analyst,architect,qa-engineer,coder,tdd-engineer,frontend-engineer}/*.md`)
plus their `kaizen/history.md` entries, targeting the three defects U6's live-dispatch
acceptance pass found (`docs/test-reports/cpg-agent-adoption-report.md`: DEF-1 `coder`
moderate, format; DEF-2 `architect` major, freshness check reasoned-around; DEF-3
`tdd-engineer` major, silent no-CPG task). Baseline: this was reviewed as the repo's
uncommitted working-tree diff against the prior HEAD (`f7e2d8f`), per the brief. Mid-review
the tree was committed as `bafc3a7` ("fix(cpg-agent-adoption): U7 — harden freshness-check
sequencing and CPG: line anchoring") by another process; I re-diffed `f7e2d8f..bafc3a7` and
confirmed it is byte-identical to the working-tree diff I had already captured (`diff -q`
empty), so this section's findings stand against the now-committed content unchanged. Read
the six agent files in full (not just hunks), all six `kaizen/history.md` additions, the
`docs/plans/cpg-agent-adoption-coordination.md` U7/U8 row edits, `docs/test-reports/
cpg-agent-adoption-report.md` in full (defect text and the "Feedback & recommendations"
section, which is where the actual fix wording traces from), and `docs/plans/
cpg-agent-adoption.md` §2.3/§3 (design intent this fix must not violate). No files were
mutated; `git diff`/`git show` and read-only `python3 -c "import yaml..."` frontmatter checks
only.

**Verdict: approve with suggestions.** Zero blockers. The fix is genuinely well-targeted —
both tightened sentences trace directly to the test report's own "Feedback & recommendations"
§1/§2 (nearly verbatim: "query the freshness marker in the same tool call/step... before
deciding whether the answer needs cross-verification" → the freshness sentence; "requiring it
verbatim rather than 'include a line matching this shape'" → the `CPG:` line sentence), design
intent (§2.3/§3: agent judgment on staleness threshold, no self-triggered rebuild) is
unchanged, and no scope creep occurred (see Findings below). Two minors and a nit, all
wording-precision issues, not soundness issues.

`CPG: not applicable — this diff is agent-prompt Markdown and a coordination-ledger doc; no
source tree with a loaded CPG is under review here, same reasoning Pass 2 and the U6 test
report both used for this same feature.`

### Findings

**Minor — `frontend-engineer.md`'s freshness-check sentence is missing the "tool call/"
qualifier the other five files all carry, contradicting the U7 ledger row's and `cobb`'s own
commit message's "identically" claim.**

Evidence: `claude/frontend-engineer/frontend-engineer.md:18` reads "…query the freshness check
(`skills/cpg-analysis/references/freshness.md`) **in that same step**, before deciding…",
while `analyst.md:45`, `architect.md:36`, `coder.md:18`, `qa-engineer.md:28`, and
`tdd-engineer.md:35` all read "…**in that same tool call/step**, before deciding…" — confirmed
by direct grep across all six files. `frontend-engineer`'s own new `kaizen/history.md` entry
quotes the shorter wording verbatim, so this isn't a copy error I introduced by reading — it's
what actually landed. `docs/plans/cpg-agent-adoption-coordination.md`'s U7 row states the fix
was "Tightened, identically across all six wired agent files," and `bafc3a7`'s commit message
says "Applied identically across all six wired agents" — both overclaim for this one clause.
Substantively this is low-stakes: `frontend-engineer` wasn't one of the three defect-carrying
dispatches (D1/D2/D3 were `coder`/`architect`/`tdd-engineer`), and the hard-stop clause ("this
is not a separate, optional judgment call") is present verbatim in `frontend-engineer.md` too
— but the specific "bundle into the same MCP call" framing that the test report's
recommendation #1 called out as the fix for DEF-2's exact rationalization ("a staleness gap
here would have shown up as a grep/CPG mismatch") is weaker here than in the other five.

*Suggested improvement:* add "tool call/" to `frontend-engineer.md:18` for literal consistency
with the other five, or — if the shorter phrasing is intentional (e.g. because `frontend-
engineer`'s discovery step isn't framed around a single `mcp__cpg__query` call the way the
others are) — say so explicitly in the U7 ledger row and commit message rather than claiming
uniform, identical treatment.

**Minor — the trailing "this is not a separate, optional judgment call" has an ambiguous
pronoun referent, in the exact sentence written to close a major defect (DEF-2) that was
itself a reasoning-around-a-soft-instruction failure.**

Evidence (all six files, e.g. `claude/architect/architect.md:36`): "…query the freshness check
… in that same tool call/step, **before deciding whether the result needs further
cross-verification** — this is not a separate, optional judgment call." "This" most
immediately follows "the result needs further cross-verification," so a literal-minded reading
could parse the non-optional part as *the cross-verification decision* rather than *running
the freshness query itself*. That inversion would leave open exactly D2's actual move: D2 (per
`docs/test-reports/cpg-agent-adoption-report.md` DEF-2) skipped the freshness query and
substituted grep/CPG agreement as its own judgment call about whether cross-verification was
needed — the intended fix is that *skipping the query* is what's forbidden, not that *the
cross-verification decision itself* is mandatory. In context (read together with "in that same
tool call/step" immediately before it) the intended reading is recoverable, and this is a
genuine, well-targeted tightening either way — but the antecedent is doing more grammatical
work than the sentence structure comfortably supports, in the one clause where residual
ambiguity has the highest cost (this is the DEF-2 fix, and DEF-2 was already a case of an
agent finding room to reason past a softer version of this same sentence).

*Suggested improvement:* tighten the antecedent, e.g. "…query the freshness check in that same
tool call/step, before you decide whether the CPG's answer needs further cross-verification —
running the freshness check itself is not optional, and skipping it in favor of a substitute
check (e.g. grep agreement) doesn't satisfy this." (The parenthetical directly forecloses D2's
specific rationalization, which the current wording only does by implication.)

**Nit — "query the freshness check" is an imprecise verb/object pairing; the test report's own
recommended phrasing ("query the freshness marker") reads more cleanly.**

Evidence: `skills/cpg-analysis/references/freshness.md` documents a Cypher query against a
`:CpgBuildInfo` **marker node** — "the check" is the reference document itself (a recipe), not
something you "query." `docs/test-reports/cpg-agent-adoption-report.md`'s own recommendation
#1 phrased the same fix as "query the freshness **marker** in the same tool call/step" — the
landed wording ("query the freshness **check**") swapped the cleaner noun for a slightly
mismatched one across all six files (the parenthetical citation `(skills/cpg-analysis/
references/freshness.md)` immediately after "the freshness check" makes clear what's meant,
so this doesn't create real ambiguity — it's a polish-level nit, not a comprehension problem).

*Suggested improvement:* "query the freshness marker (per `skills/cpg-analysis/references/
freshness.md`)" would match the report's own recommended language and avoid the "querying a
reference document" mismatch, but this is optional — low priority relative to the two minors
above.

### Consistency check (six files)

Confirmed by direct grep and full-file reads: five of six files (`analyst`, `architect`,
`coder`, `qa-engineer`, `tdd-engineer`) carry the **identical** phrasing pattern for both
edits, adapted only for each file's pre-existing sentence joints (e.g. "Note what it says in
your findings" / "…in your Context & findings" / "…in your report" — each matches that file's
existing terminology, not a copy-paste seam). `frontend-engineer` matches the `CPG:`-line edit
pattern exactly but diverges on the freshness-sentence edit per the minor finding above. No
file shows a grammatical break, an orphaned clause, or an awkward insertion point — each
edited sentence reads as a natural continuation of its surrounding paragraph, including
`frontend-engineer.md`'s numbered-list item 4 and `qa-engineer.md`'s semicolon-joined
"Read the sources of truth" bullet, both of which interpose extra clauses between the anchor
phrase and the edit.

### YAML frontmatter

Confirmed via `python3 -c "import yaml; yaml.safe_load(...)"` against all six files: this
round touched **zero** frontmatter lines (every diff hunk starts well past each file's `---`
block — earliest is `coder.md`'s hunk at line 18, frontmatter ends at line 4). The pre-existing
strict-parse failure on `tdd-engineer.md`/`frontend-engineer.md` (an unquoted colon inside the
`description:` plain scalar) reproduces identically pre- and post-diff, confirming Pass 2's
n-1 finding is still accurate and still not diff-introduced — moot for this gate, as expected.

### Scope check

`git diff --stat` against `f7e2d8f..bafc3a7` touches exactly 13 files: the six agent `.md`
files (4 lines changed each, `+2/-2`, matching the ledger row's "4 diff lines each" claim),
their six `kaizen/history.md` files (append-only, one new dated section each), and
`docs/plans/cpg-agent-adoption-coordination.md` (the U7 row updated to `delivered` + a new U8
row). Confirmed via `git diff --stat` with the six-file-plus-coordination-doc pathspec
excluded that nothing else in the repo changed. No frontmatter, roster, `claude/README.md`,
`skills/README.md`, `docs/BACKLOG.md`, `docs/HISTORY.md`, or `skills/cpg-analysis/*` touch —
matches the brief's scope claim exactly.

### Behavioral-confirmation caveat

This is a **static review of a wording fix for a behavioral problem** (a live agent reasoning
past a soft instruction, three different ways, across three live dispatches). Nothing in this
pass can prove DEF-1/DEF-2/DEF-3 are actually closed — that requires a live re-dispatch of at
least `coder`, `architect`, and `tdd-engineer` against comparable tasks, which is a `qa-engineer`
acceptance-pass job, not this gate's. What this pass *can* and does confirm: the new wording is
genuine (traces to the test report's own recommendations, not invented), well-targeted (each
clause maps to a specific defect's specific failure mode), non-broken (all six files parse,
read grammatically, and match the pre-existing agent's voice), and free of scope creep. Same
altitude boundary Pass 2 flagged for AC-1…AC-5 ("prompt-level commitments, not
runtime-observable from a static diff read") and the U6 report itself flagged for its own
result ("static prompt-wiring review and live-dispatch behavior are genuinely different failure
surfaces for this class of feature") — this pass doesn't attempt to cross that boundary, and a
follow-up live-dispatch acceptance pass (an "U6b" of sorts, re-running D1/D2/D3-equivalent
tasks) is the only way to close it.

### What's solid

- **The fix wording is traceable, not invented.** Both tightened clauses map near-verbatim to
  the U6 test report's own "Feedback & recommendations" §1/§2 — this is evidence-driven
  iteration, not a guess at what might help.
- **Design intent is untouched.** `docs/plans/cpg-agent-adoption.md` §2.3/§3 (agent judgment on
  staleness threshold, no self-triggered rebuild, "not a new section/checklist/ceremony") is
  unchanged; confirmed the plan document itself isn't in this diff at all.
- **Kaizen history entries are honest and specific**, same as Pass 2 found for U4b — each of the
  six carries the exact before/after wording and an explicit defect-to-clause mapping, which is
  what made this gate's consistency check fast and unambiguous.
- **No scope creep, no frontmatter touch, no restructuring** — a genuinely narrow, well-scoped
  wording round, exactly as the U7 brief specified.

### Open questions

None. Nothing in this diff needed the caller's input to resolve; the frontend-engineer wording
gap and the pronoun-ambiguity finding are both concrete enough for `cobb` to act on directly if
it chooses to.
