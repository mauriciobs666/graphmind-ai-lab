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
