# Typed Decision Models — Plan Review, Subagent Labelling Amendments (plan4)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/typed-decision-models4.md`

## Pass 1

### Scope & verdict

**Scope.** `docs/plans/typed-decision-models4.md` v1.0 (uncommitted, 4,550 words) against:
- the method note `docs/plans/typed-decision-models-ml.md` v1.8: §Q1.1–§Q1.4, §Q6, and handed
  items 7–11;
- `docs/reviews/typed-decision-models.md` Passes 9–10;
- the requirements decision log's 2026-10-31 entries and FR-8;
- v1.3, plan2 and plan3 §4, and the coordination ledger;
- built model-bench: `runner.py` at the four probe sites and the three item drivers' `except`
  clauses, `decision_stats.PairedItems`/`parity_cell`, `stats.wilson_interval`, and the
  working-tree `report._render_replicate`;
- `docs/plans/small-model-benchmarking.md` §3.6.

I tested the harness premises three ways:
- this repo's 214 subagent transcripts and `.meta.json` files;
- `claude --help` (2.1.316);
- `code.claude.com/docs/en/sub-agents`, fetched on 2026-10-31.

I also ran one Read-tool probe with a 232,101-character one-line JSON file.

**Verdict: needs changes. Seven majors, no blocker.** The architecture is right: a separate
driver session, an on-disk ledger, agent definitions as dataset provenance, the audit as the
backstop, and the per-cell design effect at report time. Most of its premises hold. But:
- the P8-m2 ruling relies on an outage path the runner does not implement;
- a 50-item guard request cannot be read whole by the Read tool;
- the pilot depends on units that are ordered after it;
- the instruction pin and the U5.7 smoke cannot fail;
- the token rule reads the wrong `usage` entry;
- reconciliation contradicts the audit table;
- the per-cell design effect needs item ids that `PairedItems` does not carry.

CPG: considered, not relevant — no `cpg_modelbench` or `decision-models` graph exists; the model-bench sites the plan cites were read from source.

**Stopping condition (falsifiable; the next revision meets every line):**
1. §3.10 either has U2.3 run a re-probe on a failed between-item probe (`probe()` ≠ `api-v0`
   gives exit 3, per SMB §3.6 clause (iv)), or it drops the claim that the item's own call
   detects an outage. §3.10's precedence sentence is stated per surface (F1, F12).
2. §3.2 fixes the request file layout so that Read can page it, and sizes `maxTurns` from the
   largest batch. U5.7's smoke runs a batch at the cap with items at the render cap (F2).
3. Every §3.6 mechanism the pilot uses has one owning unit, and §4.3's graph orders it before
   U5p (F3).
4. The instruction pin is a declared set, written at `install` (empty for an `omitClaudeMd`
   subagent), and never taken from the first run. U5.7's done-when asserts the injected-context
   expectations listed in F4, each with a planted-red case.
5. §3.7 says which entry's `usage` counts per `message.id`. U5.5 has a fixture whose entries
   differ, with the expected value written by hand (F5).
6. §3.3 and §3.5 give one rule for an issued batch with no transcript. The transcript is located
   by a key the audit does not check (F6).
7. U3e.2's brief names the `decision_stats`/`DecisionPointPack` change that gives it per-cell
   item ids and the `genRunId` map (F7).
8. Each minor is either addressed or declined in one line.

**Growth.** This is a new document, so growth-stops does not apply. I found no restated history
and no finding→resolution prose. §5 is **not** the place to cut, because U5.4–U5.10's done-when
cells are "§5 rows", so cutting §5 removes the done-conditions. Small trims that are available
are in F17. At 4,550 words for a whole new subsystem plus four model-bench amendments, the size
is proportionate.

### Findings

**F1 (major) — §3.10's outage backstop does not exist in the runner.** *Verified by reading the
code.*
- The SMB plan does have clause (iv) (`docs/plans/small-model-benchmarking.md`, the §3.6 timeout
  bullet): after a timeout, or a connection failure, the runner re-probes and exits 3 if the
  server has gone. So the citation is accurate as a plan citation.
- But `runner.py` calls `probe()` only once, at capture step 2 (`runner.py:1839`).
  - `_drive_single_call_items` (`:482-489`) turns a timeout into `withheldFor="timeout"` and
    continues.
  - `_drive_decision_items` (`:591`) and `_drive_handoff_items` keep a failed item and continue.
  - Nothing catches `LMStudioUnreachable` around the drivers, so it ends as a traceback.
- Under the ruling, a dead server therefore gives `[]`, then either a traceback (P8-m2's own
  complaint, one line later), or timeouts that are scored as model failures on single-call
  roles. Clause (iv) exists to prevent exactly that.
- *Fix (`architect`):* in `_between_item_residency`, on a failed probe, call `probe()`. If it
  returns anything other than `api-v0`, exit 3 (clause (iv)). Otherwise return `[]`. Add a
  U2.3 row: a dead-server stub exits 3, and a probe-only failure with a live `probe()`
  completes the run. The other route is to delete the outage sentence and add a risk.

**F2 (major) — A 50-item guard request is too large for one Read, and a one-line JSON file cannot
be paged.** *Verified with a live Read probe.*
- I read a 232,101-character compact JSON file. The tool returned the first 82,805 characters
  and stopped: "59563 tokens, cap 25000; … very long lines and cannot be paginated by line".
- §3.2 specifies canonical compact JSON for `bodySha256`, but never says the file is not
  written that way. Guard items render at up to 6,000 characters (note §Q1.3), so 50 of them can
  exceed the 25k-token cap even when pretty-printed.
- A truncated read gives missing keys. That is a counted void, so the batch is re-issued, voids
  again, and drops as `retryCapExceeded`: spend with no labels. `maxTurns: 6` leaves room for
  about four Reads.
- U5.7's 3-item smoke cannot expose any of this.
- *Fix (`architect`):*
  - write the request as JSON Lines: a header line, a rubric line, then one item per line.
    `bodySha256` stays over the canonical body;
  - tell the subagent to page with `offset`/`limit`, and derive `maxTurns` from the worst case
    (cap × render cap ÷ 25k tokens, plus Write and reply);
  - make the smoke batch 50 items at the render cap.

**F3 (major) — The pilot (U5p) depends only on U5.8, but the mechanisms it runs are in U5.9 or in
no unit.** *Verified by reading §3.6, §3.8, §4.1 and §4.3.*
- The pilot uses three of §3.6's mechanisms:
  - generation runs of 25 with `genRunId` and `genRound` (§3.6, first bullet);
  - per-vote reshuffle into batches (second bullet);
  - the batch cap.
- U5.9's brief has "votes, tranche, top-up planner, spend halt, diagnostics", and U5.9 comes
  after U5p (`U58 --> U5p`, `U58 --> U59`). Cutting seeds into generation runs is in no unit's
  brief at all: U5.3 has `seeds.py` only.
- *Fix (`architect`):* move "generation runs" and "votes" into U5.8, or make `U59 --> U5p`.
  Then name an owner for every §3.6 bullet.

**F4 (major) — The instruction pin is trust-on-first-use, and the U5.7 smoke cannot fail on
injected context.** *Verified (pin text, transcripts, docs); outside-repo behaviour unverified.*
- §3.5 says the pin "starts as the set the first run records". So whatever the first run loads
  becomes the reference, for example a managed-policy file, which `omitClaudeMd` does not drop
  (docs). Note §Q1.2 asks for an allowlist, which is a declared set.
- The smoke's done-when is "ingests with no void". With a pin taken from that same run, the
  instructions check cannot redden.
- Two other channels are unchecked. In this repo, a subagent whose allowlist has no MCP tool
  still received `mcp_instructions_delta` for `claude-artifacts`, `claude.ai Claude Docs` and
  `cypher` (transcript `c6eed315…/subagents/agent-ae9584ca598b7b325.jsonl`, entry 20).
  `session_context` carries `userEmail` (entry 5).
- *Fix (`architect`):*
  - `install` writes the expected set: empty for both subagents. Any deviation halts.
  - Add an audit row: any `mcp_instructions_delta`, or any `mcp__` name in
    `deferred_tools_record`, is an environmental halt.
  - U5.7's done-when asserts four things: an empty `instructions` set, no `gitStatus` key in
    `session_context`, no MCP attachment, and `output_config.effort` equal to the pin. Each has a
    planted-red fixture.

**F5 (major) — "Each entry repeats `usage`" is false. Deduplicating by `message.id` must say which
entry counts.** *Verified on 214 transcripts.*
- 8,558 `message.id`s span several entries, and 120 of those carry different `usage` values
  (e.g. `msg_011Cg35vHA9HhyX93AbjChQs`: `output_tokens` 3, then 2087).
- Summing output tokens over the first entry per id gives 10,283,664. Summing over the last gives
  10,603,266, about 3% more.
- The pilot's F and c are what the stakeholder decides on (§3.8). A first-entry implementation
  passes U5.5's "counts its tokens once" row as written, and under-projects.
- *Fix (`architect`):* §3.7 says to take the last entry's `usage` per `message.id`. U5.5's
  fixture has three entries with growing `output_tokens`, and the expected value is the last
  one, written by hand. A first-entry mutant reddens.

**F6 (major) — §3.3 and §3.5 disagree on an issued batch with no transcript, and the transcript is
found by the field it audits.** *Verified by reading.*
- §3.3's first row halts on "no transcript … for an issued batch". §3.5's reconcile re-issues it
  silently, with no attempt counted.
- The transcript is located by "first `user` entry equals the expected spawn prompt". A prompt
  mismatch therefore looks like "no transcript", so §3.3's second row can never fire. After a
  restart, reconcile re-issues the batch without limit, at full spend each time. A Claude Code
  update that moves the transcript directory does the same.
- A transcript with no final assistant entry, because the driver died mid-run, ingests as
  "answer missing". That is a counted void for an environmental cause.
- *Fix (`architect`):*
  - Locate the transcript through the driver's own transcript: the `Agent` `tool_use` whose
    input names the request path, then `.meta.json` `toolUseId` (verified present), then the
    `agentId`.
  - "Never spawned" means no such `tool_use`. A `tool_use` with no transcript halts.
  - An incomplete run (no final assistant turn, no `<synthetic>`) gets no attempt.
  - The old request and answer leave `exchange/` before the re-issue.

**F7 (major) — U3e.2 cannot group a cell's pairs by `genRunId`: `PairedItems.outcomes` carries no
item ids.** *Verified (`decision_stats.py:378-403`).*
- `outcomes[metric]` is `tuple[tuple[bool, bool], ...]`, ordered by `itemId`, but without the ids.
  `_render_replicate` receives `dp.strata`, and has no `itemId → genRunId` map.
- §3.9 and U3e.2's brief do not say which interface changes. That leaves the implementer to
  re-derive `pair_items`' filtering, or to widen a frozen dataclass that U3c.1 gated.
- *Fix (`architect`):* name the change, for example
  `PairedItems.outcome_ids: Mapping[str, tuple[str, ...]]`, parallel to `outcomes`, plus a
  `DecisionPointPack.gen_run_ids` map read by `validate_pack`. Add a U3e.2 row: ids align with
  pairs after a drop.

**F8 (minor) — Premise 3 is only half the story: the driver's own `Agent` call can override
`effort` and `model`, and the hook does not gate that call.** *Verified against the docs.*
- The docs say: "The [per-invocation] parameter overrides the `effort` field …
  `CLAUDE_CODE_EFFORT_LEVEL` takes precedence over both". The `model` parameter also takes an
  alias.
- The audit halts either way, but it files the halt as environmental, and the resolve path
  invites a re-pin.
- *Fix (`architect`):* `guard_hook.py` also handles the driver's `Agent` calls.
  - It denies any `subagent_type` other than the two.
  - It denies a `prompt` that differs from the ledger's pending spawn prompt.
  - It denies any `model`, `effort`, `isolation` or `run_in_background` key.
  - The audit splits "effort ≠ pin" into an env-var cause and an `Agent`-parameter cause (a
    driver halt).

**F9 (minor) — Match hook-log lines by `tool_use_id`, not "tool name and input".** *Partly
verified.*
- Transcripts record each hook firing with its `toolUseID` (`hook_success` entries 185–186 of the
  architect transcript above).
- The main agent's frontmatter hooks also fire inside its subagents: teco's
  `guard-repo-writes.sh` ran on the architect's Write. So `guard_hook.py` will log each subagent
  call twice.
- Name-plus-input matching cannot tell two identical calls apart.
- Hook input carries `agent_type` (docs). It probably carries the tool-use id as well, but that is
  unverified: U5.7 must show it.

**F10 (minor) — Classifying every `<synthetic>` as a usage stop is too broad.** *Verified.*
- Main-session transcripts here hold three kinds of `<synthetic>` entry:
  - `error: "rate_limit"`;
  - `error: "authentication_failed"` ("Login expired");
  - a non-error "No response requested.".
- Subagent transcripts so far hold only `rate_limit` (19).
- *Fix:*
  - a usage stop is `error == "rate_limit"`;
  - any other `isApiErrorMessage` is an environmental halt;
  - a non-error synthetic entry is ignored.
- State the precedence when several §3.3 rows match: halt, then usage stop, then counted.

**F11 (minor) — A seed-aware generator writes the text the labeller sees. Nothing screens that
text for intent or seed metadata.** *Inferred.*
- §3.2 closes the request's keys, but `item` values are free text written by a model that holds
  the intended answer and the difficulty.
- Meta-text such as "(near-miss: the budget is withheld)" would carry intent into a blind vote and
  inflate intent-label agreement.
- *Fix:* `data-scientist` rules whether a screen is needed and what it checks (seed-axis value
  tokens such as `near-miss`/`clear`/`intended`, bracketed asides). `architect` places it in
  ingest as a counted generator void, and the generator template forbids commentary.
- With F4's pin fixed, this is the only remaining path for seed knowledge to reach a labeller.
  I found no path for real data (Appendix A, (c)).

**F12 (minor) — §3.10's precedence sentence is false on the `decisions` and `handoff`
surfaces.** *Verified (`runner.py:591-598`, handoff `except` clause).*
- On those surfaces a failed decision call is a *kept* item whose `withheldFor` is
  `_load_withheld_for(…, resident)`. So with `[]` it becomes `"load"`, not "its own
  disposition".
- U2.3's row "(or its own failed call's disposition)" therefore has no single expected value.
- *Fix:* state the rule per surface, and give each surface its own expected value in the row.

**F13 (minor) — The top-up and pilot rules leave edge cases open.** *Verified by reading.*
- (a) The spend halt projects generation runs only. Labelling runs, three votes per item,
  dominate cost (150 sonnet runs against 97 opus runs, note §Q1.3). Project total tokens with
  the pilot's F and c instead. Also say what a stakeholder "no" produces: finalize with short
  cells and print n?
- (b) The screen is defined on "the 100 items", but voids can leave fewer. Write it as
  `10·(Δ count) ≥ n` over items carrying all three votes (`data-scientist` confirms).
- (c) The boundary tests per `model-bench/REVIEW.md`:
  - the spend halt at exactly half, which must not halt;
  - the retry cap at 2 voids (re-issue) against 3 (drop).
- (d) "Labelling restarted" does not repair a **generator** re-pin. Say that a generator re-pin
  restarts the point's generation.
- (e) The message-type re-check in `finalize` has no stated outcome when it fails.

**F14 (minor) — `model-bench` units still collide.**
- §4.3 runs U2.3 in parallel with U4, U3d and U3e.*. All of them write `model-bench/docs/HISTORY.md`
  and `AGENTS.md`, which is the reason §4.2 gives for "U2.3 after U3c.2".
- U5.2b sits outside the U5.x chain that §4.1 serializes on `decision-models/docs/HISTORY.md`.
- U5r.guard waits on U3d, and therefore on U4, although guard's campaign needs neither. Only its
  import does.
- *Fix (`teco` for serialization, `architect` for the graph):* give model-bench one serial order
  (e.g. U3c.2 → U2.3 → U4 → U3d → U3e.1 → U3e.2), and split U5r into a campaign step and an
  import step.

**F15 (minor) — The Bash allow rule and the settings location are under-specified.**
- "Not the venv's `python -m decisionmodels.driver`" reads as a prefix match, which `… ; cat
  ../store/seeds…` would pass. Match the exact argv with `shlex`, allow only a subcommand from the
  closed set, and reject shell metacharacters.
- `settings.json` must be installed at `exchange/.claude/settings.json`. §3.5 says only
  "`exchange/` (… `settings.json` …)".

**F16 (minor) — The smoke has no entry point.** `init` accepts `--phase pilot|full` only. U5.7's
"scratch campaign on fixture text" needs a named flag or a test-only phase. Without one,
U5.7 is ordered before the code it needs (U5.8's pilot phase).

**F17 (nit) — Small corrections and trims.**
- §2 says "every line carries `cwd`", but `api-request-shape` and `api-request-blob` entries lack
  it.
- §2 says "Claude Code 2.1.316, read here", while the transcripts record 2.1.314. The executing
  binary is the fact that matters.
- §4.1's live-unit rules restate note §Q1.2/§Q1.3 (counts-only commits, pilot first, point
  order). A cite is enough.
- Root `AGENTS.md` is already 3,345 words against the ~2,500 smell. U5.1's new bullet should be
  one line.

### What's solid

- Definitions live outside `.claude/` and are byte-installed per campaign with hash checks. That
  is the right reading of "dataset provenance". It also keeps `dm-labeler` unspawnable from repo
  sessions, where project agents are found by walking up (verified in the docs).
- A dedicated, human-started driver session with all state in an fsync'd append-only ledger is
  sound. `teco` never spawning runs is correct, given the injected repo context it would carry.
- Verified premises: `--strict-mcp-config` (from `--help`); frontmatter `effort`, `model`
  (full id), `omitClaudeMd`, `initialPrompt`, `Agent(a, b)` and trust-gated project hooks (docs);
  `.meta.json` `agentType`; `instructions` and `session_context` attachments;
  `params.thinking` and `params.output_config.effort`; the first `user` entry as the spawn
  prompt (transcripts). Subagents never load auto-memory (docs).
- Halting voids are kept apart from counted voids. The closed request schema has a positive
  control. The intent-filter mutant in U5.9 and the `m̄` mutant in U3e.2 are good tests.
- The design-effect formula matches note §Q6, with Kish `m₀`, the 1.0 floor and limits only.
  Using float arguments in `wilson_interval` is safe: it has no integer check (`stats.py:97`).
- Call 7 is sound. A shared vote 1 makes the two `D` values positively correlated, so fr10.py's
  `SE = √(0.30/100)` is a conservative bound (`fr10.py:89`).

### Open questions

- **Stakeholder:** Q-1, unattended relaunch, remains as the plan states it. Also F13(a): when the
  spend halt is declined, does the point ship with short cells and printed n, or stop?
- **`data-scientist`:** F11, whether a generator-text screen is needed and what it checks. F13(b),
  the screen's denominator under voids.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: F2, F4 and F5 together. Each is a harness
premise that held in the plan's reading but failed against a transcript or a live probe.

- A plan step that relies on agent-harness behaviour (frontmatter fields, hook firing, transcript
  fields, tool size limits) cites a live probe or transcript for each premise. It also names the
  smoke whose done-condition turns red if the premise is false, sized at the procedure's
  worst-case input.

### Appendix A — Premise ledger

| Premise | Status | Evidence / who proves it live |
|---|---|---|
| `--strict-mcp-config` + `--mcp-config` | verified (flag) · unverified for claude.ai connectors | `claude --help`. U5.7 smoke: no `mcp_instructions_delta` (F4) |
| frontmatter `effort`; env override | verified, plus the `Agent`-parameter override | docs (F8) |
| `omitClaudeMd` drops CLAUDE.md | verified (docs, v2.1.271+) · git-status effect unstated | U5.7 smoke |
| hooks skipped in an untrusted folder | verified (docs, project-level agents) | U5.7: a planted untrusted run halts |
| no repo CLAUDE.md/MEMORY.md/gitStatus outside the repo | MEMORY.md verified (docs). Others **unverified**: no outside-repo transcript exists on this machine (`~/.claude/projects/-home-mauricio/` holds only `memory/`) | U5.7 smoke (F4) |
| transcript fields (`agentType`, `instructions`, `session_context`, `effort`, `thinking`) | verified | 214 transcripts |
| `usage` repeated per `message.id` | **false** in 120/8,558 | F5 |
| `<synthetic>` = usage stop | **too broad** | F10 |
| SMB clause (iv) catches outages | in the plan, **not in the code** | F1 |
| one Read reads a request | **false** at 50 guard items | F2 |

**(c) FR-8 sweep.** The paths I checked, and what blocks each:
- MCP tools: the allowlist on both definitions, plus `--strict-mcp-config`.
- Repo files: the hook, the cwd permission boundary and the path audit.
- Other batches and votes: one request in `exchange/` at a time.
- Seeds in `store/`: the hook, and `store/` lies outside the cwd.
- Generator transcripts under `~/.claude/projects`: the hook and the path audit.
- The driver's knowledge: the prompt template plus the `fork` ban (F8 adds prevention).

Nothing that is still open carries real falkor-chat data. Seed and intent knowledge can still
reach a labeller only through generated text (F11), or through injected context once F4's pin
fixes it.

## Pass 2

### Scope & verdict

**Scope.**
- Plan4 v1.1, uncommitted, 5,969 words.
- The note's uncommitted in-place revision (`git diff -- docs/plans/typed-decision-models-ml.md`):
  - §Q1.2 "Generated-text screen", "Reruns" and "Majority of three blind votes";
  - §Q1.3 "Spend halt", "On a stakeholder 'no'" and the pilot "Screen".
  Nothing else gates these note sections.
- Two live Read probes, run in the scratchpad.
- A survey of `hook_*` attachments across this repo's transcripts.
- Re-reads of `runner.py:133`, `:1839-1841` and `lmstudio.probe`.

**Verdict: needs changes. Two majors, both introduced by fixes, plus five minors.** All 17
Pass 1 findings are fixed except F2's arithmetic (P2-M1). The design is now stable. What remains
is numbers and edge cases inside the new mechanisms (see *Stopping signal*).

CPG: considered, not relevant — no `cpg_modelbench`/`decision-models` graph exists; model-bench sites were read from source.

### Pass 1 disposition

**Stopping-condition lines:**

| Line | Status | What I rechecked |
|---|---|---|
| 1 | met | §3.10 re-probes. `runner.py:1839-1841` confirms `RunRefused(_PROBE_MESSAGES[…], exitCode=3)` and that `probe()` never raises. U2.3 has the unreachable row, and per-surface precedence is in place. |
| 2 | **not met** | The JSONL layout, `maxTurns` formula and smoke are right in kind, but the page size fails at the render cap (P2-M1). |
| 3 | met | U5.8 owns generation runs, screen and votes, and §4.3 orders `U58 --> U5p`. |
| 4 | met | Expectations are declared at `install`, with rows in §3.3 and planted-red fixtures in U5.7. |
| 5 | met | Last-entry rule (§3.7), with a hand-written 3/900/2087 fixture and a first-entry mutant. |
| 6 | met as asked, but its no-attempt classes can loop forever (P2-M2) | |
| 7 | met | `outcome_ids`, `gen_run_ids` and U4's FR-10 ids each have a unit. |
| 8 | met | Every minor is addressed; see below. |

**Findings:**
- F1, F12 — fixed (line 1).
- F2 — **not fixed**: P2-M1.
- F3–F7 — fixed (lines 3–7).
- F8 — fixed: `tool_use` key rows and hook denial of `Agent` calls.
- F9 — fixed in kind (matching by `toolUseID`), but it opens P2-m1.
- F10 — fixed, with precedence stated.
- F11 — fixed (note screen plus `screen.py`), but it opens P2-m2.
- F13 — fixed:
  - (a) total-token projection and a decline path, but it opens P2-m3;
  - (b) `10·Δ ≥ n` with an n ≥ 80 top-up, but it opens P2-m4;
  - (c) the equality cases are tested;
  - (d) generator re-pin;
  - (e) tranche re-check outcome.
- F14 — fixed: two serial chains, and U5c and U5i are split.
- F15 — fixed (`shlex` match, closed subcommand set, settings path).
- F16 — fixed (`--phase smoke --fixture`).
- F17 — fixed.

**Interactions the coordinator asked about:**
- **F14 chains.** U5.2b now sits at the end of the decision-models chain and waits on U4, which
  is third in the model-bench chain. It blocks only U5c.message-type, as intended. The pilot needs
  U5.1–U5.8 and no model-bench unit, which holds. U5.9 runs alongside the live pilot, which
  writes only `pilot/` and the ledger. Sound.
- **F11 screen vs `genRunId`/`genRound`/top-up.**
  - Regeneration keeps `seedId` and gives the new text its own `genRunId`. Regeneration runs pool
    across strata but draw no new seeds, so the top-up's stratum restriction is untouched.
  - The design-effect grouping sees a few small extra groups, which is harmless.
  - Regenerated items' `genRound` is unstated. It should be the round of the regenerated seed;
    folded into P2-m2.
- **F13a vs F5 and the pilot.** Both sides use the same last-entry measurement, which is
  consistent. One gap: P2-m3.
- **F6 vs the 3-attempt cap.** Yes, a batch can loop forever: P2-M2.

### New findings

**P2-M1 (major) — A 10-line page at the 6,000-character render cap goes over Read's cap. The
"≥ 3 characters per token" premise is false for dense text.** *Verified by a live probe.*
- I wrote a request in §3.2's layout: 50 item lines of pt-BR-style chat text (the storefront's
  locale) at 6,000 characters each. JSON escaping took lines to ≤ 6,210 characters.
- `Read(offset=3, limit=10)` failed: "File content (31020 tokens) exceeds maximum allowed tokens
  (25000)". That is about 2.0 characters per token. Nothing is returned.
- The labeller spends turns on failed reads and retries, so it overruns `maxTurns`. Under §3.3 the
  result is either a counted void or the uncounted loop of P2-M2.
- U5.7's smoke would catch this only if its fixture is as dense as real text, and the plan doesn't
  say the fixture's language.
- *Fix (`architect`):* the writer splits pages by a character budget, not a line count: at most
  40,000 characters per page, which is 20k tokens at 2 characters per token. Line 2 lists the page
  offsets, and the spawn template itself says "in pages of `limit` lines from the offsets on
  line 2", so the first Read is not wasted. At the render cap that is 6 item lines per page, and
  `maxTurns = pages + 4` gives ⌈52/6⌉ + 4 = 13 at cap 50. Add a U5.4 row asserting that every page
  is ≤ 40,000 characters, and run the smoke fixture in pt-BR at the cap.

**P2-M2 (major) — Three "re-issue, no attempt" classes have no cap, so a batch can loop
forever.** *Verified by reading §3.3/§3.5; the transcript shape at `maxTurns` is unverified.*
- The classes are "never spawned", "incomplete" and usage stop. A batch that always ends
  without a final assistant turn is re-issued forever at full spend, and never reaches
  `retryCapExceeded` or a halt. Examples: one that always exhausts `maxTurns` mid-tool-call (P2-M1),
  or one that always crashes the session.
- Whether a `maxTurns` stop leaves a final assistant entry is not established. The docs say only
  that output is "marked as partial".
- *Fix (`architect`):*
  - Cap the no-attempt re-issues per batch. For example, a second "incomplete" for the same batch
    halts as a driver void.
  - Usage stops stay uncapped: they wait for the next window and spend nothing further.
  - Classify a `maxTurns` stop explicitly as counted, since the run finished and failed. U5.7's
    smoke records what a `maxTurns` stop looks like in the transcript.

**P2-m1 (minor) — The `hook_success` row assumes a record that has never been observed for an
exit-2 deny or a silent exit 0.** *Verified (survey).*
- This repo's transcripts hold 4,547 PreToolUse `hook_success` entries. Every one has JSON
  stdout: 4,464 `allow`, 120 `ask`, 9 empty-decision.
- No exit-2 deny appears, and no hook with empty stdout.
- If an exit-2 deny leaves no `hook_success`, a labeller's out-of-path Read halts as "hooks
  inactive" instead of being counted. If a silent allow leaves none, every run halts.
- *Fix:* `guard_hook.py` always exits 0 and prints `{"hookSpecificOutput": {"permissionDecision":
  "allow"|"deny", …}}`, which is the form the repo's guards already use and that is recorded.
  The audit reads the decision: `deny` is run-attributable, and a missing record is
  environmental. §3.4's "denies (exit 2)" changes accordingly.

**P2-m2 (minor) — The screen's seed-axis tokens include ordinary words, so the 5% halt can fire on
clean text.** Note §Q1.2 counts a hit on "`intended`, `intent`, `difficulty`, a domain or
condition id **anywhere**".
- "Difficulty" and "intended" are ordinary words in support and shopping chat.
- Domain ids are natural words if U5.3 names them so (for example `electronics`).
- A false-positive rate near 5% halts a round, and no template fix clears it.
- *Fix (`data-scientist` rules; U5.3 places it):* hyphenated or unnatural seed-axis tokens
  (`near-miss`) and ids in a non-word form (`dom_07`) match anywhere. Ordinary words match only in
  meta positions (brackets, after `label`/`answer`/`category`). Add a U5.8 row: "had difficulty
  paying" passes. Also state that a regenerated item keeps its seed's `genRound`.

**P2-m3 (minor) — The spend projection uses guard's pilot F and c for every point.** For message
type and join-in, whose items are shorter, that over-projects and halts early.
- *Fix (`data-scientist` confirms):* project each point from its own round 1:
  `(top-up items / round-1 items) × round-1 labelling tokens + top-up gen runs × round-1 per-run
  generator mean`.
- State that "tokens" sums the four classes on both sides.

**P2-m4 (minor) — The items the pilot adds for `n ≥ 80` are not voted at batch size 50.** One
extra run gives 25 items, so votes 1 and 2 run at batch 25, and `D(50,50)` mixes batch sizes.
- *Fix (`data-scientist`):* label the added items' votes 1–2 in batches padded to 50 with fresh
  added items, or state that the mix is accepted.
- Unrelated: the note's revision is labelled `Version: 1.8`, the same label as the gated text at
  `01f6f2f4`. Bump it to 1.9, and update plan4's "v1.8" citation (`data-scientist`, `architect`).

**P2-m5 (minor, growth) — The revision restates note facts instead of citing them.**
- §3.6's screen bullets repeat the note's 3-hit drop, 5% halt and pooling.
- §3.8 repeats the screen's `n`, `Δ`, `10·Δ ≥ n` and `n < 80`.
- §3.3 repeats the generator re-pin rule.
- The item-call outage gap appears three times: §2, §3.10 "Out of scope" and R-21.
- Of the +1,419 words, the location rule, paging, interface changes, decline path and screen
  mechanism are new scope (the note rulings) or necessary rework. These duplicates are not.
- *Fix (`architect`):* keep only the mechanism (unit, function, ledger event, test) and cite the
  note for each rule. State the outage gap once, in R-21.
- I found no finding→resolution prose. The plan cites only P8-m2, by ID, and keeps a one-line
  revision note.

### Stopping signal

The second pass found no defects in the design. Every new finding is an arithmetic or edge-case
error inside a mechanism that Pass 1 added, and each is checkable on a line. **No Pass 3 is
needed.** `teco` closes it with a line check against this falsifiable condition:
1. §3.2 pages by a character budget of ≤ 40,000, with offsets on line 2 and a spawn template
   that pages from the first Read. The `maxTurns` formula uses that page count. U5.4 asserts the
   budget, and U5.7's smoke fixture is pt-BR at the render cap (P2-M1).
2. §3.3 caps each no-attempt class per batch, except usage stops, at a stated count that leads to
   a halt, and classifies a `maxTurns` stop (P2-M2).
3. `guard_hook.py` emits a JSON decision and never uses exit 2, and the audit reads `deny` as
   run-attributable (P2-m1).
4. P2-m2 to P2-m4 are ruled in the note by `data-scientist` and placed in the plan. P2-m5's
   duplicates become citations.

Beyond these, the plan is ready. Whatever is left (`maxTurns` slack for a three-part generator
Write, the smoke's fixture details, the unverified `--strict-mcp-config` behaviour toward claude.ai
connectors, R-22) belongs to the U5.x unit gates and U5.7's smoke, which are built to catch it.
