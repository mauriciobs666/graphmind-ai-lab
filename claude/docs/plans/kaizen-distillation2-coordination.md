# Kaizen distillation — team-wide pass 2

> **Status:** active · **Owner:** `teco` · **Tracks:** — (—) · **Extends:** `claude/docs/plans/kaizen-distillation-coordination.md`

Second routine curation pass over the shared `kaizen_team` FalkorDB graph
(`skills/agent-maintenance/SKILL.md` §5), covering the 196 raw `:KaizenEntry`
nodes accumulated since pass 1 closed. Procedure is unchanged and lives in
the skill: `cobb` verifies each entry (re-deriving the fact, not just
confirming the citation still exists), routes it (prompt / knowledge base /
project docs / discard / kept-open), logs the disposition in the producing
agent's `kaizen/history.md` (plus `plan.md` for kept-open actionable items,
with the `entryId` dedup check), tags `MENTIONS` for entries really about a
different agent, and only then resolves the edge or clears the node via the
curator shapes.

**Two deliberate differences from pass 1**, both at the user's direction:

- **Strictly sequential — one agent at a time**, not parallel batches of six.
- **Large inboxes are chunked** by date range, capped at ~12–15 entries per
  dispatch. Pass 1's cost data (`kaizen-distillation-coordination.md`) shows a
  20-entry unit burning 240.8k tokens / 95 tool uses; a 44-entry unit would run
  out of turns and leave a partially-cleared inbox with an incomplete history
  log. Each chunk gets its own dated `history.md` disposition entry, which is
  normal §5 bookkeeping, not a workaround.

**No independent review gate** — precedent from pass 1, unchanged: this is
`cobb`'s sole-owned, already-specified procedure with its own embedded
verification step, not a design or implementation deliverable. The §4 team
certification pass is the periodic audit of `cobb`'s distillation work and is
run separately on request.

`frontend-engineer` had zero raw entries at open — no unit dispatched.

**No legacy-shape (`author`-property) entry survives anywhere in the graph** —
verified at open, `count = 0`. Every entry in this pass is current-shape, so
§5's legacy read is not needed for any unit. `devops`'s unit does include one
entry with **no `PRODUCED` edge left** —
`8301b20f-3e57-4761-a333-f1998bcbfcf1`, whose producer edge pass 1's U8
already resolved, reachable now only through the `MENTIONS`→`devops` edge that
same unit attached. It is not a legacy entry; it is a current-shape entry
mid-way through resolution. Resolving that one `MENTIONS` edge leaves
`otherRemaining == 0`, so the node itself goes with it.

`teco` commits each accepted unit's files by explicit path — `cobb` runs as a
delegated subagent here, so the universal interactive-mode commit grant does
not apply to it (`claude/AGENTS.md`, "Git-commit authority").

## Ledger

Order is smallest inbox first (user's choice), so the six light inboxes clear
before the heavy ones. Counts are raw entries in scope at open.

| Unit | Agent (scope) | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | cobb (1: 2026-09-06) | `a2c2c175f4d6976cb` | accepted | kept open as `claude/cobb/kaizen/plan.md` K-020 + `history.md`; node `DETACH DELETE`d | none (see above) → — | 99.1k tok, 20 tools |
| U2 | security-expert (2: 08-26, 08-30) | `a4621faebf86763bd` | accepted | `claude/security-expert/security-expert.md` (step-3 clause) + `kaizen/history.md`; `claude/graph-dba/falkordb-quirks.md` + `kaizen/history.md`; both nodes deleted | none → — | 126.8k tok, 47 tools |
| U3 | devops (3: 2 produced 09-02 + 1 `MENTIONS`-only 08-23) | `a43632153a90a40c1` | accepted | `claude/devops/ops-quirks.md` (2 entries, scope broadened) + `devops.md` + `kaizen/*`; `claude/AGENTS.md`, `claude/README.md` catalog rows; 3 nodes deleted | none → — | 133.6k tok, 40 tools |
| U4 | qa-engineer (7: 08-28…08-31) | `a175af41b18b446d7` | accepted | `claude/qa-engineer/qa-testing-techniques.md` (2 sections) + `kaizen/*` (K-007 carries 3 entries awaiting a `falkor-chat/` home); 7 nodes deleted | none → — | 135.7k tok, 55 tools |
| U5 | tico (8: 08-26…09-02) | `a85a1d743ae070fa5` | accepted | `claude/tico/tico.md` (2 rules folded into existing bullets) + `kaizen/*` (K-015); `claude/AGENTS.md` git-race paragraph rewritten; 8 nodes deleted | none → — | 137.5k tok, 37 tools |
| U6 | graph-dba (9: 09-02) | `a9bcd0c2ab80b7622` | accepted | `claude/graph-dba/falkordb-quirks.md` (6 entries, 2 merged + corrected) + `kaizen/*` (K-008); `claude/qa-engineer/qa-testing-techniques.md` + `kaizen/history.md`; 9 nodes deleted | none → — | 156.3k tok, 54 tools |
| U7 | architect chunk A (11: ≤ 09-02) | `a8fdbd6dcd140b4e3` | accepted | `claude/architect/architect.md` + `kaizen/*` (K-004, K-005); root `AGENTS.md`; `cypher-mcp/README.md`; `claude/data-scientist/lm-studio-model-notes.md` + `kaizen/history.md`; 10 nodes deleted, 1 `PRODUCED` resolved (`MENTIONS`→`qa-engineer` kept alive) | none → — | 170.7k tok, 79 tools |
| U7b | architect chunk B (6: 5×09-03 + 1×09-07 arrived mid-pass) | `a035ccc6941014b46` | accepted | `claude/architect/architect.md` (1 bullet, 2 entries folded) + `kaizen/history.md`; `skills/python-web-quirks/SKILL.md` (2 sections) + `skills/agent-standards/claude-code.md` + `skills/README.md`; `claude/data-scientist/lm-studio-model-notes.md` (folded into U7's section) + `kaizen/history.md`; `claude/cobb/kaizen/history.md`; 6 nodes deleted — **`architect` closed out, 0/0** | none → — | 173.6k tok, 66 tools |
| U8 | tdd-engineer chunk A (12: ≤ 08-30) | `a23d066e2aad247f8` | accepted | `claude/tdd-engineer/tdd-engineer.md` (1 merged Principles bullet) + `kaizen/*` (K-007…K-010); `skills/python-web-quirks/SKILL.md` (1 new + 1 generalized) + `skills/agent-standards/claude-code.md` + `skills/README.md`; `claude/data-scientist/lm-studio-model-notes.md`; `claude/cobb/kaizen/history.md`; 12 nodes deleted | none → — | 182.7k tok, 74 tools |
| U9 | tdd-engineer chunk B (8: ≥ 08-31) | `aa5d3ef4bb834eab9` | accepted | `claude/tdd-engineer/tdd-engineer.md` (1 sentence onto U8's bullet) + `kaizen/*` (K-011); `skills/python-web-quirks/SKILL.md` (folded into U7b's route-table section) + `skills/README.md`; `claude/cobb/kaizen/history.md`; 8 nodes deleted — **`tdd-engineer` closed out, 0/0** | none → — | 141.2k tok, 42 tools |
| U10 | coder chunk A (12: ≤ 08-29) | — | queued | `claude/coder/kaizen/*`, graph cleared | none → — | — |
| U11 | coder chunk B (8: 08-31…09-02) | — | queued | `claude/coder/kaizen/*`, graph cleared | none → — | — |
| U12 | coder chunk C (7: 09-03) | — | queued | `claude/coder/kaizen/*`, graph cleared | none → — | — |
| U13 | data-scientist chunk A (10: ≤ 08-30) | — | queued | `claude/data-scientist/kaizen/*`, graph cleared | none → — | — |
| U14 | data-scientist chunk B (9: 08-31…09-02) | — | queued | `claude/data-scientist/kaizen/*`, graph cleared | none → — | — |
| U15 | data-scientist chunk C (9: 09-03…09-06) | — | queued | `claude/data-scientist/kaizen/*`, graph cleared | none → — | — |
| U16 | teco chunk A (8: ≤ 09-01) | — | queued | `claude/teco/kaizen/*`, graph cleared | none → — | — |
| U17 | teco chunk B (12: 09-02) | — | queued | `claude/teco/kaizen/*`, graph cleared | none → — | — |
| U18 | teco chunk C (11: 09-03…09-06) | — | queued | `claude/teco/kaizen/*`, graph cleared | none → — | — |
| U19 | analyst chunk A (12: ≤ 08-30) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U20 | analyst chunk B (6: 08-31…09-01) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U21 | analyst chunk C (15: 09-02) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U22 | analyst chunk D (11: 09-03) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |

Deliverable paths above are the guaranteed minimum (every pass touches the
agent's own kaizen files and the graph); each row is rewritten on delivery with
the actual promotion targets — agent prompts, knowledge bases, project docs.

## The graph is live during this pass

The 196-entry snapshot at open is **not** a fixed target: other sessions keep
running while this sweep proceeds, and their agents keep writing new
`:KaizenEntry` nodes. Observed concretely between U2 and U3 — `teco` went 31 →
33 while three unrelated entries were being cleared, so the graph total read
192 where naive arithmetic predicted 190. Consequences for the remaining
units, none of them a defect:

- **Re-query the agent's entry list immediately before each dispatch**, and
  pin the brief to explicit `entryId`s or a closed date range. Never let a unit
  scope itself with an open-ended "everything this agent has".
- A chunk defined as "the newest date range" will drift upward. That is fine —
  the last chunk for an agent takes whatever exists when it is dispatched, and
  anything arriving after that is simply pass 3's problem.
- **This pass will never observe an empty graph**, and shouldn't try to. Done
  means every entry in the pinned per-unit scope is dispositioned, not that
  `count(:KaizenEntry)` reaches zero.

## Observed cost, and the one re-chunk

Six units in, the rate is stable at **≈17k tokens per entry** (U1 1 entry /
99.1k · U3 3 / 133.6k · U4 7 / 135.7k · U5 8 / 137.5k · U6 9 / 156.3k) — the
per-run floor dominates, so small inboxes are disproportionately expensive and
the marginal entry is cheap. That is what makes the ~12-entry cap the right
shape rather than a smaller one.

`architect` (16) was re-split at dispatch into **U7 (11, ≤ 09-02)** and **U7b
(5, 09-03)** on this evidence: 16 × 17k projects to ~270k, past the point where
a run risks turning over. Re-splitting at dispatch is normally friction worth
avoiding — the ledger is drawn at decomposition for a reason — but here the
decomposition was drawn before any cost data for *this* pass existed, and six
units of measured rate beat the estimate it was drawn from. The remaining
chunk boundaries in the table were sized under the same estimate and should be
re-checked against this rate as each agent comes up.

## Follow-ups

- Pass 1's open follow-up (`coder` K-005, the `verify_workflows.sh`
  false-negative) is **closed** — `tdd-engineer` fixed `Repository._read_structure`
  under `falkor-chat/docs/plans/workflow-diff-absent-key-coordination.md`;
  nothing to re-route.
- **U1 → `cobb` K-020**: `cypher-mcp/server.py:881` advises `docker start
  falkordb-dev`, which no launch path in this repo can satisfy. A one-line
  string fix in another component's code, outside `cobb`'s write remit — needs
  an implementer, small enough to fold into any other `cypher-mcp` touch.
- **U2 → §5's legacy read is dead.** Zero `author`-property entries survive
  anywhere in the graph, so the dual legacy/current read that
  `skills/agent-maintenance/SKILL.md` §5 step 1 mandates "while any legacy
  entry still exists" now has a false precondition. The skill anticipates this
  ("can eventually be dropped"). A `cobb` edit, deliberately deferred until
  this pass closes rather than changing the procedure mid-sweep.
- **U3 → `scripts/audit-team.sh` exits FAIL, pre-existing.** Three check-7
  hits (git email, username, home path) across five already-committed docs:
  `claude/docs/plans/bypass-permissions-subagent-gap{,-coordination}.md`,
  `claude/docs/reviews/bypass-permissions-subagent-gap.md`,
  `docs/plans/doc-reference-convention.md`, `docs/plans/salesperson-ui.md`.
  Not introduced by this pass — `cobb` grepped every file it touched and came
  back clean. Owners are spread across agents, so it wants its own unit.
- **U4 → `qa-engineer` K-007**: three live-verified `falkor-chat` gotchas that
  belong in `falkor-chat/docs/SERVER.md` §1.7, carried ready-to-paste because
  the target is outside `cobb`'s write remit — plus one **code** fix,
  `falkor-chat/config/opencode.example.json` cannot be run verbatim
  (`modelconfig._build_providers` eagerly substitutes every catalog provider's
  `apiKey` at `from_env()`, so the unused `openai` block's
  `{env:OPENAI_API_KEY}` kills startup). An example config no fresh box can run
  is the defect, not a doc gap. Needs routing to a `falkor-chat` doc owner and
  an implementer.
- **U5 → `tico` K-015, and it is worse than the entry claimed.**
  `docs/manuals/graph-ontology.md` §2 still documents `kaizen_team` in its
  flat, pre-M8 shape: the property table calls `author` "the *only* attribution
  mechanism today", the doc states "**Relationships:** none", the Mermaid
  diagram draws four author-stringed nodes, and an FAQ answers "zero
  relationship types — is the graph broken?". Every one of those is now false —
  `author` is gone from all 180-odd nodes and the graph is entirely
  `PRODUCED`/`MENTIONS` edges — and its sample query returns nothing. This is
  the one **end-user-facing** document kind in the repo, so the staleness is
  more costly than in an engineering doc. `manuals/` is `tico`'s, not `cobb`'s.
- **U5 → `entryId` prefix grepping is unreliable in this dataset.** These ids
  are hand-shaped, not `uuid4`, so 8-char prefixes repeat: `cobb` hit two
  false-positive dedup matches (`e3a1f6b2…` against `analyst`'s history,
  `b3f2a1d4…` against `qa-engineer`'s), both different and already-cleared
  entries. Every remaining unit's brief must say to confirm date **and**
  subject before concluding a prior pass already opened a `K-` item.
- **U5 → `falkor-chat/AGENTS.md` never mentions `server/tests/eval/`** — the
  component's always-loaded context file omits the evaluation harness it
  ships. Not filed as a `K-` item (the fact is covered for the consumer that
  exists); worth one line the next time falkor-chat's context file is revised.
- **U6 → `graph-dba` K-008**: two verified CPG-freshness facts whose right
  homes are `skills/joern-cpg/SKILL.md` (the "No `--exclude` flag" bullet) and
  `skills/cpg-analysis/references/freshness.md` (**Limits**) — the latter
  currently states its `.git`-less limitation *as if inherent*, and one of the
  entries shows it is avoidable. Seven agents read that recipe. Kept open on
  remit, not doubt; parking them in a `claude/graph-dba/` file to stay in-remit
  would be exactly the hoarding §5 forbids.
- **U6/U8 → three scratch graph keys need cleanup**: `scratch_cobb_u6`
  (~155 nodes), `scratch_cobb_u6_other` (1 node) and `probe_u8_rename_dst`,
  created for write-probes that could not be settled by reading. `cobb`
  correctly did **not** delete any of them — `GRAPH.DELETE` is destructive and
  reserved to `graph-dba`/`devops` behind their guards. No existing key was
  reused or mutated. **One attribution discrepancy, recorded rather than
  resolved:** U8 reported leaving no key behind and listed
  `probe_u8_rename_dst` as pre-existing from an earlier unit, but U6 — the only
  earlier unit that ran `RENAME` probes — reported exactly two keys, both
  `scratch_cobb_u6*`. The name suggests U8's own. Redis exposes no cheap
  creation timestamp, so this is not decidable after the fact; all three go to
  `graph-dba` for cleanup regardless. The lesson is the same one U7b promoted:
  namespace scratch artifacts per unit, or leave none.
- **U7 → a `qa-engineer` top-up unit is now owed at pass close.** U7 tagged
  `b7d5e214` `MENTIONS`→`qa-engineer` (a general test-design rule: an "assert
  every survivor by label" done-condition cannot catch an over-broad delete
  when spared rows share labels with targets — the assertion must positively
  name a specific seeded non-target row). `qa-engineer`'s own unit (U4) had
  already closed, so that node sits alive with its `PRODUCED` edge resolved and
  one `MENTIONS` edge outstanding. This is the ordinary FR-5 deferral, not a
  defect — but the pass should not be declared closed while it is outstanding.
- **U7 → `architect` K-004 and K-005**, both blocked on remit, both verified
  true. K-004: `llm.py`'s `{"tool_calls": […]}` branch runs *before* the K-035
  `_BARE_CALL_OPEN` guard (`llm.py:311-320`), so `x({"tool_calls":[…]})`
  resolves by probe order alone, with zero test coverage — wants a
  `falkor-chat/docs/BACKLOG.md` test-gap item plus one docstring sentence.
  K-005: `ws:acme`'s label census belongs in `falkor-chat/AGENTS.md`.
- **U7b → the session scratchpad is shared across every delegate, and this
  pass proved it the hard way.** `$CLAUDE_CODE_SESSION_ID` inside a subagent
  resolves to the **parent's** id, so every `cobb` unit in this coordination
  writes into one directory keyed by `teco`'s session. U7b found U7's working
  files (`arch_hist.md`, `ds_hist.md`, `v0models.json`, 08:10–08:27) still
  sitting in what it took to be its own scratchpad; `teco` confirmed
  independently — U6's and U7's files are all still there together. The hazard
  is therefore **not** limited to parallel dispatch, which is how it was
  originally recorded: strictly sequential units leave stale files a later
  delegate can read as its own. Now widened in
  `skills/agent-standards/claude-code.md` § "Bash tool environment". Practical
  consequence for any future pass: have each unit namespace its scratch files,
  or write nothing there at all (U7b used inline `python3` heredocs and left
  nothing behind).
- **U3 → `salesperson/build.sh:68`**: the `elif command -v node` fallback
  accepts any `node` on `PATH` without the `/mnt/` rejection its own
  `npm`-only branch applies. Harmless today (only `npm` leaks in from
  Windows). Parked in `claude/devops/kaizen/plan.md`; `salesperson/` is not
  `cobb`'s to edit.
