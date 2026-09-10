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
| U10 | coder chunk A (12: ≤ 08-29) | `aacf69b25b40cbff5` (retry; `a695c632adfd6d91d` died on a rate limit) | accepted | `claude/graph-dba/falkordb-quirks.md` (+43 lines, 1 prior entry corrected) + `kaizen/history.md`; `claude/coder/kaizen/*` (K-006 consolidated, K-005 closed); 12 nodes deleted | none → — | 180.7k tok, 68 tools |
| U11 | coder chunk B (8: 08-31…09-02) | `a567c2024835b0516` | accepted | `claude/analyst/review-techniques.md` (2 techniques) + `kaizen/history.md`; `claude/graph-dba/falkordb-quirks.md` (1 corrected) + `kaizen/history.md`; `claude/data-scientist/lm-studio-model-notes.md` + `kaizen/history.md`; `claude/coder/kaizen/*`; 8 nodes deleted | none → — | 172.4k tok, 59 tools |
| U12 | coder chunk C (7: 09-03) | `accaa936d2c07807c` | accepted | `skills/python-web-quirks/SKILL.md` (1 fold, 1 merged section from 3 entries, 1 new section, frontmatter) + `skills/README.md`; `claude/coder/kaizen/*` (K-006 4→6 rows); 7 nodes deleted — **`coder` closed out, 0/0** | none → — | 165.7k tok, 64 tools |
| U13 | data-scientist chunk A (10: ≤ 08-30) | `a21cdcb05d80c9e1f` | accepted | `claude/data-scientist/data-scientist.md` (1 clause) + `lm-studio-model-notes.md` (folded, 1 stale claim corrected) + `kaizen/*`; **8 of 10 discarded** — 3 falsified, 1 obsolete, 4 already published at the point of use; 10 nodes deleted | none → — | 178.8k tok, 67 tools |
| U14 | data-scientist chunk B (9: 08-31…09-02) | `a62ee471d5c0c8407` | accepted | `claude/data-scientist/data-scientist.md` (Uncertainty bullet) + `kaizen/*` (K-003 kept open); **7 discarded, 1 promoted, 1 kept open**; 8 nodes deleted, 1 `PRODUCED` resolved (`MENTIONS`→`tico` kept alive) | none → — | 195.5k tok, 78 tools |
| U15 | data-scientist chunk C (10: 09-03…09-07, incl. `6ef71251` arrived mid-U13) | `ae7d55eeefd996462` (died on a rate limit after all writes, before the clear); recovery `aecf048606b51a19f` | accepted | `claude/data-scientist/data-scientist.md` (3 promotions) + `lm-studio-model-notes.md` (JIT bullet rewritten in place) + `kaizen/*` (K-004 kept open); **5 promoted, 4 discarded, 1 kept open**; 10 nodes deleted — **`data-scientist` closed out, 0/0** | none → — | recovery 80.6k tok, 26 tools (dead run unreported) |
| U16 | teco chunk A (8: ≤ 09-01) | `a91f2d49861381ba8` | accepted | `claude/AGENTS.md` (concurrent-write paragraph, closing sentence replaced) + `claude/teco/teco.md` (review-gate clause) + `kaizen/history.md`; **2 promoted, 6 discarded**, both promotions in-place sharpenings; 8 nodes deleted | none → — | 153.0k tok, 49 tools |
| U17 | teco chunk B (12: 09-02, incl. one corrupt all-`PLACEHOLDER` node) | `a3b31a7efe468e5c0` | accepted | `claude/teco/teco.md` (6 promotions) + `claude/AGENTS.md` (2) + `kaizen/history.md`; `claude/cobb/kaizen/plan.md` (K-021); **8 promoted, 4 discarded**; 11 nodes deleted, 1 `PRODUCED` resolved (`MENTIONS`→`architect`) | none → — | 187.9k tok, 55 tools |
| U18 | teco chunk C (13: 09-03…09-06) | `a881239125792e0e0` | accepted | `claude/teco/teco.md` (5 in-place sharpenings + 1 new bullet), `skills/agent-standards/claude-code.md` (agentId resolution scope), `claude/teco/kaizen/history.md`+`plan.md` (K-016 → blocking), 3x `MENTIONS` (2 analyst, 1 data-scientist), graph cleared | none → — | 179.1k tok, 68 tools |
| U18b | teco chunk D (14: 09-07, all arrived after pass open) | `adcb31af6bc3fc428` | accepted | `claude/teco/teco.md` (12 statements from 9 entries, **zero new bullets** — 11 lines changed, 11 removed); `skills/agent-standards/claude-code.md` (2 permission-classifier entries merged); `claude/teco/kaizen/history.md`+`plan.md`; 5 `MENTIONS` edges over 4 entries (3 tdd-engineer, 2 analyst); 10 nodes deleted, 4 `PRODUCED` resolved — **`teco` closed out, 0 produced / 0 mentioned** | none → — | 224.6k tok, 69 tools |
| U19 | analyst chunk A (12: ≤ 08-30) | `adb247a3e028c0606` (killed by a session rate limit at its first tool call, 09-07; resumed in place by `SendMessage` 09-08 — nothing had landed, all 12 edges intact) | accepted | `claude/graph-dba/falkordb-quirks.md` (3 entries: 2 folded, 1 new regex bullet) + `kaizen/history.md`; `claude/analyst/review-techniques.md` (2, both edits to existing material — one **corrected a wrong import-resolution mechanism the file had been carrying**) + `kaizen/*`; **7 discarded**, 3 of them additionally carrying a false or misattributed claim; 1 `MENTIONS`→`devops`; 11 nodes deleted, 1 `PRODUCED` resolved. **Zero new bullets in any always-loaded prompt** — `analyst.md` and `claude/AGENTS.md` untouched | none → — | 184.9k tok, 50 tools |
| U20 | analyst chunk B (11: 08-31…09-01 + the first 5 of 09-02) | `a6db6af4910e607bc` (killed by a session rate limit mid-promotion, 09-08; resumed in place by `SendMessage` after the reset — four promotions already on disk, nothing logged, all 11 entries intact; **resumed a second time**, which corrected a real accessor defect and refuted my stamp finding) | in-flight | 7 promoted / 4 discarded, all 11 cleared; `claude/analyst/{kaizen/history.md,kaizen/plan.md,review-techniques.md}`, `skills/agent-standards/claude-code.md`, `skills/python-web-quirks/SKILL.md`, `skills/README.md`, `claude/graph-dba/falkordb-quirks.md`, `claude/data-scientist/lm-studio-model-notes.md`, +3 `kaizen/history.md` | teco re-derivation → **accepted**; my "wrong stamps" finding was itself wrong (see below), the accessor defect it surfaced was real and is fixed | 236.4k tok, 35 tools |
| U21 | analyst chunk C (10: all of 09-02) | `a9b3f7141f0403e35` (resumed twice) | accepted | 6 promoted / 4 discarded (2 **falsified**), all 10 cleared + curator-cleared my false `7c4e91a2…`; `claude/graph-dba/falkordb-quirks.md`, `claude/analyst/review-techniques.md`, `claude/analyst/kaizen/{history,plan}.md`, `claude/graph-dba/kaizen/history.md` | teco re-derivation → **accepted** (2 resumes: evidence recount, then provenance of an unattributed bullet) | 224.0k tok, 96 tools |
| U22 | analyst chunk D (11: 09-03) | `a9a502324e0cf4ca5` | accepted | 8 promoted / 3 discarded, all 11 cleared; `review-techniques.md`, `python-web-quirks/SKILL.md`, `agent-standards/claude-code.md`, `lm-studio-model-notes.md`, `skills/README.md`, +3 `kaizen/history.md`; `falkordb-quirks.md` **held** | teco re-derivation → **accepted**, paired-control reproduction of the pipe finding | 238.8k tok, 112 tools |
| U23 | analyst chunk E (13: 09-07, arrived after pass open) | `a096fa1ae04ee4cae` (resumed once, for the AST correction) | accepted | 12 promoted / 1 discarded, all 13 cleared, **13 entries → 4 edits** (six became one six-part section, three a fold); `claude/analyst/{review-techniques.md,analyst.md,kaizen/history.md}`, `claude/graph-dba/{falkordb-quirks.md,kaizen/history.md}`, `claude/cobb/kaizen/{history,plan}.md` | teco re-derivation → **accepted after one correction**: 68→65, 42→40, per-file 3/3/14; re-derived at both revisions under 5 definitions. `rq()` defect confirmed by executing the helper | 190.6k + 210.5k tok, 71 tools |
| U24 | analyst chunk F (12 of the 14 held at dispatch: all 09-08) | `a554b6fb7d89af6f0` | accepted | 10 promoted / 1 discarded / **1 kept open** (`graph-dba` K-009), **12 entries → 7 edits**; `claude/analyst/{analyst.md,review-techniques.md,kaizen/history.md,kaizen/plan.md}`, `claude/graph-dba/{falkordb-quirks.md,kaizen/history.md,kaizen/plan.md}`, `claude/cobb/kaizen/history.md`, `skills/python-web-quirks/SKILL.md`, `skills/README.md`; 2x `MENTIONS`→`graph-dba` | teco re-derivation → **accepted, no correction needed** — the dataclass narrowing reproduced across three annotation shapes, both stdlib line refs exact, NaN asymmetry and all counts exact | 248.5k tok, 94 tools |
| U25 | analyst chunk G (the 2 deferred; re-queried at dispatch 09-09 — `analyst` held exactly these 2, nothing new arrived) | `a271cfb21a2dbe1af` | accepted | 2 promoted / 0 discarded, both `DETACH DELETE`d, **2 entries → 1 file, 2 sections**; `claude/analyst/{review-techniques.md,kaizen/history.md}`, `claude/cobb/kaizen/history.md`. **`analyst` produced inbox now 0.** | teco re-derivation → **accepted, no correction needed** — second consecutive clean unit; tombstone byte-identity reproduced under an independent anchor pair to the entry's own cited digest, growth figures digit for digit | 147.6k tok, 39 tools |
| U26 | `graph-dba` (its 1 produced entry; its 2 `MENTIONS`-only nodes are U24's kept-opens and are **out of scope**) | `afe9fd679115a7900` (resumed once, to fold the K-008 note into the item body) | accepted | 0 promoted / **1 discarded** / 0 kept open, `DETACH DELETE`d; the entry was accurate when written and **dead 40 minutes later** (`6012ddb`). Collateral find: `6012ddb` overtook **both** K-008 facts. `claude/graph-dba/kaizen/{history.md,plan.md}` | teco re-derivation → **accepted after one correction** (K-008 asserted "both facts verified true" above a note refuting it; rewritten present-tense, 2266→2075 words) | 145.3k + 155.8k tok, 33 tools |
| U27 | `data-scientist` (its 1 produced entry; its 1 `MENTIONS`-only edge is out of scope). Re-derivation **pinned to `d45e5ff`**, not `HEAD` — `model-bench/` is another session's live area | `ad47fd00ddc79e335` | accepted | **1 promoted / 0 discarded**, cleared; folded onto an **existing** prompt bullet (28→28 bullets, +80 words), model-bench-specific half discarded as already published. `claude/data-scientist/{data-scientist.md,kaizen/history.md}`. **`data-scientist` produced inbox now 0.** | teco re-derivation → **accepted, no correction — and it corrected *my brief*** | 161.8k tok, 34 tools |
| U28 | `graph-dba` **code unit** on `skills/joern-cpg/scripts/pipeline.sh`: fix **K-009** (`rq()` returns 0 on a bare runtime-error reply) + confirm and close/re-scope **K-008**. Not a distillation unit | `a71e467eb629d98ad` (resumed once, to correct a case count) | **accepted** (`48882d8`, `682fbed`) | **K-009 fixed by removing the blacklist**, not widening it: success is now recognised positively (reply's last line must begin `Query internal execution time:`), non-query commands refused rc 2. Second defect found and fixed in the same block (a read-back that never ran reported as *did not land*). **K-008 closed**, both facts confirmed by execution. 7 files | `analyst` (diff-scoped, `a8f29fed07dd1847e`) → **approve with suggestions**, two passes (`7edf98c`, `4bfe925`). Pass 1's 2 majors closed in `48882d8`; Pass 2 retracted 2 of its own claims in the implementer's favour and judged its own Pass 1 recommendation wrong by execution, then raised Major 7 — closed in `682fbed` against a **falsifiable stopping condition** (4 mutation shapes must redden + clean-tree control), all met. Review at `docs/reviews/rq-execution-gate.md` | 141.2k + 150.5k + 156.4k tok, 59 tools |
| U30 | curator clear of the 2 entries held pending K-009 (`b7f3c2a1…`, `4f9c21ae…`) — orphans on `MENTIONS`→`graph-dba`. Fresh `cobb`, not U24's (248.5k ctx) | `a714ad00b16e9eccf` | accepted | **both cleared** after re-executing `rq()` at `682fbed` with two passing controls; abort reproduced under a harder shape (4,999 producible rows → one bare line). `graph-dba` now **0 produced / 0 mentioned**, absent from the census. `claude/graph-dba/kaizen/history.md` | teco re-derivation → **accepted, no correction** — 83/11/72 reconciled exactly | 130.0k tok, 34 tools |
| U31 | **the orphan backlog** — 11 `MENTIONS`-only nodes, 08-30→09-07, the oldest population in the graph; item 3 of the resume plan, **never before attempted** because every prior unit was producer-organised | `ab74825681a290ac9` | accepted | **9 promoted / 1 already-promoted / 1 kept open**; 10 nodes cleared, 11 of 12 edges resolved; **19 files**, incl. a new `tdd-engineer` knowledge base. Orphans **11 → 1**. The dispatch hypothesis was **refuted** — see below | teco re-derivation → **accepted, no correction** — 73/1 census, bullet counts 10/8/19 unchanged, `HEAD == BASE` on all 18 tracked | 214.6k tok, 69 tools |
| U32 | **generation four** of the `rq` guard defect: a literal command after a **backslash line continuation** is inside the check's stated reach and outside its line-based mechanism. Fresh `graph-dba` (U28's carries 217k) | `aa1435d89cf1f9eb5` | accepted (`4df5e45`) | **closed against a coverage probe, not a shape list**: 35 call-site forms on 3 axes, each adjudicated twice (bash vs the *extracted* delivered reader); 30 covered, **5 blind and stated as the bound**. **Generation five caught in flight** by the probe (bash joins continuations with *nothing*, not a space). Suite 15→16 PASS | teco re-derivation → **accepted, no correction** — my own bypass now FAILs at `pipeline.sh:432` in an isolated copy | 141.8k tok, 26 tools |
| U33 | one bound stated **two ways**: `test-stamp-wiring.sh:340` scopes it to a `GRAPH.<word>` run, `pipeline.sh:361` does not — so `rq "$Q" PING` sits inside the latter's stated reach and outside the mechanism. Consistency defect, not a mechanism gap | `aa1435d89cf1f9eb5` (resumed, not respawned) | **accepted** (`00bebdc`, `f682bfa`) | **converged on the scoped claim**; third sub-case (*not spelled `GRAPH.…` at all*) now named identically in both files, and a sentence in `test-stamp-wiring.sh` that had drifted the same way tightened too — **which I had not flagged**. Pinned by probe form `B15` (`rq "$Q" PING`), 35→36 forms. **No third copy of the bound**: `SKILL.md` and `freshness.md` are inventory lines, both over-stating the requirement (site 1 passes *nothing*), corrected and given a **pointer** rather than a third copy. Suite unchanged 16 PASS / 0 FAIL / 3 sites | teco re-derivation → **both controls re-run independently** — the widened-regex mutation reddens **6** blind rows and exits 1, so the probe pins the bound's *scoping*, not only its form list; file restored byte-identical and green. My own unfiltered repo sweep agrees: no third statement of the bound. **Sent back** for 3 structural fixes to `history.md` (orphaned bullet, missing blank line, one stale *five-form* figure the follow-up itself created); all 3 applied and re-verified at the seam, `grep` for both stale strings now empty, the entry's three figures agreeing | 170.6k tok, 13 tools |
| U34 | **`graph-dba` inbox — 1 entry** (`b7d3f0a2…`, bash joins a backslash continuation with **nothing**), plus a **rider**: `cobb`'s own retraction write-up in `claude/tdd-engineer/guard-testing-techniques.md` is half wrong — right about the scoped file, false about `pipeline.sh` as it then stood. Smallest-first resumes here | `acaa8fc45105704e6` | **accepted** (`813fd71`) | **1 promoted, 0 discarded** — my discard steer **overruled**, and rightly: the fact already stood in **three** places, not the one I named, but all three stated it as *what happened here*; what was missing was the **portable rule**, now point 4's closing sentence. A fourth copy in `devops/ops-quirks.md` rejected on the arc's own lesson. Rider applied: the retraction paragraph now says it was right about one file and wrong about the other, ending on *checking a bound means checking every place it is stated*. **4 files** | teco re-derivation → **accepted, no correction** — probe greps re-run independently (36 forms / 6 blind, matching); `A15` confirmed the *flagged* continuation form and `A14` the blind wrapper, so point 1's six-form list is right; graph census re-queried: **77 nodes, `graph-dba` absent**, 1 deliberate orphan; tree touched exactly 4 files | 137.4k tok, 41 tools |
| U35 | **`qa-engineer` inbox — 1 entry** (`7f3c9a21…`, `TestClient` defaults to `raise_server_exceptions=True`, so an acceptance test on the default can never see the real bare 500). Two open questions handed over, not decided: one section or two in `python-web-quirks`, and whether `SKILL.md:47`'s audience line is stale for omitting `qa-engineer` | `ac2f1a4ba4ac40333` | **accepted** (`3f769ca`) | **1 promoted**, and re-derivation **narrowed the claim**: the client-parity `False` buys is `ServerErrorMiddleware`'s, not the flag's — outside it `TestClient` synthesizes a headerless, empty-bodied 500. Three-arm evidence incl. a **live uvicorn control**. Two sections kept, with a stated trigger (K-026) instead of a later judgement call. Audience list was stale in **4** places, not 1 — `6b0a401` had removed the reciprocal clause from all four consumer prompts. **7 files** | teco re-derivation → **accepted, no correction** — census 77/`qa-engineer` absent; root `AGENTS.md` edit is one clause and the >700 bar is clean. **My own char-count contradicted cobb's until I fixed my instrument**: counting raw YAML text gave 4,685/1,032, `yaml.safe_load` gives **4,584/1,003** — cobb's figures exactly | 158.3k tok, 65 tools |
| U36 | **`tdd-engineer` inbox — 3 entries**, all model-bench S1: a `len()`-derived-constant wiring probe, and two views of one zero-variance degeneracy (a constant bootstrap sample makes a clamp undetectable). Their `suggestedHome`s **disagree** on the pair. Rider: `skills/agent-standards/SKILL.md`'s frontmatter **does not parse as YAML** | `ad439355235f4f7a6` | **accepted** (`4ca4098`) | **3 promoted, merged to 2 sections in a new KB** (`estimator-test-fixtures.md`, 89 lines) — the pair *was* one mechanism, argued from the captures' own history: one implementer, one unit, used a degenerate sample as a clamp's worked case **and** declined one as unconstructible. Re-derivation added a condition the captures missed: hiding a clamp needs zero width **and** the point inside the support. **My YAML rider inverted** — see below. K-027+K-028 → **K-029**; K-025 row fixed. **8 files** | teco re-derivation → **accepted, no correction** — census **74**, `tdd-engineer` absent; strict-parse scan re-run over all **24** frontmatter files, now **3** failures not 4, all agent definitions; `agent-standards`' description **byte-identical at 475 chars**; `claude/AGENTS.md` at **2,491 words**, at its own ~2,500 bar | 191.6k tok, 61 tools |
| U37 | **`analyst` inbox — 5 entries**, re-queried at dispatch. **Three are about work this pass itself gated, and at least two look already-promoted** — so the **discard bar is the subject**, not the promotions. `RESULTSET_SIZE` is already stated in 4 places and `graph-dba` recorded that it answered `analyst`'s own open question *in the negative*. Two need a **routing call**: `git grep -c <rev>` has no home in this layout, and the median-latency done-condition is closest to `qa-engineer` | `a03b3f08a13aeeb01` | **accepted** (`af85412`) | **4 promoted, 1 discarded** — my discard steer **held this time**: `c3d81f7a` was already published *including* its narrow sub-claim, argued from the `falkordb-quirks.md` bullet **read whole** (710–724), not a grep hit. Both routing calls decided against my steer's default: `git grep -c <rev>` **half promoted** — its first half was already the section's rule, the unrecorded half is the BRE-vs-`-E` dialect trap — and the median entry ruled to **`analyst`, not `qa-engineer`**, on the catalogs' own words (fires where a done-condition is *written or gated*). Re-derivation **sharpened** the `$?` entry: the `if !` consumes the status, so the command substitution is innocent and removing it does not repair the call site. **5 files** | teco re-derivation → **accepted, no correction** — every load-bearing figure re-derived independently: `git grep -E` → **exit 128** vs BRE's `report.py:3`+`results.py:3` at `c523a35`; bash arms **0/0/2**; `app.py:369`/`config.py:247` as cited; probe **36 rows / 3 axes / 6 blind** confirmed (my pipe-delimited instrument was wrong again — the heredoc is whitespace-separated); `review-techniques.md` **11,003 w / 29 §§ / 0 lines >700**; `analyst.md`'s 7 long lines **pre-existed at HEAD**; `claude/AGENTS.md` untouched at **2,491 w**; every KB-carrying agent now named in `README.md`. Graph: all 5 ids gone, `analyst` **0 produced / 0 mentioned / node absent** | 209.7k tok, 64 tools |

Deliverable paths above are the guaranteed minimum (every pass touches the
agent's own kaizen files and the graph); each row is rewritten on delivery with
the actual promotion targets — agent prompts, knowledge bases, project docs.

| U38 | **`cobb`'s own inbox — 8 entries**, re-queried at dispatch. **`cobb` gates its own captures** — the first unit in this pass whose producer and distiller are the same agent, so the usual independent check on the producer's framing is absent. And **almost none of these belong in `cobb`'s own artifacts**: the homes are `graph-dba`'s FalkorDB KB, `tdd-engineer`'s guard KB and `python-web-quirks`. **Routing is the subject, not promotion.** Two strong leads handed over: `2f944aa9` looks published at `falkordb-quirks.md:736`, and `3b8ded5d` **contradicts three shipping documents** plus a `pipeline.sh` comment dated a day after it | `aa35aec2cab8c5750` | **accepted** (`e79fb61`) | **4 promoted (one a half), 4 discarded** — routing-first was right: **not one entry landed in `cobb`'s own artifacts**; they went to `analyst`, `graph-dba`, `tdd-engineer` and `python-web-quirks`. Both my leads resolved, in opposite directions: `2f944aa9` was a **full** discard, not the half I predicted (the sub-claim I thought was new is at `cypher-mcp/README.md:118–127` in the entry's own words), and `3b8ded5d` was **superseded**, as hypothesised — but my discard steer on `c4f1a2be` **failed again**, U34's lesson repeating: four sites state *an undefined function aborts at 127*, none states that **`set -u` aborts at 127 too**, which is what breaks the exact-rc oracle those four prescribe. **The half-promotion is a defect in the procedure this pass runs on**: §5 step 1 reads `k.fact`/`k.evidence` through a tool that truncates at 300 chars and never said so. **10 files** | teco re-derivation → **accepted, no correction** — bash arms re-run: unbound at top level, in a function, and command-not-found all **127**, set-but-empty **0**; `CYPHER_MCP_MAX_CELL` default **300** confirmed at `README.md:383` and the paging mechanism re-derived live (`size(evidence)=1032` → `size(substring(…,250))=782`); the `81b43cd` worked example holds — `freshness.md:181` claimed the load *"overwrites it wholesale, `NOTE` and `MARKER_ORIGIN` included"* while the stamp at that same commit was `SET b.BUILT_AT = …` over named properties touching **neither** key; supersession confirmed against the **emitted Cypher** (`git-provenance.sh` → `SET b = {$_CPG_MAP}`) and `5417f0e`'s timestamp, hours after the capture; `python-web-quirks`' description **byte-identical at 4,584**; `claude/AGENTS.md` untouched at **2,491 w**; `review-techniques.md` 11,445 w / 30 §§ / 0 lines >700 | 211.2k tok, 78 tools |

| U39 | **`coder` inbox — 9 entries**, re-queried at dispatch. **`coder` is the only agent this pass has drained that owns no knowledge base** — `claude/coder/` holds just `coder.md`, `hooks/` and `kaizen/` — so every promotion must land in another agent's artifact, a skill, or a component's docs tree, or force the decision to create one. **That decision is a stop-and-ask fork**, briefed as such. Three entries are falkor-chat-specific `project docs`; two visible pairs (`ThreadPoolExecutor` ×2, AST-guard reach ×2). Lead: `c7f1a3d2` looks published at `python-web-quirks/SKILL.md:603–629` **and in that skill's description** | `a28c977672e764a6d` | **accepted** (`88d13db`) | **6 promoted (2 halves), 3 discarded** — **neither fork fired**: no `coder` KB created, no `falkor-chat` file touched. The homes were `python-web-quirks` (the two `ThreadPoolExecutor` entries **merged on U36's criterion — one mechanism, not one topic** — plus a new `app.state` lifespan section), `tdd-engineer`'s guard KB and prompt, and `analyst`'s residual item 5 **rewritten in place**. The three `project docs` entries landed **nowhere near** `falkor-chat/docs/`: their premises are already in `storefront.py`'s docstring and `SERVER.md`'s own row, figure-for-figure. `suggestedHome` again predicted **zero** of nine homes. **10 files** | teco re-derivation → **accepted, no correction** — the 5 executor arms re-run: `qsize/threads` `0/0 → 0/1 → 0/1 → 0/1`, and **`shutdown(wait=True)` leaves `qsize=1`** (the `None` sentinel), so the entry's own oracle is unsound in both directions; `BrokenThreadPool → BrokenExecutor → RuntimeError` confirmed; the AST frontier comprehension returns **`_ex` *and* `_run_turn`**; `git grep -cF` at `e79fb61` reads `stats.py:1` beside `test_stats.py:2`, and the constructed 2-vs-3 case reproduces. **My own near-miss:** I read the section's two demonstrations as one and thought the 2-vs-3 figure was false against `test_stats.py` — the delivered text says *a file carrying `X[0]` once on one line and twice on the next*, and does so plainly. Budgets match to the word on all six live files; `claude/AGENTS.md` **2,491 w**; `python-web-quirks` description **byte-identical at 4,584** | 237.0k tok, 80 tools |

| U40 | **`architect` inbox — chunk 1 of 2**, split on the date boundary: the **seven 2026-09-07 entries** (15 total is over the ~12 chunk bar). **Six of the seven are one theme** — grep-based done-conditions and residuals — whose home is `review-techniques.md`'s *closed six-item* list, a section **three consecutive units have now touched**. The failure mode briefed against is a seventh item that items 1–6 already cover. Near-certain discard flagged for disproof: `b1f2c7a4` reads like **item 6 itself**, which U23 promoted from `analyst`'s near-twin `b1f0c7a4…` | `a5f6321e6acf3612d` | **accepted** (`a1e2234`) | **4 promoted, 3 discarded, all 7 cleared; zero new items, zero new sections** — every promotion a **fold** into existing prose, and all four into `claude/analyst/review-techniques.md`. The six-item list was read whole before any disposition and stayed six. `f4ee08b9` → **item 2**, supplying the selection rule item 2 presumed you already had (*the attribute the new union member **lacks***, derivable from the type change alone). `05192600` → **item 5** with a third shape and, crucially, the *consequence*: the cheap way to resolve a self-contradictory done-condition is to skip the test, so a mutation ships green. `b1f2c7a4` → **half**: my near-certain discard was right on the diagnosis (item 6 already carries both halves in its own words) and wrong on the entry — the remedy **from the plan author's chair** was absent, and *narrowing the path is the tempting wrong fix*. `0a4b6b2e` → **derived check 2**, routed to `analyst` not `architect` on whose decision it changes: the check is structurally invisible to the author's own sweep, `architect` owns no KB, and its only fold target `architect.md:51` is **1,617 chars**. `f6119439` **discarded as false** — cobb's own first re-derivation reproduced it and was worthless twice (ran under the ugrep shim; glob `*.py` against `conftest.py`, a filter that could not bite); the arm where the glob **cannot** match inverted it. **3 files** | teco re-derivation → **one correction sent** — six of seven hold. `--include` filters by base name regardless of how the file arrived: `--include='*.txt'` on an explicitly named `.py` file → **no output, rc 1**, with and without `-r`, GNU grep 3.11. ERE/BRE split reproduces (rc 0 vs rc 1; the constructed name is labelled *by construction* and correctly does not exist in the file). `stats.py:159` at `5878014` is one line carrying both `_percentile` calls. `start_workflow_run`'s **own** docstring (`services.py:2028`) states the raise — discard sound, and cobb's withdrawal of the `repository.py` class-docstring promotion was right. Budgets exact: **11,568 → 12,045 w**, 30 → **30** §§, **0** lines >700; `claude/AGENTS.md` **2,491 w** untouched; `python-web-quirks` description byte-identical to `88d13db`. Census **53** = cobb's 52 + its own post-count capture; `architect` **8**. **The correction:** derived check 2's *rule* is sound and kept, but its **live instance is stale in the present tense** — `continuous_verdict()` was given a required keyword-only `support` at `-ml` v1.17 *in answer to* the very v1.13 raise cobb cites (`plan:25` records the closure; `-ml:1615`, `:1555–1566`; pinned by `test_stats.py:2380` and `:2390`). An analyst applying the section's own re-derive caution would find `support` present and discount the rule with the instance. **Correction accepted:** the rule is byte-identical, the instance is re-tensed as a *worked, closed* finding, version-anchored (v1.13 raise → v1.17 absorption → v1.14 closure) and **pinned to executable tests** rather than to documents — so re-deriving it now confirms the paragraph. cobb also extracted the general remedy my correction only implied: *when a producer cannot forward an argument, move the derivation **into** the producer rather than widening the seam between the two documents.* Its own diagnosis of the error is the one worth keeping: *it cited a changelog faithfully and inherited its tense along with its content* — the citation was accurate **at** v1.13, and the question never asked was whether the document had moved since. Final: **12,156 w** (base 11,568, +588), 30 §§, 0 lines >700 | 199.0k tok, 68 tools |
| U41 | **`architect` chunk 2 of 2 — the eight 09-08/09-09 entries — plus one `cobb` entry folded in deliberately** (`2746ee65`). The fold is a considered deviation from *one agent at a time*: three of the nine land on **the same paragraph** of `skills/agent-standards/claude-code.md`, and splitting them across two units a day apart is precisely the concurrent-edit hazard this coordination keeps logging. Still one agent running. Two grounded leads pointing **opposite ways**: `c3f7a1e2` is a likely full discard (published at `python-web-quirks/SKILL.md:648-662` in that text's own words, landed by U39 from `coder`'s near-twin), while `3f5b1c02` is briefed as a **predicted half — I told cobb in advance where I expect my own steer to fail.** The published table measures a pool that *did* start a thread; the entry's claim is the **empty-`_threads`** case, a different mechanism with the same reading. Five straight failures of the same shape (U34, U38 `c4f1a2be`, U40 `b1f2c7a4`) say the missing piece is the **generalization, not the fact** | `a74ea49194ca86329` | **accepted** (`a6a676b`) — killed once mid-survey by a rate limit and resumed in place; both surfaces verified clean before the resume, nothing needed unwinding | **6 promoted (2 halves), 3 discarded, all 9 cleared — `architect` is now absent from the graph, 0 produced / 0 mentioned.** **Both leads split exactly as briefed, in opposite directions.** `c3f7a1e2` full discard *and its one unpublished claim is false*: the entry says all four raises are bare `RuntimeError` so type cannot discriminate, while `python-web-quirks:656-658` already carries the **corrected** version — `BrokenThreadPool → BrokenExecutor → RuntimeError` is separable by type where the shutdown pair is not, so the shipped text **supersedes** the entry rather than equalling it. `3f5b1c02` promoted in **full** — the sixth straight failure of my discard steer, and the first I **predicted in the brief**. The `claude-code.md` fold carried three entries in one edit as designed. **3 files + 4 histories** | teco re-derivation → **accepted, no correction** — the cold-pool arm re-run independently: `submit` raised, `qsize` **1** / `_threads` **0**, `shutdown(wait=True)` returned in **0.000002 s**, `ran=[]` after settle, post-shutdown `qsize` **2** against the warm row's **1**; the ordering sharpening reproduces too — `['abandoned', 'reviver']`, the refused job running **ahead** of the one that revived the pool. Shim arms confirmed with each binary named: ugrep **7.8.4** vs GNU **3.11**, an explicitly named gitignored file returns **2 under both**, recursion diverges (GNU 49 / shim 0), and `--no-ignore-files` restores the shim to **49**. My own count differed from cobb's on the recursion arm **because the pattern differed** — which corroborates its decision to ship **no figure** from the entry's unstated-pattern *84 files* claim. Citations exact at the entry's own sha (`test_storefront_api.py:3939`, `:3995` — I had guessed the wrong file, cobb had it right). All four budgets exact to the word; 0 new sections; `claude/AGENTS.md` **2,491 w** | 193.1k tok, 50 tools |
| U42 | **`teco` chunk 1 of 4 — thirteen 2026-09-08 entries**, including the malformed 40-char id `7b41d process-e2a-…` (literal space, U17's corruption class). **The first unit of this pass with only two parties**: `teco` wrote the entries, the brief and the gate, so **no independent party has ever judged their framing**. Briefed accordingly — *the discard bar is the subject, not the promotion bar* — with the coordination doc's prose sections (`105–502`, ~10k words) named as the surface to read **whole**, since that is where I narrated the same findings as they happened. Four leads, all disprovable: `c1e8b530` was published by cobb's **own** U41 edit an hour earlier; `3f8c1a92` + the malformed entry are the same kill-resilience family as U41's `00c5498f`; `d7a4e916`/`f4c1a7e2`/`c9a4515c` look like restatements of `## The same defect shape`; and `b9f27c04` is a **partial I can name precisely because I hit it today** — I asserted `count == 1`, it passed, and the row still landed in the wrong table | `a904a5d98e993c179` | **accepted** (`117df76`) | `claude/teco/teco.md` (+583 w) · `skills/agent-standards/claude-code.md` (+587 w) · `claude/analyst/review-techniques.md` (+162 w, item 7) · `cobb/kaizen/{history,plan}.md` (K-030 opened) | teco re-derivation → **12 of 13 hold; one shipped claim falsified.** All 13 cleared (`entryId IN [...]` → **0** leftovers); census **50** = cobb's close of 49 + one further arrival from the concurrent coordination, orphan untouched. Three-way word check clean on all four files — `a6a676b` == `HEAD` == baseline across **16** intervening model-bench commits, none touching our paths — and every reported figure exact (`teco.md` 7,318→**7,901**, `claude-code.md` 10,269→**10,673**, `review-techniques.md` 12,489→**12,651**, `claude/AGENTS.md` **2,491** untouched). `7c1f0a94` discard confirmed at `claude/AGENTS.md:197`, and cobb is right that the path-limited commit is the better remedy than the entry's own. Hook finding verifies structurally: `guard-coordination-doc-writes.sh` only at `~/.claude/agents/teco/teco.md:11`, `settings.json`'s `hooks` an **empty object**. **The correction — the pass's recurring shape once more, correct mechanism with an over-wide reach claim.** `0da6c8d6` shipped *"`ls -t` on `tasks/` enumerates **every** id the session ever dispatched"*. False: this session's `tasks/` holds **24** entries that are **two unrelated kinds** — **17** symlinks to real subagent transcripts and **7** plain files that are my own persisted Bash tool-results (two written by this gate's own `git diff` calls). That conflation is where **66** came from, and it reads 24 an hour later because the tool-results are pruned and the symlinks are not. The canonical store is the symlink's target directory, `~/.claude/projects/<slug>/<session>/subagents/agent-<id>.jsonl` — **45** here, against `tasks/`'s **17**; `comm -23` puts **28** ids in `subagents/` with no `tasks/` entry at all (38% coverage). Spot-checking two ids could not catch it: both happened to be symlinked. The finding **survives stronger** — the filesystem does enumerate every id, one directory over — and gains a surface cobb had no reason to look for: a sibling `agent-<id>.meta.json`, 45 of them, keyed `agentType`/`description`/`model`/`spawnDepth`/`requestShape`, which retires the brief-grepping heuristic it shipped. The `teco.md` *"no agent-enumeration **tool**"* correction stands unchanged and is better supported. Also flagged: version stamp **2.1.266** vs `claude --version` **2.1.267** today. **Two near-collisions cobb could not see, both vindicating its own complete-id rule:** it cleared `3f8c21a9` while `3f8c21ad-6e94-…` stays live from the other coordination (differing at char **8**), and `d41b9c37` against live `d41b7a92` **Correction applied, then killed by the sixth rate limit (HTTP 429) mid-repair** — the first kill of this pass to land *after* acceptance. `claude-code.md` was already at **45** insertions carrying the rewritten bullet, while `cobb/kaizen/history.md` still held the falsified account verbatim; resumed in place after the reset for the record side only (see `## Six platform failures`, rule 4). **cobb's revision beat my correction on diagnosis:** I attributed 66 → 24 to pruning, and it found that a `…/*/tasks/*.output` glob spans **every** session directory under the project slug — **78** files, which I confirmed — so the instrument aggregated across a boundary it never named, a likelier origin of 66 than pruning. Two figures inside that correction were themselves wrong (**6** sessions and **five** session ids, against ten directories of which **four** hold files) and went back with it. **Both repaired and re-verified.** cobb found the exact origin of its own `66`: a directory count over `…/*/tasks` returns **6**, and there are **three** denominators — **10** session dirs exist, **6** carry a `tasks/` subdir, **4** hold any output — so the obvious sanity check returns the one that is neither, which is what makes a wrong per-session figure look checked. All three reproduce here. It also **withdrew my pruning explanation** as unmeasurable and wrong: `tasks/` read 24 across five hours unchanged, while the *glob* grew 66 → 78 as the concurrent session dispatched — two different quantities, never a drop. I re-read the glob at **95** two hours later (`c7dd44c9` 41 → 58), which confirms the growth thesis rather than the shipped figure, correctly framed as an as-of reading. Version stamp **2.1.266** kept and justified: `grep -o '"version":"2\.1\.[0-9]*"'` on the U41 transcript returns it uniquely, so it is the as-of stamp for the observation, and 2.1.267 would make it wrong. `history.md` rewritten — the only surviving hits for the retracted claims are inside its own retraction narrative. Final: `claude-code.md` **10,856 w**, `claude/AGENTS.md` **2,491** untouched | 257,666 tok / 108+5 tool uses / 155 min across two kills |
| U43 | **`teco` chunk 2 of 4 — twelve entries, the remainder of 2026-09-08**. **Framed as U42's inverse.** U42 was the two-party unit; this one mostly is not — `PRODUCED.sessionId` puts **seven** of the twelve in *other* `teco` coordinations (`a2d1489d…`, `018NnCEHvvYdHyqEatoYxLsY`), three at `null`, and only `2d8f30b7`/`4e7a15c3` in this session. The independence problem shrinks; a different one replaces it — several are about mechanisms **this pass has since published**, in text written days after the capture, so familiarity from our own narration is not a discard reason. Six disprovable leads, each naming the published site to read: `3a93075c` against the Guardrails stopping-signal paragraph; `0e2a0bf5` head-on against the grant bullet U42 just landed (disjointness, where `claude/AGENTS.md:197` owns atomicity); `e41b7d06` against the **closed three-item list** of what makes units sequential — a fourth item would force re-checking its *and only the first is visible in a diff* framing; `50f0ae6b` against both sections U42 touched; `2d8f30b7` against the coordination doc's own depth-limited-scan section; `6fbc6ecb` against the `guard-testing-techniques.md` U41 grew. Five entries deliberately unled. Carries U42's own diagnosis of its miss as the standing check — *the reach claim was the one part you had no instrument pointed at* | `a41e3696567e1ceb3` | **accepted** (`3804263`) | `claude/teco/teco.md` (+377 w) · `claude/analyst/review-techniques.md` (+499 w, 1 new §) · `claude/tdd-engineer/guard-testing-techniques.md` (+306 w, 1 new §) · `teco/kaizen/history.md` · `cobb/kaizen/{history,plan}.md` · `claude/README.md` | teco re-derivation → **accepted, no correction.** 9 promoted, 3 discarded, 12/12 cleared (`stillPresent` **0**). **The headline reproduces exactly.** `50f0ae6b` collided with a *third* section, not either I flagged — § *A "this already exists" claim is a grep away from confirmation* — whose published 2026-09-08 measurement **does not reproduce**. Re-derived against the live tree: `never re-widened` raw **2**, whitespace-flattened **2** (so hard wrap, the mechanism the section named, contributes **nothing**), flattened *and* emphasis-stripped **5** (lines 17, 4383, 4911, 7193), and `never re-scoped` — the phrase the published measurement counted — **0** under every normalisation. True rule, false mechanism, remedy blind to the real one; repaired per the file's own tombstone doctrine, rule kept and the unreproducible measurement removed rather than re-argued. The shipped repair bounds its own reach in its last sentence, which is the U43 instrument working. Two re-derivations changed a disposition and both check out: `continuous_verdict`/`ContinuousVerdict` are now implemented and pinned at `model-bench/tests/test_report.py:1798`, so `900c9508`'s rule shipped without its stale instance; and `claude/coder/coder.md` contains **0** occurrences of `mutation`/`mutant`/`rejected` while `tdd-engineer.md:41` already carries the rejected-design mutant rule — so `b83e5c17` correctly routed to the *briefing* side rather than being discarded. The new `f6c3b820` section re-derives cleanly too: bare-name grep **1** vs definition-form **0**; `bash -n` **rc 0** on a script that dies **rc 1** on its first executed line (bash 5.2.21(1)); and `skills/joern-cpg/scripts/pipeline.sh` now shows 4 `replay_stamp` hits with `:419` the definition and `:398` a comment — the count moved without the check changing, as the section says. Budgets exact to the word; sections **+1** each in both knowledge bases (`##` 30→31 and 3→4). **cobb caught its own reach-claim defect twice**: it softened `b7f1c2a4` from *coordinator-only* to the context-isolation mechanism, and it drafted the `suggestedHome` tally as *0 of 12* straight from **my** asserted prior before counting and finding **8 of 12** — my brief had stated a running statistic as fact and nearly bought a fabricated figure back. Captured as `f2b6ae44` | 225,663 tok / 86 tool uses / 18.9 min |
| U44 | **`teco` chunk 3 of 4 — twelve entries, the 2026-09-09 bulk**. **The most foreign chunk yet, and the most recently contradicted.** Seven of twelve are from other coordinations — six from `c7dd44c9…` (model-bench), one from `018NnCEHvvYdHyqEatoYxLsY` — and those six all carry `createdAt` back-dated to exactly `2026-09-09T00:00:00Z`, invisible to any recency ordering: the hazard cobb flagged at U42, and the chunk where it bites. Several sit on territory **this pass published in the last two units**, in text days newer than the capture — the sharpest test of the discard bar, which U43 showed cuts both ways. Seven leads, each naming its site: `c93a7e15` against the *Brief for the kill* bullet cobb itself wrote at U42; `b1f4c2a7`/`9ba6b4f4` against the grant bullet three units have now built up; `b62faad9` against step 5's re-derivation caution; `ba410bab` against `4e7a15c3`, which cobb shipped **yesterday**; `96848806` possibly a near-twin of `d41b7a92` **in the same chunk**, with the warning that clearing both is the irreversible half; `d7a91e35` flagged as probably not a coordination fact at all. Four unled. **No running statistic quoted**, deliberately, after U43's `suggestedHome` near-miss — priors to test, not facts to carry. Asked to judge whether four statements about shared-tree commits are fragmentation or correct separation, and explicitly **not** to consolidate (K-026) | `a8673e0e6bc9481d9` | **accepted** (`8973269`) | `claude/teco/teco.md` (+785 w) · `claude/AGENTS.md` (+62 w, a **repair**) · `skills/agent-standards/claude-code.md` (+333 w) · `claude/analyst/review-techniques.md` (+269 w) · `teco/kaizen/history.md` · `cobb/kaizen/{history,plan}.md` (K-030 → high; K-031, K-032 opened) | teco re-derivation → **11 of 12 hold; one shipped claim false — a negative measured in the wrong directory.** 12 promoted (3 as halves), 0 wholly discarded, 13 cleared including `a90c5f31`. **The false one:** `511f0797` shipped *the `tasks/` view is gone outright as of 2026-09-10* on **0 of 76** session dirs and **0** `*.output`. All four figures reproduce — **under `~/.claude/projects/`, where `tasks/` has never lived.** Measured at the real base, `/tmp/claude-<uid>/<slug>/`: **10** dirs, **6** with `tasks/`, **108** `*.output`, this session holding **19 symlinks + 10 plain files** including cobb's own `a8673e0e6bc9481d9.output`, and a plain file written at **07:45 — during this gate, by its own commands**. The `tool-results/` half is a **dual write, not a move**: `bi680mzav` exists as both `.txt` and `.output` at 07:21. The path cobb needed was published in its **own U42 bullet two paragraphs above**. **Why the reach-check slid off it:** the instrument asks what the widest set a sentence covers is, and a *negative* claim has no positive set to point it at — the counter-question is *where would it be if it existed, and did I look there?* Relayed as such. The mtime discriminator itself is sound and kept (**715,557 B at 1 s old** vs **8** and **31** min stale), as is the `teco.md` clause, which never names `tasks/`. **Verified and accepted:** both `b1f4c2a7` halves reproduce in a scratch repo, git **2.43.0** — `git diff --cached` **1** added line against **3** shipped, worktree intact, index rewritten; and the entry's own remedy (`git add` + no-pathspec commit) **swept a concurrent staged file** into the commit. cobb's reframing is right and the published conditional was the defect: the staged blob had never existed in the worktree, so it was no partial stage. `d9306096`'s headline is false against this ledger and cobb caught it — **U22** is recorded accepted with no correction two units before U24. `96848806`/`d41b7a92` separate convincingly (criterion falsified at Pass 15, instrument at Pass 16, neither remedy subsuming the other) and promote once into one bullet, so clearing both loses nothing. Budgets exact to the word. **Three structural calls accepted and correctly escalated, not taken:** K-032 (delete the `claude/AGENTS.md` roster enumerations rather than add the missing topic — a second copy of `README.md`'s catalog in the one always-loaded file), K-031 (the deciding fact is how often a non-`teco` agent commits here; `git log` cannot answer it, one author), and the knowledge-base question for `teco.md`, argued from **routing evidence** — 9 of 12 landed there for want of anywhere else — rather than from word count. **Corrected and re-verified.** cobb confirmed each point independently rather than on my assertion, and brought two figures I had not: all **19** symlinks resolve into `subagents/agent-*`, and the dual write is **byte-identical** — `bi680mzav` at **55,495 bytes** in both locations, 11 ms apart, with 9 of this session's 10 plain files having a twin. Coverage re-measured on fresh numbers rather than inherited: **19 of 47** (40%) against the 17-of-45 of a day earlier, so U42's completeness argument is what the paragraph now rests on. `claude-code.md` **11,189 w** (+333); zero residual hits for the retracted phrasings. **No tombstone left**, correctly — the false version never shipped, so nothing published could have been believed, and the story lives in the kaizen histories. cobb's own diagnosis is the keeper: *a claim that something does not exist has no members, so the probe had nothing to bite on and passed vacuously.* Captured as `3ad9c07e`. **Note for the close:** `claude/AGENTS.md` is now **53 words past** its ~2,500 smell, incurred by a repair rather than an addition — K-032's proposed trim (delete the roster enumerations) is the compensating move and is a stakeholder call, not taken | 242,117 tok / 99 tool uses / 23.3 min |
| U45 | **`teco` chunk 4 of 4 — seventeen entries, the final `teco` chunk.** **Seventeen, not twelve**, on U44's own evidence rather than a relaxed rule: `aad72232` records that dispatch cost is dominated by a per-run floor, so a five-entry runt would be disproportionately expensive — and cobb is asked to say so if it thinks that entry is wrong, since it would resize the rest of the pass. **Split into two groups, briefed differently.** Group A, ten from other coordinations, ordinary work. **Group B, seven I wrote today about the units cobb just did** — four of them exist *because cobb told me to write them* at the U42 gate, and two record defects **in cobb's own work in this pass** (`f2b6ae44`, my brief priming its false tally; `3ad9c07e`, the vacuous reach check on the `tasks/` negative). That is U42's two-party problem in its purest form plus a conflict of interest running the other way, so the instruction is explicit: judge them as a stranger's, name any that is self-serving or over-general from n=1, and **measure `5fc1dfea` against the whole graph rather than reasoning about it** — *I would rather lose all seven than ship a flattering one.* Six leads, including two live contradictions: `aad72232` against `teco.md`'s cost-free dispatch-sizing rule, and `97b8fc02` against the four rules already in `## Six platform failures` — a coordination doc is archived at close where a prompt is not. Carries the **U44 reach-check amendment** from the start: for a negative, *where would it be if it did exist, and did I look there?* Closing question is the pass's own exit — **what a defensible close condition looks like over a store a concurrent producer keeps filling, and what the pass should hand over rather than finish** — answered independently, my own view formed first. **Killed by the seventh rate limit (HTTP 429) after full verification, mid-promotion** — `<result>` read *"All seventeen verified. Now the promotions… P1…"*; checked, not trusted: `teco.md` already carried P1's merged text uncommitted, `kaizen/history.md` untouched, graph unmoved at 17. Resumed in place by `SendMessage` with the verified state stated back and an instruction not to redo the landed write (see `## Seven platform failures`) | `a614d75e1d1734ca3` | in-flight | homes unknown until routed | teco re-derivation → — | — |

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
- **A `closed out, 0/0` note in this ledger is a statement about its unit's
  pinned scope at the moment that unit ran — never a claim the agent is
  finished.** Census taken after U18 (2026-09-07): `analyst` 56 (up from the 53
  this pass planned four chunks against, 12 of them dated today), `teco` 14,
  **`architect` 7**, **`coder` 1**, **`graph-dba` 1** — the last three all
  marked closed out at U7b, U12 and U6 respectively, all refilled with entries
  dated today, plus one new `teco` entry that appeared mid-U18. Team total 85
  (79 produced + 6 `MENTIONS`-only). The refill is not drift to correct: those
  are real learnings from other sessions running today, and they are pass 3's
  scope unless a top-up unit is dispatched here. What it does mean is that
  **the top-up list below is a floor, and must be re-queried immediately before
  the pass is declared closed** rather than read off this table.

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

## Seven platform failures, all cleanly recovered

Seven units were killed mid-run by session rate limits (HTTP 429) — U10, U15,
U19, U20, U41, U42 and U45 — and **not one lost a disposition.** They landed at
seven different points, which is what makes them useful together. U42 is the
first to be killed **after** its deliverable was accepted, during a
gate-driven correction, and it is the one that shows the failure mode the
other five do not: a kill can leave the **artifact** corrected and the
**record of it** stale. U45 is not a new shape — it lands at effectively the
same point as U20 (mid-promotion, first write landed, no log, no clear, source
graph untouched) — which is itself worth having: the same failure recurring at
a different unit, a different specialist, and a different target file is
confirmation the rule generalizes, not a fresh case to theorize from.

| Unit | Died at | Actual state found | Recovery |
|---|---|---|---|
| U10 (`a695c632adfd6d91d`) | starting verification | nothing written; all 27 `coder` edges intact | plain re-dispatch |
| U15 (`ae7d55eeefd996462`) | after every file write, before the clear | complete `history.md` entry on disk, `plan.md` K-004 missing, all 10 nodes intact | narrow **state-recovery** unit for the two remaining steps only |
| U19 (`adb247a3e028c0606`) | first tool call | nothing written; all 12 edges intact | **resumed in place** by `SendMessage` |
| U20 (`a6db6af4910e607bc`) | mid-promotion (`<result>`: *"Now the promotions. Starting with `review-techniques.md`"*) | **four** promotions fully written to disk (`review-techniques.md` +21, `falkordb-quirks.md` +19/-1, `claude-code.md` +40/-5, `python-web-quirks/SKILL.md` +50/-4); `analyst/kaizen/history.md` untouched, zero dispositions logged; all 11 entries intact | **resumed in place** with a written state report |
| U41 (`a74ea49194ca86329`) | mid-survey, before any write (`<result>`: *"Now the four unled entries. Let me survey the candidate landing sites."*) | **nothing written on either surface** — `git status` empty at `3e9e86a`, and the graph unmoved: `architect` still **8**, `cobb` still **2**. The §5 ordering invariant was never breached, so no entry sat cleared-without-a-history-record | **resumed in place** by `SendMessage`, with the verified state stated back to it and one added instruction: **write-and-clear per entry, never batched across entries** — batching is what turns a kill into lost work |
| U42 (`a904a5d98e993c179`) | **after acceptance**, mid-correction (`<result>`: *"Every figure reproduces, and my own instrument was worse than the gate caught."*) | **the corrected artifact was already on disk** — `claude-code.md` at **45** insertions against the 33 I had gated, carrying the rewritten enumeration bullet in full — while `cobb/kaizen/history.md` still carried the **falsified** account of the same finding, verbatim | **resumed in place** by `SendMessage` after the reset, for the record-side repair only |
| U45 (`a614d75e1d1734ca3`) | mid-promotion, after full verification of all 17 (`<result>`: *"All seventeen verified. Now the promotions, per entry. P1 — merging `7c4a1e93`… and `3ad9c07e`… onto the re-derivation bullet."*) | `claude/teco/teco.md` already carried P1's merged text, uncommitted; `claude/teco/kaizen/history.md` untouched, no log entry; graph unmoved — all **17** teco entries still live, including both P1 entries | **resumed in place** by `SendMessage`, with the verified state stated back to it and an explicit instruction not to re-derive or overwrite the already-landed P1 text — log and clear it, then continue |

Four rules come out of it, and all four are now load-bearing for this pass:

**Write and clear per entry; never batch the writes across entries.** U41 is
the case that names the rule, by nearly costing what U20 nearly cost: it was
killed after substantial reading and analysis with **nothing on disk**, because
its writes were still ahead of it. Per-entry, a kill costs one entry and leaves
the graph and the history files agreeing at the stopping point; batched, it
costs the whole survey. §5 already orders *read → history → count → clear*
**within** an entry — the gap it does not close is between entries, and that is
the gap a kill falls through.

**Establish actual state before acting — never read it off the notification.**
A killed run's `<result>` is merely the last line it emitted, which can read as
far less progress than actually landed; U15's said it was about to write the
history entry, and the complete entry was already on disk. U19's read as if it
had done nothing, and that time it was true — but only a check could tell the
two apart. The check is two commands (`git status`, one census query) and it
decides between a re-dispatch, a narrow recovery brief, and a resume.

**Prefer resuming over re-dispatching.** A subagent's `agentId` resolves for
the lifetime of the **spawning session**, including after a transient-error
kill, so `SendMessage` picks the delegate up from its own transcript instead of
paying to re-explain the brief to a cold spawn. U19 confirmed this live. What
does *not* survive is a session reboot — the transcript file persists under the
parent session's directory, but a new session has no path to resolve it — so a
checkpoint record needs an explicit cold-start fallback, not just an id
(`skills/agent-standards/claude-code.md`).

**On a kill during a *correction*, check the record, not just the artifact.**
The first five kills all landed with the unit's own work incomplete, so `git
status` answered the only question worth asking. U42 landed differently: it died
**after** its result was accepted and gated, midway through applying a
correction I had sent it. The corrected artifact was fully on disk —
`claude-code.md` carrying the rewritten enumeration bullet, 45 insertions where
I had gated 33 — while `claude/cobb/kaizen/history.md` still recorded the
**falsified** version of that same finding, word for word, including the two
figures the correction existed to retract. A diffstat says *changed*; it does
not say *which of the two surfaces changed*. So when the killed work was a
repair rather than a first pass, the check is not "was anything written" but
**"do the artifact and its record now agree"** — and the tell is cheap, because
a correction that has landed in only one of them leaves the retracted figures
still greppable in the other.

The reason all three recovered at all is §5's
**append-to-`history.md`-before-mutating** ordering, which exists for exactly
this. The two writes are independent tool calls, not one transaction; a run
that dies between them leaves an entry harmlessly duplicated or partially
resolved, never silently lost. A run that dies before either leaves nothing.

## I proved a negative with a depth-limited scan, and was wrong

Verifying U20 I reported that its environment table — `fastapi 0.139.0 /
starlette 1.3.1 / uvicorn 0.49.0 / anyio 4.14.1 / redis-py 8.0.1 /
falkordb-py 1.6.1` — matched nothing on this machine, and that `fastapi` was
not installed anywhere at all. **Both claims were false.** All six versions are
exactly right, in `falkor-chat/server/.venv`, which is the venv U20 actually
used (every command in its run was prefixed
`cd falkor-chat/server && ./.venv/bin/python`) and the correct one for
`python-web-quirks`, whose subject is a FastAPI app.

The cause was entirely in my probes. There are **five** venvs in this repo:

```
model-bench/.venv   falkor-chat/server/.venv   deprecated/salesperson/.venv
mcp-monitor/.venv   cypher-mcp/.venv
```

I searched with `find . -maxdepth 3 -name pyvenv.cfg`, and
`falkor-chat/server/.venv/pyvenv.cfg` sits at depth **4** — one level past the
cut, because that venv is nested under `server/` rather than at the component
root like the other four. I then "confirmed" `fastapi`'s absence machine-wide
with `find / -maxdepth 8 -name 'fastapi-*.dist-info'`; the real path is at
depth **11**. Two scans, both truncated just short of the answer, agreeing with
each other — which is what made the false finding feel corroborated.

Root `AGENTS.md` already states the rule I broke, in the curation section: a
scan whose purpose is **proving a negative** must be unfiltered. I read that as
being about `--include` globs and file extensions and did not carry it to
`-maxdepth`, which is the same filter wearing different clothes. **Any bound on
a search — depth, glob, path prefix — invalidates a negative result, and a
second bounded scan does not corroborate the first.**

**What the pass got right, and should keep doing.** The verification was not
wasted: re-deriving U20's claims *did* surface a real defect the report had not
flagged — `conn.retry._retries == 0`, written as the check proving the lab is
safe from retry write-amplification, raises `AttributeError`. My explanation of
it (a version move in redis-py 8.x) was wrong; the true cause is better, and
`falkordb-quirks.md` now carries it:

| read off | falkor-chat (1.6.1 / 8.0.1) | cypher-mcp (1.6.2 / 8.1.0) |
|---|---|---|
| `FalkorDB(...).connection.retry` | `AttributeError` | `AttributeError` |
| `...connection.get_retry()` | `None` | `None` |
| pooled `Connection.retry._retries` | `0` | `0` |
| plain `redis.Redis` pooled conn | `10` | `10` |

Verified by me on both venvs. The split is **which object you hold** — client
versus its pooled `Connection` — and is identical across both version pairs, so
it is not a version fact at all. An `AttributeError` there means you are
holding the client, not that the driver changed.

**The transferable lesson is about the finding, not the fact.** A verifier who
re-derives a claim and finds a discrepancy has two candidate explanations —
the delegate is wrong, or the probe is — and the second deserves the same
scrutiny as the first *before* the finding is written down. I committed mine
(`a74735d`) before testing my own instrument.

## The same defect shape, four times running: right method, wrong evidence line

U20 and U21 failed in exactly the same way, and neither failure was in the
thinking. U20's retry-safety promotion was correct in substance and cited an
accessor (`conn.retry._retries`) that raises `AttributeError` on the object it
names. U21's plan-hashing promotion is a genuinely useful technique whose
evidence line says "five committed revisions … 20 step rows present in all
five" of a window that actually holds **16** revisions and in which the row
count changes — `S7c` enters at `732f5e0` (v1.19) and persists, so 3 revisions
carry 20 rows and 13 carry 21. In both cases the delegate had the correct fact
in hand: U21's own report quotes the v1.19 note announcing `S7c`.

So the thing that needs verifying is **not** the claim a delegate is making —
that is the part it reasoned about and usually gets right. It is the
**citation attached to the claim**: the version stamp, the accessor
expression, the revision count, the sha. Those are summary artifacts, produced
at the end of a long run, and they are where a confident, well-formed, wrong
number appears. `claude/teco/teco.md` already demands "state only figures you
directly observed" for units routed to a cheap model; both of these ran on the
inherited model and produced it anyway.

U23 makes it four, and adds two sub-shapes worth naming separately.

**The figure drifted between the report and the promotion.** U23's report to me
said the package held **40** tuple-target assignments. The text it wrote into
`review-techniques.md` says **42**. Nothing else changed; the delegate verified
one number and shipped another. A gate that only re-derives what the *report*
claims would have passed this — the promoted file is the artifact, and it is
the one to measure.

**The corrected figure got the attention; the inherited one rode through.**
U23 correctly refuted the entry's per-file breakdown (6/16/23) and diagnosed
*why* it was wrong — three scopes in one sentence. Having spent its
verification there, it passed the entry's **headline** 68 as "exactly right"
without re-deriving it. It is 65. The report even names the trap it then fell
into: *"the tell I should have read before drafting a finding was that two of
its other figures already matched exactly."* Finding one defect in an evidence
line is the moment the rest of that line is *least* likely to be checked.

Sharpest detail: the paragraph carrying both wrong figures closes by warning
that "mixing the two scopes inside one evidence line is how this measurement
goes wrong." Stating the rule and breaking it in the same breath is not
carelessness — it is what a summary artifact does when it is assembled from
memory at the end of a long run instead of read off an instrument.

**Standing check for the remaining chunks:** for every promoted section, re-run
the *evidence*, not the assertion — count the things it says it counted,
**measured in the file that shipped, not in the report** — on an instrument you
have tested against known data first. Three of the four defects this pass
turned up were found this way, and the one time I skipped testing my own
instrument I filed a false finding against a correct delegate (see the
retraction above).

**And test the instrument even when you have used it before.** Verifying U23 I
built an AST counter that double-counted nested functions, read 79 where the
promotion said 68, and had a refutation half-drafted before the discrepancy in
my *own* two runs stopped me. The fix (enclosing-scope attribution via parent
chains) gave 65. Had the buggy instrument happened to read 68 I would have
confirmed a wrong figure instead of catching it — a false *negative* costs
exactly as much as the false positive of U20 and is much harder to notice. What
made the result trustworthy in the end was enumerating **five** definitions of
the same count (65 / 65 / 79 / 290 / 304) and showing none of them yields 68,
rather than trusting the one that agreed with my hypothesis.

## Two sessions' uncommitted work inside one file

The shared-tree hazard escalated on 2026-09-08 from *different files* to **the
same file**. `claude/graph-dba/falkordb-quirks.md` now holds 480 uncommitted
words from two coordinations at once: 369 from U21's promotions, and 111 from
the concurrent CPG session — a `Properties removed` bullet citing
`cpg_falkorchat`, `CpgBuildInfo`, `MARKER_WRITTEN_AT` and
`skills/cpg-analysis/references/freshness.md`, every one of them a subject
U21's brief explicitly barred.

I asked `cobb` to account for the 111-word delta rather than assume it was a
byproduct of its own probing, and it disclaimed the bullet on five independent
grounds (wrong graph, wrong subject, wrong section of the file, exact
arithmetic, and zero mentions in any of its three history entries). All of it
checks out: the bullet is absent from `HEAD`, sits in a section neither of
cobb's two `Edit`s touched, and lines 190-199 measure exactly the 111 words.

**So the file is held out of U21's commit** (`04205c5`), and cobb's two
promotions into it stay on disk, uncommitted, until the CPG session commits
its own bullet. That is the right trade: a path-limited commit is only safe
while each *path* belongs to one coordination, and this file no longer does.
Committing it would have swept another session's unreviewed, unattributed work
into a kaizen commit under my name.

**What this changes for the remaining chunks.** `git status` and a path list
are no longer sufficient to establish ownership — a file can be *partly* mine.
The discriminator that worked is a three-way word count, `a506a34` vs `HEAD` vs
worktree: where `HEAD` already exceeds the baseline the other session has
committed its share and the worktree delta is wholly mine (this is why
`claude/cobb/kaizen/history.md` was safe to commit at 811 words while
`falkordb-quirks.md` was not). Run it per file before every integration
commit, not per unit.

## The tree hazard ran the other way: another session committed *our* work

`375af25 fix(joern-cpg): close Pass 6 — the oracle, the replay wiring, the
credential` contains U22's kaizen dispositions. The CPG session committed
`claude/cobb/kaizen/history.md` (+153) and `plan.md` (+108) as part of its own
six-file change, and those hunks include U22's chunk-D entry and its
21-character-collision note. Nothing was lost — the content is intact in
`HEAD`, verified — but this pass's work is now filed under a joern-cpg commit
message, and `git log -- <kaizen path>` will attribute it to that coordination.

**Why "commit by explicit path" did not prevent this, and what actually
follows.** The earlier note in this document said path-scoped commits keep
concurrent sessions from sweeping each other. That holds only while each
*path* belongs to one coordination — and `claude/cobb/kaizen/{history,plan}.md`
never can. **Every coordination that dispatches `cobb` writes there**, by
construction: it is `cobb`'s own log, not a given unit's deliverable. So it is
a structurally shared path, and no commit discipline on either side can
separate two coordinations' entries inside it.

That makes the three-way word count (`baseline` vs `HEAD` vs worktree) the
load-bearing check rather than a convenience. It is what showed `NOW == HEAD`
on both `cobb/kaizen` files — the tell that someone else had already committed
our content — and it is the same instrument that, one unit earlier, showed the
opposite condition on `falkordb-quirks.md` and stopped me sweeping *their*
bullet. One check, both directions.

**Not repaired, deliberately.** Rewriting history to re-attribute those hunks
is a tree mutation and off-limits; the content is correct and present, and the
cost of the mis-filing is a confusing `git log`, not a lost disposition.

## `analyst`'s inbox refills faster than the pass drains it

Re-queried at U23's dispatch: `analyst` holds **24** produced entries, not the
13 the U23 row was drawn against — 09-07 (13) and **09-08 (11)**, a population
that did not exist when this pass opened. U23 clears 09-07; **U24 is owed for
09-08**, at whatever size it has reached by then.

This is the live-graph property recorded earlier, but it now has a scheduling
consequence worth stating plainly: **this pass cannot be closed against a
snapshot taken at its open.** The team total has moved 93 → 89 → 86 → 87 across
these units *while entries were being cleared*, because other agents keep
writing — `teco` alone went 14 → 23 during the pass, mostly from this
coordination's own verification findings. Before declaring the pass closed,
re-query per agent and drain what is actually there, however many rounds that
takes.

The **`MENTIONS`-only backlog stands at 11**, and none of them hangs off a
produced entry — so no unit clears them incidentally and they need a unit of
their own. They carry **12** edges between them: one node is tagged by both
`analyst` and `tdd-engineer`, which is why the node and edge counts differ.
Spread: `analyst` 3, `tdd-engineer` 2, and one each for `qa-engineer`, `tico`,
`devops`, `architect`, `data-scientist`. They are the oldest population in the
graph — 08-30 through 09-07 — precisely because every pass so far has been
organised by producer, and these have no producer.

**The refill is measurable inside a single unit.** U23 reported a post-clear
census of 76 nodes with `analyst` at 12. I re-queried perhaps twenty minutes
later, during verification: **78 nodes, `analyst` at 14**. Two arrived while I
was checking the work of the unit that had just drained that agent's inbox. So
U24's size is not merely unknown until dispatch — it is unknown *at* dispatch,
and the only sound close condition is a re-query that comes back empty, not a
count planned in advance.

## Stopped here — how to resume

The stakeholder asked to **stop after U24**, the unit currently in flight. The
pass is **not finished**, which is why this document stays `Status: active`:
archiving it would freeze the ledger that is the resume point.

**State, verified rather than assumed:**

- Everything through **U24 is accepted and committed** (`cfed0a0`, `bd924b1`,
  `468822b`). `git status claude/` is clean; nothing is held back.
- **U24 was the first unit of this pass to need no correction.** It was also
  the first briefed with the full accumulated lesson set, and the first to
  report instruments tested under *two* definitions before use — counts taken
  both as collected node IDs and as unique def names, the AST blindness
  reproduced with both a name-set and a site-qualified reader, the `rq()` probe
  run with a passing control alongside the failing ones so a uniform `0` could
  not be misread as a working helper. The briefing cost is now visibly cheaper
  than the correction round-trips it replaces.

**A correction worth keeping, because it is this pass's own recurring defect
turned on its author.** An earlier revision of this section recorded U24 as
"killed by a host reboot, had written nothing — re-dispatch, do not resume."
The reboot had not happened. I had observed *"nothing on disk, 12 entries
intact"*, which is equally consistent with a healthy unit mid-verification, and
wrote the *prediction* into the ledger in the past tense. The evidence was
real; the tense was invented. `teco.md` already forbids stating a pending
delegate's result before it arrives — the rule survived the moment it was
written for, because the sentence did not feel like a prediction. Checking the
transcript mtime (4 seconds old, 815 KB) settled it in one command.

**Distinguishing a live unit from a dead one, since an empty `git status` does
not:** the task transcript's mtime is the cheap tell. A recent mtime means it
is working; a stale one plus an empty worktree means it died before promoting.

**What remains, in order** — rewritten 2026-09-09 after **U42**; the previous
list was written after U40, and item 2 has now moved four times.

1. ~~`analyst`'s inbox~~, ~~the `MENTIONS`-only backlog~~, ~~routing the `rq()`
   fix~~ — **all done.** U25 and U37 drained `analyst` (empty, and its `:Agent`
   node is gone from the graph); U31 drained the orphan backlog down to **one
   deliberate survivor**; the `rq()` fix landed as `graph-dba` K-009 and closed
   in U33. U32/U34/U35/U36 drained `devops`, `data-scientist`, `graph-dba`,
   `qa-engineer` and `tdd-engineer`.
2. **The producers still holding entries.** Re-queried live at the U42 gate,
   **2026-09-09**: `teco` **31**, `analyst` **8**, `coder` **4**,
   `tdd-engineer` **4**, `cobb` **1**, `data-scientist` **1**, plus the 1
   orphan — **50** in the graph. U41 closed `architect` outright: it is absent
   from the graph, 0 produced and 0 mentioned. U42 took `teco` 42 → 30 and the
   count has already ticked back to 31. **Every non-`teco` figure here is
   larger than it was two units ago** — `analyst` 3 → 8, `tdd-engineer` 1 → 4,
   `data-scientist` 0 → 1 — and all of it is model-bench subject matter from
   the concurrent coordination, which is item 5's problem, not this list's.
   Next: **`teco` chunks 2–4** (~30 entries, ~12 date-ordered each, per the
   standing decision), then a final sweep of whatever the other producers hold
   at that moment. **Re-query at each dispatch; never dispatch against a figure
   recorded here** — this list has been wrong at every single dispatch it has
   been read at, and the two units since it was last rewritten are no
   exception.
3. **The one orphan is not a residue and must not be swept.**
   `e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e` is alive on its `tico` edge on
   purpose — it is the routing signal for `tico` K-016. A close pass that
   deletes it to reach zero destroys the signal.
4. **`teco`'s inbox is the pass auditing itself**, and U42 proved the
   asymmetry is worth briefing on rather than merely noting. It is the largest
   inbox in the graph and most of it is this coordination's own verification
   findings, so it is the one unit whose entries were written by the agent that
   gates them. U42 was briefed *the discard bar is the subject, not the
   promotion bar* and returned **4 discards of 13** — the highest discard rate
   of the pass — with every discard justified against a published site rather
   than against my framing. Brief chunks 3 and 4 the same way. **What that
   framing does not reach** is the class U42 identified in return: capture fires
   when the pass finds it was wrong and never when it was right, so the
   coordination's own load-bearing *successes* are absent from the graph
   entirely. Three named in the U42 report — the three-way word count, the
   structurally-shared `cobb/kaizen/` path, the ≈17k-tokens-per-entry cost
   curve. They are being captured as ordinary `teco` entries for pass 3, **not**
   written into this document, since this is the document entries are checked
   against.
5. Only then: re-query every agent, drain what is actually there, and flip this
   document to `archived`.

**The close condition is a re-query that comes back empty — never a planned
count.** Two entries arrived while U23's own verification was running; U37's
verification saw the total move 69 → 70 between `cobb`'s post-clear census and
mine, from a **concurrent `teco` session's** capture, not from anything this
unit did.

**And after U40 that condition is known not to be reachable unilaterally.** I
read the four refilled entries' content directly: all four carry `sessionId`
**null** and all four are **model-bench subject matter** — a `urllib` exception
ladder that misses the body phase, an editable install's `.pth` meta-path
finder, `IncompleteRead` not subclassing `OSError`, a CLI guard mutation turning
an assertion failure into a pytest `OSError` on `input()`. They are the
**concurrent model-bench coordination's `analyst` and `coder`** writing into the
shared graph while this pass drains it. So the refill is neither noise nor
stragglers: it is a **second writer**, and it stops when that coordination stops,
not when this one does. The honest close is *empty of everything this pass was
scoped to*, with whatever the other coordination has since written handed on as
a fresh inbox — **a stakeholder call, not mine to take**, and the last thing this
pass needs a decision on before it can be archived.

## Follow-ups

- **`docs/plans/small-model-benchmarking.md:3272` ships a false mechanism as a justification —
  needs routing to that plan's owner, and this pass must not touch it.** The parenthetical reads
  *"`--include` does not suppress a file named explicitly on the command line — **checked**, because
  this command depends on it"*. It is false, and I re-derived the falsification independently of
  U40's: `grep -Fn <tok> tests/conftest.py --include='*.txt'` returns **no output, rc 1**, with and
  without `-r`, on `/usr/bin/grep` 3.11 — `--include` filters by **base name regardless of how the
  file arrived**. The command it justifies is **safe today** (`conftest.py` does match `*.py`), so
  nothing is broken; the hazard is that a correct command is resting on a false mechanism, which
  would license an unsafe re-scoping later. Two reasons it is not ours to fix: the plan is
  `architect`-owned, and **another session is executing against that tree right now** — U40 read it
  read-only and staged nothing. Note the tell for the KB: *"checked, because this command depends
  on it"* is a **stated method with a false conclusion**, the same shape as the four
  right-method/wrong-evidence-line failures above.
- **`suggestedHome` has now predicted the home for 0 of 17 entries across U38–U39.** All eight of
  U38's said *knowledge base* or *project docs*; all nine of U39's likewise, and **three marked
  `project docs` landed nowhere near a component docs tree**. The producer knows the fact, not the
  shelf — which is why a **routing-first** brief (argue the home from the receiving artifact's own
  scope line) outperformed a promotion-first one twice running. Not a `K-` item: no change to the
  write shape follows, and asking a capturing agent to route better would be asking it to know
  something it structurally does not.
- **`claude/tdd-engineer/tdd-engineer.md:40` is now 1,502 characters — the longest line in that
  file**, after U39 folded the round-trip-blindness rule onto the mutation bullet. The fold was the
  right routing (it is that bullet's mirror case: there a *surviving* mutant is not the test's
  fault, here it is), but the bullet now wants a split into *prove the mutant* / *pick the mutant*.
  Deliberately **not** done in U39 — restructuring as a side effect of a distillation is exactly
  what K-026 exists to prevent. Owner: `cobb`, as its own unit.
- **`claude/analyst/review-techniques.md`'s residual section opens "Six ways the residual
  *passes* on an incomplete edit" while item 5 is a false *failure*** — a pre-existing inaccuracy
  U39 found and left for the section's next reviser rather than widening its own scope to fix.
  One-line fix whenever that section is next touched. **Closed in U40**, and **renumbered in U42**
  when a seventh item landed: the opening now reads *"Seven ways the residual **misreports** the
  edit — six of them by passing an incomplete one, and item 5 by doing that *and* failing a
  correct one"*. Kept here as the record that the follow-up was carried, not dropped — and as a
  worked instance of a quote going stale in the document that records it while the quoted text
  stays correct.
- **`claude/architect/architect.md:51` is 1,617 characters — the file's longest line**, and it is
  now the *second* recorded instance of the same shape as `tdd-engineer.md:40` above. It matters
  more than a style nit: `architect` owns **no knowledge base**, so its always-loaded prompt is the
  only landing site any future `architect` promotion has, and that line being over budget is what
  pushed U40's `0a4b6b2e` to `analyst` instead. Already flagged in `architect/kaizen/history.md` at
  U31; recorded twice now. Owner: `cobb`, as its own unit.
- **`coder` refilled 0 → 2 and `cobb` sits at 2** — `coder` was drained to zero by U39 and has two
  new entries (`e3f1c2a4…`, `c1e8f4b2…`, both `sessionId` **null**); `cobb` holds `2e14550b…`
  (written by U39's own cobb) and `2746ee65…` (U40's). Neither is in any brief yet. Both need a
  top-up unit before the pass closes — which is the standing reason the close condition is **a
  re-query that comes back empty**, never a planned count.
- **The refill has a cause, and it means the close condition as stated can never be met while
  another coordination is live.** I read the four refilled entries' content and edges directly.
  All four `null`-`sessionId` ones are **model-bench subject matter** — a `urllib` exception ladder
  that misses the body phase, an editable install's `.pth` meta-path finder, `IncompleteRead` not
  being an `OSError`, a CLI guard mutation turning an assertion failure into a pytest `OSError` on
  `input()`. These are the **concurrent model-bench session's delegates** capturing into the same
  shared graph while this pass drains it. `analyst` and `coder` are exactly the agents that
  coordination is running. So the refill is not noise and not slow stragglers: it is a second
  writer, and it stops when *that* coordination stops, not when this one does.
  **Consequence for the close:** "re-query comes back empty" is the right shape but is not
  reachable unilaterally — the honest close is *empty of everything this pass was scoped to*, with
  whatever the other coordination has since written handed on as a fresh inbox. A stakeholder call,
  not mine to take.
- **`$CLAUDE_CODE_SESSION_ID` resolves to the *parent* session inside a subagent — confirmed from
  the graph, and it retro-explains U38's mis-attribution.** Both `cobb` entries
  (`2e14550b…`, `2746ee65…`) carry **this session's** id `1501c506-…` although neither was written
  by this session — they were written by U39's and U40's `cobb`. `2e14550b`'s own subject *is* that
  fact, so the entry is self-demonstrating. This is why U38 named the right extra node and then
  attributed it to the wrong session: **`sessionId` names the coordination, never the agent**, and
  reading it as the latter is what produced that error.

- **The §5 truncation defect predates this pass and silently bounds its own record.** U38 found
  that `skills/agent-maintenance/SKILL.md` §5 step 1 tells a distiller to read `k.fact`/`k.evidence`
  through a tool that cuts every cell at `CYPHER_MCP_MAX_CELL` (default **300**) and appends
  `…(+N chars)` — and never said so. Any disposition in pass 1, or in pass 2 before U38, argued
  from a long cell **without** paging was argued from the half of the entry that states the claim,
  missing the half that bounds it. Nothing is re-openable — those nodes are cleared and their
  dispositions logged — so this is a **standing caveat on this pass's own record**, not a defect
  to fix retroactively. The recipe is now in §5 step 1.
- **`suggestedHome` is not a routing signal.** All eight U38 entries said *knowledge base* or
  *project docs*; every actual home was decided by the **receiving artifact's scope line**, and
  twice landed somewhere the capture had not contemplated. Not filed as a `K-` item — no change to
  the write shape follows — but it is why a routing-first brief beat a promotion-first one, and it
  generalises: the producer knows the fact, not the shelf.
- **Naming the extra census node is the right practice, and it still needs the attribution
  checked.** U38 correctly identified the one node separating its census from mine
  (`7c4a1e93-5b28-4d06-9f31-2ae80c6b5d47`) but attributed it to *"the concurrent `teco` session"* —
  it was written by **this** session, during U37's acceptance. The node identification is what
  mattered and it was right; the attribution was asserted rather than checked, which is the pass's
  own recurring defect in miniature.

- **A post-clear census is a snapshot of a graph other sessions are writing to, and the
  delta is not evidence of a miscount.** U35 caught `cobb` reporting a figure taken before
  its *own* subsequent capture. U37's delta has a different cause with the same shape:
  `cobb` reported **69** nodes / `teco` **36**; my verification twenty minutes later read
  **70** / **37**, the extra being `3f8c21ad-6e94-4b7f-a1d2-9c5e0b7a4318` (2026-09-09
  15:05Z), written by a **concurrent `teco` session** working `model-bench`. Reconcile a
  census delta by naming the extra node before treating it as a clearing defect — the
  producer counts that matter for the *next* dispatch are the ones queried at that
  dispatch, not either of these.
- **My grep instrument has now been the defective party four times in this pass.** U35: raw
  YAML text vs `yaml.safe_load` for a description's char count. U36: the inverted strict-parse
  rider. U37: a pipe-delimited extraction over `test-stamp-wiring.sh`'s `FORMS` heredoc, which
  is **whitespace**-separated — it returned `0 rows`, a result that reads exactly like a missing
  block rather than a wrong parser. In every case the delegate's figure was right. A
  re-derivation that comes out *empty* or *stable* deserves the same suspicion `teco.md`
  already assigns to one that comes out clean.

- **That YAML finding inverted under execution, and the inversion is the durable part.** I
  routed it as *"under a strict parser `cobb`'s own standards skill does not load at all"*.
  `cobb` established the opposite first-hand: the harness runs a **more permissive** parser,
  and all four files load with their **complete** description, text after the offending colon
  included. **I corroborated that by a method independent of `cobb`'s** — the three agent
  definitions appear in *this* session's own agent-type listing with the post-colon text
  intact (`tdd-engineer`: *"…the efficient path: a bug fix (reproduction test first)…"*).
  So the useful output was never a fix: **strict YAML is the wrong instrument to audit Claude
  Code frontmatter with** — an audit built on `yaml.safe_load` would fail three live agent
  definitions on day one. The three are deliberately left alone; only the `skills/` one was
  converted to a folded scalar, on portability grounds, value byte-identical. This is the
  pass's own recurring lesson — *my instrument was the thing that was wrong* — arriving one
  step before the check it would have justified.
- **`claude/AGENTS.md` is at 2,491 words against its own ~2,500-word smell.** Not caused by
  this pass, but this pass keeps adding to it: every new knowledge base earns a clause there.
  The next unit that would extend it should compact instead.
- **`skills/agent-standards/SKILL.md`'s frontmatter is not valid YAML — found here, by
  accident, while re-deriving a char count.** `yaml.safe_load` rejects it: *mapping values are
  not allowed here*, line 2 col 404. The cause is a `: ` inside an **unquoted plain scalar** —
  the description's *"Perishable: every fact is dated"*. Every other `SKILL.md` in the repo
  (four in `skills/`, five in `opencode/skills/`) parses. Whether Claude Code's own loader is
  strict or lenient is **unknown and is the whole question**: under a strict parser `cobb`'s
  own standards skill does not load at all. Routed to `cobb` as a rider on U36, alongside
  K-027/K-028 — the audit those three want is the same one: *a skill's frontmatter parses,
  fits the 1,536-char listing budget, and names the right audience.*
- **`K-025` has no summary row in `cobb`'s `plan.md` Active table** — U31 added only the detail
  block. `cobb` noticed this in U35 and correctly left it, being out of that unit's scope.
  Small, and `cobb`'s own file.
- **`falkor-chat/AGENTS.md:603` is 980 characters** — the only line in the repo over the ~700
  bar, and by the convention's own diagnosis that means a cell being used as a changelog.
  Pre-existing, untouched by this pass, and not `cobb`'s to write.
- **A retraction can be correct about one file and wrong about another.** While
  writing up the arc, `cobb` probed the check's stated bound adversarially, found
  `PING`/`INFO`/`keys` pass unflagged, and had "generation six" half-drafted
  before withdrawing it — the finding had been drawn from the bound's
  **examples** rather than its **claim**. That retraction is right for
  `test-stamp-wiring.sh`, whose claim is scoped to a `GRAPH.<word>` run. It is
  **wrong for `pipeline.sh`**, whose copy of the same sentence drops the
  scoping — and `cobb` had not read that file. So the near-finding was half
  right, about the file nobody checked. **One bound stated in two files is one
  claim and two chances to be stale**; the pass's other recurring defect
  (corrected here, inherited there) in its purest form.
  - **Open, routed to `cobb`.** U33 converged the two script files, but the
    retraction as *written up* survives in `claude/tdd-engineer/guard-testing-techniques.md`
    (§*The worked example*, the `**One retraction…**` paragraph): it reasons from
    the scoped wording and concludes flatly that the finding "is not" a
    generation — true of `test-stamp-wiring.sh`, false of `pipeline.sh` as it
    then stood. `cobb` wrote that paragraph and owns the file, so the one-clause
    correction goes there, not to `graph-dba`; rider it onto `cobb`'s next unit.
    The lesson the paragraph should end on is **checking a bound means checking
    every place it is stated**, which is stronger than the one it currently
    teaches.
- **`cobb` contributed the stopping rule this arc had been missing**, and it was
  in nobody's brief: *you may stop widening a static guard when a miss is bounded
  to a **false failure** rather than a false pass* — because the check is a lint
  over a property the runtime already enforces, not the safety boundary itself.
  Establish that cost asymmetry **before** accepting a narrow claim; without it,
  narrowing is just conceding. That is what licenses this check to stop at a
  stated bound instead of chasing a shell parser.

- **What finally stopped the regress: pinning the claim to the mechanism.** Four
  generations of this defect were closed by fixing the *mechanism* and rewriting
  the *claim* — and generation N+1 always arrived because nothing tied the two
  together. U32's block records that widening the reader **without** rewriting
  the stated bound turns a `blind` row **red**: the claim is now an assertion the
  suite evaluates, not prose beside it. That is the structural answer to "a guard
  whose stated reach exceeds its mechanism", and it is worth generalising beyond
  this file.
- **The probe carries controls on itself.** A coverage probe that agrees with its
  author on every row proves nothing, so U32 mislabelled a row (probe fails) and
  reverted the reader to its generation-three form (the two continuation rows
  redden) — demonstrating the probe *would* have caught the defect that prompted
  it. Same discipline as a passing control beside failing probes, one level up:
  the instrument is tested against a known-bad version of itself.
- **Two documents went stale the instant the fix landed**, both in other agents'
  files and both routed to their owners rather than fixed in place:
  `claude/tdd-engineer/guard-testing-techniques.md` (says the check is open) →
  `cobb`; `docs/reviews/rq-execution-gate.md` lines 297-301 (quotes the
  superseded comment) → `analyst`, as a compact Pass 3. A fix that closes a
  finding invalidates every document that described it as open — sweep for those
  in the same breath as the fix.

- **The `MENTIONS` tag is a queue with no consumer — a structural defect, not a
  backlog.** U31 refuted this coordination's own dispatch hypothesis (that the
  orphans were promoted-but-uncleared residue) and found something better: each
  orphan is a **deliberate forward routing**. A unit promoted the producing
  agent's half, tagged `MENTIONS → <second agent>` for the half belonging to
  someone else, and resolved the `PRODUCED` edge — leaving the node alive on
  purpose. `teco`'s own U18b states the intent verbatim: *"each surfaces in the
  tagged agent's own distillation pass."* **It never does.** A distillation unit
  is scoped `MATCH (a:Agent {agentId})-[:PRODUCED]->(e)`, which by construction
  cannot see a `MENTIONS`-only node. **Six of the seven tagged agents were
  recorded "closed out, 0/0" while holding an unrouted edge.** This ledger
  logged two instances as one-off deferrals owed at pass close; it was never
  two. Now `cobb` **K-025**, with two candidate fixes: scope units by *either*
  edge kind, or stop letting the tag imply a promotion.
- **My stopping condition was itself an enumeration, and that was its flaw.**
  U28's generation-three closure was gated on **four named mutation shapes**
  reddening plus a clean-tree control. It held honestly. A **fifth** shape — a
  literal command after a backslash line continuation — was found within the
  hour by `cobb`, and I reproduced it: the anchor is a line-based `grep`, so the
  count stays at 3 and the token is never inspected. A list of shapes I happened
  to think of cannot establish coverage of a space. This is the worked example
  of the very entry U31 promoted into
  `claude/tdd-engineer/guard-testing-techniques.md` — for a static-analysis
  guard, a **mutation test** and a **coverage probe** answer different
  questions. U32 is therefore gated on an **enumeration of call-site forms with
  a disposition for each**, not a count of mutations that reddened.

- **The close condition is a re-query, and the refill proves it again.** Census
  taken after U30 (2026-09-09): **83 nodes — 72 produced, 11 orphan**. Produced:
  `teco` 35, `architect` 15, `coder` 8, `cobb` 7, `analyst` **3**,
  `tdd-engineer` **3**, `qa-engineer` 1. **`analyst` was drained to 0 by U25
  earlier the same day and `tdd-engineer` was closed out back at U9 — both have
  refilled.** `graph-dba` and `data-scientist` are the only agents this pass has
  taken to 0-and-still-0. A pass over this graph does not converge on a planned
  count; it converges only on a re-query that comes back empty, and even that is
  a moment rather than a state.

- **One defect shape, three generations, each produced by the previous fix.**
  K-009's error **blacklist** overclaimed its reach → replaced by a positive
  trailer test plus an **`rc 2`** refusal, which overclaimed (no call site could
  read it) → replaced by a **static check**, which overclaims in turn (its anchor
  requires `$(rq `, its token match is uppercase-only, so a bare-statement call
  site and a lowercase `graph.delete` both pass). Every generation was caught,
  and every one was caught **only by execution** — each had a plausible reading
  of the code under which it was fine. The stopping rule applied at generation
  three: when the marginal finding is produced **by** the fixes rather than found
  **in** the original artifact, stop narrowing prose and give the implementer a
  **falsifiable done-condition** (here: four named mutation shapes that must
  redden, with narrowing the stated bound an equally acceptable closure).
- **A gate's own recommendation is the least reliable thing it produces.**
  Pass 1 recommended capturing `rc=$?` at the three call sites. Pass 2 refuted it
  **by execution**: inside `if ! V="$(f)"`, `$?` is **0** for a function
  returning 1 *and* one returning 2 — the `if` consumed it, so the recommended
  branch could never fire. The implementer's chosen closure (delete the guard)
  was better than the reviewer's proposal, which is why a finding is routed as
  *judge this*, never *apply this*.

- **The pass's own promoted lesson predicted a defect in the pass's own work,
  three units later.** U25 promoted `analyst` entry `7c1d4a92` into
  `review-techniques.md`: a helper whose contract is a **side-effect variable**
  is silently defeated when its caller invokes it in `$(…)`, because command
  substitution runs in a subshell. U28 then added an `rc 2` refusal path to
  `rq()` — which is called at all three sites as `if ! VAR="$(rq …)"`. The
  refusal cannot escape the subshell, and `! VAR=$(…)` collapses every non-zero
  to "false" besides, so the tri-state has **no reader at all**. `analyst`
  confirmed the guard can be deleted outright with the suite unchanged at 14
  PASS / exit 0. **A guard no test can redden is a guard in name only** — and
  the knowledge needed to catch it at authoring time was already in the
  reviewer's own knowledge base, promoted by this same coordination.
- **`RESULTSET_SIZE` is a silent correctness limit, not a display limit.**
  `GRAPH.CONFIG GET RESULTSET_SIZE` = **10000** on this instance. A query
  exceeding it is capped and **still emits a normal statistics trailer**:
  `UNWIND range(1,200000) AS x RETURN x` returns rc 0, 10,003 lines, last line
  the trailer — so a trailer-anchored gate passes a reply that lost 190,000
  rows. The trailer proves the server **ran the query to completion**, never
  that the reply carries every row matched. Any caller needing completeness must
  check the row count against an expectation. Verified read-only by `teco`,
  independently of `analyst`'s own measurement.

- **A null result can be the right evidence for a rule, and the reporting defect
  is what makes it look wrong.** U27's entry paired a methodological rule
  (compute the which-instrument audit *before* any value-modifying transform)
  with an exhaustive sweep finding **zero** differences — which reads as an
  argument *against* the rule. Both are correct: the sweep measures **printed
  numbers**, the rule is about **attribution**, so the null says adopting the
  rule is **free**, not that ordering is irrelevant. The evidence line recorded
  only the null half. When gating an entry whose evidence seems to contradict
  its claim, check whether the two are measuring **different quantities** before
  concluding either is wrong.
- **My own brief carried a wrong figure into a unit, and the delegate caught
  it.** I wrote the sweep's grid as "4 × 6 = 24 grid points"; 28,912 is already
  the **sum** of `C(n+3,3)` over n ∈ {12,30,38,40} (455 / 5,456 / 10,660 /
  12,341), and the only multiplier is the six design effects. My reading implied
  693,888. **A brief is the one input no gate reads** — this one was corrected
  only because the delegate recomputed instead of adopting my framing, which is
  exactly what the accumulated-lesson list asks for. The lesson generalises: a
  figure a coordinator states as background fact is load-bearing on the
  delegate's done-condition, and it is unreviewed by construction.
- **`claude/data-scientist/data-scientist.md` has seven lines over 700
  characters** (longest 1,247; the U27 edit took line 49 to 1,182). All but the
  edit are pre-existing and the ~700-char smell is written for `*AGENTS.md`, not
  agent prompts — but this is an always-loaded file and the density is the same
  problem. Worth a compaction pass by `cobb`; not this pass's scope.

- **`graph-dba` owns one code unit on `skills/joern-cpg/scripts/pipeline.sh`, not two.**
  U25 and U26 between them settled what belongs in it: **K-009** (the `rq()`
  helper returns 0 on a bare FalkorDB runtime-error reply — the live defect) and
  **K-008's disposition** (confirm Fact 2 delivered and Fact 1 superseded, then
  close or re-scope). Two entries that *looked* like they belonged — U25's `$(…)`
  subshell defect and U26's stamp race — are both **dead**, fixed by `271c899`
  and `6012ddb` respectively. Dispatch as one unit on that file; do not let a
  distillation unit and a fix unit run against it concurrently.
- **A distillation unit's collateral findings can outweigh its dispositions.**
  U26 discarded its only entry — 0 promoted — and was still worth running: it
  established that a **parked backlog item's premise had gone false**, which no
  producer-organised query would have asked. An inbox of 1 is not a low-value
  unit, and "nothing promoted" is not "nothing found".

- **Three stray `:Agent` nodes in `kaizen_team` carry no edges at all** —
  `_qa_selftest_producer_4e24af1e`, `_qa_selftest_producer2_4e24af1e`,
  `_qa_selftest_mentioned_4e24af1e`, surfaced by U25's gate query (which listed
  agents rather than only producers). Test residue from a `cypher-mcp`
  self-test that cleaned up its entries but not its agent nodes. Harmless, but
  they inflate any `MATCH (a:Agent)` count and will read as real agents to the
  next reader. Deleting an `:Agent` node is **not** one of the six authorized
  write shapes, so this needs the maintainer or a widened curator remit — do
  not improvise it.
- **A delegate attributed one of my own commits to "another session."** `cobb`
  closed U25 noting the coordination doc "was dirty when I started and is clean
  now — another session committed it mid-run." It was **this** coordination
  (`2d23482`, the U25 dispatch record). The inference was from absence of
  evidence — *not me, therefore someone else* — in a repo where that is usually
  true. It cost nothing here, but it is the same shape as attributing content
  by who did **not** write it, which this pass has already been bitten by; a
  brief that tells the delegate the coordinator commits the ledger mid-unit
  removes it.

- **Live defect in shipped tooling, confirmed by execution — route to
  `graph-dba` (owner of the `joern-cpg` skill).**
  `skills/joern-cpg/scripts/pipeline.sh`'s `rq()` helper returns **0 on a
  runtime-error reply**. `9124a1f` closed the loud half of this trap (the
  stamp's output was going to `/dev/null`, so an error was invisible) and
  replaced it with a prefix match at line 304:
  `errMsg:*|ERR\ *|WRONGTYPE*|*"read only"*|*"read-only"*`. That set is
  incomplete. FalkorDB returns some runtime errors **bare**, with no prefix at
  all. I ran the helper rather than reading it — extracted verbatim, pointed at
  a live graph:

  | probe | reply | `rq()` |
  |---|---|---|
  | `RETURN (((` | `errMsg: Invalid input …` | **1** (caught) |
  | `RETURN nosuchfunc(1)` | `Unknown function 'nosuchfunc'` | **0** (missed) |
  | `MATCH (n:KaizenEntry) RETURN keys(n.fact)` | `Type mismatch: …` | **0** (missed) |

  A call site with no expected-substring third argument therefore treats a
  failed query as success — the same class of silent pass `9124a1f` set out to
  close, surviving in the fix. The general fact (`redis-cli` exits 0 on an
  error reply and prints it to **stdout**; the paired control is that a
  *connection* failure exits 1 to **stderr**, so `$?` is reliable for the
  unreachable case and blind to the rejected one) is now published in
  `claude/graph-dba/falkordb-quirks.md` by U23. The **fix** is outside `cobb`'s
  write remit and unclaimed.

  Its live kaizen entry is `b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94` (`analyst`,
  09-08) — in the graph, out of U23's pinned scope, and **not** to be cleared
  until the code is fixed.

  This one bears on the pass's own method: I have driven `redis-cli` throughout
  these units. Every clear was verified by a `count` read rather than by an
  exit status, so no disposition rests on it — but that was convention, not
  design, and it held by luck.

- **A second verified full-eight-character `entryId` collision.**
  `b3f2a6d4-9e1c-4a2b-8f7d-2c6e1a9b5d40` (09-01, Claude Code settings merge
  semantics) shares all eight leading characters with
  `b3f2a6d4-8c1e-4a7f-9d2b-1e6f5a0c3d7a` (08-29, `conftest.py` fixtures,
  discarded and deleted in U19) — different facts, different dates, different
  subjects, and they landed in **adjacent chunks of the same agent's inbox**.
  The first collision (`b7e41c92-3f8a…` / `b7e41c92-3d5a…`) was already
  enough to justify `claude/cobb/kaizen/plan.md` K-021, an `entryId`-shape
  guard in `cypher-mcp/server.py`'s write authorizer; two independent
  collisions in one pass, in a graph of a few hundred entries, make the case
  that eight characters is simply not a key here. Every §5 curator operation —
  read, tag, count, resolve, clear — is keyed on `entryId`, so a short-prefix
  match can silently disposition the wrong entry. Every brief in this pass now
  pins complete ids and says so.

  **U22 found a third, and it defeats the guard K-021 proposes.**
  `b7f3a1c2-5d84-4e19-9a6f-2c8e71d40b93` (`analyst`, 09-03) and
  `b7f3a1c2-5d84-4e19-9a06-3c2e8f14d7b0` (`architect`, 09-07) agree on **21
  characters**, diverging at index 21 (`6f` / `06`). Both are well-formed
  uuid4s — I checked them against the regex, and both pass — so a **shape**
  guard cannot see this pair. K-021 now records that limit rather than implying
  a shape check is sufficient; the real guard is uniqueness at write time.

  **U23 found a fourth, and it is the first with both members live.**
  `b1f2c7a4-3d59-4e18-9f60-7a2c5d8e41bb` (`architect`, 09-07) and
  `b1f2c7a4-9e33-4d61-8a52-0c6d5e77a913` (`analyst`, 09-08) share all eight
  leading characters. Every earlier pair had one member already dispositioned;
  these are both unprocessed and sitting in two different agents' inboxes, so
  the collision is now a **live** hazard for the units still to run, not a
  post-hoc curiosity.

  The architect member is also a near-twin of `b1f0c7a4-9d2e-…`, which U23
  promoted an hour earlier — same subject (a grep-with-a-count edit table
  verified by a residual), ids differing by one character inside the first
  eight. Whichever unit takes it must read the section U23 wrote before
  promoting it again, or the same technique lands twice.

  Four collisions in one pass, in a graph of a few hundred entries, is not
  chance. U23's reading — folded into K-021 — is that these ids are **hand-shaped
  by the writing agent**, so they cluster on whatever the agent recently typed;
  that is exactly why the pairs keep landing in adjacent chunks of one inbox
  rather than scattering.

- **Falsified evidence in an `active` plan doc, surfaced by U19 — needs routing
  to `graph-dba` (its owner), not fixed here.**
  `falkor-chat/docs/plans/oversized-indexed-property-guard-graph.md:205` asserts
  "this schema has zero `RELATIONSHIP`-type constraints
  (`grep -n RELATIONSHIP scripts/bootstrap_schema.sh` -> no matches)". `teco`
  re-ran that grep: `falkor-chat/scripts/bootstrap_schema.sh:265` now reads
  `gconstraint "$g" UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId`, added
  2026-08-24 by `8d7dcfb` (K-050 fusion). The document's conclusion (do not
  bound `tr["on"]`) may still hold on other grounds, but its stated evidence is
  false and the document is `Status: active`. This is the exact failure mode
  U19 promoted into `review-techniques.md` in the same run — an honestly-run
  negative grep that decayed reads exactly like a current one — so the instance
  and the rule surfaced together.

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
- **U14 → a second `MENTIONS` node is now owed at pass close, this one to
  `tico`.** U14 tagged `e1a6c4d2` `MENTIONS`→`tico` (falkor-chat's
  `ModelGateway.__init__`→`_build_providers` resolving `{env:}` substitution
  for *every* declared provider, so a harness pointed at
  `config/opencode.example.json` dies on the example file's unused `openai`
  provider). `PRODUCED` resolved, node alive on the `MENTIONS` edge. Kept open
  as `data-scientist` K-003, whose proposed home is
  `falkor-chat/docs/manuals/llm-provider-config.md` §2 — a `tico`-owned
  document, and correctly **not** `falkor-chat/AGENTS.md`: an always-loaded
  context file is the wrong price for a fact that binds only when someone
  writes a harness. Two `MENTIONS` nodes now stand (`qa-engineer`'s
  `b7d5e214`, `tico`'s `e1a6c4d2`); the pass does not close while either does.

- **U14 → the model-bench file collision cost nothing, but it was real.** Four
  of chunk B's nine entries were model-bench statistics facts, dispatched under
  a hard no-write constraint on `docs/**small-model-benchmarking*` and
  `model-bench/**` because a concurrent session had a plan gate in flight
  against them. All four turned out already published — three inside
  `model-bench`'s own shipped code and tests — so nothing was blocked. The
  constraint still earned its place: by the unit's end the concurrent session
  had moved from those docs onto `model-bench/modelbench/fingerprint.py` and
  three test modules, which a writing unit would have collided with.

- **U14 → three `claude/scripts/audit-team.sh` check-7 FAILs stand, none in
  scope.** All three are personal-identifier leaks elsewhere in the repo. Not
  this pass's work; worth a routed unit of its own.

- **U17 → the commit-footer conflict, resolved by the stakeholder mid-pass.**
  U17 promoted a rule into `claude/AGENTS.md` telling every agent to omit the
  `Claude-Session:` footer, on the grounds that the harness `system-reminder`
  carrying it is a platform default with no knowledge of this user's settings
  (`includeCoAuthoredBy: false`), memory, or repo docs. That contradicted this
  session's own practice — all 40 most recent commits carry the footer, six of
  them mine. **Put to the stakeholder, who confirmed the memory wins: drop the
  footer.** The promotion ships as written; commits from U17 onward carry no
  trailer; the six already made keep theirs, since rewriting history is out of
  bounds.

- **U17 → an `entryId` collision on the full first eight characters, verified.**
  `b7e41c92-3f8a-…` (09-02, chunk B) and `b7e41c92-3d5a-…` (09-03, chunk C)
  share all eight leading characters and are entirely different facts. Seven
  units this pass hit false-positive dedup matches on 8-char prefixes; this is
  the first pair that a prefix check *cannot* distinguish. Every §5 curator
  operation is keyed on `entryId`, so a colliding id makes a clear silently
  take the wrong node. `cobb` filed **K-021** proposing an id-shape guard in
  `cypher-mcp/server.py`'s write authorizer — deliberately on id shape only,
  not on content: content validation automates a judgment the curator already
  makes, and any agent can route around it with a differently-useless string.
  Not implemented; component code, outside `cobb`'s write remit.

- **U16 → a standing-memory disagreement, flagged for the human, not acted
  on.** The user's `subagent-permission-mitigation` memory names running the
  parent session in **`acceptEdits`** as the mitigation for auto mode's
  classifier review of subagent actions. `.claude/settings.json` now carries
  **no `defaultMode` key at all** — `6f719ae` pinned `bypassPermissions` on
  2026-08-29 and `4bb96e1` reverted it — so the parent runs on the harness
  default (`auto`). The forensics that justified the revert indicted
  `bypassPermissions` only; `acceptEdits`, whose earlier evidence showed
  *partial per-run stickiness* that `auto` does not give, was never re-tested.
  So the shipped config may have discarded a real if partial mitigation on
  evidence that never examined it. A config decision for the stakeholder, not
  a distillation one.

- **U16 → three already-closed agents are accumulating again.** At U16's close
  `architect` held 5 entries and `graph-dba` 1, all dated 2026-09-07, and
  `analyst`'s newest is the same date. Capture is outpacing the pass on agents
  it has already closed. The top-up units owed at pass close are now
  `qa-engineer` (1 `MENTIONS`), `tico` (1 `MENTIONS`), `architect` (5) and
  `graph-dba` (1).

- **U12 → `architect` needs a top-up unit, like `qa-engineer`.** U7b closed
  `architect` out at 0/0 on 2026-09-03. By U12's close (2026-09-07) it had
  **four new entries**, all created that same day by a concurrent session:
  `0a4b6b2e`, `f4ee08b9`, `b7f3a1c2`, `73df8a0d` (three plan-authoring
  techniques plus one falkor-chat `services.py:2085` fact). This is the
  live-graph property the pass has been recording from the start — "closed
  out" means *every pinned entry was dispositioned*, never `count() == 0` at
  some later moment. A top-up unit is owed before the pass closes, alongside
  the `qa-engineer` one U7 opened.

- **U11 → a parking-lot lesson is now promotable.** `claude/coder/kaizen/plan.md`
  parks U10's "a mutant must be proven to change behavior before its survival
  is read as a coverage gap" lesson, parked *because no owning knowledge base
  existed*. U11 falsified that premise by putting a sibling mutation technique
  into `claude/analyst/review-techniques.md`. It stayed parked only to respect
  the unit's eight-entry scope — a ~15-line promotion whenever wanted.
- **U3 → `salesperson/build.sh:68`**: the `elif command -v node` fallback
  accepts any `node` on `PATH` without the `/mnt/` rejection its own
  `npm`-only branch applies. Harmless today (only `npm` leaks in from
  Windows). Parked in `claude/devops/kaizen/plan.md`; `salesperson/` is not
  `cobb`'s to edit.

- **The working tree is shared with a live concurrent coordination.**
  Discovered while recovering U20 on 2026-09-08: between `05a60ce` (this
  pass's last commit) and the recovery, **19 commits** landed from another
  session — a model-bench/CPG coordination — moving `HEAD` to `29538d6`, and
  two `git status` runs three minutes apart disagreed because that session
  committed in between. It writes into files this pass also writes:
  `claude/cobb/kaizen/{history,plan}.md`, `skills/cpg-analysis/references/freshness.md`,
  `skills/joern-cpg/*`. Three consequences, all live for the remaining units:
  - **Commit by explicit path, never `git add -A`.** The other session did
    exactly this at `29538d6` and so left U20's four uncommitted promotions
    untouched. Had either side swept, it would have committed the other's
    half-finished work under a misleading message. This is the guardrail
    earning its keep, not a theoretical one.
  - **Re-read immediately before editing.** A delegate resumed after an
    interruption may hold an in-context copy of a file that a *different*
    coordination has since rewritten and committed; writing it back wholesale
    destroys that work with no error on either side. Targeted edits against
    freshly-read content only.
  - **`HEAD` is not a stable baseline** — it moved 19 commits under this pass
    without any action of mine. This is the same hazard U18 promoted into
    `claude/teco/teco.md` ("`git show <sha>:<path>` with an explicit commit
    sha, never `HEAD`"), now confirmed live from the opposite direction: there
    the mover was my own integration commits, here it is another session's.
