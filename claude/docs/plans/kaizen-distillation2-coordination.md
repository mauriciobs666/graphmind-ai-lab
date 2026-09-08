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
| U24 | analyst chunk F (12 of the 14 held at dispatch: all 09-08) | `a554b6fb7d89af6f0` | in-flight | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U25 | analyst chunk G (2 deferred + whatever has arrived) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |

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

## Four platform failures, all cleanly recovered

Four units were killed mid-run by session rate limits (HTTP 429) — U10, U15,
U19 and U20 — and **not one lost a disposition.** They landed at four different
points, which is what makes them useful together:

| Unit | Died at | Actual state found | Recovery |
|---|---|---|---|
| U10 (`a695c632adfd6d91d`) | starting verification | nothing written; all 27 `coder` edges intact | plain re-dispatch |
| U15 (`ae7d55eeefd996462`) | after every file write, before the clear | complete `history.md` entry on disk, `plan.md` K-004 missing, all 10 nodes intact | narrow **state-recovery** unit for the two remaining steps only |
| U19 (`adb247a3e028c0606`) | first tool call | nothing written; all 12 edges intact | **resumed in place** by `SendMessage` |
| U20 (`a6db6af4910e607bc`) | mid-promotion (`<result>`: *"Now the promotions. Starting with `review-techniques.md`"*) | **four** promotions fully written to disk (`review-techniques.md` +21, `falkordb-quirks.md` +19/-1, `claude-code.md` +40/-5, `python-web-quirks/SKILL.md` +50/-4); `analyst/kaizen/history.md` untouched, zero dispositions logged; all 11 entries intact | **resumed in place** with a written state report |

Two rules come out of it, and both are now load-bearing for this pass:

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

## Follow-ups

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
