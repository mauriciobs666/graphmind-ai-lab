# Kaizen-team distillation — U4 (`tdd-engineer`'s 7(+1)-entry inbox)

> **Status:** active · **Owner:** `analyst` · **Tracks:** `docs/plans/kaizen-team-distillation-coordination.md` (unit U4)

## Scope & verdict

Reviewed the uncommitted working-tree diff produced by `cobb`'s standing kaizen-distillation
procedure (`skills/agent-maintenance/SKILL.md` §5), scoped to `tdd-engineer`'s 7 raw
`:KaizenEntry` nodes plus one fresh mid-pass entry in the shared `kaizen_team` FalkorDB graph.
Six files diffed directly (`git diff`): `claude/tdd-engineer/guard-testing-techniques.md`,
`claude/tdd-engineer/kaizen/{history.md,plan.md}`, `skills/python-web-quirks/SKILL.md`,
`claude/data-scientist/lm-studio-model-notes.md`, `claude/data-scientist/kaizen/history.md`.
Baseline is the current working tree against `HEAD` (nothing committed yet). Out of scope: U1/U2/
U3's own gates (already committed, cited only as shape reference), and
`kaizen-team-distillation-coordination.md`'s own content beyond identifying unit boundaries.

**Verdict: needs changes.** Both promoted facts, all three discards, the K-012 backlog item, the
mid-pass cross-agent promotion, and the graph-clear state all check out against primary source —
most independently re-verified live, not just re-derived from cobb's narrative (see "What's
solid"). K-013's "kept open, not promoted, not discarded" disposition is also the right call,
independently corroborated by my own second reproduction attempt. One blocker: the new
`ruff format` KB paragraph asserts a specific, numbered "real instance" (a 1418-line
`git diff --stat` reduced to a 269-line hand-reconstruction) that I could not find any trace of
anywhere in this repo's history — not in `git log` (all branches, no reformatting/ruff-format
commit exists), not in any `HISTORY.md`/kaizen file. It reads as fabricated color grafted onto an
otherwise entirely true and independently-verified general claim, shipped into a "live-verified"
knowledge base consumed by every agent doing Python work in this repo.

**CPG: not applicable — this unit reviews prose/documentation edits to agent knowledge-base,
history, and plan files with no code-level component of their own; every source claim inside
those edits (the union/disjointness algebra, the `ruff format` behavior, the hazard-curve/
RediSearch/p95-rank discards, the `assert_index_scan` gap, the ANN-recall repro, the Ministral/
Qwen3 system-message behavior) was verified by direct inspection of the cited Python/Bash source
and by live reproduction (ruff itself, `GRAPH.PROFILE`/`GRAPH.QUERY` against disposable FalkorDB
graphs), not via a CPG traversal — no CPG is loaded for `model-bench`/`falkor-chat` scripts in
any case (`cpg_falkorchat` covers the server, not `scripts/test_queries.sh` or `model-bench/`).**

## Findings

### Blocker — the new `ruff format` KB entry states a specific numbered "real instance" I could not find any evidence of

`skills/python-web-quirks/SKILL.md:939-947` (new): after correctly describing the live-tested,
independently-reproducible general behavior (a 5-line mixed-spacing sample fully reformatted by
`ruff format`, which I re-ran myself with the same ruff 0.14.14 and got the identical result —
whitespace normalized, two blank lines inserted around `def`), the paragraph adds: *"one real
instance: 3 new long lines fixed via `ruff format` on a test file produced a 1418-line
`git diff --stat`, where hand-reconstructing the same fix from `git show HEAD` plus the new
section kept it at 269 lines"*. I searched for this instance three ways and found nothing:
`git log --all --oneline --stat | grep -B5 "1418 "` (no hit), `git log --all --pretty=... | grep
-i "ruff format\|reformat"` (no commit subject matches either term, on any branch), and a repo-wide
grep for "1418"/"269 lines"/"diff --stat" across every `.md` (the only "1418" hits are an unrelated
coincidence — `model-bench/docs/HISTORY.md:3058/3136`, `stats.py`'s own line count in an
unrelated Pass-9 fix round, not a diff-stat figure, and not about `ruff format` at all). A
"live-verified knowledge base" (`skills/python-web-quirks/SKILL.md`'s own framing, mirrored by
`lm-studio-model-notes.md`'s explicit banner) that ships a specific, quantified incident with no
locatable source undermines exactly the evidentiary discipline the rest of this pass demonstrates
(cobb re-verified the other five dispositions against primary source in each case). **Fix:**
either cite the actual commit/PR this happened in (if it exists on an unlisted branch or was
squashed/amended away — check `git reflog` too) and correct the description to match it exactly,
or delete the "one real instance" clause entirely and let the paragraph stand on the
independently-reproducible general claim, which is true and needs no supporting anecdote.

### Assessment — K-013's disposition ("kept open, not promoted, not discarded") is correct, corroborated by an independent second reproduction attempt

Ran my own minimal repro, deliberately not looking at cobb's script first beyond the shape
described in `claude/tdd-engineer/kaizen/plan.md`: a fresh disposable dim-4 cosine vector index
(`CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) OPTIONS {dimension:4,
similarityFunction:'cosine'}`), two `Chunk` nodes with identical `vecf32([1,0,0,0])` embeddings,
confirmed both returned by `db.idx.vector.queryNodes(k=4)`, then one `SET` per node
(`documentCurrent=false`, applied one at a time and then both) with an ANN re-query after each
step. **Both nodes stayed in the result set in every variant** — the same negative result cobb
reports, from an independently-authored probe rather than a re-run of cobb's own script. This is
a second, differently-shaped data point against the raw claim's specific 1-2-event threshold, and
it still leaves the question genuinely open rather than closed: `document-ingestion2-rca.md`
Appendix B's own established mechanism is that recall degrades as a function of **cumulative
churn on the same long-lived index** (the RCA's own repro needed ~150-200 cycles at k=4, on an
index that had already absorbed hundreds of prior create/delete cycles from a whole pytest
session) — a truly clean 2-node index with 1-2 total writes is nowhere near that regime by the
RCA's own curve, so a negative result here doesn't yet distinguish "the raw claim describes a real
but distinct low-churn mechanism specific to identical embeddings" from "the raw claim's own
`ws:test`-based observation was itself already sitting on pre-existing session churn it didn't
account for" (exactly the ambiguity K-013's own text names). Promoting into
`falkordb-quirks.md` now would ship a claim two independent minimal probes couldn't confirm as
ground truth; discarding would drop a claim that indisputably drove a real Stage C test change.
**Kept open, routed to `graph-dba` for a churn-seeded repro, is the correct disposition** — no
change requested. One low-cost suggestion: fold this session's negative result (a second,
independently-authored minimal-probe failure to reproduce) into K-013's own plan.md entry as
corroborating evidence for whoever picks it up next, so `graph-dba` doesn't have to re-run the
trivial case before moving straight to the churn-seeded one.

## What's solid

- **Both promotions' underlying technical claims verified against source.** The union/
  disjointness "clean move doesn't redden either assertion" claim matches
  `model-bench/modelbench/scoring/toolcalls.py:81-101` exactly — `ITERATION_SUMMARY_DISPOSITIONS`/
  `ITERATION_SUMMARY_EXCLUDED` are `frozenset` literals asserted against `convo.TURN_DISPOSITIONS`
  for both union-completeness and disjointness, and the set-algebra argument (a relocation
  preserves both invariants; only an incomplete reclassification or a drop reddens either) is
  correct by construction, not just plausible. The `ruff format` behavior itself (reformats the
  whole file, not just touched lines) is independently reproduced (see Blocker above for the one
  unsupported embellishment on an otherwise-sound entry).
- **All three discards independently re-confirmed as already-documented, not taken on the
  report's word.** Hazard-curve `H` truncation: `toolcalls.py:507-531`'s own docstring states the
  `None`-computes-every-position behavior, and the live call site (`toolcalls.py:985`) passes
  `h=None` — confirmed by direct read and grep. RediSearch fuzzy-token stripping: covered in more
  depth at `claude/graph-dba/falkordb-quirks.md:79-84`, same function
  (`repository._escape_fuzzy_token`), same QA Defect 1 incident. `math.ceil` p95-rank mirror trap:
  published verbatim, including the mirror-trap reasoning and the `x<=200` caveat, in
  `model-bench/tests/test_scoring_toolcalls.py:543-552`'s own `_independent_p95_rank` docstring —
  confirmed by direct read.
- **K-012 re-verified live, independently of cobb's own probe.** I built my own disposable graph
  (`CREATE INDEX FOR ()-[r:SUPERSEDES]-() ON (r.matchId)`, one edge, `GRAPH.PROFILE` on the exact
  query shape `falkor-chat/scripts/test_queries.sh:1694` uses) and got `Edge By Index Scan |
  [r:SUPERSEDES]`, confirming `assert_index_scan` (`test_queries.sh:52-62`, checks only `"Node By
  Index Scan"`/`"Node By Label Scan"`) is genuinely blind to this shape. The cited "458/459 —
  one known failure" baseline is also real, not invented: `falkor-chat/docs/HISTORY.md:293`
  documents exactly this pre-existing §14.8 SUPERSEDES.matchId gap as the suite's one standing
  failure as of a prior, unrelated fix. Correctly kept open (a `falkor-chat/scripts/` code fix,
  outside `cobb`'s write remit), consistent with K-007..K-011's precedent.
- **The mid-pass cross-agent promotion is accurate and non-duplicative.** Read
  `convo.py:379-396`'s `_prologue_system_message` docstring directly: it documents only the
  Ministral HTTP-400-on-second-system-message half, never Qwen3's tolerance of the same shape —
  matching the new `lm-studio-model-notes.md` section's claim that the Qwen3 half is the entry's
  actual new content. Read the existing adjacent "role alternation" section in the same file and
  confirmed it documents a different mechanism (same-role-repeats-consecutively vs. a hard cap on
  a second system-role message) — not a restatement.
- **Graph state matches the report, on both shapes.** Live re-query of `kaizen_team`:
  `MATCH (a:Agent {agentId:'tdd-engineer'})-[:PRODUCED]->(k:KaizenEntry) RETURN count(k)` → `0`;
  legacy `MATCH (k:KaizenEntry {author:'tdd-engineer'}) RETURN count(k)` → `0`. `GRAPH.LIST`
  shows no leftover probe/verification graphs attributable to this pass.

## Open questions

None for `cobb` — the one blocker has a mechanical fix (cite the real instance correctly, or
delete the clause). K-013's disposition needs no further input from this review; the fold-in
suggestion above is optional, not a gate condition.
