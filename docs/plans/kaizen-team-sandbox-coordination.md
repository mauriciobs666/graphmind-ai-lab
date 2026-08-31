# Agent-team graph sandbox — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (M<n> TBD)

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `analyst` | `ac9ceb1483591ccda` | gated | `docs/reviews/kaizen-team-sandbox.md` | `analyst` → needs changes (3 Major, 2 Minor, none requirements-level) | 102.6k tok / 22 tool uses |
| U1b | `architect` | `a90193ecac610bacb` | delivered | `docs/plans/kaizen-team-sandbox.md` (Version 2, revision note added) | — → — | 86.5k tok / 20 tool uses |
| U1c | `analyst` | `ac9ceb1483591ccda` | gated | `docs/reviews/kaizen-team-sandbox.md` `## Pass 2` | `analyst` → needs changes (3 orig. Majors fixed; 1 new Major: undeclared `jq` dep, no fallback, not installed on this box; 2 new Minors) | — |
| U1d | `architect` | `a90193ecac610bacb` | delivered | `docs/plans/kaizen-team-sandbox.md` Version 3 (jq→python3 fallback, 7-field template, port-reuse fix) | — → — | — |
| U1e | `analyst` | `ac9ceb1483591ccda` | gated | `docs/reviews/kaizen-team-sandbox.md` `## Pass 3` | `analyst` → needs changes (3 Pass-2 items fixed; 1 new Major, empirically confirmed by running the python3 fallback: JSON round-trip reformats `.mcp.json`'s untouched `cypher` entry, breaks AC-4's byte-diff check; 1 new related Minor: teardown-before-mcp.json-removal leaves a stale entry an "add-if-missing" provisioning run won't correct) | — |
| U1f | `architect` | `a90193ecac610bacb` | delivered | `docs/plans/kaizen-team-sandbox.md` Version 4 (text-anchored splice, no jq/python3 dep; add-or-correct) | — → — | — |
| U1g | `analyst` | `ac9ceb1483591ccda` | accepted | `docs/reviews/kaizen-team-sandbox.md` `## Pass 4` | `analyst` → **approve with suggestions** (empirically verified splice mechanism via a 3-scenario fixture; 1 non-blocking Minor: name the safe shell idiom explicitly, avoid `awk -v` escape-reinterpretation footgun) | — |
| U1h | `architect` | `a90193ecac610bacb` | delivered | `docs/plans/kaizen-team-sandbox.md` Version 5 (names safe `sed '/anchor/r tempfile'` splice idiom, §4 step 1) | — → — | — |
| U2 | `teco` | — | accepted | commit of `docs/plans/kaizen-team-sandbox.md` (V5), `docs/reviews/kaizen-team-sandbox.md`, this coordination doc | — → — | — |

## Notes

- Requirements doc (`docs/requirements/kaizen-team-sandbox.md`, Status: Ready for design) already
  committed at `e723166`.
- Plan doc went through 4 review rounds before a clean gate: Pass 1 needs changes (3 Major: `.mcp.json`
  git lifecycle undecided, AC-5 recording gap for undocumented work, AC-2 verification polluting
  production schema — plus 2 Minor); Pass 2 needs changes (all 3 Majors + both Minors fixed, but the
  fix for Major #1 introduced a new Major: undeclared `jq` dependency, not installed on this box, no
  fallback — caught because `analyst` live-checked `which jq`/`jq --version` on this machine rather
  than trusting the plan's prose); Pass 3 needs changes (jq→python3 fallback fixed, but `analyst`
  **empirically ran** the python3 fallback against a copy of the real `.mcp.json` and found it
  reformats the untouched `cypher` entry, breaking the plan's own AC-4 byte-diff check — a bug static
  reading alone would have missed); Pass 4 **approve with suggestions** (architect's replacement
  text-anchored splice — which also removed the jq/python3 dependency entirely — empirically verified
  via a 3-scenario fixture: insert/stack/delete-middle/delete-last, valid JSON at every step, `cypher`
  entry never reformatted, byte-identical after full teardown; one non-blocking Minor on documenting
  the safe shell idiom, folded in as Version 5 before commit since it was a one-line doc addition with
  no design impact). None of the 4 passes' findings reopened the requirements doc.
- **Takeaway for future infra/process plan reviews:** static reading caught the first-round design
  gaps, but the two deepest bugs (the `jq` absence, and the JSON-reformatting bug) were only found
  once `analyst` actually **ran** the described shell mechanism against a realistic fixture instead of
  just reading it — worth defaulting to for any plan whose deliverable is executable scripting, not
  just application code.
- No CPG relevant: `GRAPH.LIST`-equivalent check (`kaizen_team` labels query) confirms only
  `KaizenEntry`/`Agent` — matches the plan's own §1 finding that no `cpg_cypher-mcp`/`cpg_claude`
  graph is loaded for this design's touched paths (`cypher-mcp/`, `claude/`, provisioning scripts).
- **Follow-up flagged, not acted on:** the plan's step 4 (§4, steps 1-7) is real forward-looking
  implementation work (devops provisioning/teardown scripts, graph-dba schema/data-copy
  verification, cobb's `claude/AGENTS.md` pointer) with no `docs/BACKLOG.md` entry yet — that file's
  `C-` numbering ties its hundreds digit to an assigned milestone, and this plan's `Tracks:` is still
  `— (M<n> TBD)`. Not adding a guessed ID here; flagged in the final report for the
  stakeholder/`architect` to assign a milestone and open the backlog item accordingly.
- §6's own flagged open governance question (should the "Sandbox & promotion" subsection convention
  be promoted into root `AGENTS.md`) is unresolved by design — the plan itself routes it to the
  stakeholder/`teco`, not a defect. Also carried into the final report.
