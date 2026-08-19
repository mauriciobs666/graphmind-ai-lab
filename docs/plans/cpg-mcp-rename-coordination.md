# CPG MCP server/tool rename — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (`docs/requirements/cpg-mcp-rename.md`)

## Goal & definition of done

Deliver `docs/requirements/cpg-mcp-rename.md` (Status: Ready for design, FR-1…FR-7). Rename the
MCP server/tool from `cpg`/`mcp__cpg__query` to `cypher`/`mcp__cypher__query` and relocate
`cpg/mcp/` to match, as a single atomic change, updating every currently-**active** reference
across the repo (60+ files per the requirements doc's own estimate) while leaving genuinely
CPG-specific naming (the `cpg-analysis`/`joern-cpg` skills, `graph-dba`'s Joern pipeline,
`cpg_<component>` graph names, archived documents) untouched.

**Sequencing decision (teco, 2026-08-19):** this rename lands **before** `generic-cypher-mcp2`
(M6, team-wide kaizen-graph rollout, also `Ready for design`) is put into design. M6 will produce
new active plan/coordination/review documents that reference the MCP tool by name; landing the
rename first means M6 is designed and built directly against the final `cypher` name instead of
being renamed a second time right after. M6 stays queued, not started, until this coordination
closes.

## Unit ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `architect` | `aeb46318c1156c6c4` | delivered | `docs/plans/cpg-mcp-rename.md` — relocates `cpg/mcp/` → top-level `cypher-mcp/`; full 8-axis identity mapping (tool/server-key/path/Docker image+label/env-var prefix/shell-fn names/log prefix); status-driven grep-sweep discovery (not a fixed list, direct fix for M5's B1 precedent) with a 5-step per-hit classification and a before/after FR-6+FR-7 proof gate; `.mcp.json`/`.claude/settings.json` wiring; `image-tag.sh` confirmed location-independent (one clean rebuild); proposes this delivery as **M6** (C-601…C-605), flagging the resulting collision with `generic-cypher-mcp2.md`'s already-claimed `(M6)` header and proposing a `(M6)`→`(M7)` bump on that sibling doc as part of step 3b — explicitly flagged for plan-gate confirmation, not silently applied; 4-step table (1 `coder` relocate+rebuild, 2 `coder` harness+agent-surface wiring, 3a/3b `cobb` parallel claude+skills vs. docs+mcp-monitor+falkor-chat sweeps, 4 `qa-engineer` acceptance) with explicit oversized-step justification; AC-1…AC-6 each mapped to a concrete static/live check. §6 flags 4 risks/open items for the plan-gate reviewer, including the M6/M7 call, the living-log surgical-substitution interpretation, the env-var/Docker-namespace scope call, and one review document whose subject is a quoted historical diff. (One transient platform failure — weekly API limit — mid-run; resumed cleanly from confirmed on-disk state per teco's state-recovery brief, no rework needed.) | plan gate (`analyst`) → — |
| U2 | `analyst` | `a4c622a3a7b44bf71` | delivered | `docs/reviews/cpg-mcp-rename.md` — **B1** (blocker): §3.2's discovery/proof `git grep` pattern misses bare unquoted `cpg` tool-identity references (verified real misses: `server.py:2`, and worse `server.py:131`'s `SERVER_INSTRUCTIONS` text every connecting agent sees, `docs/BACKLOG.md:240,376`, `docs/HISTORY.md:465`, `claude/graph-dba/falkordb-quirks.md:277`) — since this same pattern is the plan's own AC-1 final proof gate, it would report a false-clean pass, reproducing the exact "fixed-artifact under-covers" failure (M5's B1) this design set out to fix; suggested fix: add `\|\bcpg\b` (case-insensitive) to the alternation. **M1** (major): the 3-document self-referential exemption list omits this very review doc (`docs/reviews/cpg-mcp-rename.md`, `Status: active`, produced before the sweep runs) — sweep would incorrectly rewrite its own renamed-from-X findings; widen exemption to the whole `cpg-mcp-rename*` family. 2 minors: nonexistent `docs/plans-coordination/` dir cited in §3.2 glob (coordination docs live in `docs/plans/`); §6's 4th open item (quoted historical content in `cpg-mcp-joern-agent-string-fix.md`) is resolvable outright — reviewer read the file, confirmed its 4 hits are plain path citations, not quoted string content. Also gave explicit views on all four §6 open items as requested (image-tag.sh location-independence confirmed by full read; 84/7 baseline + `build.sh --verify-inputs` both confirmed still holding). | plan gate → **needs changes** |
| U1-fix | `architect` | `aeb46318c1156c6c4` | delivered | `docs/plans/cpg-mcp-rename.md` (Version 1.1, §7 dated revision note) — B1 fixed: widened §3.2 grep with `\bcpg\b`, deliberately **case-sensitive** (diverges from the review's literal case-insensitive suggestion — live-tested both: case-insensitive balloons 94→141 files pulling in the `` `CPG:` `` evidence-trail convention and "Code Property Graph (CPG)" prose as noise; case-sensitive widens 94→135, still catches all 6 of the review's confirmed misses, zero noise-category hits); rule 5 gained a bare-token sub-bullet + 2 new CPG-domain exclusions the wider net surfaces; §4/§5 now state the ~135-file triage volume honestly. M1 fixed: 3-document exemption list → basename-prefix rule (`cpg-mcp-rename*` across the 5 doc kinds), closing the real 4th member (the review doc itself) and future-proofing test-plan/test-report. Both minors fixed (removed nonexistent `docs/plans-coordination/` ref; closed the `cpg-mcp-joern-agent-string-fix.md` open item outright per the review's own confirmed finding). Review's "sound, no action" verdicts on M6→M7 and env-var rename folded into §6 as confirmations. | re-gate (`analyst`) → — |
| U2-regate | `analyst` | `a4c622a3a7b44bf71` | accepted | `docs/reviews/cpg-mcp-rename.md` §Pass 2 — re-derived every claimed fix rather than trusting the revision note: B1 confirmed closed (live-ran both grep variants, exact match to plan's claimed 141/135 counts, all 6 Pass-1 misses now caught, diffed the 6-file case-sensitive/-insensitive delta and confirmed all genuine noise — the case-sensitivity deviation from the reviewer's own literal suggestion is empirically the right call); M1 confirmed closed (basename-prefix rule covers the review doc itself, exempts exactly the 4 real family members, no over-exemption); both minors confirmed closed; both §6 confirmations checked accurate. 2 new **non-blocking** observations for the implementer: (a) step 3a's `claude/*/*.md` glob under-reaches `claude/docs/requirements/security-expert.md` but the step's own done-condition wording independently forces it into scope anyway; (b) discovery `git grep` is tracked-files-only by default, currently harmless only because rule 2 independently protects this delivery's own untracked docs — cheap `--untracked` addition would remove the coincidence. | plan gate → **approve with suggestions** |

**Plan gate closed** (2 passes, final verdict approve with suggestions). Implementation units below
are sized 1:1 against `docs/plans/cpg-mcp-rename.md` (Version 1.1) §4's own step table.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U3 (step 1, C-601) | `coder` | `a71f83cd1f1914fa6` | in-flight | `cypher-mcp/` (relocated from `cpg/mcp/`) | — |
| U4 (step 2, C-602) | `coder` | | queued | `.mcp.json`, `.claude/settings.json`, agent tool-surface wiring | code re-gate (`analyst`) → — |
| U5 (step 3a, C-603) | `cobb` | | queued | `claude/`+`skills/` sweep | (folded into U4/U6 re-gate) |
| U6 (step 3b, C-604) | `cobb` | | queued | `docs/`+`mcp-monitor/`+`falkor-chat/` sweep, BACKLOG.md M6 section, generic-cypher-mcp2.md M6→M7 bump | code re-gate (`analyst`) → — |
| U7 (step 4, C-605) | `qa-engineer` | | queued | `docs/test-plans/cpg-mcp-rename.md` + `docs/test-reports/cpg-mcp-rename-report.md` | acceptance → — |

Sequencing (per plan §4, with teco's own same-file serialization applied): U3 → (U4 ∥ U6) → U5 →
U7. U4 and U6 have no file overlap (agent/harness wiring vs. docs/mcp-monitor/falkor-chat) so
dispatch together once U3 lands. U5 (step 3a) is serialized **after** U4 rather than run parallel
with it as the plan's own dependency diagram would allow — U5's file list (`claude/analyst/
analyst.md`, `claude/architect/architect.md`, `skills/cpg-analysis/SKILL.md` body prose) overlaps
U4's file list (same 3 files, different lines) and this repo's standing practice is to serialize
same-file units rather than run parallel agents against a shared file, even when the touched lines
don't collide. U7 depends on all three.

## Notes

- Requirements doc flags this as "a wide, cross-component rename, not a cosmetic single-file
  tweak" — 60+ files across `cpg/mcp/`, `claude/AGENTS.md`, multiple agents' operative prompts
  and kaizen history, `skills/`, and `docs/plans|reviews|test-plans|test-reports`, plus
  `mcp-monitor/`'s docs (cites the tool as an example).
- FR-4 explicitly excludes archived documents from the sweep — the plan needs a reliable
  mechanism (e.g. a `grep` sweep filtered by each hit's own document `Status:`) to tell active
  from archived, not a fixed file list (the M5 precedent's B1 finding — a fixed file list omitted
  two agents' own operative prompts — is exactly the failure mode to design around here).
- FR-3's relocation target for `cpg/mcp/` is left to the architect's judgment (requirements doc
  doesn't name one); `.mcp.json` wiring, `docker-run.sh`/`build.sh`, and the content-hash image
  tag all need to keep working post-move.
