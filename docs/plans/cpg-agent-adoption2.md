# CPG agent adoption — freshness check centralized on teco

> **Status:** active · **Owner:** `cobb` · **Tracks:** cpg-agent-adoption (M4) · **Extends:** `docs/plans/cpg-agent-adoption.md`

Redesigns one slice of the archived `cpg-agent-adoption.md` — the FR-5/FR-6 **ownership** of the
freshness check, not its mechanism. Stakeholder decision, 2026-08-19, made while reviewing the
team's prompt verbosity. Everything else in that document (AC-2's `CPG:` spot-check line, the
six-agent CPG-orientation contract, the skill architecture, the `:CpgBuildInfo` marker and
`references/freshness.md` recipe themselves — still graph-dba-owned/built) is untouched and stays
authoritative.

## Decision

The per-consumer freshness check (`skills/cpg-analysis/references/freshness.md`) is dropped from
`analyst`, `architect`, `coder`, `tdd-engineer`, `frontend-engineer`, `qa-engineer` and
centralized on `teco`, which runs it once at dispatch — when briefing a unit whose specialist
will likely consult a CPG — and states the result in the brief.

**Accepted trade-off:** a specialist invoked standalone, not through `teco`, no longer checks CPG
freshness at all. Deliberate scope cut, not an oversight: it removes a real capability from the
un-coordinated path in exchange for six shorter, duplication-free prompts (the freshness
paragraph was byte-identical boilerplate across all six, ~130 words each).

## What changed

| File | Change |
|---|---|
| `claude/{analyst,architect,coder,tdd-engineer,qa-engineer,frontend-engineer}/*.md` | Freshness-check clause removed from each CPG-orientation paragraph; "check whether a CPG exists and use it" stays as-is. |
| `claude/teco/teco.md` | Added `mcp__cypher__query` to `tools:`; new "CPG freshness" bullet in §3 (Delegate with complete briefs); Guardrails note flagging the grant as not yet live-verified. |
| `skills/cpg-analysis/SKILL.md` §4, `references/freshness.md` header | Consumer/routing updated from "any consuming agent" to `teco`, at dispatch. |

## Follow-up required — live-verify the tool grant

`teco`'s frontmatter already carries `Grep`/`Glob` that are declared but not actually granted at
runtime (verified 2026-08-10 by probing a live run — a real precedent for "declared in
frontmatter ≠ live"). `mcp__cypher__query` was added the same way here and has **not** been
live-probed. Before relying on this design in a real coordination: confirm with a live `teco` run
that the tool actually resolves. If it doesn't, `teco`'s own Guardrails now say to report the
CPG-freshness duty as blocked rather than silently skip it — but that's a documented fallback, not
a substitute for checking.

## AC-2 is untouched

The `CPG:` verdict-line convention (`used <graph> — <clause>` / `considered, not relevant —
<clause>` / `not applicable — <clause>`, `docs/plans/cpg-agent-adoption.md` §3) is orthogonal to
freshness — it evidences *whether* an agent used the CPG, not whether it checked staleness — and
keeps its original wording, unchanged, in all six files.
