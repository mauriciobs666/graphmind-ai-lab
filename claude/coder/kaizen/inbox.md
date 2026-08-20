# Kaizen — Learnings Inbox: coder

> **FROZEN — 2026-08-20.** This file is a historical snapshot only (no entries had accumulated at
> migration time). `claude/cobb/kaizen/history.md`'s 2026-08-20 entry records the team-wide
> switch; `coder` no longer appends here. New raw learnings are written directly into the
> `kaizen_coder` FalkorDB graph and are immediately queryable by any agent:
> `mcp__cypher__query(graph='kaizen_coder', cypher='MATCH (e:KaizenEntry) RETURN e.date,
> e.fact, e.evidence, e.context, e.suggestedHome, e.author ORDER BY e.date')`. Content below is
> preserved for historical reference and will not change.

> Append-only capture of durable, non-obvious environment facts the `coder` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-19 — `git mv <dir1> <dir2>` moves a tracked directory's *untracked* contents too (it's a filesystem rename, not an index-only op), but a Python venv's internal absolute-path artifacts (script shebangs, `pyvenv.cfg`) still point at the old path and must be regenerated
- **Evidence:** `git mv cpg/mcp cypher-mcp` (relocating the `cypher-mcp` MCP server per `docs/plans/cpg-mcp-rename.md` step 1) carried the gitignored `.venv/`, `__pycache__/`, `.pytest_cache/` along with the tracked files — `ls cypher-mcp/` showed `.venv` present immediately after the `git mv`, no separate copy needed. But `./cypher-mcp/.venv/bin/pytest cypher-mcp/tests -q` failed with `cannot execute: required file not found` — the venv's `bin/pytest` shebang still hardcoded `#!/home/.../cpg/mcp/.venv/bin/python`, a path that no longer existed. Fixed by `./cypher-mcp/setup.sh --recreate`.
- **Context:** cpg-mcp-rename U3 (relocating `cpg/mcp/` → top-level `cypher-mcp/`, renaming every internal identity string) — the offline test suite needed to run clean from the new path before the container gate.
- **Suggested home:** knowledge base (a directory-relocation checklist item: after `git mv`-ing a directory containing a Python venv, always `--recreate` it rather than assuming the moved venv still works).
