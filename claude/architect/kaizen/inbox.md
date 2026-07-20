# Kaizen — Learnings Inbox: architect

> Append-only capture of durable, non-obvious environment facts the `architect` agent
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

## 2026-07-19 — falkor-chat's "byte-identity lock" on `executor._drive_loop` reproduces only by SHA, and only via a line-number-independent extraction

- **Evidence:** `falkor-chat/docs/plans/m3-executor-coordination.md` quotes the lock as SHA
  `71055f756280` with three different byte counts (2839, 2844, 2860); only 2860 is correct. The SHA
  reproduces from `sed -n '333,392p' server/falkorchat/executor.py | sha256sum | cut -c1-12` — i.e.
  it is pinned to *line numbers*, which shift whenever anything above the method changes. Verified
  equivalent that survives edits elsewhere in the file:
  `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12`.
- **Context:** designing K-024's `kind:'process'` proof flow, whose hard constraint is "do not touch
  `_drive_loop`" — every unit's done-condition needs a verification command that stays valid.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`, next to where the lock is quoted)

