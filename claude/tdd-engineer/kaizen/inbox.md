# Kaizen — Learnings Inbox: tdd-engineer

> Append-only capture of durable, non-obvious environment facts the `tdd-engineer` agent
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

## 2026-07-15 — A reachability-`skip` does NOT gate a live test when the live dep is normally up; only marker deselection does

- **Evidence:** `falkor-chat/server/tests/conftest.py` gates its FalkorDB integration tests with
  `if not _falkordb_reachable(): pytest.skip(...)`. Copying that pattern for an LM-Studio-backed
  test is a trap: LM Studio was reachable (`curl localhost:1234/v1/models` → 200), so the skip
  never fires and the "network-free" default `pytest` silently starts making LLM calls (the U14
  test takes 20–40s+ vs. the suite's 6s). Registering `markers = ["live: ..."]` **plus**
  `addopts = '-ra -m "not live"'` in `pyproject.toml` deselects instead: default run =
  `312 passed, 1 deselected` (verified), and a command-line `-m live` overrides the addopts `-m`
  (verified: `pytest -m live --collect-only` → `1/313 tests collected (312 deselected)`).
  The reachability-skip is still worth keeping *inside* the live test — but as the "don't fail for
  env reasons" net, not as the gate.
- **Context:** K-022 U14 — writing the first marker-gated live LLM e2e in `falkor-chat`, where the
  hard constraint was that the network-free baseline stay green and fast with LM Studio running.
- **Suggested home:** prompt (a general test-gating rule: gate on a marker when the dep is usually
  present; reachability-skip only guards against env absence, it does not opt tests out)

## 2026-07-15 — FalkorDB's empty-`UNWIND` row collapse silently turns "no rows" into an `IndexError` at the *caller*, not a graceful empty write

- **Evidence:** `falkor-chat/server/falkorchat/repository.py:1397` (`materialize_snapshot`) does
  `row = res.result_set[0]` after a `MERGE`-and-`UNWIND $transitions` query. Calling it with
  `transitions=[]` (a legitimate single-terminal-step def) raised
  `IndexError: list index out of range` — the bare `UNWIND []` collapsed the whole row stream
  before the `RETURN`, so the query wrote nothing AND returned nothing. `AGENTS.md` documents this
  quirk only for the §4 mention write-block (which is defended by an
  `UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END)` guard); `materialize_snapshot`
  has no such guard. The failure surfaces as an unrelated-looking Python `IndexError`, which is
  easy to misread as a repository bug rather than the known engine quirk.
- **Context:** K-022 Defect B — authoring a drive-level reproduction test needed a minimal workflow
  def; the natural shape (one terminal step, zero transitions) crashed on setup and briefly looked
  like a RED for the wrong reason. Worked around by using a 2-step/1-transition def.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` — generalize the empty-`UNWIND` note
  from "the mentions write-block" to "any `UNWIND $list` whose caller indexes `result_set[0]`";
  `materialize_snapshot` is a second, unguarded instance)
