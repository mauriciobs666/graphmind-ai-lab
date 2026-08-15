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

## 2026-08-15 — Two sibling pytest dirs without `__init__.py` that each ship a `conftest.py`, when a third file does a bare `from conftest import X`, collide non-deterministically on the bare module name `conftest`

- **Evidence:** falkor-chat's `server/tests/conftest.py` (root) defines `TEST_EMBEDDING_DIM`; `server/tests/test_tools.py`/`test_graphrag.py` do `from conftest import TEST_EMBEDDING_DIM` (a bare import relying on pytest's "prepend" import-mode sys.path insertion, not a package-relative one). Once a sibling test dir (`server/tests/eval/`, itself lacking `__init__.py`) also shipped its own `conftest.py`, running the whole-repo default suite (`pytest -q`, no path arg) failed collection: `ImportError: cannot import name 'TEST_EMBEDDING_DIM' from 'conftest' (.../tests/eval/conftest.py)` — Python's `sys.modules['conftest']` cache got clobbered by whichever dir's `conftest.py` pytest imported second, and the bare import in `test_tools.py` picked up the wrong one. Reproduced deterministically by temporarily removing `tests/eval/conftest.py` (root suite passes) and by removing my own unrelated `tests/eval/*.py` additions (collision persisted — proved the cause was the second `conftest.py` file existing at all, not anything about the new files' own content). `pytest tests/eval -q` alone (no root `tests/*.py` in the same session) never triggers it — the collision only appears once both dirs are collected in one session (i.e. the default no-path `pytest -q`).
- **Context:** K-026 Unit 3 (GraphRAG eval harness judge layer) — discovered as a pre-existing defect introduced by a concurrently-landed Unit 2b (`server/tests/eval/conftest.py`), not by anything in this unit's own deliverables; flagged to the coordinator rather than fixed unilaterally (fixing it correctly needs a repo-wide decision: add `__init__.py` throughout `tests/` — which then also breaks every existing *bare* sibling import like `from metrics import ...`/`from judge import ...` used inside `tests/eval/*.py` unless those are rewritten too — or rename one `conftest.py`, or switch `test_tools.py`/`test_graphrag.py` to a non-bare import).
- **Suggested home:** project docs (a callout in `falkor-chat/AGENTS.md`'s testing-hazards section, alongside the existing `DESIGN.md` §14.7 pytest gotchas) — this is a repo-specific structural landmine, not a generic Python/pytest fact worth a skill entry on its own, though the underlying pytest "prepend import mode has no package namespacing without `__init__.py`" mechanism is a fact `python-web-quirks` could plausibly want too if it recurs elsewhere.
