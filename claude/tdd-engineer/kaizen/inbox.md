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

## 2026-08-16 — Once `tests/eval/__init__.py` exists (see 2026-08-15 entry), a *new* test file that needs something from `tests/eval/conftest.py` itself must import it as `from eval.conftest import X` — a bare `import conftest`/`from conftest import X` silently resolves to `tests/conftest.py` (root), not the sibling one

- **Evidence:** `falkor-chat/server/tests/eval/conftest.py` defines `_falkordb_reachable()`; `server/tests/conftest.py` (root, no `tests/__init__.py`, so pytest imports it as bare top-level module `conftest`) defines a same-named function testing a different bug. Root conftest is always collected first, so by the time any eval test module imports, `sys.modules['conftest']` already points at the root file. Verified directly: a scratch test doing `from conftest import _falkordb_reachable, EVAL_WS` inside `tests/eval/` raised no error but silently returned the wrong function (root's, missing `EVAL_WS` entirely would have been the tell — caught it by printing `sys.modules.get('conftest')` vs `sys.modules.get('eval.conftest')`, which showed two distinct module objects, root vs `tests/eval/conftest.py` respectively). The correct, verified-working import is `from eval.conftest import _falkordb_reachable` — `eval` is importable as a package because pytest already inserted `tests/` (the parent of the `__init__.py`-bearing `eval/` dir) onto `sys.path` during its own conftest-loading walk, and `eval.conftest` is the exact same module object pytest itself loaded as the directory's conftest plugin (confirmed no duplicate execution/side effects).
- **Context:** K-026 Unit 2b analyst-gate fixes (B-1: `_falkordb_reachable()`'s write-mode-query bug) — needed a new pytest test file (`tests/eval/test_conftest_probe.py`) exercising a function defined in `tests/eval/conftest.py` directly.
- **Suggested home:** project docs — append to the same `falkor-chat/AGENTS.md` testing-hazards callout the 2026-08-15 entry is slated for; this is the concrete "how to safely reach into a sibling conftest.py" answer the earlier entry left open, not a separate issue.
