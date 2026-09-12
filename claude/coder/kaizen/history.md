# Kaizen — Change History: coder

> Dated log of actual changes to the `coder` agent. Most recent first.

## 2026-09-11 — Added a `Bash` guard (`guard-coder-broad-bash.sh`) — closes the Bash confirmation gap `acceptEdits` never covered

- **What:** `cobb` added a second `PreToolUse` hook to `coder.md` (`matcher: Bash`, alongside the
  existing `Write|Edit` guard): `coder/hooks/guard-coder-broad-bash.sh`, a thin wrapper over the
  new shared `claude/scripts/guard-broad-bash.sh` core. `permissionMode: acceptEdits` auto-approves
  file edits but has no effect on Bash, so every `pytest`/`git`/`npm` call still hit the plain
  confirm prompt even after the Write/Edit guard shipped (2026-08-28) — user-reported live
  2026-09-11 ("i tried accept edits but then any command ask so we just moved the problem
  around"). The new guard reuses `guard-destructive-ops.sh`'s own destructive-pattern matching
  (piping the same stdin through it) rather than duplicating the catalog: a genuinely destructive/
  shared-state command still escalates to `ask`, everything else gets an explicit `allow`.
- **Why:** Direct user-reported friction; closes a real gap in this agent's own hook wiring, not
  the settled Task/`Agent`-delegation classifier limitation (that one is unaffected by this fix and
  remains permanent). Full design/verification detail in `claude/cobb/kaizen/history.md`,
  2026-09-11.
- **Plan items:** —

## 2026-09-10 — `kaizen_team` distillation pass 2, unit U53 (`coder`'s second, fresh 1-entry chunk): 1 promoted, 0 discarded, 0 kept open — cleared, **folded into `coder.md` itself**

- **What:** `cobb` ran `agent-maintenance` §5 over `coder`'s one fresh `PRODUCED` entry
  (`a1e3c9d2-6f4b-4a2e-9c1a-3d5f7b8e2c41`, dated 2026-09-10), re-queried fresh at dispatch — a new
  arrival from a different concurrent session, since `coder`'s inbox was fully drained at U49
  (`0059ec5`). Current-shape (`(:Agent {agentId:'coder'})-[:PRODUCED]->`). Paged past the
  `cypher` tool's 300-char truncation via `substring(k.fact/k.evidence, n, 240)` cross-checked
  against `size()` (336-char fact, 422-char evidence) before judging it.
- **The fact:** when another agent concurrently modifies shared files in the same working tree, a
  full-suite failure can be pre-existing and unrelated to your own diff — confirm by checking
  whether the failing fixture/assertion touches code your diff never modified, before assuming
  your change broke it. Evidence: model-bench Step 0 unit (U115), 11 `test_report.py` failures
  traced via `git --no-pager diff` to a concurrent sibling unit (U114); the offending fixture was
  untouched by the reporting unit's own diff, confirming the failures were pre-existing/concurrent
  rather than self-caused.
- **Checked for prior coverage before promoting** (this pass's own recurring failure mode is a
  stale/duplicate promotion): `claude/teco/teco.md`'s U48-promoted shared-tree fact
  (`f3a1c9e2`, "shared git tree wiping a delegate's edits") is a **different mechanism** — data
  loss/reversion detected by the *coordinator* integrating results — not test-failure
  misattribution diagnosed by the *implementer* mid-run; not a duplicate. `coder/kaizen/history.md`'s
  own 2026-08-?? entry (`c1e8f4b2`'s neighbor, promoted into `coder.md` step 5: report the
  *attributed* delta when a suite baseline can move under you) is the closest sibling — same step,
  same "shared tree" theme, but a distinct discipline: that clause is about honest **reporting** of
  counts, this fact is about **diagnosing** a failure's cause before reporting anything. Grepped
  `skills/python-web-quirks/SKILL.md`, `claude/tdd-engineer/guard-testing-techniques.md`,
  `claude/analyst/review-techniques.md`, and `claude/AGENTS.md` for "pre-existing"/"full-suite"/
  "sibling unit"/"concurrent" — no existing statement of this specific diagnostic technique found.
- **Routing.** `suggestedHome` said `prompt`; agreed. Ruled out `skills/python-web-quirks/SKILL.md`
  — the fact is a general shared-working-tree diagnostic discipline, not a Python/pytest-library
  quirk (the skill's scope is Python's own runtime/library behavior, not git-coordination
  process). Ruled out `teco.md`/`claude/AGENTS.md` — those are the *coordinator's* integration-time
  concerns (has a delegate's work been wiped/reverted); this fact fires at the *implementer's* own
  verification step, before any report reaches a coordinator. Folded into `claude/coder/coder.md`
  step 5 ("Verify and report"), directly after the existing attributed-delta sentence it
  complements — same bullet, same shared-tree theme, no new section.
- **Files touched:** `claude/coder/coder.md` (+50 words: 1,269→1,319, one sentence added to the
  existing step 5 bullet, no new sections) · this file.
- **Graph state after:** `coder` 0 produced / 0 mentioned (drained again). `tico` orphan
  (`e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e`) untouched. `model-bench/` untouched (no test/backup/doc
  file there was read for write purposes, only the entry's own evidence text).

## 2026-09-10 — `kaizen_team` distillation pass 2, unit U49 (`coder` inbox, 4 raw entries): 4 promoted, 0 discarded, 0 kept open — all 4 cleared, **none of them into a `coder` artifact**

- **What:** `cobb` ran `agent-maintenance` §5 over the four `coder`-produced `kaizen_team` entries
  dated 2026-09-09 — unit U49. All four were current-shape
  (`(:Agent {agentId:'coder'})-[:PRODUCED]->`); re-queried fresh at dispatch, confirmed unchanged
  from the brief.
- **Every cell was paged past the 300-char truncation** via `substring(k.fact/evidence/context, n,
  240)` projections, cross-checked against `size()`, before judging any of the four.
- **All four re-derived by actual execution, not just re-checked against their own citation** —
  no fact was narrowed, widened, or found false:
  - `7e2c1a4e` (`typing.Protocol`/`vars(cls)`): built a plain and a `@runtime_checkable` two-method
    Protocol on CPython 3.12.3; `{n for n in vars(cls) if not n.startswith("_") and
    callable(getattr(cls, n, None))}` returned exactly `{'foo', 'bar'}` in both cases. Confirmed
    as stated.
  - `e3f1c2a4` (`http.client.IncompleteRead` not an `OSError` subclass): `issubclass(IncompleteRead,
    OSError)` → `False`, MRO `(IncompleteRead, HTTPException, Exception, BaseException)`; an
    `except OSError` around a raised `IncompleteRead` does not catch it, confirmed by raising it
    inside a live `try/except OSError`. Confirmed as stated.
  - `312f05b6` (`HTTPError.read()` / `json.loads` NaN): built a real
    `urllib.error.HTTPError(url, code, msg, hdrs, fp)` with a `fp.read()` that raises
    `ConnectionResetError` — propagated out of `exc.read()` uncaught by any guard around
    `urlopen()` alone. Separately, `json.loads("NaN"/"Infinity"/"-Infinity")` returned
    `nan`/`inf`/`-inf` with no error, and `json.dumps({"x": float("nan")})` round-tripped to
    `'{"x": NaN}'`. Confirmed as stated; the two halves are unrelated topics (urllib body-read vs.
    JSON numeric coercion) and were routed separately.
  - `c1e8f4b2` (CLI guard mutation + `input()` under pytest capture): reproduced live with a
    scratch pytest file (model-bench's own venv's pytest 9.1.1, run outside `model-bench/` to
    respect the concurrent session's claim on that tree) — a function whose only body is
    `input(...)` fails under default capture with `OSError: pytest: reading from stdin while
    output is captured!`, traceback rooted in `_pytest.capture.DontReadFromInput.read`. Confirmed
    as stated; the fact is a general pytest mechanism, not model-bench- or CLI-specific.
- **Routing.** `suggestedHome` said `knowledge base` on three, `unsure` on one; all four landed in
  `skills/python-web-quirks/SKILL.md` — the shared cross-agent KB already carrying general
  Python/pytest facts beyond its "web" name (circular imports, `.pyc` caching, `pytest.raises`),
  confirmed by grep to hold no prior coverage of any of the four topics before this unit. Two
  became a new section each (`typing.Protocol`/`vars(cls)`; `json.loads` NaN/Infinity); one folded
  into the existing urllib taxonomy section as a second, read-phase paragraph (same mechanism,
  next phase, rather than a new section) — this merged `e3f1c2a4` and `312f05b6`'s urllib half;
  one became a new section (pytest `input()`/capture). The CLI-guard entry was explicitly
  considered for `claude/tdd-engineer/guard-testing-techniques.md` (mutation-testing overlap) and
  `claude/qa-engineer/qa-testing-techniques.md` (testing-mechanics overlap) and routed past both:
  the tdd-engineer KB is scoped to static-analysis *text* guards (AST/lint/grep readers), not a
  runtime CLI-argument validator; the qa-engineer KB is scoped to QA's own black-box tooling
  mechanics, not an implementer's mutation-test kill-signal recognition. `coder` still has no
  knowledge base of its own and this unit did not create one — no entry needed it.
- **Files touched:** `skills/python-web-quirks/SKILL.md` (+841 words: 3 new sections, 1 section
  extended; frontmatter `description` updated with all four topics) · `skills/README.md` (catalog
  row synced to match) · this file.
- **Graph state after:** `coder` 0 produced / 0 mentioned (drained). No other producer's count
  touched by this unit.

## 2026-09-09 — `kaizen_team` distillation pass 2, unit U39 (`coder` top-up, 9 raw entries): 6 promoted (2 of them halves), 3 discarded, 0 kept open — all 9 cleared, **none of them into a `coder` artifact**

- **What:** `cobb` ran `agent-maintenance` §5 over the nine `coder`-produced `kaizen_team` entries
  dated **2026-09-07 … 2026-09-09** — unit U39 of
  `claude/docs/plans/kaizen-distillation2-coordination.md`, a top-up after U10/U11/U12 closed
  `coder` out at 0 on 2026-09-07. All nine were current-shape
  (`(:Agent {agentId:'coder'})-[:PRODUCED]->`); no legacy `author`-property entry exists anywhere in
  the graph, so §5's legacy read was skipped.
- **Every cell was paged before dispositioning**, per the step-1 recipe added after U38 found the
  `cypher` tool truncates at `CYPHER_MCP_MAX_CELL` (300). Four of the nine `fact` cells and five of
  the nine `evidence` cells exceed 300 chars; the longest `fact` is 620 (`c7f1a3d2`) and the longest
  `evidence` 577 (`4e9b1c07`). Read via `substring(k.fact, 0|240|480, 240)` projections checked
  against `size()`. Two dispositions turn on the tail: `c7f1a3d2`'s enumeration of the three
  pre-`put` raise sites lives past char 480, and `12c5ab37`'s per-record-kind sentinel clause past
  char 480.
- **`suggestedHome` was again not a routing signal** (U38's follow-up, reconfirmed). Three entries
  said `project docs` and **none** landed in project docs; the two `prompt` entries split, one to a
  prompt and one to a knowledge base. Every home was decided from the receiving artifact's own scope
  line.
- **`coder` has no knowledge base, and this unit did not create one.** The scope fork the brief
  flagged never triggered: all six promotions had an existing home — `skills/python-web-quirks/SKILL.md`
  (2), `claude/tdd-engineer/guard-testing-techniques.md` (2), `claude/tdd-engineer/tdd-engineer.md` (2)
  and `claude/analyst/review-techniques.md` (1, counting the `b2d6f480`/`c7f1a3d2` halves once each).
  A first KB for `coder` would have been a file, a `README.md` row, a `claude/AGENTS.md` clause and a
  permanent maintenance surface bought to hold nothing.
- **No `falkor-chat` file was touched.** Three entries were falkor-chat-specific; each split into a
  live constraint on that system (already published there) and a portable technique (promoted into an
  agent artifact). `falkor-chat/AGENTS.md` and `falkor-chat/docs/` are unchanged.

### The nine dispositions

1. `c7f1a3d2-5b84-4e19-9a0c-2f6d8b31e740` (09-08, `knowledge base`) — **discarded, less one clause.**
   `ThreadPoolExecutor.submit` enqueues the work item at `thread.py:178` before `_adjust_thread_count()`
   at `:179` may fail. Already published in full: `skills/python-web-quirks/SKILL.md`
   § *"Holding an application lock across `ThreadPoolExecutor.submit()`…"* — read whole, the section
   plus its second-ordering-fact paragraph — carries the same two line numbers, the same
   `threading.Thread.start` reproduction, the same-exception-type trap and the same
   discriminate-on-the-message consequence; the skill's own frozen description states it too. The one
   thing the section lacked was the entry's enumeration of the raises *ahead* of the `put`: it named
   only `:169-173`. Added as a clause — the third is `BrokenThreadPool` at `:167`, which subclasses
   `RuntimeError` (`BrokenThreadPool → BrokenExecutor → RuntimeError`, checked on 3.12.3) and so is
   the one pre-`put` raise that type-discrimination *can* separate.
2. `906d4b08-6c19-47ec-8ef3-6f31ea0c266e` (09-08, `knowledge base`) — **PROMOTED** into that same
   section of `skills/python-web-quirks/SKILL.md`. The surviving half of the pair: `_work_queue.qsize()`
   cannot prove *"nothing was submitted"*. **Merged with entry 1 rather than sectioned separately,
   on U36's criterion** — this is not the same topic, it is the same *mechanism*: `:179` starting a
   worker inside the same `submit()` call is simultaneously why an exhaustion refusal leaves the job
   running and why the queue is empty when the test looks. Re-derived, not confirmed: five arms on
   CPython 3.12.3, `max_workers` 2–4 — fresh/no submit `qsize 0, threads 0`; after one submit
   `0, 1`; after the job finished `0, 1`; after `shutdown(wait=True)` **`1`**`, 1`. **Two bounds the
   entry did not state, both promoted with it:** `_threads` grows only when no **idle** worker can be
   reused (`_idle_semaphore.acquire(timeout=0)`, `:185`), so two sequential submits leave it at 1 and
   only overlapping ones take it to 2 — it answers *"has this executor ever run work?"*, not *"did
   this call submit?"*; and `qsize()` is not stably 0 either, because `shutdown()` puts a `None`
   wake-up sentinel on the queue.
3. `307487c5-f4f6-4a3e-8dad-58a85c587bf2` (09-08, `project docs`) — **PROMOTED** into
   `skills/python-web-quirks/SKILL.md` as a new section, *"Anything the lifespan puts on `app.state`
   does not exist until the `with TestClient(app)` block is entered"*. Routed to the skill rather
   than to `falkor-chat/docs/`: the falkor-chat half (`app.state.storefront` is set in `_lifespan`,
   `app.py:382`, not in `create_app`'s body) is a fact any reader of that function has, while the
   portable half is a FastAPI/Starlette lifecycle trap the skill's three existing `TestClient`
   sections do not cover. Reproduced on a minimal app at the pinned venv (starlette **1.3.1**,
   fastapi **0.139.0**): `app.state.thing` before the block raises `AttributeError: 'State' object
   has no attribute 'thing'` from `starlette/datastructures.py:686` — the entry's exact cited line —
   and the identical object is there inside it. The entry's second half was kept and sharpened into
   a three-way rule: whether a patch reaches the route depends on the router's closure shape, and
   `build_storefront_router` closing over the **object** (`storefront_api.py:878`,
   `services = shop._services`) and resolving `.post_message` per call (`:1220`) is the patchable
   one; rebinding the owning attribute is not, and a router that binds the bound method at build
   time is not either.
4. `3f6c1a52-9d24-4b7e-8a10-5c2e77b4d901` (09-07, `project docs`) — **PROMOTED** into
   `claude/tdd-engineer/guard-testing-techniques.md`, § *"A hand-written 'which object is this'
   resolver has two axes"*. **Judged a distinct fact from the VALUE-axis escape list already there,
   and opposite in polarity** — that list (`x = self`, tuple unpacking, a conditional expression, a
   container round-trip, *"then a call argument"*) enumerates ways the reader is **blind**; this is
   a way its *frontier* over-approximates. Re-derived by execution rather than from the entry's test
   run: `{c.attr for c in ast.walk(tree) if isinstance(c, ast.Attribute) and ast.unparse(c.value) in
   {'self'}}` over `self._ex.submit(self._run_turn, 1)` returns `_ex` **and** `_run_turn`, and
   `_storefront_reach` (`tests/test_storefront_api.py:3237-3248`) expands its frontier with exactly
   that comprehension. Promoted as the general property with its two non-obvious consequences —
   work deferred to a thread pool is *inside* the guarded reach, and a method only referenced is
   walked anyway — not as the falkor-chat instance.
5. `b2d6f480-71ae-4c93-8e15-5a3f0d92c6e1` (09-08, `knowledge base`) — **half promoted, half
   discarded.** The headline (an allowlist keyed on class **names** loses all force for the second
   raise of an allowlisted name; site-qualify instead) is already published verbatim as the closing
   paragraph of `guard-testing-techniques.md` § *"When the docstring states SEMANTIC reach…"* — read
   whole — and is documented again in the guard's own source (`_raise_sites` docstring,
   `tests/test_storefront_api.py:3466-3500`, including the measured `get_state` counter-example).
   The **unpublished** half is the entry's second clause: assert the exemption by **equality**, not
   by subtraction. Promoted as a paragraph there, with the delivered shape read whole and cited
   (`set(STOREFRONT_RAISES_TODAY) - storefront_family == frozenset({"RuntimeError"})`, plus the
   list-of-sites pin `== ["enqueue_turn"]`).
6. `12c5ab37-b418-44b5-8aaa-3db5742009f6` (09-07, `knowledge base`) — **PROMOTED** into
   `claude/tdd-engineer/tdd-engineer.md`, folded onto the existing mutation bullet as one sentence.
   The general form is what shipped: an **identity** assertion is invariant under any change *both*
   halves share, so `from_dict(to_dict(x)) == x` cannot pin a serialization decision (omit-vs-null,
   key name, ordering) however many records it round-trips — assert the serialized form. That bullet
   already says *"a surviving mutant is not always a weak test"*; this is its mirror, where the
   survival **is** the test's fault. Verified at `e79fb61`: `fingerprint.py`'s `to_dict` omits the
   key for exactly one record kind (`:359-360`), `from_dict` uses a per-record-kind sentinel
   (`missing: str | None = None if arm_kind == "deterministic" else ""`), and
   `tests/test_fingerprint.py:398` carries the assertion that killed the mutant
   (`assert "callSurface" not in reference_arm.to_dict()`).
7. `4e9b1c07-3a52-4d68-b1f0-9c7d24ae5b83` (09-08, `prompt`) — **PROMOTED** into
   `claude/tdd-engineer/tdd-engineer.md` as a new Principles bullet: when a plan explicitly rejected
   an alternative, the mutant that validates the suite is *that alternative*, not the absence of the
   chosen mechanism. **Home ruled from artifact scope, not from the captor.** `coder.md` carries no
   mutation-testing doctrine at all (its only `mutat` hits are about mutating the *environment*), so
   landing a mutant-selection rule there would open a topic with no anchor; `teco.md:93` is a
   coordinator's *ask* (*"break the implementation deliberately"*) rather than a selection method,
   and the entry's own evidence shows the coordinator's ask was fine — the implementer's mutant was
   not. `tdd-engineer.md` is the one prompt that already owns "which mutant, and why".
8. `c3f7a1b2-9d64-4e58-8a30-71b0c2e4d9aa` (09-08, `project docs`) — **discarded: both premises are
   already published, in the two artifacts a reader consults, and the entry is one step from them.**
   Read whole: `Storefront.__init__`'s docstring (`falkor-chat/server/falkorchat/storefront.py:382-430`)
   already states that `services` is read once at construction for `_repo` (`:430`, with the
   reason) and that `trigger` is *"the turn worker's only collaborator … never through its own
   `self._services`"*; `falkor-chat/docs/SERVER.md`'s `FALKORCHAT_STOREFRONT_TURN_WORKERS` row
   (line 135, read whole) already carries the measurement this entry was captured while producing,
   figure for figure (*"five simultaneous arrivals, the fifth one's reported position:
   `turn_workers=1` → `3`, `=2` → `2`, `=4` → `0`"*); and `falkor-chat/AGENTS.md:108` already warns
   that a default `pytest` run wipes the shared `reference` graph, which is the hazard the entry's
   method avoids. What is unpublished is only the assembly of those three into *"so a stub plus a
   parking trigger drives it"* — a corollary, and not worth a line in a context file at its density
   bar.
9. `f3e2a1c4-7b6d-4e2a-9c1f-8a5d6e2b7c91` (09-09, `prompt`) — **PROMOTED** into
   `claude/analyst/review-techniques.md`, § *"A grep-pinned edit table is an edit list, not a
   completeness proof"*, by **rewriting numbered item 5 in place** rather than adding a seventh. Item
   5 was already the section's one false-*failure* case (a done-condition that must spell the retired
   token); this entry is the same mechanism one step more general — the residual counts **lines in
   files**, so a comment or docstring does it too. Verified by execution rather than by citation:
   `grep -c` returns **2** where `grep -o | wc -l` returns **3** on a file with the token once on one
   line and twice on the next; and at `e79fb61`,
   `git grep -cF 'SUPPORT_DIFF_PROPORTIONS[0]' -- model-bench/` reads `stats.py:1` beside
   `tests/test_stats.py:2`, the repaired state the entry describes.

- **The K-026 trigger shape was checked and does not fire.** The brief flagged that landing here
  would make three consecutive units extending one section of `review-techniques.md`. It does not:
  this is an **in-place rewrite of an existing numbered item**, not an added bullet, and the section
  is unchanged in length by one line. No parent heading or split is proposed. One pre-existing
  inaccuracy noted and deliberately left: that section opens *"Six ways the residual **passes** on
  an incomplete edit"*, and item 5 was already a false-failure case before this edit — a one-word
  header fix belonging to whoever next revises the section, not a side effect of this unit.
- **Plan items:** none opened. `coder/kaizen/plan.md`'s parking-lot item *"A mutant must be proven
  to change behavior before its survival is read as a coverage gap"* was annotated, not closed —
  the general rule it wanted stated *is* already in `tdd-engineer.md`'s mutation bullet, which this
  unit extended twice; only its `re.fullmatch` instance is still unhomed.

## 2026-09-07 — `kaizen_team` distillation pass 2, unit U12 (chunk C of three, closing `coder` out): 7 raw entries processed — 5 promoted into `skills/python-web-quirks/SKILL.md` (2 folded, 3 merged into one new section, 1 new section), 2 kept open as K-006 rows — all 7 cleared

- **What:** `cobb` ran `agent-maintenance` §5 over the seven `coder`-produced `kaizen_team` entries
  dated **2026-09-03** — unit U12 of `claude/docs/plans/kaizen-distillation2-coordination.md`, the
  last of `coder`'s three chunks. All seven were current-shape
  (`(:Agent {agentId:'coder'})-[:PRODUCED]->`); zero legacy `author`-property entries remain
  anywhere, so §5's legacy read was skipped. With this unit `coder`'s `kaizen_team` capture is
  empty.
- **Versions confirmed before judging any version-stamped claim** (all probes run from the pinned
  `falkor-chat/server/.venv`): fastapi **0.139.0**, starlette **1.3.1**, redis **8.0.1**, pytest
  **9.1.1**, CPython **3.12.3**.
- **Every entry re-derived, never confirmed from its own citation** — five by executing a probe
  script against the pinned venv, two by reading the scripts and fixtures themselves. Three raw
  claims gained a consequence the entry did not state, and one narrowed. Nothing was verified by
  running the falkor-chat suite: `62bc71d6` is *about* a fixture that wipes the shared `reference`
  graph, so it was established by reading `conftest.py`, `seed_catalog.sh` and `verify_catalog.sh`,
  never by executing them.

  1. **`62bc71d6-d444-4592-956e-88e97133b4db` (the `wf_repo` fixture wipes `reference` on setup
     only, so fixture `Product` rows outlive the run) — KEPT OPEN, folded into K-006 as row 5.**
     Whole chain verified in source without running anything: `tests/conftest.py:101-110`'s
     `wf_repo` runs `MATCH (n) DETACH DELETE n` on `reference` and returns — no `yield`, so no
     teardown; `scripts/seed_catalog.sh:124` is `MERGE (p:Product {productId: row.productId}) ON
     CREATE SET …`, which cannot remove a row it did not seed; `scripts/verify_catalog.sh:33` pins
     `EXPECTED_COUNT=15` exactly, so any survivor fails it. **Not a discard**, and the reason is
     the U10 `9050f193…` shape: the fact is already written down at the point of use — and, better
     than that, already *solved* in code, by `tests/test_storefront.py:839-851`'s `catalog_repo`
     yield-fixture, whose docstring states the entire chain including the `verify_catalog.sh`
     mismatch. But `docs/SERVER.md` §1.7's first bullet — which documents this exact fixture and
     this exact setup-only wipe — stops at the consequence for workflow *defs* and prescribes a
     remedy (`seed_workflows.sh`) that does not touch the catalog. The next author of a
     `reference`-touching test reads §1.7, not another test file's fixture docstring.
  2. **`16ab10b7-fc1e-4635-b29c-1d7a51fb2eb1` (`test_queries.sh` does not execute the code under
     test) — KEPT OPEN, folded into K-006 as row 6.** The absolute claim survives re-derivation
     intact, which is not what the brief's prior turned out to be: `grep -n 'python\|pytest\|
     falkorchat\|\.venv' scripts/test_queries.sh` returns **zero** hits, and the script holds 27
     shell query constants driven through `redis-cli` — it genuinely never reaches `repository.py`.
     The entry's *own instance* has since been repaired (the `FILTER=` constant at `:1385` and the
     abstention header at `:1375` both now carry `productId`), so the finding is spent as a defect
     and durable as a property of the gate. Already documented, generally and well, at
     `docs/QUERIES.md` §15.1 — transcribed 2026-09-03, the same day this entry was captured, and
     ending on the lesson verbatim ("a transcription gate goes green on a wrong transcription, so
     the doc block and the shell constant have to be checked against the code, never against each
     other"). Kept open anyway, narrowly, for the same reason as entry 1 and against a live
     counter-claim: `AGENTS.md:200` tells every agent that `test_queries.sh` "must pass before any
     schema or **query** change is committed", which is precisely the reading the §15.1 note
     exists to defeat, and §15.1 is a per-query lookup nobody consults on the way to trusting a
     green run.
  3. **`3f0c9b52-6a41-4d8e-9d17-2b8a5c0e77a1` (redis-py's `TimeoutError` and `ConnectionError` are
     siblings) — PROMOTED to `skills/python-web-quirks/SKILL.md`,** as the fourth trap of the new
     exception-handler section (entry 4 below). MROs printed on redis 8.0.1: both are
     `(…, RedisError, Exception, …)`, `issubclass` **False in both directions**, and
     `redis.exceptions.ConnectionError is ConnectionError` → `False`;
     `falkorchat.db.FalkorDBUnreachableError` is `(…, builtins.ConnectionError, builtins.OSError,
     …)`, i.e. related to neither. The shipped code agrees — `storefront_api.py:717-721` registers
     all three separately — but *why* three are needed is nowhere in prose. **This is a considered
     scope call, not an oversight of the brief's warning**: redis-py is not a web framework, and
     U9's ruff-config and U10's `re.fullmatch` facts were both declined for this file. What
     separates this one is that the skill already carries its exact sibling — the `urllib`
     `HTTPError`/`URLError`/`TimeoutError` taxonomy entry, same genre ("your except clauses are not
     total, here is the real hierarchy") — and that the consequence lands in a **FastAPI handler
     map**, squarely in scope. Promoted with the version-sensitivity stated, since older redis-py
     majors *did* nest the two and a reader's memory is the wrong thing to trust. The
     `FalkorDBUnreachableError` half is used as the illustrating instance and opened no K-006 row:
     `db.py:18`'s class statement is self-documenting.
  4. **`5e3c1f42-9a77-4b6e-8c21-7d0f4a9b6e58` + `b71d8a06-3c54-4f19-9ad2-2e6c8f5b1a93` +
     `8d2e4a17…`'s second half (the exception-MRO clause) — PROMOTED as ONE merged section,
     "FastAPI/Starlette's exception-handler registry", with a consequence none of the three
     stated.** They are one fact with three faces, so a merge rather than three near-duplicates
     (the same call U8 and U9 made on their own pairs). All re-derived by probe: a bare `FastAPI()`
     carries exactly three handler keys and `fastapi.HTTPException in app.exception_handlers` is
     `False` while the starlette one is `True`; `add_exception_handler` on a registered type leaves
     the key count at 1 and the incumbent readable immediately before the overwrite (delegation
     verified end-to-end — `/shop/boom` → the wrapper, `/legacy/boom` → the captured original);
     `starlette._exception_handler._lookup_exception_handler` read at source is
     `for cls in type(exc).__mro__: if cls in exc_handlers`, so **registration order is irrelevant
     and specificity is the exception hierarchy alone**, which the entries implied but did not say.
     **The added consequence, and the reason the identity trap is worth more than a test-writing
     note:** `add_exception_handler(fastapi.HTTPException, …)` does not replace the default at all —
     it adds a *fourth* key, after which your handler wins for `fastapi.HTTPException` and its
     subclasses while the bare starlette `HTTPException` that Starlette itself raises for routing
     404s/405s silently keeps the untouched `{detail}` shape. Probed both routes; that is a live
     hole in an error envelope, not just a mis-keyed assertion.
  5. **`8d2e4a17-05b9-4c63-8f2a-71c6d9b3e004`'s first half (`responses={…}` stores unknown keys
     verbatim) — PROMOTED by FOLDING into the existing "`responses={...}` is keyed by status code
     only" section, not stacked beside it.** The brief's instruction, and independently the right
     shape: the existing section (added at U7b) closes on "There is no declaration-side key for
     anything inside the body: not an error token, not the offending `field`" — true about the
     *key* set and now qualified where it needs to be, because the per-status *value* is not
     validated. Probed: `responses={409: {"description": …, "x-storefront-error": [...]}}` on a
     router route reached through `include_router(prefix="/shop/api")` reads back verbatim off
     `route.responses` **and** appears intact in `app.openapi()["paths"][…]["responses"]["409"]`;
     the `"4XX"` range form carries extensions identically. Filed as the escape hatch the existing
     section's "structurally cannot see a finer axis" paragraph otherwise leaves the reader
     without, with the two caveats the raw entry lacked (nothing validates the extension, so a
     typo is just another passthrough; a non-serialisable value fails at `app.openapi()` time, not
     at import).
  6. **`607b0a2a-4fe0-47fa-ba88-81cb22abb6dc` (a whole-module-only test must gate on what was
     collected) — PROMOTED to `skills/python-web-quirks/SKILL.md`, new section beside the other
     pytest traps.** Checked against both prior candidates the brief named and it is neither:
     U8's `pytest.raises` section is about assertion *strength*, and `tdd-engineer.md`'s "prove a
     new assertion against the mutant" bullet is about rejecting a wrong implementation — this is
     about a test detecting that its own precondition held, the same family as the file's
     env-var-frozen-at-import entry ("your guard is a silent no-op"). Re-derived on pytest 9.1.1
     with a throwaway module outside the repo, all four selectors measured:
     `config.option.keyword` is `'probe'` under `-k` and **`''`** under a node id, under `--lf`,
     and under `--deselect`, while `request.session.items` is correct in all four. `--lf` was
     measured in the load-bearing direction (the gating test itself failing, so `--lf` re-selects
     it): `KEYWORD=''`, `COLLECTED=['test_probe']` — the guard is off exactly when a developer
     re-runs after the failure. Promoted with the measurement as a four-row table, plus the two
     implementation traps that make the collected-set form work: `originalname` (without it every
     parametrized test reads as uncollected and the guard skips always) and subtracting
     marker-deselected tests from `defined` — the second lifted from the shipped test's own
     docstring (`tests/test_storefront_api.py:4401-4410`, P12-3), which had thought further about
     this than the raw entry did.
- **`MENTIONS` tags added: none.** No entry was substantively about a different agent; every
  promotion is a general Python/framework fact that `cobb` re-derived first-hand, so all five were
  dispositioned directly rather than tagged and deferred.
- **Cleared from `kaizen_team` this pass — all seven, each a full `DETACH DELETE`:** every entry
  counted first (`producedEdges + mentionEdges`); every one read `1 + 0`, so `otherRemaining == 0`
  in all seven cases and the whole node was removed rather than just the `PRODUCED` edge. Ids:
  `62bc71d6-d444…`, `16ab10b7-fc1e…`, `3f0c9b52-6a41…`, `8d2e4a17-05b9…`, `5e3c1f42-9a77…`,
  `b71d8a06-3c54…`, `607b0a2a-4fe0…`.
  **Id collision note:** `3f0c9b52-6a41-…` is close to ids cleared in other units of this pass;
  these ids are hand-shaped, not `uuid4`, so full id, date **and** subject were confirmed on every
  one of the seven before it was touched.
- **Docs touched:** `claude/coder/kaizen/{plan,history}.md` (this file + K-006 rows 5–6) ·
  `skills/python-web-quirks/SKILL.md` (one fold, one merged section, one new section, frontmatter
  `description`) · `skills/README.md` (its catalog row, both the what and the when-to-use columns).
- **Plan items:** **K-006 extended** from four rows to six; no item opened or closed. K-002
  untouched. The parking-lot mutation-testing lesson U11 left pointing at
  `claude/analyst/review-techniques.md` was **not** moved — out of this unit's scope, and it is
  still correctly parked.

## 2026-09-07 — `kaizen_team` distillation pass 2, unit U11 (chunk B of three): 8 raw entries processed — 4 promoted (2 to `graph-dba`, 1 to `data-scientist`, 2 sections to `analyst`), 4 discarded as already documented — all 8 cleared

- **What:** `cobb` ran `agent-maintenance` §5 over the eight `coder`-produced `kaizen_team` entries
  dated **2026-08-31 through 2026-09-02** — unit U11 of
  `claude/docs/plans/kaizen-distillation2-coordination.md`. All eight were current-shape
  (`(:Agent {agentId:'coder'})-[:PRODUCED]->`); zero legacy `author`-property entries remain
  anywhere, so §5's legacy read was skipped. `coder`'s **seven** entries dated 2026-09-03 are
  chunk C, a separate unit, and were **not touched** — read only, never counted, never cleared.
- **Build confirmed before judging any version-stamped claim:** `redis-cli MODULE LIST` → graph
  module `ver 41811` — still **v4.18.11**.
- **Every entry re-derived, never confirmed from its own citation.** Four turned out to be already
  published — three of those **at the point of use** (a script's header comment, a directory
  README's own rule) rather than anywhere a docs-tree grep for the entry's own wording reaches.
  One raw entry's stated mechanism was materially wrong and was corrected in promotion.

  1. **`c1a2b3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d` (08-31 — more targeted wording guidance is not
     monotonically safer) — PROMOTED to `claude/data-scientist/lm-studio-model-notes.md`, new
     section.** Re-derived from `falkor-chat/docs/HISTORY.md` (2026-08-31, K-057) and
     `docs/reviews/salesperson-tool-reliability-ml.md` §11/§14 — both carry the numbers
     (iteration 1 shipped at 16/20 = 80% net correct; the reverted iteration 2 fell to 14/20 = 70%,
     the targeted category-omission defect unimproved at 30% vs. 20%) *and* the mechanism
     (`HISTORY.md`: "while also suppressing a multi-call self-correction pattern that had rescued
     several replies under the shipped wording"; §14.4 measures that self-correction rescuing ~2/3
     of the reps it fires on). Not a discard, because both homes are per-item lookup documents — a
     milestone history entry and one section of a K-item review — while the *generalized* rule
     (score **net** correctness on every wording iteration, not just the targeted defect rate; an
     observed self-correction is part of the baseline you can lose) is what a `data-scientist`
     designing the next prompt-wording eval needs and would not find there. **No model was loaded
     on the shared LM Studio server to verify** — per U7b's precedent, the documents were enough.
  2. **`7c1f4e2a-9b83-4d15-a6c0-2e8f5d31b904` (09-02 — `git mv <dir>` moves a gitignored `.venv/`,
     whose shebangs keep the old absolute path) — DISCARDED.** The live half verifies exactly as
     written: `deprecated/salesperson/.venv/bin/streamlit` still begins with the pre-move
     `…/salesperson/.venv/bin/python3` shebang, and `pyvenv.cfg`'s `command =` records the pre-move
     path too. But the base fact — a virtualenv is not relocatable — is documented CPython
     behaviour, the same bar that discarded `a1b2c3d4…`'s `re.fullmatch` fact at U10; and
     `git mv` moving untracked-but-present contents follows directly from it being a filesystem
     rename. The one project-specific consequence (the moved README's `./.venv/bin/python …`
     instruction no longer works) is already covered by `deprecated/README.md`'s own first rule:
     "**Nothing here is expected to still run**; a retired component's own docs describe it as it
     was when it was retired." Nothing to add, and no `K-` item: a retired component's broken venv
     is precisely what that rule says will not be fixed.
  3. **`7c1e9d24-3b06-4a58-9f21-2ad4b6e0c913` (09-02 — a `WorkflowDefSnapshot` materialized into a
     workspace is create-only per (key,version)) — DISCARDED, already documented on both sides.**
     Mechanism verified in source: `Repository._PUBLISH_CYPHER` sets every property under
     `ON CREATE SET` (`d.name`/`d.kind`, `st.config`, `rel.guard`), so a differing resubmit writes
     nothing. `falkor-chat/docs/DESIGN.md` (§ the materialize decision, ~:170) already states it —
     "Properties (`name`, `kind`, step `config`, transition `guard`) are create-only — a differing
     resubmit of those stays a silent no-op, unchanged" — and `scripts/verify_salesperson.sh`'s
     header (check 6) already carries the operational corollary the entry actually contributes,
     verbatim: "`config` is create-only, so re-seeding cannot repair it; the failure text says to
     bump the version, which is the only fix." Not merely documented — **guarded**: check 6 is a
     shipped assertion. **No K-006 row was added for this one**, deliberately, against the routing
     brief's default: it is not an open doc ask.
  4. **`b4f0a7e1-95c2-4d3a-8e77-1c60f2ab5d38` (09-02 — graph content can originate from an
     UNCOMMITTED working tree) — SPLIT: general half PROMOTED to
     `claude/analyst/review-techniques.md`; falkor-chat corollary DISCARDED as already fixed and
     documented.** The corollary (`services.diff_def_snapshot` compares `reference` against the
     workspace snapshot, i.e. two *derived* artifacts against each other, so both can agree and
     both be stale) is now shipped as `verify_salesperson.sh` check 6 and stated in that script's
     own header — "Check 3 compares the two sides against EACH OTHER, so it is blind to a version
     whose config was stored from an earlier or uncommitted working tree". The **general** half is
     in no agent-facing place: `git log --all -S` structurally cannot establish the provenance of
     anything stored in a database, because the code that wrote it may never have been committed —
     a nil result is not evidence. Filed in `review-techniques.md` beside its own family ("An
     untracked plan/review doc has no re-verification baseline"), with the second, structural
     lesson attached: whenever a gate reports "in sync", ask **in sync with what**, because two
     derived sides cannot detect common-mode staleness against their source.
  5. **`86af3475-c64a-4137-be2a-256402f4ca76` (09-02 — repository Cypher in class-level constants
     can be mutated without editing source) — PROMOTED to `claude/analyst/review-techniques.md`.**
     Precondition verified in source, which is the whole technique: `repository.py` declares
     `_RESET_PARTICIPANT_CYPHER` (`:3214`), `_PUBLISH_CYPHER` (`:1732`),
     `_ENSURE_PARTICIPANT_CYPHER` and siblings as **class attributes**, and every call site
     dereferences `self._…_CYPHER` (`:3661`, `:1884`, `:2604`, `:3422`, `:3712`), so a plugin
     rebinding the attribute at import time takes effect for the whole session. Written up with the
     precondition foregrounded — a constant captured at import into a module-level name, a default
     argument, or a local will not respond to the rebind. Homed in `analyst`'s knowledge base
     rather than left parked (U10's judgement, when the only candidate homes were
     `skills/python-web-quirks/` and `claude/qa-engineer/qa-testing-techniques.md`, both out of
     scope): `review-techniques.md`'s stated scope is verification technique and it already carries
     "Verifying an uncommitted diff without mutating the working tree" — the same family, same
     hazard, no scope stretch. `plan.md`'s parking-lot note updated to point there.
  6. **`f3b1c2d4-6a58-4e19-9c07-2b8ad4e51f76` (09-02 — an alias bound by
     `head(collect(DISTINCT <node>))`) — PROMOTED to `claude/graph-dba/falkordb-quirks.md`, with
     the entry's stated mechanism CORRECTED in both halves.** Four read-only probes on the live
     `reference` graph, module `41811`. The raw entry said the alias is usable as a value but
     "cannot be re-bound as a pattern node" and that "`head()` over the list is what degrades
     them". Both are narrower than claimed: `MATCH (x) WHERE x.name IS NOT NULL RETURN x.name`
     re-binds the alias fine (one row, the same node — not a re-scan), so a **bare** node pattern
     is not the trigger; only a **relationship** pattern raises (`MATCH (x)-[r]-(y)` and
     `OPTIONAL MATCH (x)-[:REL]->(c)` both → `encountered unexpected type in Record; expected
     Node`, raised whether or not such an edge exists). And it is not `head()`:
     `collect(DISTINCT p)[0]` fails identically, so the trigger is **index extraction from the
     collected list**. `WITH collect(…) AS xs UNWIND xs AS x` keeps the node fully usable —
     measured, and now the entry's stated workaround. This is the sixth time in this pass that
     re-derivation changed a claim rather than confirming it.
  7. **`b1f0c3ae-6a7e-4d2c-9f31-2c7a4d5e8b90` (09-02 — FalkorDB dynamic property access by variable
     key) — PROMOTED to `claude/graph-dba/falkordb-quirks.md`.** Re-derived with both controls on
     module `41811`: `MATCH (n) WHERE any(k IN keys(n) WHERE n[k] = 'Wireless Mouse Pro') RETURN
     labels(n), count(n)` → the one real match; the identical query for a value present on no node
     → `0`, so it is not vacuously true either way. Promoted with the paired-control requirement
     stated as part of the rule, since without it the "stored nowhere" assertion the entry exists
     to enable degrades to a silent pass.
  8. **`8f3a1c62-5d47-4b91-9a2e-0c7b6e41d5aa` (09-02 — this monorepo has no root-level pytest
     configuration) — DISCARDED, already promoted.** Confirmed still true (`ls pyproject.toml
     pytest.ini setup.cfg tox.ini` at the repo root → all four missing), and confirmed already
     published: the root `AGENTS.md` opening paragraph, rewritten at unit U7 of this same pass,
     already carries every clause this entry has — no root pytest config, run a component's suite
     **with that component as the working directory**, because from the repo root pytest sets
     `rootdir` to the monorepo, ignores the component's own `testpaths`, and walks into other
     components' tests. Nothing new to fold; a second home would only drift.
- **`MENTIONS` tags added: none.** Every entry that belonged to another agent's knowledge base was
  re-derived by `cobb` first-hand and dispositioned into that file directly, so none needed to be
  deferred to a future pass. (The 2026-08-25 pass tagged three entries precisely because confirming
  them needed write-capable probes; that did not apply to any of these eight — all four promotions
  were verified read-only or from source.)
- **Cleared from `kaizen_team` this pass — all eight, each a full `DETACH DELETE`:** every entry was
  counted first (`producedEdges + mentionEdges`); every one read `1 + 0`, so `otherRemaining == 0`
  in all eight cases and the whole node was removed rather than just the `PRODUCED` edge. Ids:
  `c1a2b3d4-5e6f…`, `7c1f4e2a-9b83…`, `7c1e9d24-3b06…`, `b4f0a7e1-95c2…`, `86af3475-c64a…`,
  `f3b1c2d4-6a58…`, `b1f0c3ae-6a7e…`, `8f3a1c62-5d47…`.
  **Id collision note:** `c1a2b3d4-5e6f-…` is one character from the `a1b2c3d4-e5f6-…` entry
  cleared at U10, and `f3b1c2d4-6a58-…` is close to ids cleared in three other units. These ids are
  hand-shaped, not `uuid4`; full id, date and subject were confirmed on every one before touching
  it.
- **Docs touched:** `claude/coder/kaizen/{plan,history}.md` · `claude/graph-dba/falkordb-quirks.md`
  + `claude/graph-dba/kaizen/history.md` · `claude/data-scientist/lm-studio-model-notes.md` +
  `claude/data-scientist/kaizen/history.md` · `claude/analyst/review-techniques.md` +
  `claude/analyst/kaizen/history.md`.
- **Plan items:** none opened, none closed. **K-006 was deliberately not extended** — both
  falkor-chat entries in this chunk turned out already published (one of them additionally guarded
  by a shipped assertion), so neither is an open doc ask. K-002 untouched.

## 2026-09-07 — `kaizen_team` distillation pass 2, unit U10 (chunk A of three): 12 raw entries processed — 4 promoted to `graph-dba`'s knowledge base, 4 discarded as already documented, 4 kept open as K-006 — all 12 cleared

- **What:** `cobb` ran `agent-maintenance` §5 over the twelve `coder`-produced `kaizen_team`
  entries dated **2026-08-27 through 2026-08-29** — unit U10 of
  `claude/docs/plans/kaizen-distillation2-coordination.md`. All twelve were current-shape
  (`(:Agent {agentId:'coder'})-[:PRODUCED]->`); zero legacy `author`-property entries remain
  anywhere, so §5's legacy read was skipped. `coder`'s **fifteen** entries dated 2026-08-31 and
  later are chunks B and C, separate units, and were **not touched** by this pass.
- **Build confirmed before judging any version-stamped claim:** `redis-cli MODULE LIST` → graph
  module `ver 41811` — still **v4.18.11**, the build the FalkorDB entries were written against.
- **Every entry re-derived, never confirmed from its own citation.** Four turned out to be already
  published — twice **at the point of use** (a script comment, a method docstring) rather than
  anywhere a docs-tree grep reaches. Two raw claims were sharpened or narrowed in promotion.

  1. **`b3f2a1c4…` (`$param IS NULL OR` null-guard vs. a literal-sentinel `coalesce`) — PROMOTED,
     folded into `claude/graph-dba/falkordb-quirks.md`.** Re-derived with four live
     `GRAPH.EXPLAIN` runs against the existing `reference` graph (read-only, no probe graph
     created): the null-guard form plans `Node By Label Scan` even with `$category` bound to a real
     value; `p.price >= coalesce($minPrice,-1.0) AND p.price <= coalesce($maxPrice,1e9)` plans
     `Node By Index Scan` with **every** parameter `NULL`; a self-referential coalesce on an
     equality predicate plans a label scan, while plain `= $category` on the same indexed property
     is a clean index scan. This **corrects** the existing 2026-08-22 quirks entry, which closed
     with "no phrasing avoids that" full scan for the unfiltered call. Folded into that entry
     rather than stacked beside it, and given two caveats the raw entry lacked: the whole-range
     index scan buys no selectivity, and the sentinel bounds silently drop a `NULL`/non-numeric
     row that the `IS NULL OR` form would have returned (`NULL >= -1.0` → `NULL`, verified).
  2. **`a3f1c2d4…` (precomputed normalized property, not a runtime `toLower()`) — PROMOTED, engine
     half only, new *Query tuning* entry.** Re-derived: `toLower(p.categoryNormalized) = 'audio'`
     plans `Node By Label Scan` + `Filter`; the plain equality plans `Node By Index Scan` — there
     are no expression/functional indexes on this build. The raw entry's `falkor-chat`-specific
     half (whether swapping the residual filter's property was safe, given `Product.price` was the
     real anchor) was **left out**: already published in `falkor-chat/docs/QUERIES.md` §15.2 and
     verbatim in `Repository.filter_products`'s own docstring.
  3. **`f3c6021e…` (a step `config`/transition `guard` reads back as an opaque JSON string) —
     DISCARDED, already documented.** Verified true (`services._serialize_opaque`, `:275`), and
     documented twice: `falkor-chat/docs/DESIGN.md` states `Step.config` and `TRANSITION.guard` are
     **opaque serialized strings parsed app-side only (rule 8)**, and `_serialize_opaque`'s own
     docstring restates it at the point of use. The entry's practical corollary (`json.loads()`
     before comparing in a test) follows directly. Its `suggestedHome: 'prompt'` was rejected
     outright — a single component's serialization convention never belongs in an always-loaded
     agent prompt.
  4. **`2ee298a8…` (`filter_products` exact, case-sensitive category match) — DISCARDED,
     superseded by the fix.** The bug is gone: `Repository.filter_products` now normalizes the
     caller's `category` with `extraction.normalize_name` and compares against a precomputed,
     indexed `Product.categoryNormalized`. Confirmed live — `reference` carries a RANGE index on
     `categoryNormalized`. The incident, the fix and its sargability rationale are all in the
     method's docstring plus `docs/QUERIES.md` §15.2. Same-day sibling of entry 2 above, which is
     the fix's own capture.
  5. **`c1a9e6f0…` (`collect({map literal})` over a zero-row `OPTIONAL MATCH`) — PROMOTED, new
     *Cypher dialect* entry in `falkordb-quirks.md`, mechanism sharpened.** Re-derived read-only:
     on a node with no matching edge, `collect(l)` → `[]`, `collect(l.qty)` → `[]`, but
     `collect({q: l.qty})` → `[{q: null}]`, size **1**. The raw entry reported the symptom; the
     added contrast gives the actual rule — a map literal is a non-null value whose *fields* are
     null, so it survives a collect that drops bare nulls. Filed beside the `sum(CASE …)` → `0.0`
     and global-`collect()`-over-zero-rows entries it belongs with.
  6. **`c1f2a8b4…` (a step of ANY type parks iff `config.waitsForHuman`) — KEPT OPEN, K-006.**
     Verified by reading `executor._drive_loop` (`:634` — OUTCOME B tests
     `config.get("waitsForHuman")` and never `step.type`) and `services._validate_def_spec`
     (`:1446` — the requirement is applied only to `WAITING_STEP_TYPES`). The core is already
     documented at `services.py:85` ("the executor's OUTCOME B keys on exactly that flag") and
     demonstrated by a shipped `agent`-typed precedent. What is **not** documented is the
     validation asymmetry: a `decision`/`agent` step that needs to park but omits the flag
     publishes clean and fails only at runtime. That one clause belongs in `falkor-chat/docs/DESIGN.md`,
     outside `cobb`'s write remit.
  7. **`29b6274a…` (a Cypher-level regression is invisible to `test_services.py`) — KEPT OPEN,
     K-006.** Verified: `tests/test_services.py:606-615`'s `FakeRepo.upsert_profile` re-implements
     `repository.py` §17's `coalesce()`-per-field semantics in plain Python, and says so in its own
     comment — so reverting the real Cypher to an unconditional `SET` cannot redden it. Target is
     `falkor-chat/docs/SERVER.md` §1.7, whose existing four testing hazards are exactly this genre;
     `falkor-chat/AGENTS.md` already points there for suite-specific hazards.
  8. **`b7e1f0a2…` (`falkordb-py` names an un-aliased `RETURN` column by its expression text) —
     PROMOTED, new *Ops, config & tooling* entry in `falkordb-quirks.md`.** Re-probed live from
     `falkor-chat/server/.venv` (falkordb-py **1.6.1**): `RETURN p.name AS name, p.price AS price`
     → `header == [[1,'name'],[1,'price']]`; the same query with no `AS` →
     `[[1,'p.name'],[1,'count(p)']]`. `Repository.run_readonly_query`'s docstring already documents
     the consequence for *that* method; the client-library fact itself was in no knowledge base,
     and it binds anyone writing a generic row-to-dict mapper.
  9. **`9050f193…` (batch document POSTs race a single JIT-loading LM Studio) — KEPT OPEN,
     K-006, narrowly.** Verified as already documented **at the point of use**: a 16-line comment
     at `scripts/seed_nlq_eval_corpus.py:342-353` carries the whole fact — the two background jobs
     per chunk, the one-model-at-a-time swap, the `{"error":"Model is unloaded."}` HTTP 400, 11 of
     12 documents `failed`, and the post-then-poll-per-document fix. Not a discard, because that
     comment is scoped to one script while the constraint binds the *next* author of any live batch
     driver; the ask is one generalized line in `docs/SERVER.md` §1.7's QA gotchas half.
  10. **`a3f0e6c1…` (the shared `~/.config/opencode/opencode.json` points at an unreachable LAN
      address) — DISCARDED, already promoted.** `claude/qa-engineer/qa-testing-techniques.md`
      carries a fuller section on exactly this, promoted in unit U4 of this same pass and
      re-verified 2026-09-07 (`curl http://localhost:1234/v1/models` → 200, the configured LAN
      address → `curl` exit 7). It already states everything this entry does **plus** the rule this
      entry lacks — never edit the shared file; point `FALKORCHAT_OPENCODE_CONFIG` at a scratch
      copy for the pass only. Nothing to fold.
  11. **`55364b9a…` (a new `SALESPERSON_DEF` tool `KeyError`s the offline scaffold tests) — KEPT
      OPEN, K-006.** Verified: `tests/test_salesperson_scaffold.py` hand-mirrors the def's tool list
      in a module-level `_SCHEMAS` dict and `StubRegistry.schema()` is a bare `_SCHEMAS[name]`
      lookup (`:212`), so the coupling is real and unguarded — the def's 11 tools and `_SCHEMAS` are
      kept in lockstep by hand. Same target as entry 7, `docs/SERVER.md` §1.7.
  12. **`a1b2c3d4…` (`re.fullmatch()` already enforces whole-string matching) — DISCARDED, with the
      general lesson kept in `plan.md`'s parking lot.** Re-verified on CPython 3.12.3: an unanchored
      pattern under `.fullmatch()` correctly rejects `"count(v)) DETACH DELETE (v) //"`, and only
      switching the call to `.match()` lets it through. True — but the base fact is Python's own
      documented definition of `fullmatch`, which fails the "non-obvious environment fact" bar for a
      standing knowledge-base entry. The *generalizable* lesson (**a mutant must be proven to change
      behavior before its survival is read as a coverage gap**) has no owning knowledge base:
      `skills/python-web-quirks/` is scoped to web/async plus pytest import-timing, and
      `claude/qa-engineer/qa-testing-techniques.md` to black-box QA mechanics — stretching either
      was declined, the same judgment unit U9 applied to a ruff-config fact. Recorded in the parking
      lot so a future mutation-testing knowledge base can pick it up.
      **Id collision note:** this entry shares its 8-character prefix with a `tdd-engineer` entry
      already cleared earlier in this pass (`a1b2c3d4-5e6f-…`, an ORDER BY guard). These ids are
      hand-shaped, not `uuid4`; full id, date and subject were all confirmed before touching it.
- **`MENTIONS` tags added: none.** The four FalkorDB entries are engine/client facts rather than
  facts about another *agent*, and `cobb` was able to re-derive all four read-only — so they were
  fully dispositioned into `graph-dba`'s knowledge base directly instead of being tagged and
  deferred (the 2026-08-25 pass tagged three such entries precisely because confirming them needed
  write-capable probes `cobb` does not have; that did not apply here).
- **Cleared from `kaizen_team` this pass — all twelve, each a full `DETACH DELETE`:** every entry
  was counted first (`producedEdges + mentionEdges`), every one read `1 + 0`, so
  `otherRemaining == 0` in all twelve cases and the whole node was removed rather than just the
  `PRODUCED` edge. Ids: `b3f2a1c4…`, `a3f1c2d4…`, `f3c6021e…`, `2ee298a8…`, `c1a9e6f0…`,
  `c1f2a8b4…`, `29b6274a…`, `b7e1f0a2…`, `9050f193…`, `a3f0e6c1…`, `55364b9a…`, `a1b2c3d4-e5f6…`.
- **Docs touched:** `claude/coder/kaizen/{plan,history}.md` (this file + K-006 + the parking-lot
  entry) · `claude/graph-dba/falkordb-quirks.md` (four promotions) ·
  `claude/graph-dba/kaizen/history.md` (its own dated record of those four).
- **Plan items:** **K-006 opened** (the four kept-open falkor-chat doc asks, one table).
  **K-005 closed** — see the entry below.

## 2026-09-07 — K-005 closed: `tdd-engineer` fixed `Repository._read_structure`

- **What:** K-005 (opened 2026-08-25 by the previous distillation pass) is delivered and has been
  moved out of `plan.md`. `Repository._read_structure` now catches the "empty key" `ResponseError`
  and returns `None` per side — the same pattern as `read_index_dimension`/`services._read_or_absent`
  — so `verify_workflows.sh` no longer reports an intact `ws:<id>` snapshot as MISSING when
  `reference` has been fully `GRAPH.DELETE`d. The `falkor-chat/AGENTS.md` `test_queries.sh` row and
  a `falkor-chat/docs/HISTORY.md` entry (2026-08-25) landed with it, coordinated through
  `falkor-chat/docs/plans/workflow-diff-absent-key-coordination.md` (U1).
- **Why:** the item existed only because both fixes sat outside `cobb`'s write remit; with them
  shipped, `falkor-chat`'s own `HISTORY.md` is the durable record and a `coder` backlog item
  tracking someone else's delivered work is exactly the stale row the backlog convention forbids.
- **Plan items:** K-005 ✅ done, removed from the active table.

## 2026-08-28 — Broad-write guard added (prompt-friction fix, closes coder's FR-2 gap)
- **What:** Added `hooks/guard-coder-broad-write.sh` (thin wrapper over
  `claude/scripts/guard-broad-write.sh`, deny-list kept in lockstep with tdd-engineer's
  `guard-tdd-broad-write.sh`) and wired it in frontmatter (`hooks:` PreToolUse `Write|Edit`,
  plus `permissionMode: acceptEdits` matching the rest of the team).
- **Why:** Transcript evidence from teco session `2e0c2f42` (2026-08-27/28): every hook-free
  coder `Write`/`Edit` in an auto-mode teco orchestration hit a permission prompt (waits of 11s
  to 5.3h), while guard-carrying agents wrote silently in the same hours. An explicit PreToolUse
  `"allow"` is the one mode-independent prompt suppressor
  (`claude/docs/plans/agent-permission-friction.md` §1.3); coder was the implementer that never
  received one.

## 2026-08-25 — `kaizen_team` distillation: 10 raw entries processed (2 promoted, 2 discarded as superseded, 2 discarded as low-value, 1 kept open as K-005, 3 tagged `MENTIONS`→`graph-dba`)
- **What:** Ran `agent-maintenance` §5 over every `coder`-produced entry in `kaizen_team` — 4 legacy
  (`author: 'coder'`) + 6 current-shape (`PRODUCED` edges), all dated 2026-08-21/24/25, from K-028
  workflow-timers and K-050 M5 document-ingestion work. Each verified by re-deriving the fact
  myself (source reads, live `curl`, a Python repro script, or a fresh `mcp__cypher__query` count),
  not by re-confirming the entry's own citation.
  1. **`c4a9d1e6…` (circular-import direction) — promoted.** Empirically re-verified with a
     throwaway `pkg/a.py↔b.py` (fails both orders), a class-body variant (fails identically — the
     entry's own "not a class body" framing checked out), and a function-body variant (succeeds) —
     CPython 3.12. Added to `skills/python-web-quirks/SKILL.md` as a new section, tightened and
     corrected from the raw entry's slightly garbled wording; frontmatter `description` updated.
  2. **`d8f3b6a2…` (~8KB `Step.key` crashes the shared FalkorDB container) — discarded, superseded.**
     `falkor-chat/docs/BACKLOG.md` K-049 already tracks this exact incident (opened the same day,
     same author/context); `graph-dba/falkordb-quirks.md` already carries a **far more precise**,
     confirmed root cause (verified 2026-08-22: SIGSEGV, exact 4096/4097-byte boundary on any
     `UNIQUE`-constrained property, isolated-container repro, RCA doc) that supersedes this entry's
     own "root cause NOT confirmed, ~8KB" framing entirely. Nothing left to add.
  3. **`b7e2c9a1…` (TestClient teardown cancels tasks regardless of app lifespan) — promoted.**
     Re-reproduced fresh with a minimal FastAPI app whose lifespan never cancels its own task —
     `task.done()==True, task.cancelled()==True` right after the `with TestClient(app):` block
     exits anyway (starlette 1.3.1 / fastapi 0.139.0 / anyio 4.14.1, same versions the skill file's
     other entries already cite). Added to `skills/python-web-quirks/SKILL.md`.
  4. **`a3f1e8c2…` (unconditional wait/human transition can never suspend) — discarded, superseded.**
     This is the *same* defect `falkor-chat/docs/HISTORY.md`'s 2026-08-21 K-028 entry already
     documents exhaustively under "How it got here" — the v1/v2 unconditional-fallback design was
     caught by exactly this `coder`-authored finding during implementation, `teco` independently
     re-verified it against live source, and v3 (the shipped, QA-accepted design) replaced it with
     the `ctx.timerFired` marker-guard mechanism. The raw entry is the pre-fix stepping stone; the
     project record already supersedes it in full.
  5. **`c1e8a0a2…` (`test_queries.sh` reference-wipe false-negatives `verify_workflows.sh`'s
     snapshot check) — kept open, opened `plan.md` K-005.** Verified TRUE by tracing actual source
     (`services.py:1748`, `repository.py:1717`) — a real, currently-live bug: `_read_structure`'s
     unguarded `ro_query` against a fully-`GRAPH.DELETE`d `reference` raises an "empty key"
     `ResponseError` that `verify_workflows.sh`'s `read()` wrapper turns into a whole-diff `ABSENT`,
     falsely reporting an intact `ws:<id>` snapshot as missing too — contradicting
     `diff_def_snapshot`'s own docstring claim to handle this gracefully (true only when
     `reference` exists-but-empty, not when the graph key itself is gone). Both fixes (an
     AGENTS.md doc correction, a code fix or a BACKLOG defect) land in `falkor-chat/`, outside
     `cobb`'s write remit — see K-005 for the full trace and the recommended `teco` routing.
  6–8. **`7e3d1a2b…` (`count(*)` undercounts parallel edges), `7f3c2e1a…` (undirected
     relationship-property-filter pattern silently degrades to directed), `b2d8f4a1…` (REFINES
     `7f3c2e1a…`: the trigger is any relationship-property predicate, inline or `WHERE`, not just
     an inline filter) — tagged `MENTIONS`→`graph-dba`, not cleared.** All three are FalkorDB
     engine-behavior facts (not "how `coder` should behave"), none found already in
     `graph-dba/falkordb-quirks.md`'s "Cypher dialect & query behavior" section, and none
     independently reproducible by me — confirming a parallel-edge/write-based Cypher behavior
     needs a live write-capable probe, which is `graph-dba`'s tool access, not a curator's. Tagged
     `MENTIONS` per `agent-maintenance` §5 step 3's "substantively about a different agent" rule;
     only each entry's `PRODUCED` edge was resolved this pass (`otherRemaining=1` after — the fresh
     `MENTIONS` edge — so no `DETACH DELETE`), leaving the node live for `graph-dba`'s own future
     distillation pass to verify/promote/clear.
  9. **`a3f8c2e1…` (a pytest `FakeRepo` with one shared `since_rows` attribute can't test two
     merged repo methods independently) — discarded.** Already fixed in the actual test code
     (`self.hybrid_rows`/`self.chunk_rows`, falling back to `since_rows`) — the fix is
     self-documenting in the fixture itself; the underlying lesson ("give each mocked behavior its
     own controllable state") is standard test-double practice, not a durable/non-obvious
     environment fact worth a standing knowledge-base entry.
  10. **`b7e1d4a2…` (LM Studio reachable at `localhost:1234` in this sandboxed dev env) —
      discarded.** Re-verified live (`curl` succeeded, same as the entry). `falkor-chat/AGENTS.md`
      already flags the `pytest -m live` → LM Studio dependency; whether the local server happens
      to be running right now is transient session state, not a stable environment fact, and the
      "check reachability before assuming untestable" tip is generic testing advice not specific
      enough to earn a standing entry.
- **`MENTIONS` tags added:** `7e3d1a2b…`, `7f3c2e1a…`, `b2d8f4a1…` → `graph-dba` (all three
  `mcp__cypher__query(agent='cobb')` `MERGE`s, committed before any clearing ran this pass, per the
  ordering invariant).
- **Cleared from `kaizen_team` this pass:** `c4a9d1e6…`, `d8f3b6a2…`, `b7e2c9a1…`, `a3f1e8c2…`
  (legacy, unconditional `DETACH DELETE`); `a3f8c2e1…`, `b7e1d4a2…`, `c1e8a0a2…` (current-shape,
  `otherRemaining==0` after resolving the sole `PRODUCED` edge → full `DETACH DELETE`). **Left
  live** (current-shape, only the `PRODUCED` edge resolved): `7e3d1a2b…`, `7f3c2e1a…`, `b2d8f4a1…`.
- **Docs touched:** `claude/coder/kaizen/{plan,history}.md` (this file + K-005) ·
  `skills/python-web-quirks/SKILL.md` (two new sections + frontmatter `description`).
- **Plan items:** K-005 opened (kept-open disposition, item 5 above).

## 2026-08-25 — K-003 closed — the scope guardrail now defers to plan fidelity on the one overlapping case
- **What:** `:11` said a mid-build plan defect means *"stop and say so with a concrete proposal"*; `:35` said surface plan defects *"as notes for the user — don't just do them."* Items 2 and 3 of that list plainly mean note-and-continue, so the collision was item 1 only — but for a mid-build plan defect the two rules said halt and don't-halt. `:35` now defers: **a plan defect that blocks the current step stops the work; one that doesn't, becomes a note.** +26 w.
- **Cited by bolded name, not "(step 1)" as K-003 proposed.** `coder.md` carries both a "What you optimize for" bullet list and a numbered "How you work" list whose step 1 is **Orient** — so "(step 1)" would have pointed at the wrong rule. An ambiguous pointer is worse than a fragile one.
- **The gate lint found K-003 half-closed and the fix was extended.** The contradiction was resolved but the failure mode K-003's rationale singles out — *"it bites hardest when `coder` runs delegated: 'stop' there means returning the unit undone"* — was not. `coder.md` has **no general subagent protocol** (nothing like `security-expert.md:20`); its only two subagent branches are case-scoped to baseline blockers (`:19`) and destructive asks (`:36`), and neither reaches a mid-build plan defect. So a delegated `coder` that finished steps 1–2 of five and hit a blocking defect at step 3 had no instruction to report the completed work — §6's "skips a required artifact element", and a unit `teco` cannot resume from. Added: *"Delegated, 'stops' means returning the defect and your proposed fix as your result, alongside the steps you did complete"* — modelled on `:19`, four lines up, which already handles the structurally identical case correctly.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint 1 major (fixed in-commit), 1 nit accepted.

## 2026-08-25 — K-004 closed — `:12` now names the local-vs-project tiebreak (conventions-precedence family)
- **What:** `:12`'s conventions rule mixed a project-scope authority (*"already in the codebase"*) with a local-scope discovery heuristic (*"reading neighboring code"*). They agree in a consistent codebase and diverge in exactly the case worth a rule. One sentence appended; +25 w.
- **`:29` is not part of the collision and was left alone** — *"Its conventions… win over your defaults"* runs on the project-vs-**agent** axis, pointing away from the agent's habits just as `:12` does. Confirmed by the lint.
- **The three "judged and kept, do not re-litigate" restatements** in `kaizen/plan.md`'s parking lot are undisturbed — none touches conventions or style, and the addition is a tiebreak *between* two clauses in one bullet, not a restatement of either. **K-003** (`:11` vs `:35`, stop vs. note-and-continue) is untouched and stays open: a separate rule change.
- **The rule, byte-identical in all three implementer prompts:** *"Where a file or folder deviates locally from the project norm, match it, not the norm — mixing both in one place is worse than either applied consistently."* Identity is deliberate — three paraphrases would recreate exactly the divergence that made this a `agent-maintenance` §4 **check-5 boundary-reciprocity** problem rather than three separate nits.
- **Not relocated to a shared file, and the reason is stronger than "no plausible home."** Finding 15 says a rule binding N>1 agents belongs where all N read it, and this binds three. But no shared file owns *how to author code*, and root `AGENTS.md` would be an **actively bad** home: its document conventions are stated as absolutes (`never begins with m<digit>`, the closed role set, the closed `Status:` set), and a general "a local deviation beats the project norm" principle sitting beside them hands every agent a lever to justify deviating from them. The rule would not sit inertly there; it would undercut them. So: byte-identical duplication, **plus a mechanical identity guard** — `audit-team.sh` **check 10**, which fails when the sentence is in some but not all three, and passes when it is in all three or none (removing it everywhere is a legitimate family decision; removing it from one is the defect). Recorded as plan finding 22.
- **Why the guard was necessary and not belt-and-braces.** All three kaizen plans said "fix together or not at all" and nothing read those plans at edit time. The mitigation was prose in files nothing loads. Stage F had shipped the enforcement machinery the same day, and this is exactly the class of guarantee it exists for: deterministic, must-always-hold, therefore a mechanism rather than hopeful prose.
- **The family boundary is a test, not a judgment call:** *two co-equal scope claims in one prompt with no tiebreak between them.* `graph-dba:53` and `devops` were checked and excluded — not because they are a "different axis" (that was my first, wrong reasoning; `graph-dba:53` is structurally identical and has a real local-vs-project instance in this repo, since `cpg_*` graphs carry Joern's imported schema against the `PascalCase`/`UPPER_SNAKE` norm) but because each states **one** scope, so there is nothing to tiebreak. Recorded so the next reader does not re-open it, notice the shape match, and conclude the exclusion was arbitrary.
- **Verified:** `audit-team.sh` **PASS**, check 10 green at 3/3; negative-tested at 2/3 (FAIL, exit 1) and 0/3 (PASS). `cobb` §7 lint: 0 blockers, 1 major, 3 minors, 2 nits — all applied.

## 2026-08-24 — Prompt-waste compression, Stage C6: inventoried, **zero edits** — judged at its editorial floor
- **What:** `coder.md` was one of C6's four files (`claude/docs/plans/prompt-waste-reduction.md`, Stage C). Full class-5/6/7 inventory run; **no edit made**, and that is the finding, not a shortfall. 1,240 w before and after. Wave 1 + the wave-2 micro-fix had already taken everything class-5/6 this file ever carried.
- **Class-6 residual: 0 w.** Mechanically confirmed — zero hits for dates, FR/AC tags, authority markers, supersession trails, or `kaizen/history.md` pointers anywhere in the file. Nothing left to cut in the cheap category, permanently (plan finding 11).
- **Class-7 residual: ~34 w, all of it certified keeps** under finding 5 ("needed twice", not "said twice"):
  - **"Don't claim what you didn't run" (guardrail) vs. step 5's "Report what you actually ran and saw."** Step 5 is the report-writing *procedure* (show the output; report `passed`/`skipped`/`deselected`, since a suite can exit 0 with a chunk silently unrun); the guardrail is the absolute prohibition. Same pair `qa-engineer` certified as a keep at C5.
  - **"Ask before destructive or environment-changing actions" (guardrail) vs. step 2's "ask before installing or mutating the environment."** Two decision points — baseline setup vs. anywhere in the build — each carrying its own subagent carve-out. Also the pair `qa-engineer` certified at C5; the carve-out duplication is a **certification requirement** (`agent-maintenance` §4 check 3), not a style choice.
  - **The three scope statements** — "Minimal blast radius" (scope of edits) / "Don't silently exceed scope" (reporting obligation) / "don't silently diverge" (what to do when the plan is *wrong*). Three rules, not one restated.
- **Gate (a) inventory — trivially satisfied:** no text removed, so every rule survives byte-identical.
- **Two pre-existing defects found and routed, not bundled** (§4.0 rollback contract — both fixes are rule changes):
  - **K-003** — `:11` and `:35` prescribe *different actions for the same trigger* (a mid-build plan defect: "stop and say so" vs. "surface as notes for the user"). `cobb`'s lint dissented from this session's initial reading that the three scope statements were cleanly distinct, and it is right: this pair is a genuine contradiction, worst when `coder` runs delegated (where "stop" means returning the unit undone with no way to ask).
  - **K-004** — `:12`'s conventions ambiguity: "already in the codebase" (project scope) vs. "reading neighboring code" (local scope), which return different answers when a file deviates locally. Same family as `tdd-engineer` **K-006**, weaker instance — here the second clause reads as the *method* for the first rather than a competing authority, so the collision is inside `:12` alone. `:29` ("its conventions win over your defaults") runs on the project-vs-*agent* axis and does not collide.
- **Verified:** `audit-team.sh` PASS (check 8 row intact, file untouched); `cobb` §7 lint — clean on all seven dimensions for this file, the two findings above pre-existing.

## 2026-08-23 — Freshness-clause grammar fix (Stage B wave 2 micro-shape)
- **What:** "a `teco`-issued brief that states the graph's freshness, take it as given" → "when a `teco`-issued brief states the graph's freshness, take it as given" — closing the hanging-topic construction cobb's wave-1 lint flagged as minor; applied uniformly across all files carrying the clause. No rule change; both branches intact.

## 2026-08-23 — Prompt-waste compression, Stage B wave 1 (boilerplate sweep)
- **What:** Applied the three pilot-validated boilerplate compressions from
  `claude/docs/plans/prompt-waste-reduction.md` (§3 doctrine, Stage B), same shapes as the
  `architect.md` pilot. (1) CPG-freshness clause (§ "Orient"): dropped the "(2026-08-19)" date and
  the redundant "without re-deriving staleness yourself" tail; rule now reads "CPG freshness is
  `teco`'s responsibility, not yours: a `teco`-issued brief that states the graph's freshness, take
  it as given; running standalone, use the CPG's answers as current." (2) Interactive-commit-grant
  bullet (§ Guardrails): dropped the provenance sentence "Stakeholder decision, 2026-08-21 — see
  `kaizen/history.md`." and ", same as before"; the "(spawned via `Agent`/`Task`)" clarifier moved
  from the interactive-definition parenthetical to the carve-out sentence (was stated in both).
  (3) Learning capture: intro dropped "directly" and "identified by a real `:Agent` node it's
  `PRODUCED`-linked to," (the Cypher template below shows the MERGE + PRODUCED edge); tail dropped
  the inbox-replacement history sentence and "exactly like the old inbox was".
- **Rule inventory (gate a), edited regions — all preserved:** freshness is `teco`'s / brief taken
  as given / standalone = current (block 1); interactive-mode definition, explicit-path grant,
  full never-list, delegated-subagent carve-out, deliverable left for `teco` post-verification
  (block 2); capture trigger + graph + Cypher template, skip-known-facts, raw-capture/`cobb`
  promotes, never edit own definition (block 3). Verbatim `CPG:` three-form sentence and
  audit-check-8 tokens untouched.
- **Removed class-5/6 material, recorded where:** inbox-replacement history → this file's
  2026-08-21 "kaizen/inbox.md deleted" entry; commit-grant provenance → this file's 2026-08-21
  grant entry + `claude/AGENTS.md` § Hook machinery; freshness centralization date → this file's
  2026-08-19 "Freshness-check clause removed" entry.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint pass on the result.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Guardrails bullet: when running interactively (`claude --agent coder`, a human
  present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own verified
  code changes from the session, by explicit path, never bulk-staged/pushed/reset/rebased/
  amended; the grant does not apply when spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Verify and report"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there has already been distilled and cleared — and this file's own pre-migration content (if any) was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry below). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_coder`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_coder` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," and "never edit your own agent definition" (no write-guard clause — coder has no doc-scoped write guard). Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Freshness-check clause removed (centralized on teco)
- **What:** Dropped the CPG freshness-check paragraph from the CPG-orientation step — still checks whether a relevant CPG exists and uses it via `cpg-analysis`, but no longer queries the `:CpgBuildInfo` freshness marker itself. That responsibility is now `teco`'s alone (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`); running standalone (no `teco`-issued brief), staleness is simply not checked.
- **Why:** User-directed prompt-verbosity reduction: the freshness paragraph was ~130 words, byte-identical across six agent files. Stakeholder chose full centralization over a per-agent dedup, accepting the standalone-run capability loss.
- **Plan items:** —

## 2026-08-16 — U7 fix round: freshness-check sequencing hardened, `CPG:` line anchored (DEF-1/DEF-2/DEF-3)
- **What:** Two wording tightenings per `docs/plans/cpg-agent-adoption-coordination.md` unit U7,
  following U6's `qa-engineer` live-dispatch acceptance pass
  (`docs/test-reports/cpg-agent-adoption-report.md`). (1) The freshness-check sentence now reads
  "query the freshness check … in that same tool call/step, before deciding whether the result
  needs further cross-verification — this is not a separate, optional judgment call" (previously
  "also run the freshness check … as part of that same step") — closes DEF-2 (`architect`
  reasoned its way past the check with a grep/CPG-agreement substitute that doesn't rule out
  "stale but coincidentally consistent"). (2) The `CPG:` line instruction now reads "written
  verbatim and required in all three cases including when the CPG isn't relevant — not
  paraphrased, not dropped" — closes DEF-1 (`coder`, this agent, used the CPG correctly but wrote
  loose prose ("**CPG freshness note:**…") instead of the literal line) and DEF-3 (`tdd-engineer`
  dropped the line entirely on the not-relevant branch). Applied identically (phrasing pattern,
  not restructuring) across all six wired agents; only `coder`/`architect`/`tdd-engineer` were
  live-tested, but the near-verbatim wiring pattern means the same gap plausibly existed in
  `analyst`/`qa-engineer`/`frontend-engineer` too.
- **Why:** U6's acceptance pass found the M4 wiring (U4b/U4b-2) was correctly worded but didn't
  survive contact with a real dispatched agent's own judgment calls — all three live-tested
  dispatches failed a different way (format, skip, silence). Design intent
  (`docs/plans/cpg-agent-adoption.md` §2.3, §3) unchanged: still agent judgment on staleness
  threshold, still no self-triggered rebuild, still a suggestion not a hard rule about *when*
  something counts as stale — only the sequencing and the anchoring got tightened.
- **Plan items:** none new; closes U7.
- **Same-day addendum (U8 diff-gate follow-up):** `analyst`'s U8 diff gate
  (`docs/reviews/cpg-agent-adoption.md`, Pass 3 — approve with suggestions, zero blockers)
  flagged two minors and a nit against this same freshness sentence: (a) `frontend-engineer.md`
  was missing the "tool call/" qualifier the other five files carried, undercutting the U7
  ledger row's and commit message's "identically" claim; (b) the trailing "this is not a
  separate, optional judgment call" had an ambiguous pronoun referent — a literal reading could
  bind "this" to the cross-verification *decision* rather than the freshness *query itself*,
  exactly the room DEF-2's `architect` dispatch used to reason past a softer version of this
  sentence; (c) nit — "query the freshness check" mismatched a reference-doc noun with a query
  verb, when the actual queried object is the `:CpgBuildInfo` marker (the report's own
  recommendation said "marker"). Fixed all three: the sentence now reads "…query the freshness
  marker (per `skills/cpg-analysis/references/freshness.md`) in that same tool call/step, before
  you decide whether the CPG's answer needs further cross-verification — running the freshness
  check itself is not optional, and skipping it in favor of a substitute check (e.g. grep
  agreement) doesn't satisfy this." — byte-identical across all six files now. The `CPG:`-line
  wording from the original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — M4 cpg-agent-adoption: wired as a new `cpg-analysis` consumer

- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§6 step 3 (`cobb`-owned
  design, U4b implementation unit). **Description:** added "With a loaded Joern CPG, uses the
  `cpg-analysis` skill for impact analysis before changing a function — what calls it, what else
  would break — instead of grepping by hand." **Step 1 "Orient":** added a check for a relevant
  CPG (first-guess `cpg_<component>` naming per `skills/cpg-analysis/SKILL.md` §1) bundled with
  the freshness check (`skills/cpg-analysis/references/freshness.md`) in the same step, noting
  the result in the report and surfacing a refresh suggestion — never a silent rebuild — if
  stale. **Step 5 "Verify and report":** added the `CPG:` evidence-trail line (plan §3 — `used
  <graph> — <clause>` / `considered, not relevant — <clause>` / `not applicable — <clause>`).
- **Why:** M4 widens the `cpg-analysis` roster from three consumers (`analyst`, `architect`,
  `qa-engineer`) to six — `coder` is a new consumer because its "what calls this / what would
  break" question before changing a function is exactly the impact-analysis recipe's target, and
  both live CPGs (`cpg_falkorchat`, `cpg_salesperson`) are Python codebases `coder` already works
  in. Plan §1's `coder` row has the full roster reasoning.
- **Plan items:** none (design-driven, not backlog-driven).
- **Addendum (same day):** the description clause initially shipped in the conditional "With a
  loaded Joern CPG, uses…" framing (carried over from a state-recovery instruction that locked it
  as already-landed). The coordinator caught that this left `coder`/`tdd-engineer` as the only two
  of the six wired agents not on plan §2.1's mandated default-orientation framing — the sibling
  unit's `analyst`/`architect`/`qa-engineer`/`frontend-engineer` edits all use "Checks whether a
  relevant CPG exists as part of its normal orientation and, when one does, uses…". Reworded to
  match: "Checks whether a relevant CPG exists as part of its normal orientation and, when one
  does, uses the `cpg-analysis` skill for impact analysis before changing a function — what calls
  it, what else would break — instead of grepping by hand." Body-prompt and evidence-trail
  additions were already correct and untouched.

## 2026-08-11 — Inbox distillation, corrected: 8 entries routed (5 promoted, 1 discarded as redundant, 2 promoted late after an `analyst` review caught them missing)

- **What:** `cobb` processed all 8 entries in `coder/kaizen/inbox.md` (§5), triggered by a
  stakeholder report of context blowouts and teco's explicit request to sweep this inbox. **This
  entry replaces a first version that mis-credited two entries to `coder` that actually came from
  `analyst`'s and `architect`'s inboxes** (the `urllib` timeout taxonomy and the LM Studio `/v1`
  200-envelope quirk — both correctly logged in *those* agents' own history entries) and, as a
  result, left two real `coder` entries with no logged disposition at all. Caught by `analyst`'s
  independent review (`docs/reviews/kaizen-distillation-2026-08.md`, B-1).
- **The 8 real entries and their dispositions:**
  1. FastAPI `response_model_exclude_unset=True` omit-vs-null — **discarded**, already fully
     covered by `python-web-quirks.md`'s pre-existing "nested models" entry.
  2. A `pytest --collect-only` baseline moving under you mid-run (concurrent agent on the same
     tree) — **promoted**: `coder.md` step 5 ("Verify and report") now says to report the
     *attributed* delta (own diff's contribution) alongside entry→exit counts when the tree may be
     shared.
  3. A green pytest exit code isn't evidence an integration suite ran; read the skip count —
     **promoted**: `coder.md` step 5 now carries the same skip/deselected-count clause
     `tdd-engineer.md`'s "Verify honestly" step already had — this was a live asymmetry (the agent
     that filed the learning didn't have it yet).
  4. FastMCP/`mcp` stdio EOF-response-loss — **promoted**, merged with `devops`'s 2026-07-26
     refinement (which corrects this entry's "always the last reply" framing to "a race, can drop
     more than one") → `claude/cobb/TESTING.md` (new gotcha subsection).
  5. FalkorDB no string-repetition operator — **promoted** → `claude/graph-dba/falkordb-quirks.md`
     (this is the only agent/history pair that owns this promotion; an earlier version of
     `graph-dba`'s own history entry also claimed it and has been corrected).
  6. `audit-team.sh`'s `grep -c FAIL` overcount-by-one — **promoted** → `skills/agent-maintenance/
     SKILL.md` §4.
  7. `monkeypatch.setenv` no-op against an import-time-frozen module constant — **promoted** →
     `skills/python-web-quirks/SKILL.md` (general Python fact; the falkor-chat-specific instance
     was already independently captured in `falkor-chat/docs/DESIGN.md` §14.7).
  8. Function-local deferred-import monkeypatch timing — **promoted** → same skill file.
- **On coder/tdd-engineer convergence (flagged as a judgment call in the review's open questions):**
  made the call to converge on suite-reporting discipline specifically — both are implementers
  whose "done" claim rests on the same suite, and a skip-count blind spot is exactly the kind of
  gap that should not differ by which implementer happened to touch the code. Did **not** merge
  their broader disciplines (TDD-cycle narration, the attributed-delta clause's fuller framing,
  etc.) — those reflect genuinely different working styles (red-green-refactor vs. plan-execution)
  that shouldn't converge just because one prompt happens to be shorter.
- **Verified:** `bash claude/scripts/audit-team.sh` clean. No personal identifiers introduced.
- **Docs touched:** `claude/coder/{coder.md,kaizen/{history,inbox}.md}` ·
  `skills/python-web-quirks/SKILL.md` · `claude/cobb/TESTING.md` ·
  `claude/graph-dba/{falkordb-quirks.md,kaizen/history.md}` · `skills/agent-maintenance/SKILL.md`.

## 2026-08-09 — Description gained a `python-web-quirks` skill routing clause
- **What:** Frontmatter `description` gained one clause: in a Python web/async codebase, the
  agent consults the new `skills/python-web-quirks/SKILL.md` for asyncio/FastAPI/Starlette/
  pydantic gotchas — mirroring how `cpg-analysis` is wired into `analyst`/`architect`. No body
  change; skills are progressively disclosed and self-describe.
- **Why:** `python-web-quirks` was created distilling three general Python/web-framework facts
  from `analyst`'s learnings inbox (asyncio `create_task` GC-safety, `BackgroundTasks` vs.
  `threading.Thread` concurrency bounds, FastAPI/pydantic `exclude_unset` on nested models).
  Stakeholder wired it to `coder`/`tdd-engineer`/`architect`/`analyst` at minimum since all four
  plausibly implement or review Python web/async code. See `claude/analyst/kaizen/history.md`
  (2026-08-09) for the full distillation record and `skills/README.md` for the catalog entry.
- **Plan items:** none.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 547 → 384 chars (-29%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (coder↔tdd-engineer, coder↔frontend-engineer) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Learnings inbox distilled (first pass; 6 entries, inbox cleared)
- **What:** Ran the `agent-maintenance` §5 distillation over `kaizen/inbox.md` — 6 entries accumulated
  2026-07-16 → 2026-07-21, all falkor-chat environment facts from K-022/K-024 work, never before
  distilled. All 6 verified against the live repo; **none routed to the coder's prompt** (every one is a
  fact about *falkor-chat*, not about how the coder should behave), so the prompt is unchanged.
  Routing:
  - **Already documented → discarded (3).** (1) *Published defs are immutable per version; re-seeding
    an edited prompt silently keeps the old config, and `reference`/`ws:<id>` go stale independently* —
    now covered exhaustively by the `seed_workflows.sh` row in `falkor-chat/AGENTS.md`, consistent with
    `docs/DESIGN.md` §144/§147/§544. Re-verified the mechanism still holds (`repository.py`
    `_PUBLISH_CYPHER` is still `MERGE … ON CREATE SET`). (5) *pytest wipes `reference` at setup, so the
    last test's defs survive and masquerade as a seeded def* — same AGENTS.md row already spells out the
    false `already present — no-op` signal. (3) *the `_drive_loop` byte-identity lock's quoted byte count
    is wrong (SHA `71055f756280` right, 2844 ≠ actual 2860)* — already an open **Doc-drift** item at
    `falkor-chat/docs/BACKLOG.md:399` with the same diagnosis, and the surviving plan docs
    (`docs/archive/plans/m3-process-flow.md:396`, `…-coordination.md:38`) now say "verify by SHA only;
    every byte count quoted is wrong". Nothing to add.
  - **Promoted to project docs (3).** (4) *zero-transition publish → bare `IndexError` on a half-written
    def* — the **service-layer guard has since landed** (`services.py` `_validate_def_spec`, K-024 U4b
    O-6, rejects with `WorkflowDefSpecError` → 400), but `docs/QUERIES.md` §11.1 documented neither the
    trap nor why this query is deliberately unguarded while §4's mention block is; added a ⚠️ note there
    stating the collapse mechanism, the unrepairable-poisoning consequence, where the guard actually
    lives, and that any new caller bypassing the service layer must re-validate. (2) *pytest is
    destructive to `reference`, and a green pytest line hides ~half the suite skipping when FalkorDB is
    down* — the `seed_workflows.sh` row carried this from the seeding side, but the **M1 server pytest
    bullet**, where someone running pytest actually looks, did not; added two bullets to
    `falkor-chat/AGENTS.md` (destructive-at-setup + re-seed, and read the skip count — verified
    `conftest.py:54` `pytest.skip` guard). (6) *`ruff check .` baseline is already red* — verified still
    red today, exactly one pre-existing `I001` at `falkorchat/llm.py:13`; documented as a known baseline
    in the same AGENTS.md section so an implementer doesn't misread it as their own regression.
- **Why:** §5 — capture is cheap and unreviewed, promotion is curated; facts about *a project* belong in
  project docs where every agent sees them, never hoarded in one agent's private files. Half the inbox
  had already been absorbed into project docs by the K-024/M3-close doc sweeps, which is the loop
  working as intended.
- **Deliberately NOT done (flagged as follow-up, not landed silently):** the one-line `llm.py` import
  reorder that would make ruff green, and any repository-level guard on `_PUBLISH_CYPHER`. Both are code
  changes to falkor-chat, outside a distillation pass's remit.
- **Plan items:** none closed; K-002 evidence noted in `plan.md`.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so the user kept having to re-grant write permission to `coder` across sessions even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`. `acceptEdits` auto-accepts file edits and common filesystem commands for paths in the working directory/`additionalDirectories`, independent of session-level grants.
- **Why:** User asked why they always had to give write permission to `coder`; root cause confirmed against current Claude Code docs (`sub-agents.md`, `permissions.md`) rather than assumed.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 822 to 535 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Description: route-away clause to the new `frontend-engineer`
- **What:** the `description`'s routing tail gained a second route-away rule: UI-heavy front-end work (components, styling, accessibility, client-side state, a Streamlit screen) → `frontend-engineer`. Catalog rows (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`) updated to match; the pair was added to `scripts/audit-team.sh` `BOUNDARY_PAIRS`.
- **Why:** a UI-depth specialist implementer (`frontend-engineer`) joined the team; boundary reciprocity requires the adjacent generalist implementer to name it so routers see the contract from both sides.
- **Plan items:** none (driven by frontend-engineer's creation).

## 2026-07-09 — K-001 ✅: efficiency-based routing boundary with `tdd-engineer` (de-personalized)
- **What:** Rewrote the `description`'s routing tail. Was "for strict test-first discipline, prefer tdd-engineer" — a subjective tiebreaker; now routes by **task shape / efficiency**: a detailed plan/spec ready to execute → `coder` (tests alongside); a bug fix, safety-net refactor, test-focused work, or clear-contract feature → `tdd-engineer`. Made symmetric: `tdd-engineer`'s description (which previously shadowed coder's trigger — "whenever the user asks to implement a feature" — and never pointed back) now carries the mirror rule. Synced everywhere the rule is repeated: teco's roster (the "(this user prefers TDD — lean toward tdd-engineer)" note removed), `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, and `cobb/TESTING.md`'s rationale column.
- **Why:** User ruling on the overlap review: use `tdd-engineer` only where test-first is genuinely the efficient path, and when a detailed plan already exists the most efficient implementer wins — plus, personal-preference notes ("this user prefers TDD") don't belong in agent prompts; the user's standing preferences are **quality and efficiency**, encoded as objective routing rules.
- **Plan items:** K-001 ✅ (closed — the descriptions no longer collide; the tiebreaker is objective at routing time).

## 2026-07-09 — Subagent-awareness on the two "ask" spots (teco interface review follow-up)
- **What:** Step 2 (baseline) "ask before installing or mutating the environment" and the "Ask before destructive or environment-changing actions" guardrail now both say what to do when running as a subagent (e.g. delegated by `teco`): return the blocker / request as the result instead of trying to ask mid-run — subagents can't ask. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** Sweep after the 2026-07-09 teco interface review found the "ask" phrasing assumed an interactive session across several delegates (same fix applied to tdd-engineer, qa-engineer, graph-dba the same day; architect already handled it via questions-as-deliverable).
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-07-08 — Architect handoff arrives as a plan-document path
- **What:** Step 1 (Orient) updated: an `architect` handoff is now a plan **document** at `<component>/docs/plans/<slug>.md` — the coder gets the path in its brief, reads the file itself, and treats it as source of truth (gaps filled by reading code, not guessing). Synced with the architect's same-day change making the plan doc its default deliverable and teco's switch to path-based handoff.
- **Why:** The previous flow had the orchestrator paste the plan into the brief — lossy and unreviewable. Reading the artifact directly makes the handoff lossless regardless of who invokes the coder.
- **Plan items:** advances K-002 (transport fixed; live validation run still pending).

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software engineer" → "Software engineer") and the body opener ("You are a senior software engineer who builds" → "You are a software engineer who builds"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User raised the overconfidence concern with seniority framing; persona-prompting evidence (e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness while authority framing can hurt calibration. Quality is carried by the concrete process + guardrails ("don't fake green," "report only what you ran"), not the title. Chose the most conservative option (drop the word). Goes further than the 2026-06-05 precedent that kept "Senior" as an altitude signal — architect/coder now differ from the rest of the collection until harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `coder` subagent (`coder/coder.md`, `model: opus`). Senior implementer that executes an approved plan/spec end-to-end: orients on the plan + code, establishes a green baseline (with explicit handling for already-red and can't-run-here cases), implements in small reversible increments, tests alongside, refactors under green, and reports only results it actually ran. Inherits all tools (no `tools` key), following the `tdd-engineer` K-003 precedent of keeping the implementer flexible.
- **Why:** User asked for two complementary Claude Code subagents, "the architect" and "the coder," with an architect→coder handoff. The `coder` is the implementation half.
- **Plan items:** seeded K-001..K-002.

## Decisions recorded at creation
- **Distinction from `tdd-engineer`:** both implement, but `tdd-engineer` is *strictly* test-first (red→green→refactor as the defining discipline). `coder` is plan-driven and pragmatic: it tests behavior thoroughly and never ships untested code, but doesn't mandate writing the failing test first unless the project requires it. The `description` explicitly routes strict-TDD requests to `tdd-engineer` so auto-delegation doesn't collide. Revisit if the two over-trigger on the same prompts.
- **Why inherit all tools:** an implementer needs Read/Write/Edit/Bash plus the ability to fetch docs and delegate; mirrors the deliberate `tdd-engineer` choice (K-003 there). Revisit only if broad access causes surprise.
