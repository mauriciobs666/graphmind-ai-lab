# Kaizen — Change History: analyst

> Dated log of actual changes to the `analyst` agent. Most recent first.


## 2026-09-08 — `kaizen_team` distillation pass 2, chunk E (unit U23): 13 entries, 12 promoted, 1 discarded

`cobb` processed the thirteen `analyst`-produced `kaizen_team` entries dated 2026-09-07
(`claude/docs/plans/kaizen-distillation2-coordination.md` U23, the last of five chunks). Every
entry pinned by **complete** `entryId` at read, count and clear. This chunk was the pass's worst
prefix-collision cluster — eight ids sharing 2-4 leading characters across three families
(`b1f0e6c2`/`b1f0c7a4`; `b7f1c3a2`/`b3f1c7a2`/`b1e7c2a4`/`b18d5c47`; `3f6c2a91`/`3f6c1e28`) — and
all thirteen were read, counted and cleared on the full 36-character id. Verified after the
clears: the near-prefix neighbours are distinct nodes and every one survived its sibling's
deletion. Zero kept open, zero `MENTIONS` tags, and **no always-loaded prompt gained a bullet** —
the one `analyst.md` change replaces an existing bullet in place, as in chunks A-D.

**Twelve entries landed as one new section, two folds and one knowledge-base bullet**, not
thirteen additions. Six of the thirteen were one technique family and were written as a single
six-part section rather than six adjacent sections.

**Promoted (12 entries → 3 files):**

- `claude/analyst/review-techniques.md`, **new section** *"A grep-pinned edit table is an edit
  list, not a completeness proof"* — six entries folded into one six-part list:
  `b1f0e6c2-9a44-4d3e-8f21-7c5a2e9d4a10` (intermediate-baseline residual),
  `3f6c2a91-0b47-4d1e-9c58-71a2e4d0b8fa` (enumerate by the attribute the ship criterion reads),
  `b1f0c7a4-9d2e-4c11-8a63-5e7f24d0aa19` (two cases with no completeness property),
  `b1e7c2a4-9f3d-4a51-8c6e-2d70f1a4c983` (run the table's commands before trusting its site rows),
  `b3f1c7a2-5d64-4e18-9a2c-7f0e41d8b6aa` (retired-token residual vs. a done-condition naming it),
  `b18d5c47-2e93-4a06-8f71-c93a0e4bb210` (cross-table collisions).
- `claude/analyst/review-techniques.md`, **fold** into the existing *"A guard derived from the
  artifact it guards…"* section (its AST subsection): `b7f1c3a2-9d4e-4c18-8a6f-2e5b71d0c934`
  (two axes), `cbfd9614-5ae1-4c55-94e9-012e06ccdf00` (mutate the aliased spelling),
  `3f6c1e28-7a94-4d51-9b2e-0c8a5f2d41b7` (`ast.Assign`-only harvest under-reaches). Deliberately
  a fold, not a fourth adjacent section — the concrete falkor-chat facts these three carry are
  **already published at the point of use**, in the `_bindings` and `_alias_prefixes` docstrings
  in `falkor-chat/server/tests/test_storefront_api.py:3107` and `:3158`, in more detail than the
  entries hold (both axes named, the `me = self` two-hop stop measured at 185 vs. 184, the
  annotated-local under-reach called a house idiom). What was **not** documented anywhere, and is
  what got promoted, is the reviewer-facing generalisation plus a way to derive the binding-node
  enumeration from the grammar.
- `claude/analyst/analyst.md`, **existing bullet replaced in place** (Guardrails → Evidence over
  vibes): `2f8a5d13-6b47-4c92-a1e0-83b5cf27d904` (a review-suggested fix against a green suite is
  weak evidence) and `9c4e2b71-8a3d-4f56-b0e9-2d17c4a85f33` (a finding that suggests *design* is a
  claim that must be run). The prompt already carried *"A regex/glob/pattern you suggest as a fix
  is a claim, not a nit — run it"*; both entries are that same rule, generalised past regexes and
  given a test. Widened to any suggested fix including a design, with the operative addition —
  **a suggested fix is evidence only where you can name the assertion that would catch it being
  wrong; where you cannot, hand the design decision to the implementer and say why.** One bullet
  in, one bullet out; `analyst.md` 2477 → 2515 words.
- `claude/graph-dba/falkordb-quirks.md`, **new bullet**:
  `6eeaa03e-98f9-42ac-8c6f-d9a0d9a06791` (`redis-cli` exits 0 on an error reply). Logged in
  `claude/graph-dba/kaizen/history.md` the same day. No `MENTIONS` edge — nothing was left
  outstanding for `graph-dba` (see below).

**Discarded (1):** `be57e2aa-8c39-47c8-927b-24b3b7d60b3e` (`git rev-parse HEAD:.` is fatal,
`HEAD:./` works). The fact **re-derived exactly** — at the repo root `git rev-parse --short HEAD:.`
→ `fatal: Needed a single revision`, exit 128, message on stderr; `HEAD:./` → `7ff9aa9`, identical
to `HEAD^{tree}` (git 2.43.0, 2026-09-08) — but its own target has already fixed it and documents
it better: `skills/cpg-analysis/references/freshness.md:104-117` now prescribes
`git rev-parse --verify "HEAD:./<sourceOrigin>"`, states that a bare `HEAD:.` is fatal with the
exact error and exit code, and adds two facts the entry does not have (without `--verify`,
`rev-parse` echoes an unresolvable argument back on stdout and a script reads that as "the source
moved"; `--short` width tracks the repo's object count, so the same tree can render 7 chars today
and 8 next month). Already published at the point of use.

**Two corrections to entries' evidence, both in the citation and not in the claim** — the same
defect shape U20 and U21 produced, now three passes running:

- `3f6c1e28`'s figures are **wrong at its own commit — and so were mine, twice, before `teco`
  caught them.** The entry reads "68 annotated local assignments (`ast.AnnAssign`) — 6 in
  `storefront.py`, 16 in `storefront_api.py`, 23 in `services.py` — plus 40 tuple-target assigns
  and 1 walrus". Correct figures, re-derived by `ast` over all 28 `.py` files of
  `falkor-chat/server/falkorchat`, at the entry's own sha `00827c2` and at the 2026-09-08 worktree
  (**identical at both**): **65** function-local annotated assignments with a `Name` target,
  **40** function-local tuple-target assignments, **1** walrus, per-file **3 / 3 / 14**. Only the
  walrus was right in the entry. Its `40` is right too, but by accident of framing — tuple-target
  reads 40 under *both* local and package-wide scope, so it is not the package-wide figure the
  entry's sentence structure implies. Its `16` and `23` are *whole-file* `AnnAssign` counts
  (module- and class-level annotations included, which are not local assignments at all), and
  `16` matches the worktree rather than `00827c2`, where `storefront.py` reads 15. Its `6` matches
  nothing under any of the five definitions enumerated below. Promoted in corrected form, with the
  scoping trap stated in the fold, because it is exactly how this measurement goes wrong.
  - **My own first two attempts were also wrong, and the cause was the instrument.** I reported
    68 / 42 / per-file 3 / 5 / 14, then wrote 42 into the promoted text while my report to the
    coordinator described 40 as package-wide — the citation drifting from the verification, a
    fourth instance of this pass's recurring defect shape. Root cause: my script ran
    `ast.walk(fn)` for **every** `FunctionDef` in the module, so any node inside a nested function
    was counted once per enclosing scope. Re-run with single-visit attribution (parent links,
    each node classified once by walking to its nearest enclosing scope) gives 65 / 40 / 3 / 3 / 14.
    `teco` refuted it by enumerating five candidate definitions to show none yields 68 — local
    `Name` target 65, local `Name` incl. bare declarations 65, local ANY target 79, package-wide
    `Name` 290, package-wide ANY 304 — and I reproduce all five exactly. The 304 in the promoted
    text was right throughout, which is the tell that both instruments agreed on the space being
    measured and disagreed only on double-counting.
  - **The failure was structural, not arithmetic.** Correcting the per-file breakdown consumed the
    verification and the headline `68` rode through on the entry's authority — the inverse of the
    trap I had named in the same disposition ("the tell I should have read was that two of its
    other figures already matched exactly"). A refutation of part of a citation is not a
    verification of the rest of it.
  - **One claim survived a challenge and is stronger than I knew.** `teco` doubted "identical at
    `00827c2` and at the worktree", having seen 749 insertions across `storefront.py` and
    `storefront_api.py` between the two revisions (`git diff --stat`: 667 insertions, 84
    deletions). It holds — 65 / 40 / 1 and per-file 3 / 3 / 14 at both. The churn moved
    module- and class-level annotations (`storefront.py` whole-file 15 → 16) and left every
    function-local count untouched.
- `b1f0c7a4`'s per-command counts are **not re-derivable at any commit**, and this is a property
  of plan gates rather than an error. It cites `grep -rFn armKind` at 50 lines with `arm_kind`
  finding 18 more, 14 carrying no `armKind`, and `FORBIDDEN_BY_ARM_KIND` matching
  `fingerprint.py:128`/`:136`. An unbounded `grep -rn FORBIDDEN_BY_ARM_KIND .` over the worktree
  (no `--include`, no `-maxdepth`, no path prefix) returns 33 hits in **7 files, every one of them
  a plan, review, history or kaizen document** — the symbol exists in no source file, at
  `c523a35`, at `8fc2341`, or now. A plan gate reads the working tree, so its counts were taken
  against a state never committed. The mechanism is independently confirmed twice over by entries
  in the same chunk, so it was promoted on those citations and the caveat is written into the
  section.

**What was re-derived, and with what instrument** (baseline pinned at `f5e8326`; `HEAD` never used
as a baseline):

- `redis-cli 7.0.15` against `localhost:6379`, **paired control**: a good command and a bad one,
  compared on exit status *and* which stream carries the text. `PING` → stdout `PONG`, exit 0;
  `NOTACOMMAND` → stdout `ERR unknown command`, stderr **empty**, exit 0; a syntax-error
  `GRAPH.RO_QUERY` → `errMsg:` on stdout, stderr empty, exit 0; `set -e` with `>/dev/null` runs on
  past it. The control that made this a finding rather than an observation: `redis-cli -p 6399
  PING` (nothing listening) → empty stdout, message on **stderr**, **exit 1**. So `$?` is not
  uniformly useless, which is precisely what lets the trap survive review.
- `python3` 3.12.3: intersecting every `ast.AST` subclass's `_fields` against
  `{target, targets, optional_vars, name, names, asname, arg, rest}` → **exactly 27 classes**,
  confirming `b7f1c3a2`'s completeness claim figure-for-figure.
- `git grep` at three pinned shas for the edit-table section's three re-runnable citations —
  `_widen` → 7 lines at `5878014`, `8fc2341` and `c523a35` alike, the 4 in `tests/test_stats.py`
  all `def test_…` lines and none a call site (`b1e7c2a4` exact);
  `isinstance(metric, BinaryMetric)` → exactly 3 `report.py` lines at `c523a35` with the two `m`
  spellings and two string-literal spellings where the entry says they are, and `.mean` returning
  exactly the three bare-`else` readers (`3f6c2a91` exact, every figure);
  `armKind` 67 / `arm_kind` 25 / 19 carrying neither-camelCase at `8fc2341` (my own numbers at my
  own sha, replacing `b1f0c7a4`'s un-re-derivable ones).
- `git archive` sandboxes of `model-bench` at `c523a35` and `8fc2341` run against
  `model-bench/.venv`: **475** and **472** tests collected, matching `2f8a5d13`'s "475 passed" and
  `b3f1c7a2`'s "passed all 472 tests" exactly, at the right shas. `9c4e2b71`'s two named tests
  exist at `c523a35` (`tests/test_fingerprint.py:368` and `:596`), and its "1 failed / 474 passed"
  is internally consistent with the 475 collected.
- `git show 6012ddb:skills/joern-cpg/scripts/pipeline.sh` → line **199** reads
  `redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$STAMP" >/dev/null`, verbatim the shape
  `6eeaa03e` names at the line it names. `9124a1f fix(joern-cpg): the stamp can now fail loudly`
  introduced the `rq()` helper that captures stdout and `case`-matches `errMsg:*|ERR\ *|WRONGTYPE*`
  — which is why no `MENTIONS`→`graph-dba` was tagged: the entry's actionable half is closed, by
  the review that produced the entry.

## 2026-09-08 — `kaizen_team` distillation pass 2, chunk D (unit U22): 11 entries, 8 promoted, 3 discarded

`cobb` processed the eleven `analyst`-produced `kaizen_team` entries dated 2026-09-03
(`claude/docs/plans/kaizen-distillation2-coordination.md` U22, chunk D of five). Every entry pinned
by **complete** `entryId` at read, count and clear, and this chunk is where that stopped being a
precaution. Three of its ids share `b7f` (`b7f3c1d2…`, `b7f2c1d4…`, `b7f3a1c2…`), and the last of
them turned out to be **a third verified collision in this graph, far worse than the two 8-character
pairs already on record**: `b7f3a1c2-5d84-4e19-9a6f-2c8e71d40b93` (this entry — `analyst`,
2026-09-03, mutation-kill counts) and `b7f3a1c2-5d84-4e19-9a06-3c2e8f14d7b0` (`architect`,
2026-09-07, grep-based done-conditions) share their **first 21 characters**, diverging only at index
21 (`6f` vs `06`). Both were live simultaneously; a match on any prefix up to 21 characters would
have cleared `architect`'s entry instead of this one. Verified after the clear: the `architect`
entry is intact. Evidence folded into `claude/cobb/kaizen/plan.md` K-021. Zero kept open, zero
`MENTIONS` tags. **No always-loaded prompt was touched**: `analyst.md` and `claude/AGENTS.md`
untouched, as in chunks A, B and C.

Every promoted claim was re-derived from primary sources, and in four cases the **citation** — not
the claim — was the thing that had to be re-run: a live `TestClient` probe on
`falkor-chat/server/.venv` (fastapi 0.139.0 / starlette 1.3.1), two `git archive` sandboxes
(`model-bench` at `ab91419` run against `model-bench/.venv`; `falkor-chat/server` at `2e27835` and
`2e27835^` run against `falkor-chat/server/.venv`), a paired background-Bash probe with a
no-pipe control, read-only Cypher against the live `reference` and `kaizen_team` graphs
(module `41811`), and live `curl` against LM Studio on `localhost:1234`.

**Promoted (8 entries → 5 files; three are edits to existing material):**

  - `b7f3c1d2-9a44-4e6f-8c21-5d0e7a9b3f18` + `4e21d9f7-8a03-4c6b-b512-9fd0a3e77c15` → **one merged
    section** in `claude/analyst/review-techniques.md` ("A guard derived from the artifact it guards
    is blind along the derivation axis"). Merged because the review action is identical: a
    parametrized test and an AST guard are both generated *from* the thing they check, and both fail
    green along the derivation axis. Both re-derived by mutation. **`b7f3c1d2`'s figures were right
    and my arithmetic was wrong** — I predicted 231 for a one-field deletion (two parametrized tests
    × one field) and the sandbox returned the entry's `230`, because
    `FORBIDDEN_BY_ARM_KIND["deterministic"]` is *derived* from `_MODEL_SCHEMA_1`, so one deletion
    removed three cases across two collections. Testing the instrument before filing the finding is
    what kept that out of the report. The section gained a distinction the entry did not have:
    shrinking the forbidden set **at the constant** gave `1 failed, 229 passed`, but only because a
    non-parametrized sibling spot-checks two named fields and I happened to remove one of them;
    shrinking it **at the parametrize site** gave `230 passed`, green. For `4e21d9f7` the three-way
    mutation reproduced exactly: alias spelling reddens, `shop._services.…` direct spelling and
    `shop.enqueue_turn(…)` one-hop-out both stay green. **What I could not reproduce is the entry's
    file-level `183 passed`** — the sandbox needs a live seeded workspace and returned 89-90 errors —
    so the promoted section cites the per-test results I did observe and no file-level count.
  - `b7f3a1c2-5d84-4e19-9a6f-2c8e71d40b93` → `claude/analyst/review-techniques.md`, new section on
    mutation-kill counts as draws from a distribution. The mechanism re-verified (7 distinct orders
    of a 5-element `str` set across `PYTHONHASHSEED=0..7`, CPython 3.12.3, no `pytest-randomly`),
    and **half the entry's remedy refuted**: "pin `PYTHONHASHSEED`" does not work for the very case
    it cites. `_PresenterSessions` mints `secrets.token_urlsafe(32)` values, so at a *pinned*
    `PYTHONHASHSEED=0` eight independent runs put three different tokens at index 0. A set of fixed
    literals under the same eight seeds is stable per seed. Promoted with that split stated. The
    entry's own `1,2,1,2,0,1,1,3` sweep is carried as its original observation, attributed and
    dated — I could not re-run it without falkor-chat's live environment.
  - `6b1f0d2e-9a4c-4f77-8c3d-1e5b7a02c941` → `skills/python-web-quirks/SKILL.md`, new section,
    **corrected in a way that makes it more useful**. The claim is true — `starlette.routing.Route`
    with `methods=['GET']` reports `{'GET','HEAD'}`, FastAPI's `APIRoute` reports `{'GET'}` — but the
    entry frames `HEAD` as special, and it is not: *every* non-`GET` method loses the partial match
    and falls through to a path-matching `Mount` (proved by swapping `StaticFiles` for a permissive
    ASGI app, which answered `MOUNT:OPTIONS`/`MOUNT:DELETE`/`MOUNT:PUT`). What is special is the
    *symptom*: `StaticFiles` serves `GET`/`HEAD`, so `HEAD` reads as **404** while the others get
    `StaticFiles`' own **405**, byte-identical to a real method rejection. A bare app with no mount
    answers `HEAD` with 405. The entry's own conclusion (a `(method, path)` handler table cannot
    `KeyError` on `HEAD`) holds, for the stronger reason.
  - `e77cd2a5-fed3-4fba-876c-27327d9df724` → `skills/python-web-quirks/SKILL.md`, new section.
    Routed to the shared skill rather than to `review-techniques.md` because it is a general
    Python/pytest mechanic that `coder`/`tdd-engineer`/`qa-engineer` need as much as a reviewer.
    Both halves reproduced: `pytest-timeout` absent from `falkor-chat/server/.venv`; a
    `daemon=True` thread with `join(timeout=1.0)` around a 30 s sleep returns at 1.00 s with
    `is_alive()` true; and with a ms-resolution clock and a no-op body, an outside stamp taken
    before `Thread.start()` **passes** the `started < finished` ordering assertion off a 300 ms gap
    while an inside stamp **fails** it. Added a caveat the entry lacks — a `perf_counter`-resolution
    clock hides the trap (the delta comes back ~1 µs and still passes).
  - `7c1e4b62-3a91-4d18-9f2c-5b0ad7e61c44` → `skills/agent-standards/claude-code.md`, new bullet
    under *Bash tool environment*. Verified **with a control**, which is what makes it a finding
    rather than an anecdote: the same 60 s flushed-line script backgrounded three times — piped
    through `tail -20` the task output file was **0 bytes at ~30 s**; with **no pipe** it held
    **7 lines at ~35 s**. So the task file streams and the pipe is the buffer. Mechanism stated
    (`tail -n N` cannot emit before EOF; `head -n N` blocks the same way below N lines).
  - `df03e2c1-86b7-4d9a-8b24-7710785226d4` → `claude/graph-dba/falkordb-quirks.md`, new bullet under
    *Cypher dialect & query behavior*. Expected to be a discard-a-fortiori against U21's already
    promoted `28d78725` — instead the read found that **U21's promotion left a dangling
    cross-reference**: `falkordb-quirks.md:59` points at "the `RETURN n.prop` → `null` entry under
    *Cypher dialect*", and no such entry existed. This entry is exactly that content, so promoting
    it resolves my own prior unit's defect. Re-derived read-only (no probe graph created): a missing
    property projects as `null` with `keys(n)` unchanged and nothing raised, and `CALL
    db.constraints()` on `reference` returns four `UNIQUE` rows and **no `MANDATORY` row**. The
    entry's downstream `productId: null` observation is attributed to it and dated — all 15 current
    `Product` nodes carry the key, so that row is gone.
  - `b7f2c1d4-9e63-4a58-8c21-5f0a7d3e9b16` → `claude/data-scientist/lm-studio-model-notes.md`.
    **Mostly already published** — the 10 catalog keys, `/v1/models`' three, `runtime`/`stats`/
    `model_info` living only on `POST /api/v0/chat/completions`, `loaded_context_length` absent while
    not-loaded, and `lms.exe`-only reachability are all in that file already. One clause was not:
    `POST /api/v0/embeddings` carries **none** of the three, verified live (response holds exactly
    `data`/`model`/`object`/`usage`), which is what makes "an embeddings-only arm can never populate
    a runtime field" a design constraint rather than an inference. Promoted as that clause. The
    re-measurement also refreshed two counts in the neighbouring bullet — the catalog is **16**
    models today, not 19 — while confirming the finding's shape is stable across the turnover (the
    same four named entries still lack `capabilities`), and gave `loaded_context_length` its first
    positive confirmation: the one model a probe had JIT-loaded carried the key.

**Discarded (3 entries, all already published — none falsified):**

  - `a3f1c9e2-7b4d-4a18-9c62-5e0d8f3b1a77` (FastAPI 0.139 does not flatten `include_router`) —
    `skills/python-web-quirks/SKILL.md` already carries this in more depth than the entry, including
    `route.original_router`, the `include_context.prefix` trap, the `Mount`/`Host` unclassified-entry
    trap, and the positive-control advice. Nothing to add.
  - `c47a83b0-52d1-4e6a-9f18-3d0c6b9e7a52` (per-route `responses={}` accepts `x-` extension keys) —
    same file, the *"escape hatch"* paragraph of the `responses={...}` section, which already states
    both reads (off `route.responses` and out of `app.openapi()`), the `include_router` case, and two
    caveats the entry does not carry.
  - `a7f3c1e2-9b04-4d5a-8e61-2c7f0d3b9a15` (the scratchpad path is not isolated between parallel
    sessions) — `skills/agent-standards/claude-code.md` already carries it, and **cites this exact
    incident** (`docs/reviews/small-model-benchmarking-impl.md` Appendix C.5, 2026-09-03) alongside
    a second, sequential-reuse observation that generalises it further. Published at the point of
    use, by an earlier unit of this same pass.

**No `MENTIONS` tags.** Each entry was dispositioned to completion in this run and every promotion
landed in the home that owns its subject — tagging would only ask another agent's pass to re-decide
a settled disposition. The two cross-agent promotions (`graph-dba`, `data-scientist`) are logged in
those agents' own `history.md` instead, per the precedent set in U19 and U21.

**Plan items:** none. Nothing was kept open, and no analyst-agent improvement surfaced —
`claude/analyst/kaizen/plan.md` untouched.

## 2026-09-08 — `kaizen_team` distillation pass 2, chunk C (unit U21): 10 entries, 6 promoted, 4 discarded

`cobb` processed the ten `analyst`-produced `kaizen_team` entries dated 2026-09-02
(`claude/docs/plans/kaizen-distillation2-coordination.md` U21, chunk C of five). Every entry pinned
by **complete** `entryId` at read, count and resolve — two full-eight-character prefix collisions
are now verified in this graph, so a short prefix is not a key here. Every claim was re-derived
from primary sources: live FalkorDB probes on a disposable graph (`cobb_u21_probe`, created and
`GRAPH.DELETE`d in this run) plus `GRAPH.INFO`/`GRAPH.CONFIG GET` against the shared dev instance
(module `41811`), a synthetic markdown-it reproduction on `/usr/bin/python3`, per-row md5 hashing
across all sixteen committed revisions of a plan document, and direct reads of
`falkor-chat/server/tests/conftest.py`, `falkorchat/repository.py`, `docs/QUERIES.md`,
`docs/SERVER.md` §1.7, `docs/HISTORY.md` and `scripts/verify_salesperson.sh` — never by confirming
an entry's own citation. Zero kept open. **Zero new bullets in any always-loaded prompt**:
`analyst.md` and `claude/AGENTS.md` untouched, as in chunks A and B.

The chunk's shape matches B's — most of these are engine, tooling and project facts `analyst`
discovered while reviewing rather than review doctrine, so each routed to the home that owns its
subject. **All four discards were falsified or overtaken by work that landed after the entry was
written**, which is what a six-day-old capture looks like in a repo moving this fast.

**Promoted (6 entries → 2 files; three are edits to existing material, not new sections):**

  - `28d78725-cff4-456b-8ad9-c637ac0b12de` → **`claude/graph-dba/falkordb-quirks.md`**, two new
    bullets under *Indexing, constraints & DDL*. The entry's bottom line holds — a `UNIQUE` on a
    nullable marker property is safe, not a re-join hazard — but re-derivation **sharpened its
    mechanism**: the entry frames absent and explicitly-null as two states the constraint exempts;
    they are **one state**. `CREATE (:Chan {name:'c', pid:null})` reports `Properties set: 1` and
    `keys(n)` returns `[name]` — FalkorDB discards a null at write and never stores it, so the
    constraint has nothing to compare rather than choosing to exempt anything. Confirmed alongside:
    two same-string nodes correctly rejected (`unique constraint violation on node of type Chan`),
    and `DETACH DELETE` → re-`CREATE` of the same value clean. The entry's second half was also
    corrected: it says `EXISTS { MATCH … }` and `exists((pattern))` "both fail to PARSE", and they
    fail at **different stages** — the first is a real parse error (*"Invalid input '(': expected
    ':', ',' or '}'"*), the second parses and dies at plan time (*"Unable to resolve filtered
    alias"*). Same practical verdict, different error to recognise. The working `OPTIONAL MATCH …
    WITH x, t WHERE t IS NULL` anti-join re-ran clean.
  - `94917906-3ae1-40c6-930d-14df73cfaa04` → **`claude/graph-dba/falkordb-quirks.md`**, new bullet
    under *Ops, config & tooling*. Re-derived live: `GRAPH.INFO` returns `# Running queries` /
    `# Waiting queries` / `Object Pool` sections and takes **no graph key** — it is instance-wide,
    a detail the entry omits — and `GRAPH.CONFIG GET MAX_QUEUED_QUERIES` returns `25`. The promoted
    form keeps the entry's actual point, which is about **done-conditions, not monitoring**: a
    capacity assertion can be written against observed queue depth instead of degrading to "no
    query was rejected", a condition that only reddens once the cap has already been hit.
  - `e2bf2057-956a-421c-8b8a-f6086814f651` → **`claude/analyst/review-techniques.md`**, new section
    *"An extract-and-execute loop over a doc's code blocks does not prove the doc is well-formed."*
    Genuine review doctrine, and the mechanism reproduces exactly: on a minimal document with one
    glued closing fence, a non-greedy ` ```lang(.*?)``` ` regex finds **2** blocks while
    markdown-it-py 3.0.0 (`MarkdownIt("commonmark")`, tables enabled, `/usr/bin/python3`) finds
    **1** fence spanning 8 lines and **2** headings instead of 3; inserting a newline before the
    fence restores 2 and 3. The original instance is **closed** — `docs/plans/salesperson-ui-graph.md`
    has no glued fence left (`grep -n '[^ ]```$'` → no matches) — so only the rule was promoted,
    carrying the synthetic reproduction as its evidence rather than the entry's now-unreachable
    455-line-block measurement.
  - `941edc18-26af-411b-99cc-87c4eb39082f` → **`claude/analyst/review-techniques.md`**, folded into
    the **existing** section *"Mutating a class-level constant via a pytest plugin proves a guard is
    load-bearing without touching source"* (promoted in an earlier pass, from a different entry —
    the technique was therefore already published). What was missing is the entry's **limit**, and
    it is the half that keeps a report honest: the seam rewrites *query text only*, so it can prove
    a Cypher-level guard load-bearing and says nothing about the Python-side behaviour around it —
    exception dispatch, row shaping, the branch deciding whether the query runs at all. Promoted as
    one paragraph telling the reviewer to state that boundary when reporting the result.
  - `c3e2f0a4-7b19-4d52-9a6e-2f8c1d40b7e5` → **`claude/analyst/review-techniques.md`**, folded onto
    that same section as its measurement-validity caveat, which is where it belongs: it is a
    warning about running *that* battery, not a standalone technique. Verified at source —
    `falkor-chat/server/tests/conftest.py:86-91`'s `conn` fixture calls `MATCH (n) DETACH DELETE n`
    on the single `ws:test` graph at **setup**, once per test, and both `repo` and `wf_repo` depend
    on it, so two concurrent pytest processes wipe each other mid-test. The promoted form keeps the
    entry's numbers as the signature to recognise (43 failed / 2338 passed with two dozen failures
    far from the mutation, versus 2381 passed / 14 deselected serially on the identical tree).
    **The `falkor-chat`-side home was considered and not taken:** the hazard is absent from
    `falkor-chat/docs/SERVER.md` §1.7, which is where that component's testing hazards live and
    which does document the sibling `wf_repo` `reference`-graph wipe. Adding it there is
    `falkor-chat`'s owner's call, not `cobb`'s or `analyst`'s — noted in `plan.md` rather than
    written into another component's docs from here.
  - `1d8276d8-1b4d-474f-8603-70814598a754` → **`claude/analyst/review-techniques.md`**, new section
    *"Per-row hashing turns 'is this plan stable enough to dispatch?' into evidence."*
    **`suggestedHome` was `prompt`; overridden to the on-demand knowledge base** — it is a technique
    with a mechanism and a worked command, not a rule that changes routing in most sessions, and
    `plan.md` K-003 already argues this genre belongs in `review-techniques.md` rather than the
    always-loaded body. **The method re-derived cleanly and its specifics did not**, which became
    the promoted caveat: run over **all 16** committed revisions of `docs/plans/salesperson-ui.md`
    in `acb5a2a^..069f6ae` (v1.16 → v1.31), 21 distinct step rows appear in total but `S7c` only
    enters at `732f5e0` (v1.19), so 3 revisions carry 20 rows and 13 carry 21; of the **20 present
    in all 16, 13 are byte-identical across the window**, while `S9` takes 13 distinct values,
    `S8`/`S13` four each and `S10`/`S12a` three each. (Corrected 2026-09-08 after `teco`'s
    re-derivation: the first write-up of this entry said "five committed revisions" — the number I
    had *sampled*, not the window — and a subsample can only over-report stability, so the figure
    was an upper bound presented as a measurement. The full recount happened to land on the same 13
    rows, which is luck, not justification.) The entry's own stable set (`S3`, `S6`) and churn set
    (`S7`/`S8`/`S10`/`S12a`) only partly
    survive into this later window — `S3` still stable, `S6` no longer, `S7` now stable — so *which*
    rows are stable is a property of the revision window, and a previous pass's stable-row list must
    never be carried forward as a finding. The entry's second half — a completeness table keyed on
    `(response → rule)` reintroduces the over-generalisation it was added to prevent, while
    `(route, response)` makes the collision unexpressible — was promoted with it as a companion
    trap: check a completeness table's **key** before its rows.

**Discarded (4), every one falsified or overtaken:**

  - `1c900356-0c84-4086-b48d-bda14c9c6ce8` (the `n.prop > ''` "always-true" index-anchor conjunct)
    — **already documented, in more depth than the entry.** `claude/graph-dba/falkordb-quirks.md`
    carries it twice: the *Cypher dialect* bullet on cross-type comparison (`RETURN 42 > ''` →
    `NULL`, verified 2026-09-07) and the *Query tuning* bullet that names the idiom, says **"Do not
    use it"**, and gives both of the entry's reasons — the silent row-drop and the absence of any
    selectivity win. That bullet's own header records that it "merges and **corrects** two raw
    `kaizen_team` entries", so this material has already been through curation twice. The primitive
    was re-confirmed anyway (`RETURN 42 > ''` → `null`, `'p-aaa' > ''` → `true`) and the entry's
    timing figures point the same way as the file's. Nothing left to add.
  - `74f8b159-fd5a-41a5-9b7b-cb34f29d5d54` (falkordb-py disables retry) — **already documented and
    already corrected.** The entry's framing, that `falkordb-py` "builds its Redis connection with
    `retry._retries == 0`", is true only of the pooled `Connection`; `FalkorDB(...).connection` is
    the `redis.Redis` *client* and raises `AttributeError` on `.retry`. `falkordb-quirks.md` now
    carries the corrected account under *Ops* — the split is **which object you hold**, identical
    on falkordb-py 1.6.1 / redis-py 8.0.1 (`falkor-chat/server/.venv`) and 1.6.2 / 8.1.0
    (`cypher-mcp/.venv`), so it is not a version fact — landed by U20 after `teco` re-derived it on
    both venvs. The entry's second half (redis-py's `TimeoutError` and `ConnectionError` are
    siblings under `RedisError`, so `except ConnectionError` cannot swallow a socket timeout) is
    already published in `skills/python-web-quirks/SKILL.md`. Not re-promoted, per the U21 brief.
  - `4422cd43-1a9d-4b73-9f27-cd0a4ef2c128` (a workspace `WorkflowDefSnapshot` can diverge from
    `proof_defs.py` with **no tool able to detect it**) — **falsified: the tool now exists.**
    `falkor-chat/scripts/verify_salesperson.sh` carries a **check 6** that diffs every stored step
    `config`, on both sides, against the shipped `falkorchat.proof_defs` constant, and its header
    comment states the blindness it closes in the entry's own terms ("Check 3 compares the two sides
    against EACH OTHER, so it is blind to…"). The transferable rule had also already been promoted,
    in an earlier pass, as `review-techniques.md`'s *"Live graph/database state has no git
    provenance — and a two-sided diff cannot detect common-mode staleness."* Both halves closed.
  - `1de41323-0401-481f-900f-b1c8a50c5752` (the `§N` section-header convention is "unwritten — it
    lives only in the code") — **falsified on both clauses.** The convention is written down:
    `falkor-chat/docs/HISTORY.md:780-782` states "the file convention is §N = QUERIES.md §N", and
    `repository.py`'s module docstring carries its basis ("Each method maps 1:1 to a verified query
    in `docs/QUERIES.md`"). The specific defect is fixed too: the offending header was relabelled to
    `# ── Structured NL query generation (K-055 M6) — no QUERIES.md section ──`
    (`repository.py:3784`, the number dropped entirely), `QUERIES.md` now has a real
    `## 18. Storefront participants & resets`, and `repository.py:3141`'s header cites it
    explicitly. A duplicate-header sweep over `repository.py` returns only `§3`, which is correct —
    Channels and Threads are both QUERIES.md §3.

**No `MENTIONS` edge was added.** Two entries (`28d78725`, `94917906`) are substantively about
FalkorDB and belong to `graph-dba`'s subject, and both were promoted **into**
`claude/graph-dba/falkordb-quirks.md` in this run, with the landing recorded in
`claude/graph-dba/kaizen/history.md`. A `MENTIONS` edge exists to surface an entry in the mentioned
agent's *future* pass; tagging one whose content has already been published into that agent's own
knowledge base would only schedule a re-read of a resolved entry.

**Graph:** all 10 entries were current-shape with exactly one `PRODUCED` edge and no `MENTIONS`
(`producedEdges = 1`, `mentionEdges = 0`, so `otherRemaining == 0` on each), and each was cleared
with the curator full-node shape **after** this entry was on disk. Post-clear census is in
`claude/cobb/kaizen/history.md`.

## 2026-09-08 — `kaizen_team` distillation pass 2, chunk B (unit U20): 11 entries, 7 promoted, 4 discarded

`cobb` processed the eleven `analyst`-produced `kaizen_team` entries dated 2026-08-31..2026-09-02
(`claude/docs/plans/kaizen-distillation2-coordination.md` U20, chunk B of five). Every entry pinned
by **complete** `entryId` at read, tag, count and resolve. A second verified full-eight-character
prefix collision sits inside this chunk — `b3f2a6d4-9e1c-4a2b-8f7d-2c6e1a9b5d40` (09-01, settings
merge) vs. `b3f2a6d4-8c1e-4a7f-9d2b-1e6f5a0c3d7a` (08-29, `conftest.py` fixtures, discarded and
deleted in chunk A) — different facts, dates and subjects; a `STARTS WITH 'b3f2a6d4'` probe
returned the 09-01 row only, confirming the 08-29 node is gone. **Eight characters is not a key.**
Full untruncated text read via `redis-cli --no-raw GRAPH.RO_QUERY`. Every claim was re-derived from
primary sources — live FalkorDB probes on a disposable graph (`cobb_u20_scratch`, deleted after),
in-venv construction of a real FastAPI app and uvicorn middleware, a live `curl` of the LM Studio
catalog, a marker-package import experiment, and fresh `WebFetch` of three Claude Code doc pages —
never by confirming an entry's own citation. Zero kept open; **zero new bullets in any
always-loaded prompt** (`analyst.md` and `claude/AGENTS.md` both untouched; `claude/AGENTS.md`
sits at 2,434 w against its own ~2,500 bar and nothing here needed the headroom).

As in chunk A, most of these were not review doctrine — they were harness mechanics, framework
behaviour and box facts that `analyst` merely *discovered* while reviewing, so each routed to the
home that owns its subject rather than the one that produced the note. Only one entry was genuine
`analyst` doctrine, and it went to the on-demand `review-techniques.md`, not the prompt.

**Promoted (7 entries → 5 files; four are edits to existing material, not new sections):**

  - `e1a2c3d4-5b6f-4a7c-8d9e-0f1a2b3c4d5e` → **`claude/analyst/review-techniques.md`**, new
    section *"A document that adds a member to its own taxonomy is swept table-by-table, not
    changelog-by-changelog."* The only entry in the chunk that is review technique. Two halves:
    enumerate the taxonomy-keyed tables yourself (`grep -n '^|' <doc>`) instead of reviewing the
    sections the amendment's changelog names; and rank them, because a table the document declares
    another document **copies from** does not merely go stale — it **inverts the claimed
    provenance**, so a later reader reconciling the two edits the wrong file. The instance is
    closed, which is why only the rule was promoted: `docs/plans/doc-reference-convention.md` is
    now **v1.5.1** and its own changelog credits this finding ("Major (§24), fixed"); §9.6's row
    reads `requirements/*`, `manuals/*` (`:1590`), matching root `AGENTS.md`.
  - `b3f2a6d4-9e1c-4a2b-8f7d-2c6e1a9b5d40` → **`skills/agent-standards/claude-code.md`**, new
    bullet placed **immediately before** the existing "Placement" bullet in `## Hooks`, so the run
    reads *how files combine* → *where to put your rule*. Re-verified against
    `code.claude.com/docs/en/settings` (fetched 2026-09-08), which carries the entry's quote
    verbatim, plus the five-level stack and a whole `### Lists merge instead of overriding`
    subsection. **The promotion added three things the entry lacked.** (1) The four exception keys
    that do *not* merge (`fallbackModel`, `modelPicker`, `availableModels`, `modelSettings`).
    (2) The explicit fall-through consequence the entry only implied — a key a higher file *omits*
    resolves to the next file that sets it, which is exactly the question the entry was written to
    answer. (3) **The correction that matters: merging is not winning.** `code.claude.com/docs/en/
    permissions` states *"Rules are evaluated in order: deny, then ask, then allow. The first match
    in that order determines the outcome, and rule specificity doesn't change the order"* — so the
    unioned list is then evaluated by **rule type**, and an `allow` in `settings.local.json` cannot
    beat an `ask` in the shared project file despite the local file's higher precedence. Read
    alone, "list keys merge across files" invites exactly the wrong inference, and it bears
    directly on the adjacent U18b classifier entry and the Placement bullet it now precedes.
  - `e1f8a2c4-3b6d-4a2e-9c1f-7d5e8a0b3f42` → **`skills/agent-standards/claude-code.md`**, sharpened
    in place. Mostly already promoted (the file has carried the toggle since the 2026-09-01 Gen 4
    work, at two places, with a richer account of the failed live test). Two residuals were worth
    the edit, both re-verified 2026-09-08: the file's quote was **truncated** at *"in every kind of
    session,"* and the real sentence ends *"…and whether or not fork mode is on"*; and the
    docs inconsistency the entry flagged **still holds** — `/docs/en/env-vars` does not list
    `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` at all, while it *does* document the opposite lever
    `CLAUDE_AUTO_BACKGROUND_TASKS`, so a reader who searches only the reference page concludes no
    disable toggle exists. Dropped from the entry: its closing clause blaming a prior `cobb`
    investigation for overlooking the lever — provenance, not rule.
  - `d78fbe9e-b7fa-40dc-886a-d0694ccd631b` → **`skills/agent-standards/claude-code.md`** §
    "Bash tool environment", folded into the existing shell-shadowed-`find`/`grep` bullet as a
    third instance. Re-derived: `type rg` → a function that `exec -a rg`s
    `${CLAUDE_CODE_EXECPATH:-…/claude}` with a `command rg` fallback; `which rg` prints nothing and
    exits 1 while `rg …` works. The promoted framing separates it from its two siblings — `find`
    and `grep` wrap *different tools* (`bfs`, `ugrep`) so their hazard is **behavioural**, whereas
    `rg`'s is **detection**: any check or plan done-condition gating on `which <tool>` reports
    ripgrep as not installed here (`type` and `command -v` both see it; `which` is the one that
    can't). **The entry's second half was discarded, not promoted** — `--glob '!docs/**'` being
    root-anchored is documented ripgrep/gitignore glob semantics (a pattern containing a slash
    anchors to the root), and the review lesson it drove is already carried a fortiori by
    `analyst.md`:84 ("a plan's prescribed acceptance-check command is a claim too — run it verbatim
    before approving"), which needs no second illustration in an always-loaded prompt. The
    mechanism reproduces; **the entry's counts do not, and are stale rather than wrong** — its
    36/21/15 was measured 2026-09-02, and today `rg -n 'salesperson/' --glob '!docs/**'` returns
    66 lines of which 28 are under a `*/docs/` path, against 40 for `--glob '!**/docs/**'`
    (the `salesperson-ui` work landed in between). `rg --files --glob '!docs/**' | grep -c /docs/`
    = **246**; the same with `'!**/docs/**'` = **0**.
  - `19690317-a178-48fc-bbbd-a4dead3ee564` → **`skills/python-web-quirks/SKILL.md`**, split in two
    because the entry bundles a framework fact with a framework fact, neither of them
    falkor-chat-specific despite the entry's title. Both re-derived in `falkor-chat/server/.venv`
    (fastapi 0.139.0, starlette 1.3.1, uvicorn 0.49.0, anyio 4.14.1, redis-py 8.0.1, py 3.12.3).
    (a) **The anyio half sharpened an existing sentence in place** — the `BackgroundTasks` section
    already said "roughly 40 concurrent worker threads by default"; it now states the measured
    `total_tokens == 40`, that it is settable, and the operative constraint the hedge was hiding:
    `current_default_thread_limiter()` is **event-loop-scoped** and raises
    `anyio.NoEventLoopError` outside a running loop, so a bump written at import or in a `main()`
    preamble dies rather than applying — "raise the limiter at startup" has not yet said *which*
    startup. (b) **The uvicorn half is a new section, and the verified half is not the half the
    entry leads with.** Confirmed: `Config` defaults `proxy_headers=True` and resolves
    `forwarded_allow_ips` to `os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")`; `::1` is not in
    that trust list. Driving `ProxyHeadersMiddleware` directly, a **loopback peer that sends any
    `X-Forwarded-For` is rewritten away from loopback** — peer `("127.0.0.1",1234)` + XFF
    `203.0.113.9` → the app sees `("203.0.113.9", 0)` — unconditionally, with no proxy in the
    picture. That kills a `client == "127.0.0.1"` gate on its own. **What I dropped is the entry's
    word "fully invertible":** the other direction did *not* reproduce — a remote peer claiming
    `X-Forwarded-For: 127.0.0.1` stays remote, because `_TrustedHosts.get_trusted_client_address`
    walks the header **right-to-left and returns the first untrusted host**, so an appending proxy
    defeats an attacker-supplied left-hand entry. The spoof needs `forwarded_allow_ips="*"` or a
    proxy that passes client-supplied XFF through. Both directions are in the section, labelled.
  - `3f7c1a92-5d64-4b0e-9c31-8ae2f0d47b15` → **`claude/data-scientist/lm-studio-model-notes.md`**,
    new bullet inside the existing `/api/v0/` section (which already listed `capabilities` among
    the returned fields but said nothing about trusting it). Re-derived six days on, same box,
    `curl -s :1234/api/v0/models`, 19 models — and the re-derivation **strengthened** the claim.
    The entry says the field is unreliable; the measurement says it carries **no discriminating
    information at all**: every entry that has the key holds exactly `["tool_use"]` and none holds
    anything else, so the field never says *no*. It says `["tool_use"]` for the embeddings model
    `text-embedding-qwen3-embedding-0.6b`, and it is absent from **four** entries (the entry named
    two), spanning both kinds — `google/gemma-3-4b` and `google/gemma-3-12b` (`vlm`),
    `gemma-3-4b-vl-it-…` (`llm`), and a second embeddings model
    `text-embedding-nomic-embed-text-v1.5`. So presence does not imply a chat model and absence
    implies nothing; gate on `type` ∈ {`llm`,`vlm`}. The entry's `loaded_context_length` aside also
    reproduced (absent from all 19 while `state == not-loaded`) and went in beside it. **Routed to
    the notes file, not to the plan it was about** — `docs/plans/small-model-benchmarking.md` §3.6
    is held by a concurrent session and was out of bounds for this unit; the fact is now where
    `data-scientist` reads it, and the plan-side change is someone else's call.
  - `d8039ade-c9ae-4be0-83eb-681dc2f0b5d5` → **`claude/graph-dba/falkordb-quirks.md`**, sharpened
    in place. **The entry's headline was already documented** — the existing bullet has said
    "Default `TIMEOUT` is 1000ms — and writes ignore it entirely" since before this entry was
    written. Re-measured anyway on module `41811` (`TIMEOUT 1000`, `TIMEOUT_DEFAULT 0`,
    `TIMEOUT_MAX 0`): the entry's own write query ran **1.75 s** untouched, and a 4-way cartesian
    `MATCH` over 400 nodes was killed with `Query timed out` at **exactly 1.00 s**. What was **not**
    documented is the entry's second clause — and the entry's own evidence never demonstrated it
    either, so it had to be tested rather than believed. It holds: a
    `redis.Redis(socket_timeout=0.5, retry=Retry(NoBackoff(),0), retry_on_timeout=False)` client
    raised `redis.exceptions.TimeoutError` at 0.50 s on that write and **all 4 nodes were
    committed** — there is no cancel path. **The test also surfaced something neither the entry
    nor the file had:** the *same* call through a stock `redis.Redis(...)` left **36 nodes** where
    the query creates 4, because redis-py 8.0.1's default connection carries
    `Retry(ExponentialWithJitterBackoff(), retries=10)` whose supported set includes
    `TimeoutError` — a non-idempotent write over a short client timeout multiplies silently. The
    lab is not exposed by default and the reason is worth recording: `falkordb-py` disables retry
    — its pooled `Connection` reports `retry._retries == 0` while the `FalkorDB(...).connection`
    client has no `.retry` at all and `get_retry()` → `None` (verified on falkordb-py 1.6.1 and
    1.6.2 alike, so an object difference, not a version one) — so `falkor-chat`
    (`db.py:44`, `FALKORDB_SOCKET_TIMEOUT` default 10 s) and `cypher-mcp` (`server.py:903`) both
    get one attempt; a helper script reaching for bare redis-py does not inherit that.

**Discarded (4 — each already covered, at least a fortiori, by something that also covers a case
the entry misses; two additionally lead with a mechanism that is wrong):**

  - `a3f1c2b4-6e8a-4b2f-9d7c-1e5f8a0b2c3d` (editable install maps `falkorchat`) — **its lead clause
    states the mechanism chunk A refuted.** "…whose finder maps `falkorchat` straight to the live
    source dir **regardless of PYTHONPATH**" is the finder-beats-`sys.path` model that
    `review-techniques.md` was carrying wrongly until chunk A corrected it; only the entry's
    subordinate clause ("a PYTHONPATH override only wins if cwd is NOT `server/` itself") and its
    evidence are right. Re-derived today with a marker package under `env -i`: `sys.meta_path` is
    `[BuiltinImporter, FrozenImporter, PathFinder, _EditableFinder]`, and `import falkorchat`
    resolved to the **real** source from cwd `falkor-chat/server` with `PYTHONPATH=<shadow>`, to the
    **shadow copy** from the repo root and from `/tmp` with the same `PYTHONPATH`, and to the real
    source from the repo root with `PYTHONPATH` unset — i.e. cwd → `PYTHONPATH` → the finder's
    hardcoded absolute `MAPPING`, exactly as the corrected file states. **A methodology note worth
    keeping:** the first run of this check appeared to *refute* the correction, and the refutation
    was a bug in my own harness — `cd` persists across commands inside one Bash call, so the
    "repo root" arm silently ran from `server/` too. Isolating each invocation in its own subshell
    reproduced the corrected order first try. Discarded because `review-techniques.md`
    § "Verifying an uncommitted diff without mutating the working tree" (a) already carries the
    full three-step order **and** both consequences, including the operational rule this entry
    exists for.
  - `a3f0c8e2-2b1d-4e6a-9c7f-1d8b6a5e4c3f` (repo-root `.mcp.json` is git-tracked) — true
    (`git ls-files .mcp.json` → tracked; `git check-ignore` → exit 1; last touched by `59a03c4`),
    and already covered a fortiori by `skills/agent-standards/claude-code.md` § "Scopes,
    precedence, and the approval gate", whose scope table labels `project` scope
    "**`.mcp.json` at the repo root** … the team, version-controlled". The table also supplies the
    **answer** the entry only asks for: `local` scope (`~/.claude.json`, "you, this project —
    **untracked**") is the per-session, non-committed place for exactly the sandbox-provisioning
    entries the reviewed plan proposed adding to the tracked file. Spelling out the inference in a
    900-line reference the reviewer already loads is restatement, not coverage.
  - `f3a1c2e4-9b7d-4e2a-8c6f-1d5e7a9b3c02` (prefer file-scoped `pytest -q tests/test_llm.py`) —
    **its mechanism is wrong in the same way chunk A's `wf_repo` entry was.** The wipe does not run
    "at teardown": `tests/conftest.py`'s `wf_repo` is a plain return fixture whose body runs
    `db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")` **before returning**, i.e. at
    setup, exactly as `falkor-chat/docs/SERVER.md` §1.7 already documents ("at fixture **setup**
    … the wipe never runs at teardown, [so] a finished pytest session leaves the *last* workflow
    test's own published defs sitting in `reference`"). The consequence and the re-seed obligation
    are documented twice over — SERVER.md §1.7 plus `falkor-chat/AGENTS.md`:108-111 — and SERVER.md
    additionally gives the strictly safer form the entry never reaches for
    (`pytest --collect-only -q`, no connection and no writes). The residual advice ("run only the
    file you changed") follows directly from the documented mechanism; `tests/test_llm.py` requests
    no `wf_repo`/`conn` and collects 55 tests, matching the entry. Verified by reading, **not** by
    running the suite — running it is the hazard.
  - `7e3b1c4a-9f52-4d18-b0a6-2c5d8e114f37` (FastAPI 0.139 `_IncludedRouter`) — already promoted, and
    `skills/python-web-quirks/SKILL.md` § "Asserting over a FastAPI app's route table" carries it
    **richer** than the entry: the opaque wrapper and the zero-paths reading, the prefix living on
    `route.include_context.prefix` while the inner `.path` stays pre-prefix, nested composition,
    plus three things the entry has not — the false-green framing (the naive assertion is
    *unfalsifiable*, not merely wrong), the `Mount`/`Host` entries that break a flatten, and the
    positive-control rule. Re-derived from a synthetic app rather than `create_app` (whose module
    is dirty in another session's tree): `app.routes` = 4 `Route` + 1 `_IncludedRouter`,
    `getattr(inc, "path", "<none>")` → `<none>`, `include_context.prefix` → `/shop/api`, inner
    `.path` → `/join`, `app.openapi()["paths"]` → `/shop/api/join`.

**`MENTIONS` tags added: none.** Every one of the eleven resolved to its home this pass — the
seven promotions landed in the owning agent's or skill's file directly, and the four discards are
covered where they already sit — so a tag would have created empty work in another agent's queue
rather than cross-agent visibility, the same judgement chunk A applied. The pre-existing
`MENTIONS`-only `analyst` entries from earlier units were out of scope and untouched.

**Clearing:** count-and-decide per `agent-maintenance` skill §5 on each of the eleven. This history
entry landed on disk and was confirmed **before** any graph mutation — which mattered: this unit
was interrupted twice by platform limits, and both times the invariant meant nothing was lost.

**One finding routed outward, not fixed here (unchanged from chunk A):** nothing in chunk B bears
on `falkor-chat/docs/plans/oversized-indexed-property-guard-graph.md` or on `RELATIONSHIP`-type
constraints — checked by grepping that document for every subject in this chunk (timeout, socket,
rollback, uvicorn, editable, `include_router`, capabilities, settings.json): zero hits. The stale
grep-evidence finding chunk A reported there stands exactly as reported, no more and no less
urgent.


## 2026-09-08 — `kaizen_team` distillation pass 2, chunk A (unit U19): 12 entries, 5 promoted, 7 discarded

`cobb` processed the twelve `analyst`-produced `kaizen_team` entries dated 2026-08-25..2026-08-30
(`claude/docs/plans/kaizen-distillation2-coordination.md` U19, chunk A of five). Every entry pinned
by **complete** `entryId` at read, tag, count and resolve — this graph contains a verified
full-eight-character prefix collision (`b7e41c92-3f8a-…` vs `b7e41c92-3d5a-…`), so a prefix is not a
key. Full untruncated text read via `redis-cli --no-raw GRAPH.RO_QUERY`. Every claim was re-derived
from primary sources — live FalkorDB probes on a disposable graph (`cobb_u19_scratch`, deleted
after), a marker-package import experiment, and direct reads of current source — never by
confirming the entry's own citation. Zero kept open; **zero new bullets in any always-loaded
prompt** (`analyst.md` and `claude/AGENTS.md` both untouched: `claude/AGENTS.md` sits at 2,434 words
against its own ~2,500 bar and nothing here needed the headroom).

Most of these were not review doctrine at all — they were FalkorDB dialect facts, `falkor-chat`
codebase facts and harness mechanics that `analyst` merely *discovered* during a review, so they
routed to the home that owns the subject, not the one that produced the note.

**Promoted to `claude/graph-dba/falkordb-quirks.md`** (3 entries, all live-verified 2026-09-08
against module `41811` on a disposable graph; the established maintainer-edits-another-agent's-
knowledge-base channel, logged in `claude/graph-dba/kaizen/history.md` too):
  - `71c2f839-09a3-415e-a51b-eb3ed919f737` — two `FOREACH` clauses chain back-to-back after a
    single `WITH`, and each sees property values written earlier in the same query by a preceding
    `SET` *or* a preceding `FOREACH`. Measured directly rather than via the entry's `falkor-chat`
    test: `SET d.remaining = d.remaining - 1 WITH d FOREACH(…d.remaining = 0…| SET
    d.status='ready') FOREACH(…d.remaining > 0…| SET d.status='running')` returned `1/running` on
    `2 -> 1` and `0/ready` on `1 -> 0`; cross-`FOREACH` visibility proven separately with a
    `SET d.flag = 7` in the first clause guarding the second. Folded into the existing `FOREACH`
    bullet, which already covered nesting and multi-`CREATE` bodies but not chaining.
  - `e1c2a7b4-4f2a-4b9e-9c3d-2f6a8b1d5e70` — `=~` is a hard error ("FalkorDB does not currently
    support =~"), not a degraded match. Reproduced verbatim. New bullet; the file had **no** regex
    entry at all. Carries the `toLower(...) CONTAINS` replacement, including the
    `any(k IN keys(n) …)` whole-node form.
  - `7c3e1f2a-9b4d-4e21-8a6f-2d5b6e7c9a10` — duplicate result columns are rejected
    ("Error: Multiple result columns with the same name are not supported"), reproduced for both
    `RETURN d.id, d.id` and `RETURN count(d), count(d)`. **The entry's mechanism was corrected
    before promotion:** it claimed the error is raised "at query time — not at parse/compile time",
    but `GRAPH.EXPLAIN` on the same text errors identically, so it is raised during server-side
    validation and no rows are ever produced. What survives is the consequence — there is no
    client-side parse step, so the caller still meets it as a `ResponseError` at call time.
    Folded into the existing `result.header` bullet (same subject: un-aliased column naming). The
    `falkor-chat` half was already documented a fortiori at the point of use — `querygen.py`'s
    duplicate-`returns` guard comment names the exact error string *and* why `QueryGraphDataTool`'s
    try/except cannot catch it.

**Promoted to `claude/analyst/review-techniques.md`** (2 entries, both as edits to existing
material rather than new sections where one already covered the ground):
  - `a2f1c8e4-3b7d-4e1a-9c2f-6d8b5a7e0f11` — a `git worktree` isolates an editable-installed
    package only when Python runs with cwd inside the worktree's own package-parent directory.
    Confirmed, and **the promotion also fixed a wrong mechanism this file had been carrying**:
    §"Verifying an uncommitted diff" (a) and (d) both asserted that the editable-install
    `MetaPathFinder` "is consulted before `sys.path`", so a `PYTHONPATH`-prepended copy cannot
    shadow it. Measured: `install()` does `sys.meta_path.append(_EditableFinder)` and the live
    order is `[BuiltinImporter, FrozenImporter, PathFinder, _EditableFinder]` — `sys.path` wins,
    and a `PYTHONPATH` shadow *does* take (marker package resolved to the scratch copy). The real
    cause of the originally-observed failure is **cwd precedence** (`sys.path[0] == ''`): from the
    real `server/` the cwd entry outranks `PYTHONPATH`. Both passages now state the measured
    order — cwd, then `PYTHONPATH`, then the finder's hardcoded absolute `MAPPING` — which is also
    exactly what makes the worktree-repo-root case silently pull in the main tree. Also tagged
    `MENTIONS -> devops` (below): the environment-level half is venv/dependency territory and
    `claude/devops/ops-quirks.md` carries no Python-import entries at all.
  - `a1e6c8f0-2b7d-4e3a-9c1f-6d5b8a2f7e01` — a reused write-query precedent carries its `NULL`
    contract with it: `SET x = $x` (NULL clears) and `SET x = coalesce($x, x)` (NULL means leave
    unchanged) look alike and are not interchangeable, and plain Python `None` cannot distinguish
    "omitted" from "explicitly null" once it reaches the query params. The *project* fact is
    already documented better than the entry states it — `falkor-chat/docs/QUERIES.md` §17.1 sets
    the two contracts against each other by name, adds the "never pass `''` to mean not provided"
    corollary and records live verification in both partial-update directions — so only the
    plan-gate **check** was promoted, as a new section pointing at §17.1/§13.1 rather than
    restating them.

**Discarded** (7 — each already covered, at least a fortiori, by something that also covers a case
the entry misses; three additionally had a false or misattributed claim):
  - `b3f0e7d4-6b1a-4b2f-9c3a-2f8a7d1e5c6b` (api.py `/diff` has no `_read_or_absent` wrapper) —
    the defect is fixed and documented at the point of use: `repository.py:1823-1831`'s docstring
    states the empty-key catch, names `Services.diff_def_snapshot` as one of the callers the
    uncaught `ResponseError` used to escape to, and covers the *other* callers the entry never
    mentions. `services.diff_def_snapshot`'s own docstring documents the one-side-missing 200.
  - `a1f3c9d2-6e4b-4a7c-9d1e-2b5f7c8a3e11` (`git stash push` blocked by the auto-mode classifier) —
    `skills/agent-standards/claude-code.md` § Hooks has carried this since 2026-07-31, a month
    *before* the entry, in a stronger form (`--keep-index` included, plus the design consequence:
    prefer a substitute that needs no working-tree write). U18b's "a classifier denial is an event,
    not a state" entry independently removes the value of a second n=1 observation.
  - `7c2e9b1a-4f3d-4e2a-9c7b-6a1d5f8e2b30` (`wf_repo` wipes the `reference` graph per test) —
    `falkor-chat/docs/SERVER.md` §1.7 documents it a fortiori, adding that the wipe runs at fixture
    **setup** and never at teardown, so a finished run leaves the last workflow test's own defs in
    `reference`. `falkor-chat/AGENTS.md` carries the default-vs-`-m live` split and the review-safe
    subset; `conftest.py`'s own fixture docstring states it at the point of use.
  - `b3f2a6d4-8c1e-4a7f-9d2b-1e6f5a0c3d7a` (`repo`/`conn` run against a real `ws:test`) — true, and
    already implied where it matters (`falkor-chat/AGENTS.md`'s review-safe-subset bullet is keyed
    on "requests no `conftest.py` fixture that reaches a real `conn`/`wf_repo`"). Its
    distinguishing half is **false**: "only `test_services.py`/`test_tools.py` use in-memory
    fakes/stubs" — `test_tools.py` has 3 of its 60 test functions requesting the real `repo`/`conn`
    fixtures (`:978`, `:1038`, `:1058`, all `*_live`), while `test_services.py` has zero. AGENTS.md
    naming `test_services.py` alone as the safe subset is correct and the entry is not.
  - `7e6a9c1d-3b2f-4e5a-9c8b-2a1f6d4e8b7c` (querygen resolves WHERE values by exact match) — the
    bottom line holds but **the cited evidence does not**: the "never coerced/fuzzy-matched"
    comment is an inline note on the `label` field declaration, not on filter *values*. The actual
    guarantee is stronger and type-level — `QueryFilter.op` is
    `Literal["=", "<>", "<", "<=", ">", ">="]` with its own comment ("no `contains`/regex ops in
    v1") and `value` is always a bound parameter — so the DSL has no substring operator to
    accidentally match with. Already enforced and documented at the point of use.
  - `b6e29d17-4a5c-4f0b-8e3a-1c9d7f2b6a44` (a seed script's stale hardcoded version default) —
    the instance is repaired (`seed_salesperson.sh` header and body both read `v7`) and the class
    is covered a fortiori twice over: `review-techniques.md` § "Live graph/database state has no
    git provenance" §2 (a two-sided diff cannot see common-mode staleness) and
    `falkor-chat/AGENTS.md`'s `verify_salesperson.sh` row, which documents check 6 as the only one
    that can see a version whose stored `config` came from an earlier or uncommitted tree, plus the
    fact that `config` is create-only so re-seeding cannot repair it.
  - `f3a1b2c4-9e5d-4a7b-8c3e-1d2f6a9b7e40` (a design doc's cited grep goes stale) — not discarded
    outright: the *rule* was already there ("A 'this already exists' claim is a grep away from
    confirmation" already tells the reviewer to re-run the cited grep), so instead of a new section
    that section's lead was **sharpened in place** to name staleness as a second, distinct failure
    mode — an honestly-run negative grep that decayed reads exactly like a current one — and the
    entry's instance was added as Origin (3). Re-derived rather than taken on trust, with a live
    consequence: `scripts/bootstrap_schema.sh:265` does now carry
    `UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId` (added 2026-08-24 by `8d7dcfb`, K-050
    fusion), and `falkor-chat/docs/plans/oversized-indexed-property-guard-graph.md:205` — **still
    `Status: active`, owner `graph-dba`** — still asserts "→ no matches". Reported to the
    coordinator; not `cobb`'s document to edit.

**`MENTIONS` tags added:** one — `a2f1c8e4…` → `devops`. Its `PRODUCED` edge was resolved and the
node left live for `devops`'s own pass to judge whether `ops-quirks.md` should carry the Python
import-resolution order. The other eleven were resolved to their homes this pass, so tagging them
would have created empty work rather than cross-agent visibility. The four pre-existing
`MENTIONS`-only `analyst` entries from earlier units were out of scope and untouched.

**Clearing:** count-and-decide per skill §5 on each of the twelve. This history entry landed on
disk and was confirmed **before** any graph mutation.


## 2026-09-07 — `review-techniques.md` gained two verification techniques from `coder`'s kaizen distillation (U11)

- **What:** Two new sections, promoted by `cobb` out of `coder`'s `kaizen_team` capture (unit U11,
  chunk B).
  1. **Live graph/database state has no git provenance — and a two-sided diff cannot detect
     common-mode staleness.** `git log --all -S` structurally cannot establish where stored state
     came from, because the code that wrote it may have been an uncommitted working tree; and a
     gate that compares two *derived* artifacts against each other (`services.diff_def_snapshot`:
     `reference` vs. the workspace snapshot) is blind to staleness they share. Filed beside the
     existing "An untracked plan/review doc has no re-verification baseline" — same family.
  2. **Mutating a class-level constant via a pytest plugin (`-p`, `PYTHONPATH`) proves a guard is
     load-bearing without touching source** — the safe form of a mutation ablation in a working
     tree carrying other sessions' uncommitted changes. Written with its precondition foregrounded
     (the constant must be read off `self` at call time, not captured at import), because that is
     what decides whether the technique applies at all.
- **Why:** both are verification technique, which is this file's stated scope, and both were
  re-derived rather than taken on the entry's word: the `verify_salesperson.sh` header and check 6
  were read as the shipped fix for (1), and `repository.py`'s class-attribute declarations plus
  every `self._…_CYPHER` call site for (2). `analyst` did not produce either — a reviewer is the
  reader who needs them.
- **Files:** `claude/analyst/review-techniques.md`. Source dispositions:
  `claude/coder/kaizen/history.md` (2026-09-07, U11).

## 2026-09-02 — one-token example refresh in the placeholder-vs-expanded-key trap (U6 of `salesperson-ui`)

- **What:** `analyst.md` `:84`'s illustration of the pasted-doc-template-placeholder trap now cites
  `cpg_falkorchat` instead of `cpg_salesperson` as the repo's expanded graph key. The lesson and
  wording are otherwise untouched.
- **Why:** `cpg_salesperson` is a graph of the retired Streamlit app under a name the incoming
  `salesperson/` component will make misleading, and its fate (rename or drop) is undecided —
  `cpg_falkorchat` is unambiguously live and serves the example identically. Done by `cobb` in the
  cross-agent sweep of `salesperson-ui` unit U6.

## 2026-08-25 — `kaizen_team` distillation (unit U1 of the team-wide pass): 20 pending entries processed

`cobb` ran the agent-maintenance skill §5 procedure against every `analyst`-linked `kaizen_team`
node — 12 legacy (`author:'analyst'`, dated 2026-08-21/22) plus 8 current-shape
(`(:Agent{agentId:'analyst'})-[:PRODUCED]->`, dated 2026-08-22 through 2026-08-25; zero `MENTIONS`
edges found). Full untruncated text fetched via `redis-cli --no-raw` (the MCP tool's per-cell
truncation would have lost most of these at ~300 chars). Every surviving claim was re-derived
against the live repo/system, not trusted from the entry's own framing — grep, live FalkorDB
probes on a disposable graph, and direct reads of the current source. No entry was found to be
about a different agent; no `MENTIONS` tag added. Zero kept-open — every entry resolved to a clean
promotion or a discard this pass.

**Promoted to `review-techniques.md`** (7 entries, consolidated into 6 new sections plus one new
sub-technique (d) on the existing uncommitted-diff section, to avoid near-duplicate entries):
  - `b4f8e21a…` — mutation-testing via in-process module substitution under the real dotted name
    (new sub-technique (d), sibling to existing (a)–(c)).
  - `c7d21f4a…` + `e91a4d7c…` — merged into "Re-gating a state-machine guard/invariant fix": two
    checks (foreclosed-pattern check, call-site tracing) a "does the mechanism work" read misses.
    The falkor-chat-specific instance (K-028 v2's unconditional-fallback defect) is fully resolved
    and already documented in the now-archived `workflow-timers.md` v3 itself (lines 1015–1024) —
    only the generalized review lesson was promoted, not a project-docs claim about current system
    behavior (v3 replaced the unconditional fallback with a conditional one; the described bug no
    longer reproduces).
  - `b6f3a1d2…` + `e3a1f6b2…` — merged into "A 'this already exists' claim is a grep away from
    confirmation." Both source facts are already fully resolved/documented in their own review
    docs (`docs/reviews/mid-run-escalation.md` §"grounded in..."; `document-ingestion.md` Pass-2)
    — only the generalized technique was promoted.
  - `9f3d2b1a…` — "An untracked plan/review doc has no re-verification baseline," verbatim
    disposition per its own `suggestedHome: review-techniques` tag.
  - `9e2a9f2e…` — "A truncate → append → truncate-again pipeline can silently discard its own
    repair pass." The falkor-chat instance (extraction.py) is already fixed and documented in the
    code's own comments, citing this exact review pass by name (Pass 3 MAJOR 1) — only the general
    lesson was promoted.
  - `8b1e4f2a…` + `2f3a9b1c…` — merged into "Two checks for a multi-shape authorization/
    security-gate function" (keyword-set completeness; early-match short-circuit smuggling). Both
    findings are already fixed in the live `cypher-mcp/server.py` (`_FOREIGN_TRIGGER_RE` now covers
    all four keywords; `authorize_write()` now calls `_has_foreign_trigger_outside_strings()` after
    an author-claim match) — verified by reading the current source, not the entries' own claims.
    Only the generalized technique was promoted, project-docs facts already fully covered by the
    (already-processed) `docs/reviews/kaizen-agent-ontology.md` review doc itself.

**Promoted to `skills/agent-standards/claude-code.md`** (2 entries):
  - `b1e6f3a2…` — the protected-path carve-out is keyed on the literal dot-prefixed directory name
    (`.claude`, `.git`, etc. — the exact list re-verified live, 2026-08-25, against
    `code.claude.com/docs/en/permission-modes` § Protected paths), not on any project's own
    conventionally-named directory (this repo's dotless `claude/`) — added alongside the existing
    self-modification/classifier callout in § Hooks.
  - `b6f1e6b0…` — FastMCP's `Tool.run` wraps any tool-function exception in a generic `except
    Exception`/`ToolError`, so a per-item try/except inside a bulk tool must catch every failure
    mode, not just its own domain exception — added to § MCP Output limits. The falkor-chat
    instance (`Services.ingest_documents`) is already fixed and documented in the code's own
    docstring; only the general FastMCP fact was promoted here.

**Promoted to `skills/python-web-quirks/SKILL.md`** (1 entry):
  - `e6f2b1a4…` — pydantic `Field(min_length=1)` doesn't reject whitespace-only strings (`" "` has
    len 1); MCP callers with no schema layer are completely unguarded, not just under-guarded. The
    falkor-chat instance (`services.ingest_document`) is already fixed (`text.strip()` check,
    documented in the function's own docstring) — only the general pydantic fact was promoted.
    Skill frontmatter `description` updated to name the new gotcha.

**Promoted to `claude/graph-dba/falkordb-quirks.md`** (2 entries, edited directly per the
established maintainer-edits-another-agent's-knowledge-base-file channel):
  - `a1e6c2f8…` — `db.labels()`/`db.relationshipTypes()` asymmetry for zero-data schema elements.
    Re-verified empirically on a fresh disposable graph (2026-08-25), independent of falkor-chat's
    current data state — the original entry's `ws:acme` evidence is now stale (K-050 has since
    shipped real `HAS_CHUNK`/`ABOUT`/`RELATES_TO`/`SAME_AS` edges on that workspace), but the
    underlying FalkorDB engine behavior it described is a durable, build-specific fact, re-proven
    from scratch.
  - `e3b6f2a4…` — `db.idx.fulltext.queryNodes()` against a label with no fulltext index created at
    all silently returns zero rows, no error. Re-verified empirically (2026-08-25, disposable
    graph).

**Promoted to `falkor-chat/docs/SERVER.md` §1.7 (Testing hazards)** (1 entry):
  - `a3f5e8c2…` — `Services`/`WorkflowExecutor` each default to their own separately-defined
    `_default_clock`, unwired even in production `app.py`; a test injecting `Services(clock=...)`
    alone doesn't control `StepRun.startedAt`. Re-verified live: still true today (no wiring exists
    in current `app.py`).

**Promoted to `claude/AGENTS.md`** (1 entry):
  - `a3f1c9e2…` — `git add` then `git commit` is not atomic against a concurrent process sharing
    the working tree; a staged file can get swept into an unrelated commit during the window before
    the commit runs, even when the two processes' own files never overlap (the race is on the
    index, not on any one path). Added to the Git-commit-authority section as a check-before-commit
    discipline (`git status`/`git diff --cached --name-only` immediately before `git commit`) —
    this is a fact about the whole team's multi-agent git workflow, not analyst-specific, so it
    landed in the shared context file rather than this agent's own prompt/knowledge base.

**Discarded — already resolved/fixed and already documented, no promotable residue** (4 entries):
  - `a3f1c2e4-6b8d…` (`CYPHER` preamble literal-binding syntax) — the underlying mechanism is
    already documented in `falkordb-quirks.md` (the existing "CYPHER preamble needs Cypher
    literals" bullet); the specific K-049 repro it describes is already fully written up at
    `falkor-chat/docs/reviews/unique-constraint-oversized-value-crash-rca.md`.
  - `a3f1c2e4-7b6d…` (sequential-mutation patch/restore/sha256sum technique) — no material
    information beyond `review-techniques.md`'s existing (b)/(c) mutate-restore-verify-hash
    discipline; same zero-touch pattern, different file-tracking status than either sub-case
    already covers.
  - `b3e2c1d4…` (falkor-chat background work is fire-and-forget) — already fully documented
    (`falkor-chat/docs/HISTORY.md`, `QUERIES.md`) and the specific race it was used to flag is
    already fixed at the database layer, per the resulting plan-gate decision in
    `document-ingestion.md` (§"Concurrency note").
  - `b7e1c9a4…` (subagent `permissionMode` inheritance — the two named exceptions are exhaustive)
    — already present, more completely, in `skills/agent-standards/claude-code.md`'s own
    2026-08-24 "parent-session mode inheritance" resolution callout (§ Hooks), which the entry's
    own review target (`permission-default-mode.md`) fed into; nothing this entry adds isn't
    already there.

**Verified:** `bash claude/scripts/audit-team.sh` — PASS, all checks including the personal-
identifier sweep; git-commit-authority check unaffected by the `claude/AGENTS.md` addition (not an
agent-file edit, not scanned by that check). All 20 graph entries cleared (12 legacy
`DETACH DELETE`; 8 current-shape resolved via the `PRODUCED`-edge-delete path, all with
`otherRemaining == 0` since none carried a `MENTIONS` edge — full-node clear in every case).
- **Plan items:** none opened — every surviving entry either promoted cleanly or was already
  resolved elsewhere.

## 2026-08-25 — Output discipline: the finding budget (prompt-waste plan, Stage D — an *addition*)

- **What:** 2,325 → 2,375 w (**+50**). Stage D is the only stage of
  `claude/docs/plans/prompt-waste-reduction.md` that adds; the unit's joint budget with
  `architect.md` is ≤~120 net words and it landed at +92. Two additions here:
  - **The finding budget** (deliverable item 2, appended after the "This is fragile" example):
    *"Keep a finding's inline body to **≤~15 lines**; a longer trace, table, or log excerpt goes to
    a trailing `## Appendix` section, cited from the finding. Never drop evidence to fit the cap —
    it bounds the body, not the review's rigor."*
  - **The `-impl` prohibition**, folded into the *existing* sentence that already named the suffix
    rather than added as a new one (+6 w): "…the bare slug is the review of the **plan**, **and
    implementation findings never grow the plan review**." The naming convention was already
    stated; only the prohibition was missing.
- **Why:** The cap's motivating incident is a **2,455-line review file**. The prohibition and the
  cap attack it from two sides — one bounds a single finding, the other stops one file absorbing a
  second review. The never-drop guard is deliberately welded to the cap in the same breath, because
  a budget stated alone invites meeting it by deleting evidence, which is the one failure this
  agent must never have.
- **Gate (a) — rule inventory, addition-shaped.** Nothing removed; every existing class-1/2 clause
  checked for contradiction by the additions. Three pairs cleared: the cap vs. item 2's *"specific
  enough that the owner can act without re-deriving your analysis"* (15 lines is ample for
  evidence+why+fix, and the appendix escape plus the never-drop guard close the gap); the cap vs.
  step 4's *"Prune ruthlessly"* (**different objects** — step 4 caps the *number* of findings, the
  new rule caps *one finding's body*); and the cap vs. the **RCA skeleton**, which it must not
  reach — an RCA's "Causal chain" and "Reproduction & evidence" legitimately carry long traces.
  Placing the cap inside the *review* deliverable rather than in Guardrails is what scopes it
  correctly; that placement is load-bearing, not incidental.
- **Gate (b):** not applicable — nothing removed. **Gates (c)/(d):** `cobb` §7 lint; `audit-team.sh`
  PASS.
- **One lint correction:** I first wrote "goes to **an appendix**" — but this file's own
  *"A complete review contains:"* enumerates exactly four sections and defines no appendix, so the
  agent was told to write into a container the file's spec doesn't declare. → "a trailing
  `## Appendix` section" (+4 w), which names it in place rather than paying ~10 w for a fifth
  skeleton item.
- **One addition the plan mandated and `cobb`'s lint sent back** — *"a later pass records a closed
  finding as one disposition line, full prose only for new findings."* I had shipped it, arguing it
  was safely split from Stage E's `## Pass N` convention amendment (I owned *findings*, Stage E
  owned *section mechanics*). The split does not hold, on three grounds: Stage E's own bullet
  **already contained "one-line disposition per prior finding" verbatim**, so both halves were
  already written and Stage D was one gate away from shipping one rule into two always-loaded files;
  the rule binds **three** reviewing agents (`analyst`, `data-scientist` via `reviews/<slug>-ml.md`,
  `security-expert`), not one; and this file's established idiom for a root-owned document
  convention is a **pointer**, not an inline restatement (`:66`, "Open the document with the header
  block from root `AGENTS.md`"). **Reverted here, relocated to Stage E's rule-5 amendment**, which
  now carries the disposition tokens so nothing is lost.
- **Plan items:** none. Feeds the plan's **finding 15**.
- **Watch (observation window open):** the verdict scale, severity ranking, the `CPG:` line, the
  evidence-traps list and the `-impl` split are unchanged. What to watch is the cap misfiring —
  a finding truncated to fit 15 lines with its evidence dropped rather than appendixed. That is
  the exact failure the never-drop guard exists to prevent, so it is also the test of whether
  pairing the guard with the cap in one breath actually works.

## 2026-08-24 — Prompt-waste compression, Stage C3 (analyst-specific pass) — file measured at its editorial floor

- **What:** Five edits, one pass (per the Stage C one-pass-by-default rule), 2,510 → 2,473 w (−37, 1.5%).
  (1) § "How you work" step 3 blockquote: dropped the tail "it is not part of this always-loaded
  prompt" (meta-commentary about prompt structure, no behavioral content); the
  `review-techniques.md` pointer and its three technique examples kept — the examples *are* the
  trigger, since "specialized verification techniques" is not a condition the agent can check.
  (2) § Guardrails "Evidence over vibes" lead: "Specific traps that have bitten a review before:"
  → "Specific traps:". (3) Placeholder-token trap: the frequency generalization ("Plans routinely
  paste…") replaced with a direct imperative ("Watch for a pasted doc-template placeholder"); the
  placeholder-vs-expanded-key mechanism deliberately kept, since it is what makes the trap
  recognizable. (4) "held, pending until X lands" trap: wording compressed, both halves intact.
  (5) "A deliverable that already exists at your target path" bullet: dropped "e.g. resuming after
  an interruption", `Offline/static` → `Static` (binds to this prompt's own "static reviewer"
  vocabulary), both branches intact.
- **Rule inventory (gate a), edited regions — all preserved:** consult `review-techniques.md` on
  demand + its three trigger examples (1); evidence-vs-inference distinction, the two hard nevers
  (suite-green, traced-path), all six traps with their mechanisms and consequences (2–4);
  not-authoritative half + grep-siblings half (4); side-effecting-claims-unverified branch and
  static-claims-inheritable branch (5). Verbatim `CPG:` three-form sentence, audit-check-8 commit-grant
  tokens, frontmatter, hooks, learning-capture block and RCA skeleton untouched.
- **Removed class-5/6 material, recorded where:** the traps' incident provenance → this file's
  2026-08-23 entry (line 90, the placeholder-token trap) and 2026-08-19 "Evidence over vibes"
  entry (which names all five original traps individually).
- **Two dedup candidates considered and rejected**, both the shape that caused the C1 pass-2
  regression: the intro's "the artifact under review stays untouched" against Guardrail 1 (kept —
  the intro is the *scope contract* stating what the agent's output **is**, the guardrail is a
  *prohibition* with a hook behind it; different speech acts, and deleting the intro sentence
  opens the prompt describing a job without saying what it produces), and step 3's "did you check,
  or does it just look wrong?" against "Evidence over vibes" (kept — step 3 is a per-finding
  *sufficiency gate* that pressures toward going and checking; the guardrail explicitly *permits*
  inferring provided you label it. Two different behaviors at two decision points).
- **Correction to that second rejection, from the lint:** the reason originally given was
  gather-time vs. reporting-time, which does not hold — step 3's tail "— say which" genuinely is
  a reporting-time labeling instruction duplicating the guardrail. What earns the keep is step 3's
  *first half*, not the split. Recorded so a later pass doesn't re-derive the wrong rationale.
- **Fixed while in there (composition conflict, pre-existing):** the placeholder-token trap cited
  `kaizen_analyst` as an example of "the repo's expanded keys" — but that graph key was **retired**
  (this file's 2026-08-21 entry, `G1`'s last two retirements). The prompt was asserting as current
  a key its own auto-loaded context says no longer exists. Now `cpg_salesperson`, live-verified
  against the FalkorDB instance's loaded-graph list.
- **Verified:** `audit-team.sh` PASS (all 13 agents, including the personal-identifier and
  check-8 sweeps); `cobb` §7 lint on the result — **0 blockers, 0 majors**, 3 minors + 2 nits, all
  four actionable ones applied (the retired-key fix, +3 w restoring the unanchored comparative's
  referent, +5 w restoring "side-effecting ones are not" as anti-trigger fencing on the
  static-claims permission, +2 w restoring the distributive "any of its"). First unit in this plan
  with no MAJOR.
- **Finding for the plan — this file is at its editorial floor.** Post-edit residual class-6/7
  inventory across the *whole* file is **under 25 words**. C3's plan target of ~1,900 w (a 26% cut)
  was unreachable without deleting rules; the measured floor is ~2,473 w. The ~350 w evidence-traps
  block is distilled class-3/4 lesson payload, not narrative — no further prose editing reaches it.
- **Plan items:** opened **K-003** (progressive disclosure — move the traps' mechanisms to
  `review-techniques.md`, keep trigger stubs), the structural lever this floor leaves.

## 2026-08-23 — Freshness-clause grammar fix (Stage B wave 2 micro-shape)
- **What:** "a `teco`-issued brief that states the graph's freshness, take it as given" → "when a `teco`-issued brief states the graph's freshness, take it as given" — closing the hanging-topic construction cobb's wave-1 lint flagged as minor; applied uniformly across all files carrying the clause. No rule change; both branches intact.

## 2026-08-23 — Prompt-waste compression, Stage B wave 1 (boilerplate sweep)
- **What:** Applied the three pilot-validated boilerplate compressions from
  `claude/docs/plans/prompt-waste-reduction.md` (§3 doctrine, Stage B), same shapes as the
  `architect.md` pilot; the full analyst-specific compression (plan Stage C3) is a separate,
  later unit. (1) CPG-freshness clause (§ "How you work" step 2): dropped the "(2026-08-19)" date
  and the redundant "without re-deriving staleness yourself" tail. (2) Interactive-commit-grant
  passage (§ Guardrails, Bash bullet): dropped the provenance sentence "Stakeholder decision,
  2026-08-21 — see `kaizen/history.md`." and ", same as before"; the "(spawned via
  `Agent`/`Task`)" clarifier moved from the interactive-definition parenthetical to the carve-out
  sentence (was stated in both). (3) Learning capture: intro dropped "directly" and "identified by
  a real `:Agent` node it's `PRODUCED`-linked to," (the Cypher template below shows the MERGE +
  PRODUCED edge); tail dropped the inbox-replacement history sentence and "exactly like the old
  inbox was".
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
- **What:** The Bash guardrail's "investigation only" bullet now also grants: when running
  interactively (`claude --agent analyst`, a human present turn-by-turn — not a delegated
  subagent), may `git add`/`git commit` its own review document from the session, by explicit
  path, never bulk-staged/pushed/reset/rebased/amended; the grant does not apply when spawned as
  a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Your deliverable"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered). It had been frozen — never written to — since the 2026-08-20 graph migration (see that date's entry below, which already confirms this file's own pre-migration content was imported into the graph verbatim at the time).
- **Why:** user-directed team-wide cleanup, "no point keeping a file already in git history." Before deleting any of the 12 agents' frozen inboxes, `cobb` live-confirmed `kaizen_team` — the single shared graph every agent's raw capture has routed through since the 2026-08-20 consolidation — holds **zero** entries for any agent: every raw capture any agent ever wrote there (including this agent's own 12-entry pass, immediately above) has since been fully distilled and cleared. Combined with the migration-time import guarantee above, nothing in this file was ever a live, undistilled input to anything — it was a pure redundant backup copy. Same session also completed `G1`'s last 2 of 12 `kaizen_<agent>` graph-key retirements (`kaizen_analyst`/`kaizen_teco`, executed by `graph-dba`), closing `docs/plans/generic-cypher-mcp2-coordination.md`'s one remaining open item.
- **Verified:** live `mcp__cypher__query` count against `kaizen_team` (0 entries) before any deletion; every entryId this file's own 2026-08-21 distillation entry (above) lists was cross-checked against this file's pre-deletion contents — all present, none missing.
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: all 12 pending `analyst`-authored entries processed

- **What:** `cobb` ran the agent-maintenance skill §5 procedure against every `kaizen_team` node
  with `author:'analyst'` (12 entries, dated 2026-08-11 through 2026-08-21 — analyst's raw
  capture since the 2026-08-20 team-wide graph migration; none of these overlap the 3 entries the
  2026-08-11 inbox.md distillation already processed, which was a distinct, file-based pass from
  before the migration). Full per-entry disposition:
  - **Promoted to `review-techniques.md`** (7 entries):
    - Reconciling a kaizen-graph distillation's claimed dispositions against ground truth
      (`481f29ed…`) — adapted from its original file-diff framing (`grep -c '^-## '` on an
      inbox diff) to the graph-based mechanism, since the inbox-diffing technique itself is now
      moot (inboxes are frozen); the underlying discipline — verify an aggregate "N processed"
      claim against the itemized ground truth — carries over unchanged, and is exactly what this
      pass's own bookkeeping had to do.
    - An uncommitted agent-prompt edit under review is already live, via the deployment symlink
      (`854e701d…`).
    - Verifying a "copied verbatim" text-block claim needs a programmatic whitespace-normalized
      diff, not a read-through (`fe2007f5…`).
    - `pytest -k` is not a substitute for the project's own `-m` marker filter when verifying a
      cited baseline (`eea48dac…`) — re-verified live: `cypher-mcp/pytest.ini`'s
      `addopts = -m "not live"` still matches the fact as described.
    - Ground truth for "may an agent edit its own definition?" — the literal "never edit your own
      agent definition" clause (`b9ed574b…`) — re-verified live: `grep -rln` today returns all 12
      non-cobb prompts (grew from 11 at entry-creation time — `security-expert` now also carries
      it), `cobb.md` still the sole exception.
    - Check live-service reachability before trusting a live-test report (`b3a1f2e4…`).
    - A brand-new untracked file has no `HEAD` baseline for the existing zero-touch mutation-test
      methods — added as case (c) alongside existing (a)/(b) (`e7f3a1b2…`).
  - **Promoted to `claude/analyst/analyst.md`** (1 entry): a new "Evidence over vibes" bullet —
    run a plan's prescribed acceptance-check command verbatim before approving; the doc-template
    placeholder token (`kaizen_<agent>`) silently matches nothing against the repo's expanded key
    (`8b881e50…`).
  - **Promoted to `claude/cobb/TESTING.md`** (1 entry): a new Gotcha — `audit-team.sh`'s `ROOT`
    resolution makes it scratch-testable, and a kaizen-files-only directory is silently skipped
    from agent enumeration rather than failing check 1 (`0b11bf16…`) — routed to cobb's testing
    doc rather than this agent's own knowledge base since the fact is about safely testing a
    script `cobb` owns, not a review technique this agent performs.
  - **Discarded, already resolved elsewhere** (3 entries):
    - The "no `SendMessage` means a nested review-gate result misroutes" open question
      (`4a4a031a…`) — fully resolved the same day this pass ran:
      `claude/docs/requirements/mid-run-escalation.md` (Status: Ready for design, 2026-08-21)
      settles it directly — `teco` performs the `SendMessage` resume, not the delegate, so
      review-gate reporters never need the grant. A stronger, more current answer than anything
      this entry could be promoted into.
    - The `r1_probe` field-semantics finding on falkor-chat's `golden_guards.jsonl`
      (`a1e6f3d2…`) — already fully documented and the underlying mismarking already fixed:
      `falkor-chat/docs/plans/golden-set-expansion-ml.md` §"r1_probe semantics (2026-08-20
      addition, analyst review)" states the exact rule, and `cs-10`/`cs-13` are confirmed flipped
      to `r1_probe: false` in that file's finalized golden set.
    - The `nc`/`ncat`/`netcat` local-marker exemption gap in `security-expert`'s
      `guard-exploitation-approval.sh` (`eadd7a90…`) — already fixed: the live script's branch
      (c) and its header comment cite this exact finding ("analyst review 2026-08-20") as the fix
      that gave `nc`/`ncat`/`netcat`-with-a-shell-flag its own unconditional always-ask branch,
      re-verified by reading the current script.
- **Verification method:** fetched every field's exact, untruncated text via `redis-cli
  GRAPH.QUERY kaizen_team ... --no-raw` (the `mcp__cypher__query` tool's own per-cell display
  truncates around 300 chars, `…(+N chars)`, and paging every field via `substring()` for 12
  entries × 4 fields would have been far more round trips than one raw redis dump). Re-derived
  each surviving claim against the live repo rather than trusting the entry's own framing —
  `pytest.ini`, `audit-team.sh`'s actual `ROOT`/enumeration logic, the `grep` for the self-edit
  clause, and the two "already fixed/documented" discards were all re-read from source, not
  assumed from the entry text.
- **Why:** user asked to "work on analyst's inbox" — `analyst/kaizen/inbox.md` is a frozen
  2026-08-20 historical snapshot (already imported and cleared, 3 entries), so the live
  equivalent is analyst's pending raw capture in the shared `kaizen_team` graph; this is the
  first full distillation pass against it since the migration.
- **Plan items:** none opened — every surviving entry either promoted cleanly or was already
  resolved elsewhere; nothing was kept open pending further verification.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_analyst`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_analyst` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 5 pre-existing
  entries were parsed out programmatically and imported into the graph verbatim (entryId assigned,
  `author: 'analyst'`), preserving every field; its own header explains the freeze and gives the
  live-read query. The trailing "Your write guard allows exactly this inbox path" clause was
  dropped — the write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP tool, so it no
  longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — "Evidence over vibes" Guardrails bullet converted to a sub-list
- **What:** The Guardrails bullet had grown into a single run-on sentence carrying 5 sub-rules (the untracked-baseline trap, the unrun-regex trap, the guard-glob cross-check, the no-shellcheck note, the "held pending" note) after four separate clause-extension edits. Restructured as one lead sentence plus a 5-item sub-list under the same bullet — no new top-level Guardrails bullet added, content unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 3 of the parked diagnosis (`cobb/kaizen/plan.md`) — flagged 2026-08-09 during the inbox-distillation review as hurting scannability, never fixed until now.
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
  paraphrased, not dropped" — closes DEF-1 (`coder`, loose prose instead of the literal line) and
  DEF-3 (`tdd-engineer`, dropped entirely on the not-relevant branch). `analyst` itself was not
  live-tested in U6, but carries the same near-verbatim wiring pattern (per the M4 design's own
  "verbatim-identical wording" precedent, U4b-2), so the fix is applied identically here rather
  than assumed unnecessary — the report explicitly flags that untested agents' compliance
  shouldn't be assumed by extension.
- **Why:** U6's acceptance pass found the M4 wiring (U4b/U4b-2) was correctly worded but didn't
  survive contact with a real dispatched agent's own judgment calls — all three live-tested
  dispatches failed a different way (format, skip, silence). Design intent
  (`docs/plans/cpg-agent-adoption.md` §2.3, §3) unchanged: still agent judgment on staleness
  threshold, still no self-triggered rebuild, still a suggestion not a hard rule about *when*
  something counts as stale — only the sequencing and the anchoring got tightened.
- **Plan items:** none new; closes U7.
- **Same-day addendum (U8 diff-gate follow-up):** `analyst` (this agent, in its U8 diff-gate
  role) reviewed this same U7 wording (`docs/reviews/cpg-agent-adoption.md`, Pass 3 — approve
  with suggestions, zero blockers) and flagged two minors and a nit against the freshness
  sentence in all six files: (a) `frontend-engineer.md` was missing the "tool call/" qualifier
  the other five carried, undercutting the U7 ledger row's and commit message's "identically"
  claim; (b) the trailing "this is not a separate, optional judgment call" had an ambiguous
  pronoun referent — a literal reading could bind "this" to the cross-verification *decision*
  rather than the freshness *query itself*, exactly the room DEF-2's `architect` dispatch used
  to reason past a softer version of this sentence; (c) nit — "query the freshness check"
  mismatched a reference-doc noun with a query verb, when the actual queried object is the
  `:CpgBuildInfo` marker (the report's own recommendation said "marker"). Fixed all three: the
  sentence now reads "…query the freshness marker (per
  `skills/cpg-analysis/references/freshness.md`) in that same tool call/step, before you decide
  whether the CPG's answer needs further cross-verification — running the freshness check itself
  is not optional, and skipping it in favor of a substitute check (e.g. grep agreement) doesn't
  satisfy this." — byte-identical across all six files now. The `CPG:`-line wording from the
  original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — M4 cpg-agent-adoption: discovery wording defaulted, freshness-check bundled, evidence-trail line added
- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§3 (U4b). (1) Frontmatter
  `description` reworded from "With a loaded Joern CPG, uses the `cpg-analysis` skill instead of
  reading files" to "Checks whether a relevant CPG exists as part of its normal orientation and,
  when one does, uses the `cpg-analysis` skill instead of reading files" — conditional →
  default-orientation framing. (2) "How you work" step 2 ("Read the real thing") gained a
  sentence: check whether a relevant CPG exists (first guess `cpg_<component>`, per
  `skills/cpg-analysis/SKILL.md` §1), and when one is found and used, also run the freshness
  check (`skills/cpg-analysis/references/freshness.md`) as part of the same step, noting what it
  says in the findings and surfacing a refresh suggestion — not a silent rebuild — if it looks
  stale. (3) The review skeleton's item 1 ("Scope & verdict") gained the one-line `CPG:`
  evidence-trail convention (`CPG: used <graph> — <clause>` / `CPG: considered, not relevant —
  <clause>` / `CPG: not applicable — <clause>`).
- **Why:** M4 (`cpg-agent-adoption`) widens CPG discovery from a conditional check to a default
  orientation step across the three already-wired consumers (`analyst`, `architect`,
  `qa-engineer`), bundles the freshness recipe into that same step (FR-6's surfacing half), and
  adds a spot-checkable `CPG:` evidence trail (AC-2). Per `docs/plans/cpg-agent-adoption.md`
  §2.1-2.3, §3, §6 step 2.
- **Plan items:** none.

## 2026-08-11 — Inbox distillation: 3 entries — 1 prompt addition, 2 to `python-web-quirks`/project docs

- **What:** `cobb` processed all 3 entries in `analyst/kaizen/inbox.md` (§5).
- **Promoted:**
  - The "held-note staleness" finding (a sibling document's "held pending X" claim can go stale
    when X lands via a *different* agent's files) → new clause on the Guardrails "Evidence over
    vibes" bullet: cross-check sibling kaizen history before trusting a holding document's own
    claim about a pending follow-up.
  - `urllib` timeout taxonomy → `skills/python-web-quirks/SKILL.md`.
  - LM Studio `/v1` 200-envelope quirk — merged with `architect`'s duplicate finding of the same
    fact; the falkor-chat-specific half was **already** fully documented in
    `falkor-chat/docs/DESIGN.md` §14.8 ("The `/v1` normalization rule"), so only the general,
    reusable half went to `python-web-quirks.md`.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/analyst/{analyst.md,kaizen/{history,inbox}.md}` ·
  `skills/python-web-quirks/SKILL.md`.

## 2026-08-09 — Independent safety recheck cleared; `review-techniques.md` marker removed
- **What:** Removed the "⚠️ Pending an independent analyst safety recheck before first use"
  callout from technique (b) (scratch-copy + reverse-patch) in `claude/analyst/review-techniques.md`.
  A separate, narrowly-scoped analyst session ran the independent safety recheck and returned a
  clean verdict — no embedded-instruction/manipulation concern, scope-limiting language intact.
  Also folded in the recheck's one optional suggestion: appended "the block was doing its job;
  this substitute earns the exception on its own zero-touch merits" to the existing "not a general
  license" paragraph, affirming the classifier's original block was itself legitimate scrutiny.
- **Why:** The marker existed because this exact technique had triggered an instruction-poisoning
  flag on 2026-07-31 (see that entry below) before being reframed and promoted into this file on
  2026-08-09. The recheck was the condition for treating it as a routine technique rather than
  "informational only." Requested by `teco`, relaying two independent reviews'
  (`docs/reviews/{kaizen-inbox-distillation,analyst-inbox-distillation}.md`) findings plus the
  separate recheck's result.
- **Plan items:** none.

## 2026-08-09 — Held entry 28 promoted: consolidated Kiro-facts edit landed
`cobb` closed out the "(H) Held, not cleared — entry 28" note from the same-day distillation
entry below: the race window against `architect`'s two held entries (both targeting the same
file) is over, so the `kiro-cli agent create` default `"resources": []"` fact was re-verified
(now against `kiro-cli 2.16.2`, up from `2.14.1` — held) and written into
`skills/agent-standards/kiro.md`'s CLI custom-agents `resources` config-key bullet. `inbox.md`
entry 28 cleared; `inbox.md` is back to the standard empty placeholder.

## 2026-08-09 — Full inbox distillation pass: 30 of 31 entries processed (entry 28 held for a coordinated follow-up)

`cobb` ran the agent-maintenance skill §5 distillation over the full inbox (31 entries spanning
2026-07-19 → 2026-08-08), preceded by a read-only proposal pass that verified each entry against
current repo state, then a second re-verification immediately before applying edits (repo state
was unchanged — `git status` clean both times, aside from an unrelated concurrent edit to
`claude/tdd-engineer/tdd-engineer.md`'s step 5 discovered mid-pass, which this pass's own
description-clause edit landed cleanly alongside). Grouped by disposition; every entry not listed
under "held" is now cleared from `inbox.md`.

**(A) Discard — already resolved/stale, cleared with no promotion (entries 1, 12, 15, 16, 18, 20,
25).** Re-verification found each condition no longer holds: entry 1 (falkor-chat pytest
self-skip) is already documented at `falkor-chat/docs/DESIGN.md` §14.7; entry 12
(`audit-team.sh` failing check 7) — a live run now shows full `PASS`, including check 7, the five
cited leaks having been cleaned up since; entry 15 (`pipeline.sh --reset` bypassing the
destructive-ops guard) — `claude/scripts/guard-destructive-ops.sh` now carries a dedicated C-311
branch (dated 2026-08-08) matching exactly what the entry asked for; entry 16 (Claude Code MCP
25k-token output cap) and entry 18 (`CLAUDE_PROJECT_DIR` expansion) are both already in
`skills/agent-standards/claude-code.md`; entry 20 (MCP startup-timeout doc disagreement) is
already resolved there too (`MCP_TIMEOUT` vs `MCP_TOOL_TIMEOUT` disambiguated); entry 25 (stale
"the joern agent's job" error text in `cypher-mcp/server.py`) — the live string now reads "the
graph-dba agent's job", no trace of "joern agent" left in the file.

**(B) Promoted to project docs — `falkor-chat/AGENTS.md` + `docs/DESIGN.md` (entries 4, 5, 6, 7,
10).** Consolidated into one new "Probing shared graph state without mutating it" subsection in
`AGENTS.md` (entries 5 + 10: the `publish_def`/`materialize_snapshot` graph-seam asymmetry, and
`test_services.py` as the review-safe pytest subset) plus a note on the `bootstrap_schema.sh`
Key-scripts row (entry 6, and entry 7's misfiled second "Suggested home" line, which was about
`bootstrap_schema.sh` despite being appended under the line-number-invariance entry — routed to
its actual topic here). Entry 4 (`pytest --collect-only -q` as the non-mutating way to check a
claimed test count) went to `docs/DESIGN.md` §14.7, next to the existing pytest-hazard bullets.
**Line numbers re-verified and corrected**, not copied from the inbox — the original entries cited
`repository.py:132-134`/`:937`/`:1483` etc.; current `HEAD` has the same functions at
`:156-158`/`:992`/`:1669` (drift from other commits landing between 2026-07-24 and now). Entry
7's own topic — the line-number-invariance re-gate technique — went to (F) below, not here.
Also logged in `falkor-chat/docs/HISTORY.md` (2026-08-09).

**(C) Promoted to knowledge base — `claude/graph-dba/falkordb-quirks.md` (entries 14, 17, 27).**
Entries 14 + 17 bundled into one "Ops, config & tooling" bullet (`GRAPH.PROFILE` executes writes
for real despite suppressing `RETURN` output; neither `RO_QUERY` nor an `EXPLAIN`/`PROFILE`
prefix — even after a Cypher comment — is honored as a planning directive under either query
command). Entry 27 (`sum(CASE...)` returns float `0.0`, never `NULL`, on zero-row aggregation,
and stays `float` not `int` on non-empty input) added under "Cypher dialect & query behavior",
next to the existing aggregation-pitfalls bullets. Edited directly per the established
maintainer-edits-another-agent's-knowledge-base-file channel (precedent: 2026-07-31 entry below);
no `graph-dba`-side kaizen entry needed.

**(D) Promoted to knowledge base — `skills/agent-standards/claude-code.md` (entries 13, 19, 21,
31).** Entry 13 (FastMCP `structured_output=False` — otherwise a `str`-returning tool ships its
payload twice via a spurious `outputSchema`) added to § Output limits. Entry 19 (a containerized
stdio MCP server's own labelled container is legitimately `Up` for the whole session that's
probing it, so a "docker ps --filter label=… must be empty" orphan-check is unsatisfiable from
inside an open session) added to § Lifecycle, framed as a liveness-aware-check rule rather than a
bare reviewer habit. Entries 21 + 31 (this environment's Bash tool shell-shadows `find`→`bfs` and
`grep`→`ugrep` via wrapper functions with a spoofed `ARGV0`, not inherited by a spawned
subprocess) merged into one new § Bash tool environment section, since they're the same
phenomenon discovered on two different dates. Per `skills/README.md`'s Maintenance section
("changes to `agent-maintenance`/`agent-standards` are logged in `claude/cobb/kaizen/history.md`"
— see that file, 2026-08-09).

**(E) New skill — `skills/python-web-quirks/SKILL.md` (entries 2, 11, 29).** Stakeholder decision:
Python/web-framework stack knowledge belongs in a skill consulted by the relevant personas, not
duplicated into one project's docs. Entries 2 + 29 merged into one background-task-GC/threading
note (`asyncio.create_task` fire-and-forget GC-safety warned-but-not-reproduced-under-stress,
paired with Starlette/FastAPI `BackgroundTasks`' bounded-threadpool concurrency vs. an unbounded
raw `threading.Thread` — both are about async-dispatch mechanics an implementer might get wrong
in the same code path). Entry 11 (FastAPI/pydantic `response_model_exclude_unset` silently
dropping defaulted fields on **nested**, not just top-level, models) as its own section. Wired via
a routing clause in `coder`, `tdd-engineer`, `architect`, and this agent's own frontmatter
`description` (mirroring how `cpg-analysis` is wired into `analyst`/`architect`). Registered in
`skills/README.md` and root `AGENTS.md`'s `skills/` bullet. No dedicated `kaizen/` for the new
skill — no existing skill in this repo actually carries one despite the agent-maintenance skill's
general rule (`agent-standards`/`agent-maintenance` are logged in `cobb`'s kaizen per
`skills/README.md`; `joern-cpg`/`cpg-analysis` changes are logged in `graph-dba`'s kaizen instead,
per that file) — followed the established precedent (log in the creating/maintaining agent's own
kaizen, here `cobb`'s) over the written-but-unpracticed rule; logged in `claude/cobb/kaizen/history.md`
(2026-08-09).

**(F) New on-demand file — `claude/analyst/review-techniques.md` (entries 3, 7, 8, 26).**
Stakeholder decision: specialized review techniques go on-demand (mirrors
`graph-dba/falkordb-quirks.md`), not always-loaded prompt body. Holds: the AST line-range
byte-identity hash technique (3), the line-number-invariance re-gate technique (7's actual
topic), the stub-package HEAD-vs-working-tree import technique (8), and the scratch-copy +
reverse-patch technique (26) — written in using the **already-reframed** text that survived the
2026-07-31 security review (see that entry below), not any earlier draft, and carrying an
explicit "⚠️ Pending an independent analyst safety recheck before first use" marker per the
stakeholder's instruction; a separate narrowly-scoped analyst session is checking it. `analyst.md`
gained a one-line pointer to this file (in "How you work" step 3) rather than inlining the
content.

**(G) Core prompt — `analyst.md` (entries 9, 22, 23, 24, 30).** Entry 9 (a deliverable already
sitting at the target path when a run starts — e.g. resuming after an interruption — may have
executed/side-effecting claims narrated in past tense before the command actually ran; re-verify
against the live system before inheriting them) added as its **own** Guardrails bullet — stakeholder
judged it high-severity (a resumed analyst could hand `teco` false confidence about live system
state), not folded into an existing one. Entries 22 (a `git grep`/`git ls-files` count is a bound,
not a fact, when the artifact under review or a sibling deliverable is itself untracked), 23 (a
suggested regex/glob/pattern fix is a claim, run it before writing it into a review — the specific
extglob bug that motivated this is already fully documented in
`docs/plans/doc-reference-convention.md`, no further action needed there), 24 (cross-check a named
agent's `PreToolUse` guard globs when a plan assigns it doc-write ownership), and 30 (`shellcheck`
isn't installed in this environment — `bash -n` + live execution is the substitute) folded as
**clause-level extensions to the existing "Evidence over vibes" guardrail sentence**, per
stakeholder instruction to avoid four new standalone bullets.

**(H) Held, not cleared — entry 28** (`kiro-cli agent create` default `resources: []`).
Disposition decided (promote to `skills/agent-standards/kiro.md`) but left in `inbox.md` with a
one-line "queued for consolidated follow-up" note: `teco` is coordinating a combined edit to that
shared file alongside two related facts from `architect`'s inbox, to avoid two sessions racing on
the same file.

**Inbox-authoring defects found and corrected while applying:** entry 6 had no "Suggested home"
line of its own (its dispositioning text had been appended, in error, under entry 7 — see (B));
entry 1 likewise had no "Suggested home" line (moot — turned out to already be stale). No entries
were found to conflict with each other.

## 2026-07-31 — Inbox entry reframed after an "Instruction Poisoning" flag; classifier-gap fact partially distilled
- **What:** A security check flagged the 2026-07-31 inbox entry ("Auto mode's Bash classifier blocks `git stash`…") as instruction-poisoning-shaped: a persistent, forward-looking "here's how to route around a safety classifier block" write-up, regardless of how benign the originating use was. `teco` (no write access to this inbox, no adjudication authority) routed the triage to `cobb`. Verdict: **reframe needed, not a false positive and not a full policy violation** — the underlying technique (scratch-copy + reverse-patch, zero working-tree touch) is sound and consistent with this inbox's established isolation discipline (the 2026-07-24 stub-package and review-safe-pytest-subset entries), but the entry's *framing* ("here's the workaround now that git stash is off-limits") taught evasion-shaped reasoning rather than the safety property that actually makes the substitute acceptable. Contrast: the 2026-07-25 `pipeline.sh --reset` entry is safe precisely because it reports a gap in a guard **this repo owns** (`claude/scripts/guard-destructive-ops.sh`) for that guard's maintainer to close — it never tells an agent to use the gap. The 2026-07-31 entry, by naming a *product-level* auto-mode classifier (not a repo hook) and framing the substitute as "the answer to being blocked," was the wrong shape even though the action taken was benign and verified harmless (`git status` clean before/after, independently spot-checked by `teco`).
  - Entry rewritten in place (`kaizen/inbox.md`): kept the technique and its evidence, added an explicit scope note disclaiming the "route around any classifier block" generalization, and stated plainly that the classifier itself is not a repo mechanism there's anything here to harden.
  - The classifier-gap fact (no reversible/read-only-verification carve-out) partially distilled: routed to `skills/agent-standards/claude-code.md` §Hooks as an observed (not doc-verified) harness quirk, since it's a Claude-Code-product fact of the kind that knowledge base is for, not project-specific to this repo.
  - **Not done in this pass:** full promotion of the technique itself into this prompt, and the broader backlog of other still-unprocessed "suggested home: prompt" entries in this inbox (stub-package HEAD-vs-working-tree import, review-safe pytest subset, isolatable snapshot side, byte-identity AST hash, line-number-invariance re-gate technique, etc.). This was a narrow security triage, not a full §5 distillation pass — see `cobb/kaizen/plan.md` for the follow-up item.
- **Why:** `cobb` owns inbox distillation (agent-maintenance skill §5) and, per the same skill, edits another agent's kaizen files directly as its normal channel — no hook restricts this (the `guard-review-doc-writes.sh` PreToolUse hook is wired in `analyst`'s own frontmatter and fires only during `analyst`'s own tool calls, not another agent's). `teco`'s message raised the possibility that `cobb` might lack write access here and asked it to say so plainly rather than route around its own guard if that were true; `cobb` verified empirically (an actual Edit call, which succeeded with no hook interception) rather than accepting or rejecting the claim unverified, and is recording that check here since it bears directly on the pass's own subject matter.
- **Plan items:** none pre-existing; see `cobb/kaizen/plan.md` for the new follow-up.

## 2026-07-29 — New review target: a tico-authored user manual's factual/architectural claims
- **What:** `tico` gained a new doc kind, user manuals (`<component>/docs/manuals/<slug>.md`), and the team certification pass flagged that manuals were the only doc kind with no independent-review gate. User decision: split the review — `qa-engineer` verifies the walkthroughs by driving the running app (behavioral claims), `analyst` checks everything else. Added a fourth reviewed-artifact category ("What you review") between source code and RCA: a manual's factual/architectural claims against the real code/config (same grounding discipline as a plan review), plus clarity for a non-technical end-user audience specifically — explicitly *not* the walkthroughs, which stay `qa-engineer`'s to avoid duplicating that check. Frontmatter `description` updated to name the new target and its qa-engineer/analyst split.
- **Why:** user ruling following the 2026-07-29 team certification's open observation (logged in `cobb/kaizen/plan.md`, now resolved). Routed through `teco`'s existing "independent review" default (its own kaizen carries the matching entry) rather than analyst self-selecting when to review a manual.
- **Plan items:** none — no prior plan item covered this; not adding one since it's already implemented.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — `-impl` review role documented; header block required on review docs (step 1 of `docs/plans/doc-reference-convention.md`)
- **What:** Two body edits, no frontmatter change. (1) The deliverable paragraph now names the `-impl` role: a review of an **implementation** is `<component>/docs/reviews/<slug>-impl.md`, the bare slug being the review of the **plan**. (2) One line added between the review skeleton and the RCA skeleton: *"Open the document with the header block from root `AGENTS.md`."*
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.4 found `-impl` **used 4× and documented nowhere** — the only member of the closed role set (`(none)` · `-coordination` · `-ml` · `-graph` · `-rca` · `-impl` · `-report`) missing from the prompt that produces it, and the absence had already broken a document family. The header line is the canonical M9 sentence, byte-identical across the prompts that get it, and is a pointer rather than an inlined template because root `AGENTS.md` reaches every agent through the root `CLAUDE.md` `@AGENTS.md` import. `claude/README.md` row 17 re-checked — it cites the review write paths, not the naming rule, so no catalog edit was needed.
- **Plan items:** none. (K-001's remaining RCA half is untouched; `-rca` was already documented.)

## 2026-07-25 — `tools:` allowlist gains `mcp__cypher__query` (M3 / C-304)
- **What:** Frontmatter `tools:` now ends `…, Agent, mcp__cypher__query`. `claude/README.md` row 17 updated to say the `cpg-analysis` skill reaches the graph through that MCP tool and why the allowlist entry is required. No body or `description` change — the CPG routing clause added on 2026-07-19 stays accurate, and the skill is progressively disclosed.
- **Why:** M3 replaces the CPG read path with a single MCP tool, `mcp__cypher__query(graph, cypher)` (`docs/plans/cpg-query-access.md` S5). **`tools:` is an allowlist, not a hint** — an agent that declares one does not see MCP tools absent from it, so without this line the feature would have been silently inert for `analyst` (and `architect`); `qa-engineer` and `graph-dba` declare no allowlist and inherit it. `redis-cli GRAPH.QUERY` remains the documented fallback and is the only path under OpenCode/Kiro.
- **Verification note:** this is the *edit*; the live proof (a cold `analyst` actually calling the tool) needs the server wired in S3 and is verified in S9, per the plan's m-4 split.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 707 → 469 chars (-33%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (analyst↔qa-engineer, analyst↔data-scientist) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `analyst`'s `guard-review-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed review-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-19 — CPG capability wired into the routing description (M2 / C-207)
- **What:** Frontmatter `description` gained one clause: when a Joern CPG is loaded in FalkorDB, the analyst uses the `cpg-analysis` skill (graph-dba-owned) for impact-analysis, RCA data-flow, and code-review taint queries instead of reading files. `claude/README.md` catalog entry updated to match. No body change — the skill is progressively disclosed and self-describes; the description clause is the routing signal.
- **Why:** M2 delivered the `cpg-analysis` skill (`analyst` is a named consumer for impact/RCA/code-review recipes per FR-10/11/12). C-207 makes the consumer agents' routing contract advertise the capability. cobb wired it as part of Gate-2b (skill also passed the standards vet).
- **Plan items:** none.

## 2026-07-12 — K-001: code-review half of the shakedown proven (K-022 impl review) — RCA remains
- **What:** The **code-review** half of the first-run shakedown ran for real. On falkor-chat
  **K-022 Landing 1** (executor implementation, committed `3921f87`) the analyst reviewed the
  delivered diff and produced `falkor-chat/docs/reviews/m3-executor-impl.md`: verdict
  **approve-with-suggestions, 0 blockers / 1 major (M-1) / 3 minor / 3 nit**, doc landed at the
  right path with the write-guard hook silent. This was the designated vehicle named in K-001 and
  the counterpart to teco K-003 (the team's first fully-gated run). Verdict calibration read as
  healthy — a real major surfaced, not a nitpick flood, and the two deferred seams were ruled
  acceptable-for-Landing-1 rather than inflated to blockers.
- **Why:** teco K-003 closed 2026-07-12 with the gated run committed; that same run is the
  evidence for analyst K-001's code-review half. Recording it here so the shakedown's remaining
  scope is honest.
- **Plan items:** **K-001 narrowed** (not closed): plan-review ✅ (2026-07-11) + code-review ✅
  (this entry); the **RCA** mode is still unexercised — K-001 now tracks that remainder only.
  No prompt change: no verdict-calibration weakness surfaced across the two review runs.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1449 to 525 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-review-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/reviews/*|*/docs/reviews/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/analyst/hooks/guard-review-doc-writes.sh`) to `$HOME/.claude/agents/analyst/hooks/guard-review-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/analyst` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — data-scientist route-away clause (boundary symmetry)
- **What:** Frontmatter `description` and the findings-routing guardrail now route the AI/ML/data-science **methodology** dimension of a plan or change — model/embedding choice, evaluation design, metric validity, statistical claims — to the new `data-scientist` agent, whose methodology review (`docs/reviews/<slug>-ml.md`, same verdict scale) complements the analyst's general static review. Pair `analyst:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09 to work alongside the analyst at review time; "review this ML-heavy change" plausibly matched both, so the boundary must live in both descriptions.
- **Plan items:** none.

## 2026-07-09 — Description gained the qa-engineer route-away clause (boundary symmetry)
- **What:** Frontmatter `description` now states the verification boundary explicitly: analyst judges statically — reading, reasoning, and running what already exists — and planning/executing *new* black-box/acceptance testing of the running system routes to `qa-engineer`. The prompt body already carried this split (findings-routing guardrail); the description — the routing contract every router sees — didn't. Counterpart clause added to qa-engineer in the same change; the pair is now mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): analyst↔qa-engineer was asymmetric at the description level (analyst never named qa-engineer), leaving "test this" work plausibly matching both.
- **Plan items:** none.

## 2026-07-09 — Added root cause analysis (RCA) mode
- **What:** Extended the reviewer into a reviewer-and-diagnostician: a third artifact class ("Defects and failures — RCA") with its own method (reproduce when possible, trace the actual code path, read git history; distinguish root cause vs trigger vs contributing factors; five-whys stops at the deepest cause actionable in the codebase; record ruled-out hypotheses) and its own deliverable skeleton at `docs/reviews/<slug>-rca.md` (symptom & impact → reproduction/evidence → causal chain → root cause with confirmed/inferred confidence → suggested fix + prevention). Frontmatter description updated; guardrail clarified (diagnoses only — the fix routes to the implementer, typically `tdd-engineer` with a reproduction test first, briefed by the RCA path). No hook change needed (`docs/reviews/` already covers the RCA doc). Rosters/catalogs synced (teco, claude/AGENTS.md, claude/README.md, root AGENTS.md).
- **Why:** User: "analyst is also good with RCA" — the team had no owner for cause-unknown defects; tdd-engineer starts from a known bug, qa-engineer finds and reports defects, but nobody's job was tracing a symptom to its root cause.
- **Plan items:** none (K-001 shakedown should now cover an RCA run too).

## 2026-07-09 — Created
- **What:** Initial version of the `analyst` subagent — a systematic, experienced developer acting as a pure reviewer: reviews architect plans (grounding, completeness, soundness, proportionality, test strategy) and source code (correctness → tests → fit → clarity → security/perf, in priority order), plus plan↔code conformance when given both. Deliverable is a severity-ranked (blocker/major/minor/nit), evidence-backed review with a verdict (approve / approve with suggestions / needs changes), written to `<component>/docs/reviews/<slug>.md` by default and handed off by path. Review-only contract is harness-enforced: `hooks/guard-review-doc-writes.sh` (PreToolUse, matcher `Write|Edit`, same pattern as architect's guard) escalates any Write/Edit outside `docs/reviews/` (or `/tmp`) to the human. Subagent-aware (questions/blockers return as the deliverable). Model `opus`, tools `Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent` — mirrors architect. Deployed via `~/.claude/agents/analyst` symlink.
- **Why:** The team had no review gate between handoffs — architect plans went straight to implementation and implementer code straight to QA, with nobody judging design soundness or code quality statically. User requested a systematic reviewer covering both plans and source code.
- **Plan items:** — (K-001, K-002 seeded)
