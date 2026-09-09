# Kaizen — Change History: tdd-engineer

> Dated log of actual changes to the `tdd-engineer` agent. Most recent first.


## 2026-09-09 — `guard-testing-techniques.md`: the retraction paragraph was right about one file and wrong about the other (U34)

- **What:** `cobb` corrected the `### The worked example` subsection of
  `claude/tdd-engineer/guard-testing-techniques.md` — five edits, +25/-13 lines, no new section.
  The file is `cobb`'s to maintain as the author of that subsection (U31/U32 follow-up); this is a
  documentation-correctness pass, **not** a reopening of the `rq` guard arc, which U33 closed.
- **The defect, and it is this file's own thesis turned on the file.** The closing paragraph,
  *"One retraction, recorded because it is the same lesson"*, recorded `cobb` probing the check's
  stated bound adversarially, finding `PING`/`INFO`/`keys` unflagged, drafting it as generation six
  and withdrawing it — because the bound is scoped to runs matching `/GRAPH\.[A-Za-z_.]*/` and its
  first clause covers a non-`GRAPH.` command literally. **That reasoning held for
  `test-stamp-wiring.sh`, the file `cobb` had read. It was false for `pipeline.sh`,** whose copy of
  the same sentence said only *"not one contiguous literal"* — the `GRAPH.<word>` scoping dropped —
  so `rq "$Q" PING` sat inside its stated reach and outside its mechanism. The withdrawn finding was
  a **real defect in the file nobody opened**, fixed by `graph-dba` in `00bebdc` (U33): both files
  converged on the scoped claim, a *second* sentence in `test-stamp-wiring.sh` that had drifted the
  same way was tightened, and the scoping was pinned by probe form `B15` (`rq "$Q" PING`,
  disposition `blind`), controlled by widening the token regex so the blind rows redden.
- **What the paragraph now ends on**, and it is stronger than what it taught: it keeps *read the
  stated bound before filing against it*, and adds **checking a bound means checking every place it
  is stated** — one bound written in two files is one claim with two chances to go stale.
- **Two stale figures in the same subsection, both created by U33 and both corrected here** — they
  are stale precisely because `B15` is the form the loose wording hid: **35 → 36 probe forms**, and
  **five → six blind forms**, with the sixth now named in the list (`rq "$Q" PING`, a command not
  spelled `GRAPH.…` at all). Directly observed in this run against
  `skills/joern-cpg/scripts/test-stamp-wiring.sh` at `HEAD`: `grep -cE '^[ABC][0-9]+ '` → **36**
  (A 15, B 15, C 6), `grep -cE '^[ABC][0-9]+ +blind'` → **6**. The 2026-09-09 entry below records
  five/35 and stays as written — it was correct at its date.
- **One promotion folded in, not a new section** — see `claude/graph-dba/kaizen/history.md` for the
  entry (`b7d3f0a2-6c41-4e58-9a03-2f8e1d5c7b96`). Point 4 narrated the continuation-joining trap as
  an episode; it now closes on the portable rule — *any reader that reconstructs logical lines from
  a shell source must join continuations with nothing, because a space manufactures a token boundary
  bash never creates* — replacing a closing sentence that carried no information.

## 2026-09-09 — `guard-testing-techniques.md`: the open live instance became the file's worked example (U32 follow-up)

- **What:** U31 shipped this knowledge base citing `skills/joern-cpg/scripts/test-stamp-wiring.sh`'s `rq` call-site check as an **open** fourth-generation instance — *"do not treat that check as a model."* U32 closed it (`4df5e45`, `graph-dba`). Those five lines are replaced by a `### The worked example` subsection under the coverage-probe section: **1,225 → 1,867 w**.
- **Why it is now model-worthy, and it is not because it is clean.** Four reasons, and the file states each: it took **both** closures where each belonged (widened the *unit of analysis* — a shell call site is not a line — and narrowed the claim to five named blind forms); its mechanism is a **coverage probe** of 35 forms on three axes with each row adjudicated **twice** (bash says whether `rq` really receives the command, the delivered reader says whether it flags it), over a reader extracted into one function so the probe cannot certify a re-typed copy; **the claim is pinned by the mechanism** — widening the reader without rewriting the stated bound turns a `blind` row red, which is the structural answer to the stated-reach-exceeds-mechanism defect; and the probe **caught generation five in flight**, inside the fix for generation four (bash joins continuations with nothing, not a space, so `GRAPH\` + `.DELETE` is one word to bash and `GRAPH .DELETE` to the reader), which is this file's own thesis closing on itself.
- **Verified by execution before promoting, not taken from the report.** Against the delivered reader extracted verbatim: the continuation form that previously scored `PASS all 3` is now flagged (`GRAPH.DELETE@451`) with the clean tree still at `SITES=3` and no tokens; three of the five stated-blind forms reproduce as blind, with bash confirming `rq` genuinely receives `GRAPH.DELETE`; two widenings (tab-separated call, single-quoted literal) flag correctly.
- **The new transferable half — a principled stopping rule.** You may stop widening when **a miss is bounded to a false failure rather than a false pass**. Measured with a passing control: `rq "$Q" PING` → rc 1 (`ERR wrong number of arguments`), `INFO` → rc 1 (empty reply), `GRAPH.RO_QUERY` → rc 0 with `Query internal execution time: …`. The static check is a lint over a property the runtime already enforces, which is what licenses it to stop at a stated bound. Establish that asymmetry before accepting a narrow claim — without it, narrowing is conceding.
- **One retraction, kept in the file on purpose.** Probing the bound I found `PING`/`INFO`/`keys` — contiguous literals, genuinely passed to `rq`, unflagged — and drafted them as generation six. They are not: the stated mechanism is scoped to `/GRAPH\.[A-Za-z_.]*/` and the bound's first clause already reads *"a command that is not one contiguous `GRAPH.<word>` run of characters at the call site."* I had drawn a finding from the bound's **examples** rather than its **claim**. Recorded as a rule in the file, since it is the same defect shape pointed at a reader.
- **Docs touched:** `claude/tdd-engineer/{guard-testing-techniques.md,kaizen/history.md}`. Nothing under `skills/joern-cpg/**` — read only.
## 2026-09-09 — new on-demand knowledge base `guard-testing-techniques.md`: 3 inbound `MENTIONS` promotions from the orphan backlog (U31)

- **What:** U31 of `claude/docs/plans/kaizen-distillation2-coordination.md` — the **orphan-backlog** unit, the first shaped by *edge* rather than by producer. The 11 nodes it covers carry **0 `PRODUCED` edges** and are alive only on `MENTIONS`; every earlier unit was organised by producer, so none of them could ever have been reached. `tdd-engineer` carried **3** of the 12 edges, all tagged by `teco` (U18b) as *"true, verified, and not coordination doctrine"* after that agent promoted the coordinator half of each. All three are about testing a **guard** — an AST reader, lint rule, grep check or wiring assertion whose subject is other code's *text* rather than its behaviour.
- **A new file rather than three prompt bullets.** `tdd-engineer` had no knowledge base; every other agent holding this much on-demand material has one (`analyst`, `qa-engineer`, `data-scientist`, `graph-dba`, `devops`). These techniques are narrow — they bind only when the deliverable *is* a guard — and `tdd-engineer.md` is always-loaded, the highest-price destination on the team. So: `claude/tdd-engineer/guard-testing-techniques.md` (1,225 w), plus a 4-line pointer blockquote at the head of the Principles section, matching the `devops`/`qa-engineer`/`analyst` convention.
- **`f6b820ae-9d47-4c15-83e0-72a1de5b9013` (2026-09-07) — a mutation test and a coverage probe answer different questions.** A mutation test asks *does my reproduction die?* and its enumeration is your imagination; a probe asks *what can this reader **not** see?* and enumerates every syntactic form the guard claims to catch, running the **delivered** reader over each. Promoted with both design rules (derive the enumeration from the language/runtime so a new version reddens the probe rather than silently opening a hole; ship the probe as a **test**, not a script).
- **`a8f3c521-6d09-4b7e-95a2-30fe14b7c8d6` (2026-09-07) — a resolver has two axes and only one is finishable.** TARGET axis (which binding forms bind a name) is derivable from `ast` and can be closed; VALUE axis (which expressions denote the object) is alias analysis and cannot be. Hence the trap: a probe varying only the target axis comes back empty and certifies nothing. Promoted with the stopping rule — state the rule narrowly and truly, name the exact resolution mechanism, move the load-bearing evidence to a behavioural test, and record the measured residual so the narrow claim reads as a decision rather than an omission.
- **`b2d64f19-7e35-4a80-93c6-1af7c25b0e88` (2026-09-07), implementer half — the docstring/body gap.** A docstring stating *semantic* reach over a body doing a *syntactic* match regenerates its defect after every fix. Promoted with the productive gate question (*smallest edit to production code that satisfies the docstring and survives the body*), the two legitimate closures, and the instruction to **say which you chose**. Its reviewer-facing half went to `claude/analyst/review-techniques.md` in the same unit.
- **Re-derived, not re-read.** The AST census these rest on was recomputed here under **two** definitions (CPython 3.12.3, system `python3`): the 8-name binding-field set gives **27** classes, the entry's own 5-name set gives **22** (dropping `Global`, `Import`, `ImportFrom`, `MatchMapping`, `Nonlocal`). Both are in the new file as a table, because the disagreement is the lesson — a derived enumeration still has a parameter, and *"the grammar has N binding forms"* is not a fact without its field set. The `185 passed` / `1 failed / 184` receiver-vs-attribute alias pair is carried from `teco`'s U18b verification, which read it verbatim out of the delivered guard's own docstring, and is labelled as a worked instance rather than re-measured here.
- **A fourth-generation instance is open and is cited as such**, not as a model: `skills/joern-cpg/scripts/test-stamp-wiring.sh`'s `rq` call-site check states a reach its line-oriented mechanism does not have. Confirmed by execution in this unit (see `claude/cobb/kaizen/history.md`); owned by `graph-dba`, reported to the coordination rather than fixed here.
- **Graph:** read before mutating. `f6b820ae` and `a8f3c521` — 0 `PRODUCED` / 1 `MENTIONS` ⇒ `otherRemaining = 0` ⇒ full-node `DETACH DELETE`. `b2d64f19` — 0 / **2**; the `analyst` edge was resolved first leaving the node alive (`otherRemaining = 1`), and the node was cleared only after this promotion landed.
- **Docs touched:** `claude/tdd-engineer/{guard-testing-techniques.md,tdd-engineer.md,kaizen/history.md}` · `claude/README.md` · `claude/AGENTS.md`.
## 2026-09-07 — `kaizen_team` distillation, chunk B: 8 entries (2026-08-31 → 09-03) — 2 promotions (4 entries), 3 discarded, 1 kept open; agent closed out

- **What:** `cobb`-run §5 distillation (`agent-maintenance` skill), unit **U9** of
  `claude/docs/plans/kaizen-distillation2-coordination.md`. Scope was **chunk B** — the eight
  current-shape entries dated 2026-08-31 through 2026-09-03, i.e. everything left after chunk A
  (unit U8, same date). This agent now has **zero** entries in `kaizen_team`. No legacy
  (`author`-property) entries have ever existed for it.
- **Verification changed three dispositions.** Re-deriving rather than trusting the entry text moved
  `a3f6e1c4`, `a93fb14a` and `f3b1c2ae` from "promote" to "already documented" — in each case the
  fact was already written, in more depth, at the point that changes behavior (a code comment block
  or a component `AGENTS.md`), which only a look could reveal. It also **falsified one third of**
  `773f87f4`'s premise. Dedup grep for all eight `entryId` prefixes across the repo's markdown
  returned nothing; the near-collision flagged in the brief (`f3c9a1d2` here vs. `architect`'s
  `f3c9a2e1`) was confirmed distinct on the full id, date and subject.

**Promoted (2 dispositions covering 4 entries):**

- **`a9bf4f7c` + `c31f0e58`** (both 09-02, the FastAPI 0.139 route-table trap; the second entry
  names itself the companion of the first) → **one merged block folded into the existing route-table
  section** of `skills/python-web-quirks/SKILL.md`, not a third near-duplicate section beside the two
  that unit U7b added. The fold matters because that section's standing advice
  (`frozenset(r.path for r in FastAPI().routes)`) is the *exact* idiom the new fact defeats — correct
  for the bare app it computes the doc-route exemption from, unfalsifiable for any app with an
  included router. Section heading widened to cover both halves; frontmatter `description` and the
  `skills/README.md` row updated to match. Re-verified against the entries' own probes: `app.routes`
  holds one opaque `fastapi.routing._IncludedRouter` (35 inner routes reachable only via
  `.original_router.routes`); `include_router(prefix=…)` lands on `route.include_context.prefix`, not
  on the inner `APIRoute.path`, while `APIRouter(prefix=…)` declared at construction **is** baked in;
  `Mount` appears directly with a `"/"` catch-all normalised to `""`; `starlette.routing.Host` has
  neither attribute, so the unclassified branch must raise. The positive-control lesson (assert the
  flatten sees a route you know exists, in the same test that uses it to assert one doesn't) was kept
  — every failure in this family is a *false green*, not a red.
- **`f3c9a1d2` + `2b8e40cf`** (both 09-03, mutation-testing method) → **one appended sentence on the
  existing "Prove a new assertion against the mutant" Principles bullet** in `tdd-engineer.md` — the
  bullet chunk A promoted hours earlier — never a second bullet on the same theme. They are the
  operational completion of it: that bullet tells you to break the code and confirm red, and its
  remedy is "assert the raiser's own message"; these two are the cases where following it lands you
  wrong. `f3c9a1d2`: restoring an identical copy over a duplicated string is an **equivalent mutant
  by construction** — nothing can kill it, and the killable form is the drift (edit the canonical
  copy, leave the duplicate stale), which is what a "two homes" finding actually predicts. Verified
  at `model-bench/tests/test_report.py:257`, whose assertion is `stats.mdd_clause(rp, "items") + "."
  in line` — the report *rendering* the stats module's string, so a verbatim duplicate passes and
  only drift fails. `2b8e40cf`: a guard whose downstream twin raises the same sentence is separable
  only by **ordering**, never by `match=`. Verified narrower and more precisely than the entry states:
  `stats.py:186` and `stats.py:849` do **not** raise byte-identical messages today — but both contain
  `"precondition 4"`, which is what `test_stats.py:1598`'s `match=` reads, so the echo satisfies the
  match either way. The fix already shipped in model-bench (the message names its own function;
  `bootstrap_seed=None` makes the two orders diverge) — what is promoted here is the general method,
  which model-bench's own test docstring cannot carry to another codebase.

**Discarded (3) — all "already documented", each found only by looking:**

- **`a3f6e1c4`** (08-31, a dispatch-time dedup guard keyed on raw tool-call JSON args defeated by a
  wrapper-applied default) — already documented in `falkor-chat/server/falkorchat/executor.py`
  (lines ~320-378) as a ~35-line comment block plus `_resolve_add_to_cart_dedup_args`'s docstring,
  covering every claim in the entry and more: why a per-tool resolver rather than a generic JSON-schema
  `default` lookup (neither tool declares one), why `remove_from_cart` is deliberately *absent* from
  the table (an omitted `quantity` there means "remove the whole line", not an implicit number), and a
  forward instruction to whoever adds the next write tool. `docs/reviews/salesperson-tool-reliability-ml.md`
  §15.2 carries the incident. The entry's "can be defeated" is also past tense — the loophole is closed.
- **`a93fb14a`** (09-03, a public Python function named `test*` collected by pytest in every module
  that imports it) — already in `model-bench/AGENTS.md` (line ~96) as the general rule, in one
  sentence, citing this exact incident: *"A public name starting with `test` is collected by pytest as
  a test in every module that imports it — which is why FR-17a's function is
  `models_with_stored_results`, not `tested_models`."* Nothing to add, and `skills/python-web-quirks/`
  would be the wrong home for a fact already stated where the code lives.
- **`f3b1c2ae`** (09-03, `Path(".").name` is `""` but `Path("..").name` is `".."`) — re-derived
  (`python3 -c` over `("", ".", "..")` reproduces exactly) and already documented **at the point of
  use**, six lines of comment at `model-bench/modelbench/results.py:424-429`, which states the
  asymmetry, why `"."` is redundant with the first clause, why `".."` is not, what dropping it would
  write (`results/runs/..json`), and the review finding (P2-4) that prompted the measurement. Same
  reasoning as chunk A's `b3f0a6b2`: a second copy elsewhere is a drift risk, not a gain.

**Kept open (1) — `plan.md` K-011.**

- **K-011** ← `773f87f4` (09-03) — a `select = ["E","F","W","I"]` ruff config cannot detect dead code.
  **Re-derived and true:** `ruff 0.14.14`, `ruff check --isolated --select E,F,W,I` on a file
  containing only `def _dead(x): return x` exits 0 with "All checks passed!" — `F401` is unused
  *imports* only, and nothing in that selection reaches an uncalled module-level private function.
  **But the entry's component list is wrong on one of three:** `cypher-mcp` has no ruff configuration
  at all (no `pyproject.toml`; `pytest.ini` + `requirements-dev.txt` only, and no `ruff` anywhere in
  the component). The three that do share the selection verbatim are `model-bench`, `mcp-monitor` and
  **`falkor-chat/server`**, which is the original the other two copied — each says so in its own
  `[tool.ruff.lint]` comment. Kept open rather than promoted because the right home is
  `model-bench/AGENTS.md` (beside the sibling `test`-prefix trap it already carries), **outside
  `cobb`'s write remit**, and because `skills/python-web-quirks/` is scoped to web/async plus
  pytest/import-timing traps — a lint-configuration fact would be stretching that scope to make it
  fit. K-011 records the corrected component list so the eventual write does not repeat the error.

- **MENTIONS tags added:** none. No entry in this chunk is substantively about a different agent —
  the two mutation-testing entries are method facts in this agent's own discipline, and the three
  discarded ones are project facts already at their point of use.
- **Graph cleared:** all eight nodes fully deleted. Each was read for edge counts first
  (`producedEdges = 1`, `mentionEdges = 0`, so `otherRemaining = 0` on every one), making the
  `PRODUCED` edge the last edge and a whole-node `DETACH DELETE` the correct clear rather than an
  edge resolve. This history entry was appended and confirmed **before** any graph mutation.

## 2026-09-07 — `kaizen_team` distillation, chunk A: 12 entries (2026-08-25 → 08-30) — 5 promoted, 3 discarded, 4 kept open; one new Principles bullet

- **What:** `cobb`-run §5 distillation (`agent-maintenance` skill), unit **U8** of
  `claude/docs/plans/kaizen-distillation2-coordination.md`. Scope was **chunk A only** — the twelve
  current-shape entries dated 2026-08-25 through 2026-08-30. This agent's **eight** entries dated
  08-31 and later are chunk B, a separate unit, and were **not touched**. Zero legacy
  (`author`-property) entries exist for this agent.
- **Verification changed two dispositions.** Re-deriving against live code, rather than trusting the
  entry text, moved `f3a1e6b2` out of "false, same premise as the U4 entry" (it is the genuinely
  narrow true case — see K-010) and moved `a1b2c3d4`, `b3f0a6b2` and `84864fe1` into "already
  documented" once their real homes were found. **Two 8-char `entryId` prefixes collided** with
  unrelated entries cleared in earlier units — `b3e2f6a1` (cobb, legacy 2026-08-21, `tools:`
  allow-list enumeration) and `a1b2c3d4` (qa-engineer, multi-column read corruption). Both were
  confirmed distinct by date **and** subject before concluding no prior `K-` item existed; these ids
  are hand-shaped, not `uuid4`, so a prefix grep alone is not a dedup check.

**Promoted (5):**

- **`b3e2f6a1`** (08-26, validation checks in a fixed order masking the field under test) **+
  `b2c3d4e5`** (08-30, a `"WITH DISTINCT" in compiled.cypher` substring assertion surviving a
  `zip()` column-swap mutation) — **one new Principles bullet in `tdd-engineer.md`**, placed
  before "Fast, isolated, deterministic by default": *prove a new assertion against the mutant, not
  just against red*. Promoted **merged**, because they are one rule with two faces: an assertion a
  coincidence can satisfy is green before the behavior exists, green after, and green when it is
  deleted. The existing RED-step rule ("confirm it fails *for the right reason*") does **not** cover
  this — it proves the test *can* fail, not that the assertion can reject a *wrong* implementation.
  Two independently-captured live incidents in one week is what closed the standing parking-lot
  question of whether mutation-testing guidance "earns its keep" in this deliberately lean prompt;
  the bullet is one rule + one why-clause + three concrete triggers, no incident story.
- **`b3e2f6a1`, pytest-specific half** → new section in `skills/python-web-quirks/SKILL.md`: a
  `pytest.raises(SomeError)` with no `match=` passes on an unrelated same-type raise from an earlier
  check in the same validator. Both the general rule (prompt) and the language-specific mechanics
  (skill) are needed; neither restates the other. The falkor-chat instance itself needed no write —
  `server/tests/test_services.py`'s `OVERSIZED_ISOLATED_STEP` comment block already records it in
  full, and the skill cites that rather than copying it.
- **`a3f5c9d2`** (08-30, `falkorchat.config` freezing `FALKORCHAT_OPENCODE_CONFIG`/
  `FALKORCHAT_MODEL_CONFIG` at import) → **generalized an existing** `skills/python-web-quirks/SKILL.md`
  section rather than adding one. That section was scoped to *"a pytest autouse fixture's
  `monkeypatch.setenv`"*; the entry's contribution is that the identical freeze bites a plain script
  with no pytest anywhere — `os.environ[...] = ...` inside `main()` looks early but runs after the
  top-of-file `from falkorchat import …` already computed the constant. The remedy differs too
  (placement above the imports, not `monkeypatch.setattr`), which is why it is worth stating. Heading
  widened, frontmatter `description` and `skills/README.md` entry updated to match ("two" → "three"
  pytest/import-timing traps). Re-verified live at `config.py:48-51`. The falkor-chat corollary was
  already published (`falkor-chat/docs/SERVER.md`, "A wired agent requires two config files") — no
  project-doc write needed.
- **`32ca9b55`** (08-25, an `Edit` silently reverting mid-task in a file with disclosed concurrent
  uncommitted edits) → new bullet in `skills/agent-standards/claude-code.md` § "Bash tool
  environment", as a sibling of the shared-scratchpad entry. **Not already covered:** `claude/AGENTS.md`
  documents the `git add`/`git commit` **index** race and the skill documents the shared **scratchpad**
  — this is a third surface, the working-tree bytes themselves, and a grep for it across both files
  returned nothing. Kept honest about its status: a single observation, mechanism inferred (the other
  session writing a whole-file buffer read before the edit), not re-derivable now — §5's
  "unverifiable ≠ discard" case, kept for value with the doubt stamped in the text.
- **`e3f1a9c2`** (08-26, LM Studio HTTP 400 `Error loading model` on the first call after idle,
  succeeding on immediate retry) → **folded into** `claude/data-scientist/lm-studio-model-notes.md`'s
  existing JIT auto-load bullet, not stacked as a near-duplicate section beside it: that bullet
  already establishes that the first call after idle pays the JIT load, and this entry's news is
  that the load can **fail** rather than merely be slow. No model was loaded on the shared LM Studio
  server to verify — the disposition rests on the entry's own live evidence plus the already-verified
  JIT mechanism it extends.

**Discarded (3) — all "already documented", each found only by looking:**

- **`a1b2c3d4`** (08-30, a reject-outright compile-time guard for "`ORDER BY` column not in `RETURN`"
  being over-broad because it also outlaws the superlative shape) — already in
  `claude/graph-dba/falkordb-quirks.md` (lines ~260-290, verified 2026-08-30, same day), in
  substantially more depth than the entry: the `GRAPH.EXPLAIN` plans for both forms, the three-row
  wrong-answer repro in both creation orders, the exact-duplicate collapse check, and the residual
  tie caveat. The entry's own `evidence` field points there. Nothing to add.
- **`84864fe1`** (08-28, a replayed-history tool-use breadcrumb backfiring — the model imitating the
  breadcrumb format verbatim in customer-visible output, a false-verification claim on top of the
  original fabrication) — already in `falkor-chat/docs/HISTORY.md` (2026-08-28 K-056 entry, ~90
  lines) and `falkor-chat/docs/reviews/salesperson-tool-reliability-impl.md` MAJOR 1, which is where
  the "not an inert leftover, an active severity increase" reading was established and acted on
  (the breadcrumb was reverted). Both records are richer than the entry.
- **`b3f0a6b2`** (08-29, `ws:nlq-eval` has no fusion/dedup applied, so a raw `count(e)` counts
  un-fused nodes) — **live re-verified** read-only against `ws:nlq-eval`: Organization 17 nodes / 5
  distinct names, Person 5/3, Product 7/3, Location 11/8, exactly as claimed. Discarded anyway,
  because the fact is already recorded **at the point of use**, which is the only place it changes
  anyone's behavior: `server/tests/eval/nlq_golden_set.jsonl`'s `nlq-31` rationale reads "17 raw
  entity nodes (un-fused — 5 distinct organizations named across 12 documents), live-verified". A
  second copy in a project doc would be a drift risk, not a gain.

**Kept open (4) — `plan.md` K-007 … K-010.** All four are `falkor-chat` project facts whose correct
home is `falkor-chat/docs/SERVER.md` §1.7 or `falkor-chat/AGENTS.md` — **outside `cobb`'s write
remit**, so each is recorded with the exact target doc and section rather than half-written:

- **K-007** ← `f3a8c9d2` (08-25) — a concurrently-used shared `falkordb-dev` makes `pytest -q` show
  21-53 transient failures across unrelated test files, a different set each rerun, reproducing on a
  `git stash`-clean HEAD and then going fully green. → `SERVER.md` §1.7.
- **K-008** ← `a7f3c1d2` (08-28) — the seed scripts take a **bare** workspace id and prepend `ws:`
  themselves, so passing the prefixed form `GRAPH.LIST` displays creates a bogus `ws:ws:<id>` key
  silently, exit 0. Re-verified 2026-09-07 (`bootstrap_schema.sh:111`, no guard in any of the five);
  the stray `ws:ws:acme` from the original incident is gone from the loaded-graph list. **Explicitly
  checked against the cross-reference and *not* superseded** by this agent's `Repository._read_structure`
  K-005 fix (`falkor-chat/docs/HISTORY.md`, 2026-08-25) — that fix concerns `verify_workflows.sh`
  false-negativing an intact snapshot when `reference` was fully deleted, a different mechanism that
  happens to surface in the same re-seed workflow. → `falkor-chat/AGENTS.md` "Key scripts" table.
- **K-009** ← `d8f0c1e2` (08-30) — `querygen.DatasetSchema` is also hand-constructed in
  `test_repository.py` (~3498, ~3523), so a `labels`-shape change breaks a file a `test_querygen.py`-
  scoped grep never reaches. Re-verified 2026-09-07. → a `DatasetSchema` docstring line is the
  cheaper fix than a doc bullet.
- **K-010** ← `f3a1e6b2` (08-30) — `trace=True` alone writes zero `TraceEvent`s from an ad-hoc
  in-process executor built without `tracer=GraphTracer(repo)`. **Re-derived specifically against the
  U4 finding that a twin entry rested on a false premise**, and it survives: `executor.py:605`
  (`tracer = self._tracer if run["trace"] else _NULL_TRACER`) plus `__init__`'s
  `self._tracer = tracer or _NULL_TRACER` make it two independent conditions, while `app.py:540-543`
  does pass `tracer=GraphTracer(repo)`, so the REST path is unaffected and U4's correction stands.
  The trap is confined to a hand-rolled harness. K-010 records that scoping, because the bullet is
  easy to write as the false general claim. → `SERVER.md` §1.7, beside the structurally identical
  `_default_clock` bullet.

- **MENTIONS tags added:** none. `e3f1a9c2` was the only candidate (LM Studio is `data-scientist`'s
  domain) and was **fully dispositioned into that agent's knowledge base directly** instead — per §5,
  preferred over tagging, which would only re-surface an entry already routed.
- **Docs touched:** `claude/tdd-engineer/{tdd-engineer.md,kaizen/plan.md,kaizen/history.md}`,
  `claude/data-scientist/lm-studio-model-notes.md`, `skills/agent-standards/claude-code.md`,
  `skills/python-web-quirks/SKILL.md`, `skills/README.md`.
- **Cleared:** all 12 in-scope entries removed from `kaizen_team` — see the per-entry edge counts in
  `claude/cobb/kaizen/history.md`'s matching 2026-09-07 entry.

## 2026-08-25 — `kaizen_team` distillation pass: 2 entries reviewed, both discarded (no prompt change)
- **What:** `cobb`-run distillation (agent-maintenance skill §5), unit U9 of a team-wide pass. Found 2 raw entries for `tdd-engineer`: 1 legacy (`author` property) and 1 current-shape (`PRODUCED` edge). Both re-verified against live code and discarded — neither required a prompt/knowledge-base/project-docs change.
  - **`4e6a2c0e-6f3b-4b8a-9d4e-2b7a1c9f5d31`** (legacy, 2026-08-22) — fact: a plan's literal pseudocode for a new catch-all rejection message can silently conflict with a brief's "existing regression suite stays unmodified" mandate; the old wording must survive as a superset, not be replaced verbatim. Evidence: implementing `cypher-mcp/server.py`'s `authorize_write()` catch-all verbatim per `docs/plans/kaizen-agent-ontology.md` §3.1 broke 4 pinned tests (7, 15, 15b, 16); fixed by extending the old sentence rather than replacing it. **Re-verified live:** `authorize_write()`'s final rejection string (server.py:612-619) still reads "neither an author-write (...) , a producer-write (...), nor a recognized curator shape (...)" — the pre-existing substring intact, new shapes appended to the same sentence; tests 7/15/15b/16 still assert `"neither an author-write" in out`. **Disposition: discard — already documented in project docs**, in more depth than the raw entry: `docs/reviews/kaizen-agent-ontology-impl.md` ("What's verified solid" section) records this exact deviation, the plan's literal pseudocode it diverged from, and the verification (`git show` diff, all 4 pinned tests independently re-run). The entry's own `suggestedHome` ("project docs") is satisfied by pre-existing work, not a gap.
  - **`a5153389-63b6-402c-ac43-e9f8a8e7e843`** (current-shape, 2026-08-23) — fact: a regex-gate refactor pattern — split a boolean-returning shape-matcher into a pure matcher (returns match data or `None`, stops before a trailer/tail check) plus the boolean wrapper, so a caller can distinguish a specific near-miss (shape matches through the map but fails only the trailer) from any other non-match, and emit a specific message without touching the authorization boolean. Evidence: `cypher-mcp/server.py` `_producer_write_shape_match` / `_producer_write_agent_id` / `_producer_write_trailer_message`, mutation-tested. **Re-verified live:** all three functions exist exactly as described (server.py:406-494), `authorize_write()` calls the shape-matcher a second time (line 582) specifically to emit the near-miss trailer message, and `test_server.py` carries the two `trailing_return` tests (lines 961, 980) pinning this behavior. **Disposition: discard — already fully captured**, and more precisely, in the code's own docstrings (`_producer_write_shape_match`'s docstring states the exact same rationale verbatim). Neither of the entry's own suggested homes fits: `skills/python-web-quirks` is scoped to live-verified asyncio/FastAPI/Starlette/pydantic gotchas, not general regex-refactor patterns; `agent-standards` is scoped to agent-artifact format specifics, not general coding technique. Not durable/generalizable enough to justify a new tdd-engineer knowledge-base file for one entry.
- **Why:** Team-wide learnings-graph distillation pass (`claude/docs/plans/kaizen-distillation-coordination.md`, unit U9), dispatched by `teco`.
- **MENTIONS tags added:** none — both entries are genuinely about `tdd-engineer`'s own work on `cypher-mcp`, not another agent.
- **Plan items:** none opened — no kept-open/unresolved entries this pass.
- **Cleared:** both entries removed from `kaizen_team` (legacy entry via unconditional curator `DETACH DELETE`; current-shape entry via curator full-node clear after confirming `otherRemaining == 0` — 1 `PRODUCED` edge, 0 `MENTIONS` edges).

## 2026-08-25 — K-006 closed — the conventions bullet now says which convention wins (conventions-precedence family)
- **What:** `:36`'s *"Idiomatic, clean production code"* bullet opened with *"Follow the language and project conventions you observe"* and closed with *"Match the surrounding code's style"* — two co-equal imperatives that answer differently when a file deviates locally, at a decision point the agent stands at on essentially every edit. This was the **strongest of the family's three instances**. One sentence appended, resolving them in the order they already appear; +25 w.
- **Why the C5 sweep could never have fixed it.** C5 flagged the pair as a class-7 duplicate and kept it under finding 5 — correctly, since the two sentences are not equivalent. The sharper reading is that *the non-equivalence is the defect*: an **ambiguity finding wearing class-7 clothes**. A dedup sweep's only move is to delete one, and deleting either silently picks a winner.
- **The rule, byte-identical in all three implementer prompts:** *"Where a file or folder deviates locally from the project norm, match it, not the norm — mixing both in one place is worse than either applied consistently."* Identity is deliberate — three paraphrases would recreate exactly the divergence that made this a `agent-maintenance` §4 **check-5 boundary-reciprocity** problem rather than three separate nits.
- **Not relocated to a shared file, and the reason is stronger than "no plausible home."** Finding 15 says a rule binding N>1 agents belongs where all N read it, and this binds three. But no shared file owns *how to author code*, and root `AGENTS.md` would be an **actively bad** home: its document conventions are stated as absolutes (`never begins with m<digit>`, the closed role set, the closed `Status:` set), and a general "a local deviation beats the project norm" principle sitting beside them hands every agent a lever to justify deviating from them. The rule would not sit inertly there; it would undercut them. So: byte-identical duplication, **plus a mechanical identity guard** — `audit-team.sh` **check 10**, which fails when the sentence is in some but not all three, and passes when it is in all three or none (removing it everywhere is a legitimate family decision; removing it from one is the defect). Recorded as plan finding 22.
- **Why the guard was necessary and not belt-and-braces.** All three kaizen plans said "fix together or not at all" and nothing read those plans at edit time. The mitigation was prose in files nothing loads. Stage F had shipped the enforcement machinery the same day, and this is exactly the class of guarantee it exists for: deterministic, must-always-hold, therefore a mechanism rather than hopeful prose.
- **The family boundary is a test, not a judgment call:** *two co-equal scope claims in one prompt with no tiebreak between them.* `graph-dba:53` and `devops` were checked and excluded — not because they are a "different axis" (that was my first, wrong reasoning; `graph-dba:53` is structurally identical and has a real local-vs-project instance in this repo, since `cpg_*` graphs carry Joern's imported schema against the `PascalCase`/`UPPER_SNAKE` norm) but because each states **one** scope, so there is nothing to tiebreak. Recorded so the next reader does not re-open it, notice the shape match, and conclude the exclusion was arbitrary.
- **Verified:** `audit-team.sh` **PASS**, check 10 green at 3/3; negative-tested at 2/3 (FAIL, exit 1) and 0/3 (PASS). `cobb` §7 lint: 0 blockers, 1 major, 3 minors, 2 nits — all applied.

## 2026-08-24 — Prompt-waste compression, Stage C5: one class-5 cut; file certified at its editorial floor
- **What:** Unit C5 of `claude/docs/plans/prompt-waste-reduction.md` (with `qa-engineer.md` and
  `data-scientist.md` as one commit). **2,163 → 2,154 w (−9, −0.4%).** One edit, one pass.
- **The file's citation habit (plan finding 9 — name it before cutting):** **incident
  fingerprints.** This file carries no dates, no authority markers and no supersession trails
  anywhere — its class-6 residual is literally zero. What survives instead is hyper-specific failure
  nouns naming the exact case a promoted lesson came from. `cobb` independently scanned the whole
  file and confirmed the habit inventory is **exhausted at one cut**.
- **Removed (class 5, already on record):** the parenthetical "(one item in a list of several small
  fixes)" from Workflow step 1's carried-finding branch. The originating incident is this file's
  **2026-08-21 distillation entry** (entry `a3f1c9e2…`: a backlog-carried `analyst`-gate finding
  already silently resolved by unrelated work) — and **that entry's own restatement of the promoted
  rule already omits the parenthetical**, so the prompt and the promoted form now agree. It was also
  narrowing the rule to batches, which the branch's lead phrase never did; the trigger ("a carried
  finding from a backlog or coordination doc") is untouched.
- **Gate (a) inventory — all preserved:** the red→green→refactor loop's four steps and its
  fail-for-the-right-reason and never-refactor-on-red rules; all ten Principles including symmetric
  fixture teardown, marker/tag gating over bare reachability checks, the positional-rule corpus
  rule, and the shared-helper caller-risk rule; Workflow steps 1–5 with all three task-arrival
  branches and all three baseline branches (greenfield / already red / can't run here); the CPG
  check + freshness rule both branches; the verbatim three-form `CPG:` line contract with its
  `not applicable` scoping clause; all four Guardrails including the write-guard deny-list and the
  interactive-commit grant with its full never-list and delegated-subagent carve-out; the Cypher
  capture template and call line. Audit check-8 tokens (`` `git add` ``/`` `git commit` ``,
  "delegated subagent") verified present after the edit.
- **Considered and rejected — judged keeps, all upheld by the lint:** the "double-fire or
  idempotency bug" phrase in step 5 (looks like a fingerprint, but it names the *class* of bug a
  reproduction test structurally cannot catch — generalising it makes the why circular); both
  conventions sentences in "Idiomatic, clean production code" (they answer differently when a file
  deviates locally — see **K-006**, opened because that non-equivalence is itself the defect); the
  three separate "as a subagent you can't ask mid-run" fallbacks (`cobb` strengthened this from
  three-decision-points to a **certification requirement** — `agent-maintenance` §4 check 3 obliges
  *every* "ask" phrasing to carry its own carve-out, so thinning one is a regression, not a trim);
  "(deliberately left escalating either way — genuinely unresolved, not a bug)" on the
  `docs/BACKLOG.md` guard (stops the agent reporting the escalation as a misconfiguration); and the
  near-duplicate brevity instructions "Announce each cycle briefly" / "Narrate the cycle compactly"
  — cutting an anti-verbosity rule inside a verbosity-reduction plan is the wrong trade, and Stage D
  of that same plan *adds* rules of this kind.
- **Residual after this unit: 29 w, all class-7 keeps; class-6 = 0 w.** `cobb`'s measurement. The
  file is at its editorial floor; the only remaining lever is structural.
- **Verified:** gates (a)–(e) green. `./claude/scripts/audit-team.sh` **PASS**. `cobb` §7 lint:
  **0 blockers, 0 majors, 0 findings introduced by the edit** (1 pre-existing minor → K-006, 1 nit
  keep-listed). Orphan-phrase grep across the repo clean.
- **Plan items:** **K-006 opened** (conventions precedence). Not fixed here — it is a rule change,
  and bundling it would make this commit non-revertible as a pure waste-reduction change.

## 2026-08-23 — Prompt-waste Stage B wave 2: three boilerplate blocks compressed to pilot shapes
- **What:** CPG-freshness clause, interactive-commit-grant bullet, and learning-capture intro/tail compressed to the pilot-validated wordings in `architect.md`/`coder.md` (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B), including wave 2's uniform freshness form ("when a `teco`-issued brief states the graph's freshness, take it as given").
- **Removed (class 5/6, already on record):** the grant's "same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`" — this file's 2026-08-21 grant entry; the tail's inbox-replacement sentence + ", exactly like the old inbox was" — this file's 2026-08-21 inbox-deletion entry; the freshness clause's "(2026-08-19)" date and "without re-deriving staleness yourself" restatement — this file's 2026-08-19 freshness-centralization entry; the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement (mechanics live in the Cypher template below); the grant parenthetical's "— not spawned via `Agent`/`Task` as an isolated delegate" (moved into the carve-out sentence: "**As a delegated subagent** (spawned via `Agent`/`Task`)…").
- **Gate (a) inventory — all preserved:** grant scope (own verified changes, this session, explicit path), full never-list (`-A`/`.`/`-a`, push/reset/rebase, amend), delegated-subagent carve-out + audit check-8 tokens, freshness rule both branches (brief stated → given; standalone → current), Cypher template + call line verbatim, "skip task-specific/already-documented", "raw capture: `cobb` promotes; never edit your own definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Guardrails bullet: when running interactively (`claude --agent tdd-engineer`, a
  human present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own
  verified code changes from the session, by explicit path, never bulk-staged/pushed/reset/
  rebased/amended; the grant does not apply when spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Verify honestly"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). This is the agent whose live dispatch (D3′, U9) originally surfaced the ambiguity — it picked `not applicable` for a code-level task on a component with no loaded CPG, which the plan's own wording calls `considered, not relevant`. See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there (including this agent's own distillation, immediately below) has already been distilled and cleared — and this file's own pre-migration content was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: 1 entry promoted to Workflow step 1

- **What:** `cobb` processed the sole `author:'tdd-engineer'` entry in `kaizen_team`
  (`a3f1c9e2…`, 2026-08-20). Fact: a backlog-carried `analyst`-gate finding can already be
  silently resolved by later, unrelated work that happened to touch the same function —
  verified by grepping for the exact pattern the finding named (a function-local `import json as
  _json` in `falkor-chat/app.py`) and finding zero matches, since removed as a side effect of an
  unrelated parse-robustness fix. Added one clause to Workflow step 1, alongside the existing
  architect-plan-path and analyst-RCA-path sentences: when the task is a carried finding from a
  backlog/coordination doc, re-verify it against current code (grep the exact pattern) before
  implementing the fix, rather than trusting the backlog text as still accurate.
- **Why:** User-requested distillation pass, continuing the oldest-first queue
  (data-scientist → architect → graph-dba/tdd-engineer, the last two done together as the queue's
  final small entries).
- **Docs touched:** `claude/tdd-engineer/{tdd-engineer.md,kaizen/history.md}`.
- **Plan items:** none opened — the entry landed directly.

## 2026-08-21 — Workflow step 1 restructured into a short lead + bulleted doc-path branches (team certification, §7 lint fold-in)

- **What:** Step 1 ("Understand first") had grown, across four separate edits since 2026-07-09
  (plan-path handling, RCA-path handling, the CPG check, and today's carried-finding clause
  above), into a single run-on paragraph carrying seven distinct conditional
  instructions. Restructured, no content change: a short lead sentence, then the three
  "task arrives as X" branches (architect plan path / analyst RCA path / carried backlog
  finding) as a bulleted sub-list, then the CPG-check and freshness-delegation sentences kept
  as trailing prose (matching how other agents' equivalent steps handle their own CPG clause).
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in
  (`claude/cobb/kaizen/history.md`, 2026-08-21 certificate entry) — cognitive-load dimension:
  this step had become an outlier in conditional density against its own peer steps (2-5),
  and this exact prompt region has a documented prior failure mode of a buried instruction
  going unfollowed (DEF-3, the M4 U6 live-dispatch pass — a different clause, same class of
  risk: dense, un-bulleted conditional prose in a step an agent must parse in one pass).
- **Verified:** `bash claude/scripts/audit-team.sh` clean; re-read the restructured step against
  the CPG-agent-adoption wiring it also carries — the `CPG:`-line requirement and freshness
  delegation are unchanged in meaning, only in paragraph shape.
- **Plan items:** none opened — direct fix from a live lint finding.

## 2026-08-21 — Enforcement-parity fix: Guardrails now describes the deny-list write guard (team certification, §4 judgment half)

- **What:** The 2026-08-21 rollout below (`guard-tdd-broad-write.sh`, wired via frontmatter
  `hooks:`) shipped the hook itself but never touched this file's body — `tdd-engineer.md` had no
  Guardrails line describing it at all, "silent machinery" by the §4 enforcement-parity
  definition (a wired hook not described in the prompt it guards). Added one new Guardrails
  bullet, first in the list, stating what's unrestricted (source/test files), what escalates (the
  named deny-list doc kinds + `docs/BACKLOG.md`), and the same "don't attempt it expecting a
  rubber-stamp" framing every other guarded agent's prompt already uses.
- **Why:** Caught during a user-requested full team-coherence certification (`claude/cobb/kaizen/history.md`,
  2026-08-21 certificate entry) — enforcement parity is one of §4's five judgment-checklist items,
  and this was a same-day gap from the hook's own landing, not old drift.
- **Verified:** `bash claude/scripts/audit-team.sh` clean (check 1's hook-existence/executable
  check doesn't inspect prompt *prose*, so it couldn't have caught this itself — the §4 judgment
  half exists precisely for gaps the deterministic script can't see).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-21 — Added a broad-implementer `Write|Edit` deny-list guard (agent-permission-friction FR-2)

- **What:** `tdd-engineer` previously had no `Write`/`Edit` guard at all. Added
  `claude/tdd-engineer/hooks/guard-tdd-broad-write.sh`, wired via a new frontmatter `hooks:`
  block, over a **new** shared core, `claude/scripts/guard-broad-write.sh` — the inverse shape
  from `guard-doc-writes.sh`: a deny-list (allow by default, escalate only on a match) rather than
  an allow-list, because tdd-engineer's remit is genuinely "the whole codebase, this task," not
  one doc kind. The deny-list covers every other specialist's documented deliverable-path
  convention (`docs/plans/*`, `docs/reviews/*`, `docs/requirements/*`, `docs/manuals/*`,
  `docs/test-plans/*`, `docs/test-reports/*`, agent definitions/kaizen under `claude/*`, the team
  catalog files, cobb's skill packages, `cypher-mcp/README.md`) plus `docs/BACKLOG.md` — kept in
  the deny-list deliberately so a tdd-engineer → `BACKLOG.md` write keeps escalating exactly as
  today, per the requirements doc's unresolved instance U1 (not decided either way by this
  change). Every entry doubled (bare + `*/`-prefixed) for an absolute `file_path`.
  Mutation-tested (temporarily dropped the `docs/BACKLOG.md` entries, confirmed the guard
  wrongly fell back to `allow` on that path, then restored and reconfirmed `ask`).
- **Why:** Requirements doc `claude/docs/requirements/agent-permission-friction.md` (FR-2 general,
  instances 7-8, AC-3's named tdd-engineer example): a manual confirmation was firing on ordinary
  in-remit test/source-file edits despite `acceptEdits` since 2026-07-24. Root cause (design doc
  `claude/docs/plans/agent-permission-friction.md` §1, `analyst`-reviewed across two passes,
  verdict approve): frontmatter `permissionMode` is silently ignored/overridden by the parent
  session's mode in documented cases; an explicit hook `"allow"` is the mechanism that actually
  suppresses the prompt regardless of ambient mode. `frontend-engineer`/`devops`/`graph-dba` are
  designed-for the same treatment but deliberately not touched this round (zero live evidence);
  `coder`'s friction and instance U1 stay explicitly out of scope, per the requirements doc.
- **Plan items:** —

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_tdd-engineer`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_tdd-engineer` (FalkorDB, via `mcp__cypher__query`) instead of appending
  to `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," and "never edit your own agent definition" (no write-guard clause — tdd-engineer has no doc-scoped write guard). Behavior unchanged.
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
  paraphrased, not dropped" — closes DEF-1 (`coder`) and DEF-3 (`tdd-engineer`, this agent,
  produced a thorough, correct test-gap analysis on a no-CPG component with zero mention of "CPG"
  in any form — indistinguishable from the discovery step never running at all). Applied
  identically (phrasing pattern, not restructuring) across all six wired agents; only
  `coder`/`architect`/`tdd-engineer` were live-tested, but the near-verbatim wiring pattern means
  the same gap plausibly existed in `analyst`/`qa-engineer`/`frontend-engineer` too.
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
  `cpg-analysis` skill for RCA and impact analysis before writing a reproduction test — the
  actual call path to the symptom, what else exercises the function — and for test-gap analysis
  when scoping what to test next." **Step 1 "Understand first":** added a check for a relevant
  CPG (first-guess `cpg_<component>` naming per `skills/cpg-analysis/SKILL.md` §1) bundled with
  the freshness check (`skills/cpg-analysis/references/freshness.md`) in the same step, noting
  the result in the report and surfacing a refresh suggestion — never a silent rebuild — if
  stale. **Step 5 "Verify honestly":** added the `CPG:` evidence-trail line (plan §3 — `used
  <graph> — <clause>` / `considered, not relevant — <clause>` / `not applicable — <clause>`).
- **Why:** M4 widens the `cpg-analysis` roster from three consumers (`analyst`, `architect`,
  `qa-engineer`) to six — `tdd-engineer` is a new consumer because its reproduction-test-first
  work benefits directly from RCA/impact recipes ("what's the actual call path to the symptom,"
  "what else exercises this function"), and test-gap analysis is a natural companion to "what
  should I be testing." Plan §1's `tdd-engineer` row has the full roster reasoning.
- **Plan items:** none (design-driven, not backlog-driven).
- **Addendum (same day):** the description clause initially shipped in the conditional "With a
  loaded Joern CPG, uses…" framing (matched to `coder`'s clause per the dispatch instruction).
  The coordinator caught that this left `coder`/`tdd-engineer` as the only two of the six wired
  agents not on plan §2.1's mandated default-orientation framing — the sibling unit's
  `analyst`/`architect`/`qa-engineer`/`frontend-engineer` edits all use "Checks whether a relevant
  CPG exists as part of its normal orientation and, when one does, uses…". Reworded to match:
  "Checks whether a relevant CPG exists as part of its normal orientation and, when one does,
  uses the `cpg-analysis` skill for RCA and impact analysis before writing a reproduction test —
  the actual call path to the symptom, what else exercises the function — and for test-gap
  analysis when scoping what to test next." Body-prompt and evidence-trail additions were already
  correct and untouched.

## 2026-08-09 — Description gained a `python-web-quirks` skill routing clause
- **What:** Frontmatter `description` gained one clause: in a Python web/async codebase, the
  agent consults the new `skills/python-web-quirks/SKILL.md` for asyncio/FastAPI/Starlette/
  pydantic gotchas — mirroring how `cpg-analysis` is wired into `analyst`/`architect`. No body
  change. **Note:** this edit landed via `Edit` while the file already had unrelated in-flight
  changes to step 5 ("Verify honestly") from a concurrent session — the edit applied cleanly and
  did not touch that content; flagging the concurrency here for visibility, not because anything
  needed reconciling.
- **Why:** `python-web-quirks` was created distilling three general Python/web-framework facts
  from `analyst`'s learnings inbox. Stakeholder wired it to `coder`/`tdd-engineer`/`architect`/
  `analyst` at minimum. See `claude/analyst/kaizen/history.md` (2026-08-09) for the full
  distillation record and `skills/README.md` for the catalog entry.
- **Plan items:** none.

## 2026-08-09 — Learnings-inbox distillation, first pass (5 prompt additions; 3 discards)

- **What:** Processed all 9 entries in `kaizen/inbox.md` (dated 2026-07-15 through 2026-07-31) —
  the agent's first-ever distillation pass. Applied to `tdd-engineer.md`:
  1. New **Principles** bullet — symmetric fixture teardown for shared/global state (from the
     2026-07-24 "fixture wipes at setup, not teardown" entry).
  2. New **Principles** bullet — gate optional/slow tests with the runner's marker/tag mechanism,
     not a bare reachability skip (from the 2026-07-15 "reachability-skip doesn't gate a live test"
     entry).
  3. Expanded the **Cover the edges** bullet with positional/anchored-rule coverage guidance — vary
     the anchored position (first/middle/last) and check what the consumer does with the whole
     result (merged the 2026-07-24 "single-line corpus can't test a line-anchored rule" entry and
     the 2026-07-24 "positional accept-rule anchored on one list element" entry — the same
     underlying lesson, rediscovered twice in the same K-027 review pass).
  4. New **Principles** bullet — when merging two callers onto one shared helper, compare what each
     caller *does* with the result (validating vs. acting consumer), not just how similar the
     parsing looks (from the 2026-07-24 "two extractors, opposite safety postures" entry).
  5. Rewrote workflow step 5 ("Verify honestly") to require reading `passed`/`skipped`/`deselected`
     counts, not just exit code, and to run the *whole* suite (not just new/reproduction tests)
     after a fix, keeping the concrete "adjacent, unrelated-looking pre-existing test catches
     idempotency bugs" illustration (merged the 2026-07-24 "green exit code isn't evidence a suite
     ran" entry and the *second half* of the 2026-07-31 "idempotency guard can reuse an
     accumulator" entry).
  - **Discarded, already covered:** the 2026-07-15 empty-`UNWIND`-in-`materialize_snapshot` entry
    — fully superseded by `falkor-chat/docs/BACKLOG.md` K-030 (🔵 proposed), which documents the
    same defect in more depth (both `publish_def` and `materialize_snapshot`, the partial-write
    hazard, the fix). Nothing left to add.
  - **Discarded, too narrow/inconclusive for this pass:** the 2026-07-24 "first `docker ps` hangs"
    entry (single-occurrence, root cause explicitly not isolated by the reporting agent, and reads
    more like a devops/team-wide environment-probing fact than a TDD-specific one — no shared
    cross-agent home exists today, so parking it in this prompt would misfile it) and the *first
    half* of the 2026-07-31 "idempotency guard" entry (reuse-an-existing-accumulator-instead-of-a-
    new-flag is a real but narrow craftsmanship tip, not a durable prompt-level rule). Both are
    easy to re-raise if the pattern recurs.
- **Why:** Team-maintainer distillation pass (agent-maintenance skill §5), dispositions
  stakeholder-approved after cobb's read-only proposal pass the same day. Two independent
  near-duplicate lesson-pairs (the two positional-rule entries; the skip-count and
  whole-suite-after-fix entries) were merged into single prompt edits rather than kept as separate
  redundant bullets — stakeholder's explicit call.
- **Plan items:** none. `inbox.md` cleared to empty.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** closes the standing "is opus warranted vs. sonnet?" revisit item — model tier is no longer this agent's decision.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 495 → 434 chars (-12%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (tdd-engineer↔coder, tdd-engineer↔qa-engineer) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change to `coder`. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission on every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`. `acceptEdits` auto-accepts file edits and common filesystem commands for paths in the working directory/`additionalDirectories`, independent of session-level grants.
- **Why:** Same root cause as `coder` (see its 2026-07-24 kaizen entry) — applied to the other implementer agents for consistency, at user request.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Inbound RCA handoff + qa-engineer boundary (certification fixes)
- **What:** (1) Workflow step 1 now names the second doc-path input the team routes here: an `analyst` RCA at `<component>/docs/reviews/<slug>-rca.md` — its reproduction evidence is the first RED, its suggested fix the target. (2) The `description` now routes acceptance/black-box QA passes to `qa-engineer`, making the altitude boundary symmetric at the routing-contract level (qa's side already named tdd-engineer); `tdd-engineer:qa-engineer` added to `audit-team.sh` `BOUNDARY_PAIRS` so the symmetry is scripted.
- **Why:** Team-coherence certification (2026-07-11, handoff symmetry): analyst and teco both route RCA docs to this agent, but its own prompt only named architect plans as doc-path input; and the qa↔tdd boundary was one-sided.
- **Plan items:** closes the 2026-07-11 parking-lot handoff-symmetry item (same-day).

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 606 to 446 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Efficiency-based routing boundary with `coder` (description narrowed + made symmetric)
- **What:** Rewrote the `description`'s trigger. Was "use proactively whenever the user asks to implement a feature, fix a bug, refactor…" — which shadowed `coder`'s trigger on any feature task; now scoped to where test-first is the **efficient path** (bug fix with reproduction test first, safety-net refactor, adding/improving tests, feature with a clear up-front behavior contract) and points back to `coder` for executing an already-detailed plan/spec (the pointer was previously one-directional, coder→tdd only). Catalogs synced: teco roster (personal-preference note removed), `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`.
- **Why:** User ruling on the coder/tdd-engineer overlap review: route between the implementers by efficiency, not by an assumed blanket TDD preference; personal-preference framing removed from agent prompts (the user's standing preferences are quality and efficiency). Closes coder K-001 from this side.
- **Plan items:** none active (out-of-band; counterpart of coder K-001 ✅).

## 2026-07-09 — Plan-doc-path handoff + subagent-awareness (teco interface review)
- **What:** Two workflow additions, made during the teco interface review: (1) step 1 now states that an `architect` plan arrives as a **document path** (`<component>/docs/plans/<slug>.md`) — read the file itself as the source of truth; its test-strategy section is the red→green sequence. Mirrors the line `coder` has carried since the 2026-07-08 path-based-handoff change. (2) Step 2 and the red-suite + environment-blocker branches of step 3 gained subagent-awareness: when running delegated (e.g. by teco), "ask one sharp question" / "ask whether to fix first" / "ask before installing" becomes "return the question/blocker as your result" — subagents can't ask mid-run. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** teco routes implementation to this agent *preferentially* (user's TDD preference), yet the handoff contract was documented only on `coder`; and the "ask" phrasing silently assumed an interactive session the agent doesn't get under delegation.
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-06-20 — Dropped "Senior" from description (collection harmonization)
- **What:** Frontmatter `description` "Senior software engineer who implements…" → "Software engineer who implements…". Catalog row in `claude/README.md` "Senior engineer who implements…" → "Software engineer who implements…".
- **Why:** Collection-wide harmonization. The new `architect`/`coder` agents dropped "senior" entirely over the overconfidence concern; this brings tdd-engineer in line. **Supersedes the 2026-06-05 decision** that deliberately *kept* "Senior" as a role/altitude signal — the collection now omits it everywhere, relying on concrete process + guardrails for altitude/calibration instead.
- **Plan items:** —

## 2026-06-05 — Dropped tenure-boast framing
- **What:** Removed "with decades of experience" from the `description` and "with decades of hands-on experience" from the opening body line (now "You are a software engineer who works across many languages, paradigms, and frameworks."). Kept "Senior" in the description as a role/altitude signal.
- **Why:** User feedback — the "decades of experience" framing reads as cocky and doesn't change behavior. Applied collection-wide (also graph-dba, dra-claudia).
- **Plan items:** —

## 2026-06-05 — Implemented K-005 (third cold-start branch: tests can't run in this env)
- **What:** Edited `tdd-engineer.md` workflow step 3. Added a third sub-branch alongside "greenfield" and "suite already red on arrival": **"Framework exists but the suite can't run here."** It instructs the agent to recognize an environmental block (deps not installed, missing runtime/toolchain, required build/service step) as *not* a code RED, avoid misattributing it or thrashing on setup, report the blocker plainly, propose the bootstrap step (`npm install`, `uv sync`, build, etc.), and ask before installing/changing the environment — establishing a runnable baseline before the first RED.
- **Why:** Closes K-005. The original two branches silently assumed an existing framework was runnable; in practice a present-but-unexecutable suite is common and was previously unhandled, risking false REDs.
- **Plan items:** K-005 ✅ (closed, removed from active backlog). Active backlog now empty; K-003 deferred remains the only standing decision.

## 2026-06-05 — K-003 deferred (keep tools unconstrained)
- **What:** Decision only, no file change to the agent. Marked K-003 ⚪ deferred — `tdd-engineer` keeps no `tools` key and continues to inherit all tools.
- **Why:** User chose to keep the agent flexible for now (able to spawn subagents and fetch docs mid-task) rather than restrict to a focused TDD set. Recorded so it isn't re-proposed; revisit only if broad tool access causes surprise in practice.
- **Plan items:** K-003 ⚪ deferred.

## 2026-06-05 — Implemented K-004 (discoverability: catalog + context files)
- **What:** Collection-level docs for the three Claude agents (no agent-prompt change). Created `claude/README.md` (human catalog: one row per agent with what/when/model + links to each source file and `kaizen/` folder; kaizen index; conventions). Created `claude/CLAUDE.md` (agent-context: concise per-agent pointers to source + kaizen, maintenance rules, "don't paste full prompts"). Extended repo-root `AGENTS.md` to register the `claude/` collection — added it to Structure, Component docs (pointing at `claude/README.md` + `claude/CLAUDE.md`), and "Working in this repo".
- **Why:** Closes K-004 — `tdd-engineer` (and cobb, dra-claudia) were invisible to both humans browsing and other agents; no catalog or context entry existed and root `AGENTS.md` covered only OpenCode/salesperson. Satisfies cobb's dual-audience documentation rule.
- **Plan items:** K-004 ✅ (closed, removed from active backlog). Shared deliverable also benefits cobb and dra-claudia.

## 2026-06-05 — Review #2 (no behavior change)
- **What:** Re-reviewed `tdd-engineer.md` at the user's request ("just review/advise"); no prompt edit. Re-verified K-004's discoverability claim against the repo: confirmed no `claude/README.md`, and root `AGENTS.md` documents only the OpenCode/salesperson components — the only `CLAUDE.md` lives under `opencode/agents/severino`, so the three Claude agents (cobb, dra-claudia, tdd-engineer) have no catalog and no context entry. Added **K-005** (workflow step 3 misses the "framework exists but tests can't run in this env — deps/toolchain not installed" case; risk of misreading an environmental failure as a code RED). Added a parking-lot idea on advanced test techniques (table-driven/property-based/mutation testing) as optional, low-priority enrichment.
- **Why:** User chose the review-only path; kaizen rules say record new findings in plan.md even without implementing. Verified rather than assumed the still-open items remain accurate.
- **Plan items:** added K-005; re-verified/annotated K-004; K-003 unchanged.

## 2026-06-05 — Implemented K-001 (test altitude) and K-002 (cold-start/red-baseline)
- **What:** Edited `tdd-engineer.md`. **K-001:** reconciled the "unit test" framing with the agent's broader stated scope — dropped "unit" from the `description` ("failing test", "add/improve tests"); reworded the RED step to "smallest possible test — a unit test by default; reach for an integration or contract test only when that's the genuine seam"; added a new **Right altitude of test** principle (smallest honest test, write real integration/contract tests at seams a unit can't reach, prefer many fast units + a thin higher-level layer); softened the isolation principle to "Fast, isolated, deterministic **by default**" with a note that integration/contract tests are deliberately slower/broader but still deterministic. **K-002:** rewrote workflow step 3 from "Find the test command" to "Establish a green baseline" with two explicit branches — greenfield (set up the minimal runner first as its own announced step, confirm a real baseline) and suite-already-red-on-arrival (stop, report failures, ask fix-first vs. proceed; never build on red or misattribute a pre-existing failure).
- **Why:** User asked to implement backlog items K-001 and K-002. K-001 removed a wording tension that could push the agent to force-fit unit tests where a higher-level test is the honest seam; K-002 closed the cold-start and red-baseline gaps the original step 3 silently assumed away.
- **Plan items:** K-001 ✅, K-002 ✅ (both closed, removed from the active backlog).

## 2026-06-05 — Bootstrapped kaizen + first review (no behavior change)
- **What:** Created `tdd-engineer/kaizen/plan.md` and `history.md` (the agent predated the kaizen convention and had neither). Conducted a review of `tdd-engineer.md` without editing the prompt. Seeded the backlog with K-001 (unit-vs-broader-scope tension), K-002 (cold-start / red-baseline handling), K-003 (tool-permissions decision), K-004 (catalog/discoverability gap), plus parking-lot ideas (no-auto-commit note, coverage-as-guide, flaky-test handling, opus-vs-sonnet cost).
- **Why:** User asked to work on the agent and chose "just review, advise" + "bootstrap kaizen files." Review found the prompt fundamentally solid (clean frontmatter, tight red/green/refactor loop, strong guardrails and anti-hallucination stance); the actionable findings are scope-framing, cold-start coverage, and housekeeping/discoverability — captured as backlog rather than applied.
- **Plan items:** seeded K-001, K-002, K-003, K-004.

## 2026-05-29 — Agent created (retroactively logged)
- **What:** Initial authoring of the `tdd-engineer` agent — a senior engineer that implements features and fixes strictly via Test-Driven Development (red → green → refactor). Frontmatter `name: tdd-engineer`, `model: opus`, routing-oriented `description` with proactive-use triggers (implement a feature, fix a bug, refactor with a safety net, add/improve tests). Body covers the TDD loop, principles, invocation workflow, communication style, and guardrails.
- **Why:** User wanted a dedicated test-first engineer agent (memory: "Prefers TDD"). Logged here retroactively since the agent predates the kaizen convention; date approximated from the source file's mtime.
- **Plan items:** —
