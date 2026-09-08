# Analyst review techniques — on-demand

> Specialized verification techniques, loaded on demand from `analyst.md` (not part of the
> always-loaded prompt body) — mirrors the `graph-dba/falkordb-quirks.md` pattern of keeping a
> resident prompt lean and pushing a growing, occasionally-needed fact/technique list into its
> own file. Distilled from the `analyst` learnings inbox via `agent-maintenance` skill §5; see
> `claude/analyst/kaizen/history.md` (2026-08-09) for the promotion record.

## Byte-identity confirmation via an AST line-range hash

When a plan or gate requires proving an artifact ("this function is locked, prove it's
byte-for-byte unchanged") wasn't touched — even when its line offset moved (e.g. code was
inserted above it) — a `git diff` line-range read can mislead. Instead:

1. `ast.parse` the file at each revision, walk for the target `FunctionDef`/`ClassDef` by name.
2. Hash `src.splitlines()[node.lineno-1:node.end_lineno]` extracted from `git show <rev>:<path>`
   at each revision under comparison.
3. If the hashes match, the function is byte-identical regardless of where it now sits in the
   file.
4. If they differ, a second pass comparing `ast.dump()` with docstring `Expr` nodes stripped
   distinguishes a docstring-only edit from a real behavioral change.

Origin: proved `executor._drive_loop` identical (same hash, same byte length) across three
revisions despite its line offset moving by 14 lines, which a naive `git diff` line-range check
would have mis-read as "moved, can't tell."

## Verifying an uncommitted diff without mutating the working tree

Two techniques for gathering *executed* evidence about a change that is still uncommitted (no
`git stash`/worktree available, or blocked by the harness's Bash safety classifier — see
`skills/agent-standards/claude-code.md` § Bash tool environment for that layer):

**(a) Load a `HEAD` version of a module alongside the working-tree one, via a stub package.**
First, the import-resolution order you are working against, measured rather than assumed
(2026-09-08, `falkor-chat/server/.venv`, Python 3.12, marker package): a setuptools editable
install **appends** its finder to `sys.meta_path` (`install()` does
`sys.meta_path.append(_EditableFinder)`; the live order is
`[BuiltinImporter, FrozenImporter, PathFinder, _EditableFinder]`), so `sys.path` beats it, and
`import <pkg>` resolves to **(1)** a `<pkg>/` directory in the cwd (`sys.path[0] == ''` under both
`python -c` and `python -m`), else **(2)** a `PYTHONPATH` entry, else **(3)** the finder's
`MAPPING`, which hardcodes an *absolute* path to the tree the install was made from. Two
consequences that decide whether an isolation attempt actually isolated anything:

- A `PYTHONPATH`-prepended scratch copy **does** shadow the editable install — *unless* you invoke
  from the real package's own parent directory, whose cwd entry beats it. That cwd precedence, not
  finder priority, is what defeats a `PYTHONPATH` shadow attempt run from `server/`.
- A **`git worktree` isolates the import only when Python is invoked with cwd inside the
  worktree's own package-parent directory** (`server/`). From the worktree *repo root* nothing
  local matches, rule (3) wins, and the run silently executes the **main** tree's uncommitted
  source while looking isolated. Observed cost of getting this wrong: a seed script invoked from a
  worktree repo root republished a concurrent unit's uncommitted content into the shared
  `reference` graph under the wrong version label.

The technique below never depends on that search order at all, which is why it is the robust
route, with zero working-tree writes:
```python
# 1. Extract the HEAD version of the file to scratch, unmodified:
#    git show HEAD:path/to/module.py > $SCRATCH/module_head.py
# 2. Build a stub package in-process to satisfy the module's own relative imports
#    (only needed if the module does `from . import something`):
import types, sys, importlib.util
pkg = types.ModuleType("hp"); pkg.__path__ = []; sys.modules["hp"] = pkg
sys.modules["hp.config"] = <a ModuleType exposing whatever `config` symbols the module needs>
spec = importlib.util.spec_from_file_location("hp.module_head", f"{SCRATCH}/module_head.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
# 3. `old.some_function` and the working-tree import of the same function are now both
#    live in one process — diff their behavior directly on a table of inputs.
```
Works cleanly when the module's only intra-package import is a small, enumerable set (`from .
import config`); a module with a wider relative-import fan-out needs more stubs. Use this to
independently confirm a claim like "N tests were red before the fix" without trusting the
implementer's narration, and to catch an undisclosed behavior change versus `HEAD` in the same
pass.

**(b) `cp -r` a scratch copy of the working tree, then reverse-apply the diff there with `patch
-p<N> -R`.** Stronger isolation than even `git stash --keep-index`: the tracked working tree is
never touched at all.
```bash
cp -r <component>/server "$SCRATCH/server-check"
git diff -- <file> > "$SCRATCH/x.diff"
(cd "$SCRATCH/server-check" && patch -p<N> -R < "$SCRATCH/x.diff")   # N = path depth from the copy root
# symlink .venv (and any repo-root-relative script dir fixtures shell out to) into the
# scratch copy so its pytest runs for real, including live-integration tests.
```
Confirm zero residue on the real tree with `git status` before/after. This substitution is
sound **specifically because it doesn't touch the tracked tree at all** — a strictly stronger
property than a blocked `git stash`, not a lower-visibility route to the same effect. It is
**not** a general "when something blocks you, find another way to the same effect" license: the
justification is the zero-touch property, not that a workaround exists — the block was doing its
job; this substitute earns the exception on its own zero-touch merits. Confirm your own
substitute is actually zero-touch before reusing this, not just that it wasn't caught.

**(c) A brand-new *untracked* file that is itself part of the diff has no `HEAD` baseline —
neither (a) nor (b) applies.** `git show HEAD:path` fails and `patch -R` has nothing to reverse
against. Instead: `cp` the file to scratch *before* mutating it in place, mutate/restore from
that scratch copy, and confirm byte-identity via `md5sum` plus `git status --short` (an untracked
file that's still untracked and hash-identical to the scratch copy is proof of a clean restore).

Origin: mutation-testing `claude/tdd-engineer/hooks/guard-tdd-broad-write.sh`, a new untracked
file in the agent-permission-friction implementation diff (U2 gate, 2026-08-21).

**(d) To independently re-verify a mutation-testing claim by constructing real objects that
exercise the mutated code path against a live system** (not just diffing static behavior), load
the *real* package first — this populates `sys.modules` with every unmutated sibling submodule via
its normal relative imports — then use `importlib.util.spec_from_file_location` to load *only* the
one mutated file under the package's real dotted name (e.g. `"pkg.services"`, with `__package__`
set correctly), `exec_module` it, and overwrite `sys.modules["pkg.services"]` with it before
constructing any objects. A `PYTHONPATH`/`sys.path.insert` shadow is **not** reliable
here — see the measured resolution order at the head of (a): whenever the run's cwd is the real
package's own parent directory, the cwd entry outranks `PYTHONPATH` and `import pkg` keeps
resolving to the working-tree copy. This differs from (a):
(a) loads an alternate version under a *separate* namespace to diff two versions side by side; (d)
substitutes one module *in place*, under its real name, so objects constructed afterward actually
run the mutated code when exercised — the right shape when the claim to verify is "this specific
check is load-bearing," not "what changed between two revisions."

Origin: `falkor-chat` K-028 workflow-timers diff re-gate — verified a mutation-testing claim
(`services.py`'s escalation-guard value-equality check) this way: confirmed `publish_workflow_def`
wrongly accepted a mismatched-step-key guard with the mutation in place, and correctly rejected it
without, against a live DB connection, with zero working-tree edits.

## Re-gating a fix pass by line-number invariance

When a fix pass lands on an *uncommitted* tree (so both the reviewed change and the fix pass are
uncommitted, `git diff` shows their union, and there is no pre-fix baseline — no stash, no
`.orig`), and the implementer claims "all N prior findings/pins survive unchanged": check that
every `file:line` your round-1 review quoted still lands on the same construct in the new
version.

- If a quoted line number is **unmoved**, the code above it in the file is unedited (an edit
  above would have shifted everything below).
- If a quoted line number **moved**, confirm the new blocks are **pure insertions** at the old
  position (i.e., everything below your quoted line shifted by a consistent offset, and nothing
  *above* your earliest quoted line moved at all) — that proves the fix pass only added code,
  it didn't touch what you already reviewed.

Combined with reading the bodies at their new locations, this is sufficient; line-number
invariance alone or body-reading alone is not. This is also why it matters to **quote `file:line`
for every finding and every pin you certify** in round 1 — that's what makes a later re-gate
provable without a diff.

## An uncommitted agent-prompt edit under review is already live

Agent prompts deploy via symlink (`~/.claude/agents/<name>` → `claude/<name>`, documented at
`claude/README.md`), so when the artifact under review is an agent prompt/skill under `claude/`
or `skills/`, "uncommitted" does not mean "not yet in effect" — the edit is already active for
the running team, including the reviewing session itself if it happens to touch the same file.
This raises the urgency of any blocker found there (findings ship immediately, not at commit),
and makes "restore it now via `git diff`/`git checkout`, it's still recoverable" a real,
time-boxed remedy rather than a nice-to-have.

Origin: reviewing an uncommitted diff that edited `claude/analyst/analyst.md` — the new clause
was already present verbatim in that review run's own system prompt (2026-08-11).

## Verifying a "copied verbatim" text-block claim needs a programmatic diff, not a read-through

When a finding or spec asks you to confirm text was reproduced "verbatim" (a caveat, a prompt, a
spec quote copied into code), extract both strings into variables, whitespace-normalize
(`re.sub(r'\s+', ' ', s).strip()`), and diff them programmatically — never confirm by reading
them side by side. A markdown soft line-break renders as a single space in the *source* but
silently vanishes when hand-transcribed into a multi-line string-literal concatenation; the wrap
point is exactly where a transcription is most likely to drop a character, and the place a visual
read is least likely to catch it, because both versions look correct individually.

Origin: `falkor-chat/server/tests/eval/generate_report.py`'s `_SAME_MODEL_CAVEAT_TEMPLATE`,
2026-08-16 — the extracted-and-diffed strings surfaced a run-together word
(`"borderline/subjective"`) where the source had a soft-wrapped space; a close read-through had
found no difference.

## `pytest -k` is not a substitute for the project's own `-m` marker filter

When verifying a plan's cited pytest baseline ("N passed, M deselected"), run the project's own
default/documented invocation — check `pytest.ini`/`pyproject.toml` `addopts` first — rather than
a hand-written `-k "not live"` that looks equivalent. `-k` is a substring/keyword filter over test
*names*; `-m` is marker-based deselection over test *markers*; they can silently disagree by one
or more tests even when the totals mostly overlap (a test can carry "live" somewhere in its
collected id without carrying the `live` marker, or vice versa).

Origin: `cypher-mcp`, 2026-08-19 — `pytest -k "not live"` gave `83 passed, 8 deselected`; the
project's own `addopts = -m "not live"` (`cypher-mcp/pytest.ini`) gave `84 passed, 7 deselected`,
matching `docs/plans/cpg-mcp-rename.md`'s cited baseline exactly.

## Ground truth for "may an agent edit its own definition?"

The literal clause **"never edit your own agent definition"** closes every non-`cobb` agent
prompt (`grep -rln "never edit your own agent definition" claude/`) — `cobb.md` is the only one
without it, since `cobb` is the team's designer. A plan that assigns any other agent a self-edit
of its own `<agent>.md` contradicts that agent's own prompt outright, and for the doc-guard-
carrying agents (`architect`/`analyst`/`data-scientist`/`teco`/`tico`) also trips their
`PreToolUse` write-glob into human escalation. Check both the clause and the relevant guard
script's allowed-globs before accepting a plan's claim that some agent may edit itself — a plan
may cite a precedent that, checked against `git show --stat`, doesn't actually exist.

Origin: `docs/plans/generic-cypher-mcp2.md` V2 plan-gate, 2026-08-20 — the plan assigned
`architect`/`graph-dba` a self-edit by analogy to a claimed `cobb` precedent that turned out not
to exist.

## Check live-service reachability before trusting a live-test report

Before accepting an implementer's "I ran the live suite, here are the numbers" claim at face
value, check whether the dependencies are actually reachable in *this* session (e.g. `curl
http://localhost:1234/v1/models` for LM Studio, a `redis-cli -h localhost -p 6379 ping` /
project's own live-marker for FalkorDB). When they are, independently re-run the live suite
yourself rather than only trusting the narration — a live re-run this cheap is strictly stronger
evidence than static verification alone. Reachability is a property of the sandbox/session, not
the codebase — re-check it each time rather than assuming a prior run's environment still holds.

Origin: falkor-chat's guard-calibration live suite (255 judge calls, ~155s) independently
re-executed and reproduced the exact G1/G2 numbers a `tdd-engineer` report claimed (K-027 item 4,
2026-08-21).

## Reconciling a kaizen-graph distillation's claimed dispositions

When auditing (or self-checking) a `kaizen_team` distillation pass, don't trust the history
entry's prose summary alone — reconcile it against ground truth: map every processed entry's id
to its stated disposition (promoted / discarded / kept-open) rather than trusting an aggregate
count, since a summary can under-report (omit a disposition), misattribute (list a promotion
sourced from a different agent's entries), or miscount by one on a boundary case (an entry with
no clean "## " analog, a headless record). This predates the 2026-08-20 migration to
`kaizen_team` — the same trap surfaced auditing a file-based `## `-heading inbox diff
(`grep -c '^-## '` the diff, reconcile against the history header's claimed count, map each
removed heading to a stated disposition from the diff text itself, never from the history's
prose) — the storage mechanism changed, but the discipline (verify the aggregate claim against
itemized ground truth) carries over unchanged to the graph form.

Origin: reviewing `cobb`'s 39-file 2026-08-11 file-based distillation — a claimed "8 entries
routed (6 to …, 1 discarded)" didn't add up to the diff's actual 8 removed headings, and the same
reconciliation caught four more unlogged dispositions and four wrong header counts across other
agents' inboxes.

## A reused write-query precedent carries its NULL contract with it

When a plan says "same shape as `<existing query>`", the precedent's treatment of a `NULL`
parameter is part of what is being copied — and it is invisible in the Cypher, because the two
incompatible contracts are written almost identically:

- **`SET x = $x`** — `NULL` *clears* the property. Right when the caller always sends a full
  record, so an omitted field genuinely means "unset this".
- **`SET x = coalesce($x, x)`** — `NULL` means *leave unchanged*. Required when the caller's
  arguments are individually optional (an LLM tool call with optional parameters, a PATCH-shaped
  route).

Plain Python `None` cannot distinguish "argument omitted" from "explicitly null" by the time it
reaches the query params, so copying the first shape for a caller of the second kind silently
erases previously-stored data on every *partial* update — and no test that only exercises a full
write will ever see it. At plan-gate, for any write query, ask **which of the two contracts this
query's callers need**, not which existing query it resembles.

Origin: `falkor-chat` M6 plan-gate — `docs/plans/workflow-durable-profile-graph.md` §3 drafted
`write_profile` on the `write_model_overrides` unconditional-`SET` precedent while
`SaveProfileTool`'s two arguments are individually optional; raised as a v1 BLOCKER and fixed to
`coalesce($field, c.field)`. The shipped resolution documents both contracts side by side —
`falkor-chat/docs/QUERIES.md` §17.1 (the `coalesce` form, plus the "never pass `''` to mean *not
provided*" corollary) against §13.1 (the clearing form and why it is right there).

## Re-gating a state-machine guard/invariant fix: two checks a "does the mechanism work" read misses

Verifying that a guard/invariant fix's own reasoning is internally sound is not the same as
verifying it's *complete*. Two distinct, easy-to-skip checks, both surfaced re-gating the same
fix in sequence:

1. **Does the fix foreclose an existing pattern elsewhere in the same codebase?** A fix that forces
   guaranteed forward progress on a state machine (e.g. "every wait step must carry an
   unconditional fallback transition") can silently break a *different*, already-shipped pattern
   that depended on the old, non-forcing behavior (a step that stays parked on an explicit
   negative/not-yet signal). Check whether the invariant you're approving forecloses a legitimate
   use the codebase already relies on, not just whether it closes the bug it targets.
2. **Trace every call site of the evaluation function, not just the one the fix's own narrative
   points at.** A fix framed entirely around "resume vs. first arrival" can still be silently wrong
   if the guard-evaluation function also fires at a call site the framing never considered (e.g.
   first arrival, before any suspend has happened at all). Verifying "the sort order resolves ties
   correctly" is not the same as verifying "this guard only fires at the right evaluation site" —
   enumerate every call site of the function under test, independent of the fix's own story about
   which ones matter.

Both gaps slipped past an earlier, otherwise-careful pass that reasoned from the fix's own framing
rather than from the full call-site/pattern inventory — the discipline is to build that inventory
yourself, not to trust the fix description's scope.

Origin: `falkor-chat` K-028 workflow-timers, v2→v3. Check 1 caught that v2's mandatory default
fallback arm made an explicit `{"provisioned": false}` "not yet" resume silently advance into the
timeout branch instead of re-parking. Check 2 caught that v2's fix (Pass-1/Pass-2-approved) made
the whole feature unreachable, because `_select_transition` fires on every visit including first
arrival, not only on resume — a gap two prior review passes missed by reasoning from "resume vs.
not" rather than enumerating call sites.

## A "this already exists" claim is a grep away from confirmation

A plan/review citing a worked example as "grounded in this repo's own history," or a fix-pass
claiming a finding was "recorded in `<file>`," is a specific, cheaply falsifiable claim — verify it
directly rather than accepting the narration, even when the claim reads as plausible either way.
One grep settles it: `grep -rn -i '<the cited event/term>' <the claimed location>` either finds the
citation or it doesn't.

**A pasted grep result is the same kind of claim, and unlike a fabricated one it decays**: it was
honestly run, it was true when it was run, and the document lands days later against a codebase
that moved. Re-run every cited grep at review time — above all a *negative* one ("→ no matches"),
which reads as settled and is the one nobody thinks to re-check.

Origin: three independent instances. (1) A `cobb`-authored plan cited a NULL-backfill decision as
having surfaced during a specific past investigation; `grep -rn -i backfill claude/docs/` found
zero occurrences outside the plan doc itself, and the cited investigation was unrelated
(hook/permission engineering, no migrations at all) — the example was plausible-sounding but
fabricated. (2) A fix-pass plan claimed two findings were "recorded in `falkordb-quirks.md`";
grepping the file directly at the cited line ranges confirmed both were genuinely present, not
just asserted — the same check, run the other direction, separating a closed finding from an
asserted-but-undone one. (3) The staleness case:
`falkor-chat/docs/plans/oversized-indexed-property-guard-graph.md` argues against bounding one
field because "this schema has zero `RELATIONSHIP`-type constraints (`grep -n RELATIONSHIP
scripts/bootstrap_schema.sh` → no matches)". True when run on 2026-08-21; falsified three days
later by commit `8d7dcfb` (K-050 fusion), which added `gconstraint … UNIQUE RELATIONSHIP SAME_AS
PROPERTIES 1 matchId`; the doc, written 2026-08-26, repeated it verbatim and **still carries it**
(`:205`, re-checked 2026-09-08 against `scripts/bootstrap_schema.sh:265`).

## A document that adds a member to its own taxonomy is swept table-by-table, not changelog-by-changelog

When the artifact under review adds a new member to a taxonomy the document itself defines — a
new document `<kind>`, a new status token, a new role, a new agent in a routing table — the
amendment is complete only if **every** table keyed by that taxonomy carries it. Enumerate those
tables yourself (`grep -n '^|' <doc>`, then read each header row for the taxonomy's key) rather
than reviewing the sections the amendment's own changelog names: a changelog lists what the
author thought about, so the tables it omits are precisely the ones that go stale.

Rank the tables before reading them. The highest-value one is any table the document **declares
another document copies from** — staleness there does not merely leave one table wrong, it
*inverts the claimed provenance*: the copy becomes the correct version and the declared source
the stale one, so a later reader reconciling the two edits the wrong file.

Origin: 2026-09-01, a targeted spot-check (not a full re-review) of an uncommitted v1.5 amendment
to `docs/plans/doc-reference-convention.md`. It added `manuals/` as a recognized `<kind>` in
§9.2/§9.4/§9.5 but left §9.6's "who performs the `archived` flip, by kind" table without a
`manuals/` row — while §9.6 itself states that root `AGENTS.md` copies from it, and root
`AGENTS.md` already carried the `manuals/` → `tico` pairing. Fixed in v1.5.1; that row now reads
`requirements/*`, `manuals/*`.

## An untracked plan/review doc has no re-verification baseline

A plan/review doc that was never `git`-committed leaves zero recoverable "before" state for a
diff-scoped re-verification of a claimed prior-content fix — `git log`/`git log --follow`/`git
stash list`/`git reflog` all return nothing for a path that was never tracked. A Pass-1 finding
that quoted specific pre-fix wording cannot be re-confirmed against the file itself once it's been
edited in place; an implementer's counter-claim ("that section never had the problem") becomes
unfalsifiable from the surviving artifact alone. Check `git status`/tracking state for the artifact
under review as part of scoping a diff-based re-gate — an untracked doc needs a different
verification strategy (e.g. asking the implementer to preserve the pre-fix text, or reviewing the
fix pass as a fresh read rather than a diff) rather than assuming a git-based re-check is available.

Origin: Pass 2 (post-implementation) diff-scoped review of `claude/docs/reviews/mid-run-
escalation.md` — `claude/docs/plans/mid-run-escalation.md` was untracked throughout, so a Pass-1
finding quoting exact pre-fix wording in §2.2 had no way to be re-confirmed against the post-fix
file.

## A truncate → append → truncate-again pipeline can silently discard its own repair pass

A pipeline shape where a raw list is capped, a repair pass appends new synthesized items on top,
and a *second* cap then runs over the combined list can silently discard the repair pass's own
output — if the raw list was already near the cap, the final truncation slices off exactly the
items the repair pass existed to add, with no error and no test coverage of the interaction.
Reading the stated order in a docstring or comment is not enough to catch this — verify the
interaction with a direct execution probe (run the real pipeline function against an input sized
to sit near the cap, and check what's actually in the output) rather than reasoning about ordering
from the code's narrative alone.

Origin: `falkor-chat` extraction.py — relationships capped, then stub-repair appended synthesized
entities, then entities capped again at `MAX_ENTITIES_PER_CHUNK`. A live probe showed a
repair-added stub entity got sliced off by the final `entities[:CAP]`, silently dropping the
relationship that depended on it. Fixed by capping the raw list *before* repair runs, not after.

## Two checks for a multi-shape authorization/security-gate function

Reviewing a function that authorizes a write (or otherwise gates an action) by checking a Cypher
statement — or any input — against several recognized shapes in sequence: two checks beyond
confirming each individual shape's own logic is correct.

1. **Keyword-set completeness.** When a fix adds a bare-keyword allowlist/denylist scan (e.g. to
   catch a foreign write clause chained onto an otherwise-authorized statement), cross-check its
   keyword set against every *other* keyword-set constant already defined in the same module,
   rather than trusting the fix author's own named attack reproductions. A fix modeled on two
   concrete attacks can silently omit siblings from the same taxonomy that a pre-existing test
   already existed to keep closed.
2. **Early-match short-circuit smuggling.** A function that returns as soon as the *first*
   recognized shape matches may never check whether the rest of a multi-clause statement smuggles a
   second, unrelated write clause. If the shape-matching logic scans the *whole* input text for its
   own trigger (rather than requiring the matched shape to consume the entire statement), a crafted
   input can chain a self-authorizing clause with an unrelated one and have the whole thing
   authorized as one.

For either check, construct the adversarial input yourself (a decoy authorized clause plus a
chained unrelated one; an attack using a keyword-taxonomy sibling the fix's own examples didn't
cover) rather than only re-running the fix's own named test cases.

Origin: `cypher-mcp`'s `authorize_write()`, Pass-2 review of `docs/plans/kaizen-agent-ontology.md`
(M8). Check 1 caught `_FOREIGN_TRIGGER_RE` covering only `MERGE|DELETE` against the module's own
pre-existing `_WRITE_KEYWORD_RE` covering `CREATE|MERGE|SET|DELETE|REMOVE` — silently reopening a
chained `SET`-based tampering path (including author-reassignment) that a sibling test already
existed specifically to keep closed. Check 2 caught that `authorize_write()` returned as soon as a
matching-author `CREATE` was found, without checking whether the rest of the statement chained an
unrelated `MATCH...DETACH DELETE` or a mismatched producer-write. Both fixed in the shipped code
(`_FOREIGN_TRIGGER_RE` now covers all four keywords; `authorize_write()` now calls
`_has_foreign_trigger_outside_strings()` after an author-claim match, before authorizing).

## Live graph/database state has no git provenance — and a two-sided diff cannot detect common-mode staleness

Two related traps when an investigation has to establish **where stored state came from**.

**1. `git log --all -S '<string>'` is not a provenance oracle for anything stored in a database.**
Content reaches a live graph by *running* code, and the code that ran may have been an uncommitted
working tree — so a nil `git log -S` result is not evidence the content was never shipped. Verified
in this lab: a reverted experiment recorded in `falkor-chat/docs/BACKLOG.md` as "Reverted, never
shipped" was found live as a materialized `ws:acme salesperson@v6` snapshot, and `git log --all -S`
on its distinctive prompt string returned nothing, because it had been published from a working
tree that was never committed. **Provenance questions about stored state are answered against the
store itself** (read the rows, compare them to the current source constant), never against history.
Same family as *"An untracked plan/review doc has no re-verification baseline"* above — git can only
answer about what git was given.

**2. A verification that compares two derived artifacts against EACH OTHER is structurally blind to
staleness they share.** `falkor-chat`'s `services.diff_def_snapshot` compares `reference` against
the workspace snapshot — both materialized from the same source constant at whatever moment they
were written. When both were written from an older or uncommitted version of that constant, they
agree perfectly, the topology check passes, and the version name silently denotes something the
source file can no longer produce. Whenever a gate reports "in sync", ask **in sync with what** — if
both sides are derived, the gate needs a third comparison against the source of truth, not a
tighter two-sided one. Shipped precedent for the fix: `falkor-chat/scripts/verify_salesperson.sh`
check 6 now diffs every stored step `config`, on both sides, against the `falkorchat.proof_defs`
constant, and its header comment states the blindness explicitly.

Origin: 2026-09-02, `salesperson-ui` S1 review findings F-1/F-8; distilled 2026-09-07 and
re-derived against the shipped script and `services.diff_def_snapshot`.

## Mutating a class-level constant via a pytest plugin proves a guard is load-bearing without touching source

Mutation-testing a guard normally means editing the source file, running the suite, and restoring
it — which is exactly the wrong move in a working tree carrying other sessions' uncommitted work
(a restore step that fails, or a concurrent writer, loses real changes). When the code under test
holds its statement in a **class-level constant read off `self`**, the mutation can happen entirely
in a plugin at import time:

- Write a plugin module in a scratch directory that imports the class and rebinds the constant
  (`Repository._RESET_PARTICIPANT_CYPHER = <mutant text>`).
- Run `PYTHONPATH=<scratch> .venv/bin/python -m pytest -p <plugin_module> tests/test_x.py -k …`.

Every method reads the constant off `self` at call time, so the rebind takes effect for the whole
session and the source file stays byte-untouched (`git diff` clean — which is also the evidence
that the mutation was contained). Verified 2026-09-07 in `falkor-chat/server`:
`repository.py` declares `_RESET_PARTICIPANT_CYPHER`/`_PUBLISH_CYPHER`/`_ENSURE_PARTICIPANT_CYPHER`
etc. as class attributes and every call site dereferences `self._…_CYPHER`, so the precondition
holds there today.

**The precondition is the whole technique** — check it before promising the approach: a constant
captured at import into a module-level name, a default argument, or a local, will not respond to
the rebind. Where it does hold, this is the safe form of a mutation ablation under a shared or
dirty working tree.

**State the seam's reach when you report the result.** This ablation mutates *query text* only, so
it can prove a Cypher-level guard load-bearing and says nothing about the Python-side behaviour
around it — exception dispatch, row shaping, the branch that decides whether the query runs at
all. Those are unreachable through this seam and need a different mutation. A report that says
"the guards are ablation-proven" without that boundary overstates what was tested.

**Run the battery serially — a concurrent suite run destroys the measurement.** Where the tests
under ablation are integration tests against one shared database, the same fixture that isolates
them makes two processes mutually destructive: `falkor-chat/server`'s `conn` fixture
(`tests/conftest.py`) calls `MATCH (n) DETACH DELETE n` on the single `ws:test` graph at **setup**,
once per test, and `repo`/`wf_repo` both depend on it — so a background full-suite run wipes the
graph out from under the ablation run mid-test and vice versa. The failure is loud but useless:
large, scattered, non-reproducible counts (43 failed / 2338 passed, with two dozen failures in
areas the mutation never touched), which re-ran serially on the identical tree gave 2381 passed /
14 deselected and 0 and 3 failures for the two ablations. **Before trusting any failure count,
confirm nothing else is running against the same store** — an unexplained scatter of failures far
from the mutation is the tell.

Origin: 2026-09-02, `salesperson-ui` S4 — proving the two participant-reset guards were
load-bearing while several other units' work sat uncommitted in the same tree.

## An extract-and-execute loop over a doc's code blocks does not prove the doc is well-formed

A design note whose queries an implementation step will copy verbatim often claims a "closed
verification loop": extract every fenced block, run each one, all green. That loop is blind to the
document's *markup*, and the blindness is systematic rather than incidental. A tolerant extractor
— the natural non-greedy ` ```lang\n(.*?)``` ` — happily accepts a **closing fence glued to the
last code line** (`… RETURN n AS x```), because the regex only needs the three backticks to appear.
A CommonMark parser requires the closing fence to start its own line, so it reads that block as
**unterminated** and swallows everything after it.

Reproduced 2026-09-08 on `/usr/bin/python3` (markdown-it-py 3.0.0, `MarkdownIt("commonmark")` with
tables enabled) on a minimal document with one glued fence: the regex reports **2** blocks, the
parser reports **1** fence spanning 8 lines and **2** headings instead of 3 — the second code block
and an entire section disappear into the first block. Inserting a newline before the glued fence
restores 2 fences and 3 headings. In the real instance (`docs/plans/salesperson-ui-graph.md` v1.1,
since fixed) the same two glued fences produced one 455-line block that swallowed three sections
and left only 7 of 11 tables rendering — while all five runnable blocks executed clean.

**So verify a doc's embedded code by re-parsing it with a real markdown parser, not only by running
what you extracted.** Assert the expected block **count** and a plausible maximum block length; a
single block hundreds of lines long in a document of short queries is the signature. The check
costs one command and catches a class the execution loop cannot see — including in your own review
file, where an unescaped inline ` ```cypher ` in a prose sentence does exactly the same thing.

## Per-row hashing turns "is this plan stable enough to dispatch?" into evidence

Asked to judge whether a long-revised implementation plan has converged, the reviewer's instinct is
to read the latest version and form an impression. Hash instead: extract each step row from the
plan's step table at every saved revision and compare the hashes. Rows byte-identical across all of
them are safe to dispatch against; the rows that change at every revision **localise the churn
surface**, which is the actual answer to the question asked.

Re-derived 2026-09-08 over **all 16 committed revisions** of `docs/plans/salesperson-ui.md` in
`acb5a2a^..069f6ae` (v1.16 → v1.31), matching `^\|\s*\*\*(S\d+[a-z]?)\*\*\s*\|` and md5-ing each
matched line. **The step table's membership is not constant across the window**, which is the first
thing the hash pass has to handle: 21 distinct step rows appear in total, but `S7c` only enters at
`732f5e0` (v1.19, *"split the catalog fix into S7c"*) and persists — so 3 revisions carry 20 rows
and 13 carry 21. Of the **20 rows present in all 16**, **13 are byte-identical across the whole
window** (`S0 S1 S2 S3 S4 S5 S7 S11 S12b S12c S12d S14 S16`), while `S9` takes **13 distinct values
across the 16**, `S8` and `S13` four each, and `S10` and `S12a` three each. That is the answer to
"has it converged": everything except `S9` and its four neighbours.

**Hash the whole window, never a sample of it.** A subsample can only *over*-report stability — a
row that changed in a revision you skipped reads as identical — so a stable-row list from five of
sixteen revisions is an upper bound presented as a measurement, whatever number it happens to land
on. And **which rows are stable is a property of the window, not of the plan**: re-run over the
window you are actually gating, and never carry a previous pass's stable-row list forward as a
finding.

**A companion trap in the same family of documents:** a *completeness table* added to a plan to
prevent an over-generalisation can reintroduce it, when the table is keyed on
`(response → rule)`. One row then spans several routes, and a second, unrelated meaning of the same
status code on a different route is silently swallowed by it. Keying the table on
`(route, response)` makes that collision unexpressible. When reviewing any table added as a
completeness argument, check its **key** before checking its rows.

## A guard derived from the artifact it guards is blind along the derivation axis — mutate the source, not the subject

A test parametrized over the very collection it validates, or an AST assertion scoped to the alias
the code happens to use, cannot see a change made *along the axis it derives from*. Both fail
**green**, and both are common shapes for a gate written to pin a decision.

**Deletion, not modification, is the mutation.** A parametrized test loses a *case* when its source
collection shrinks — the suite reports fewer-passed-and-green, never a failure. So compare the
**collected count**, not pass/fail. Verified 2026-09-08 on a synthetic pair (`falkor-chat/server/.venv`
pytest 9.1.1: 4 collected → 3 collected, zero failures) and re-derived against the citing case —
`model-bench` at `ab91419`, `git archive`d to a sandbox and run with `model-bench/.venv`:

- baseline `233 passed`;
- deleting `loadedContextLength` from `REQUIRED_BY_SCHEMA[1]["model"]` → `230 passed`, **zero
  failures**. The −3 *is* the lesson: that constant feeds two parametrized tests, and
  `FORBIDDEN_BY_ARM_KIND["deterministic"]` is *derived* from it
  (`frozenset(_MODEL_SCHEMA_1) - frozenset(_DETERMINISTIC_SCHEMA_1)`), so one deletion silently
  removed three cases across two collections;
- shrinking `FORBIDDEN_BY_ARM_KIND["deterministic"]` by three **at the parametrize site**
  (`sorted(...)[:-3]`) → `230 passed`, green;
- shrinking the same set by three **at the constant** → `1 failed, 229 passed` — but only because a
  non-parametrized sibling, `test_deterministic_arm_forbids_every_model_field`, spot-checks two
  named fields and one of the three I removed was `quantization`. It is a spot-check, not a
  set-level pin. **Mutate at both places and say which one you mutated**: they answer different
  questions, and only the parametrize-site form isolates the parametrized test.

**Same blindness, AST flavour.** A guard that walks one function for `services.<name>` sees only
calls spelled through the alias it reads. Verified 2026-09-08 by mutating
`falkor-chat/server/falkorchat/storefront_api.py` at `2e27835` in a `git archive` sandbox and
re-running `test_the_router_reaches_exactly_the_service_calls_the_exemptions_assume`:

| inserted into the router body | guard |
|---|---|
| `services.start_workflow_run(1)` (the alias) | **1 failed** |
| `shop._services.start_workflow_run(1)` (same layer, direct spelling) | 1 passed |
| `shop.enqueue_turn(1)` (same layer, one hop out via a collaborator) | 1 passed |

The guard carries a control assertion — `_router_bindings(source)["services"] == "shop._services"` —
which covers the alias being *renamed*, and not a call that never uses it at all. Two of the three
shapes above are how the next planned step was actually specified.

**Such a reader has two axes, and a probe that varies one certifies a mechanism it does not cover.**
An alias-resolving AST reader decides two separate questions — *which node type binds a name*, and
*which value expression counts as naming the object*. A coverage probe that enumerates every
binding node type with the value held at one literal spelling comes back empty while real misses
survive, so **enumerate both axes or claim neither**. The target axis can be derived from the
grammar instead of listed: intersecting every `ast.AST` subclass's `_fields` against
`{target, targets, optional_vars, name, names, asname, arg, rest}` yields **exactly 27 classes** on
CPython 3.12.3 (verified 2026-09-08, system `python3`) — `Name`/`Attribute`/`Subscript` in `Store`
are the sites, `Lambda`/`arguments` bind through `arg`. The value axis has no such enumeration: a
reader matching `ast.unparse(value)` against a prefix set is doing exact source-text identity, so an
alias of an alias, a conditional expression, a container round-trip and a non-literal iterable all
stop it, and only alias analysis closes that — a stop to document, not a node type to add.

**And harvesting `ast.Assign` alone under-reaches on ordinary code, not exotic code.** Census over
`falkor-chat/server/falkorchat` (28 `.py` files; run at `00827c2` and against the 2026-09-08
worktree, identical): **65** function-local annotated assignments (`ast.AnnAssign` with a `Name`
target), **40** function-local tuple-target assignments, **1** walrus. `x: T = self._foo` is a house
idiom there. Scope the census the same way you scope the reader — counting `AnnAssign` across the
whole package instead of function bodies gives 304, and mixing the two scopes inside one evidence
line is how this measurement goes wrong.

**The move:** where a gate is generated from the artifact it gates — a parametrize source, an AST
walk, a derived `frozenset` — the mutation that tests it is applied to the **source**, in the shape
the next change is *decided* to take, never a synthetic call written to be seen. Read the step's
plan row first and mutate that; a guard sequenced ahead of its consumer is worth exactly the
mutation that proves it will redden when the consumer lands.

## A mutation-testing kill count is a draw from a distribution, not a fact — and pinning `PYTHONHASHSEED` does not always fix it

Whenever the code a mutant sits in iterates a Python `set` of `str`, iteration order is
hash-randomised, so *which* test reaches the mutant first — or at all — changes run to run. Report a
kill count as a swept range, never as a number.

Mechanism re-verified 2026-09-08 (`falkor-chat/server/.venv`, CPython 3.12.3, no `pytest-randomly`
installed): `list({'alpha','beta','gamma','delta','epsilon'})` yields **7 distinct orders across
`PYTHONHASHSEED=0..7`**. Original observation (`falkor-chat/server`, mutant
`_PresenterSessions.verify -> compare_digest(candidates[0], token)` at `2e27835^`): kill counts
`1,2,1,2,0,1,1,3` over seeds 0..7 — **at seed 4 the mutant survived** a fully green run. Two review
passes that reported 2 and 1 were each faithful observations of one draw.

**The half of the usual remedy that does not hold.** "Pin `PYTHONHASHSEED`" works only when the
set's *members* are fixed. Where they are generated per run, the seed is one of two entropy sources
and pinning it fixes nothing. Measured 2026-09-08 on that same class in a `2e27835^` sandbox:
`_PresenterSessions` mints `secrets.token_urlsafe(32)` values, and at a **pinned**
`PYTHONHASHSEED=0`, eight independent runs put the 1st, 2nd and 3rd-minted token at index 0
(**three distinct answers**: #3, #1, #1, #1, #1, #2, #1, #1). A set of fixed string literals under
the same eight seeds does behave — one stable answer per seed. So: pin the seed for a set of
literals; for a set whose contents are minted, tokens, uuids or temp paths, only a **repeated-run
range** is honest.

**Consequence when adjudicating.** A disagreement between two mutation ledgers over the same mutant
is not a defect in either until you know whether either pinned anything. A single-seed *survived*
is not a coverage gap; a single-seed *killed by three tests* is not redundancy.

## A grep-pinned edit table is an edit list, not a completeness proof

When a plan prescribes a change across already-shipped code by pinning each site with a `grep`
command and a count, and closes with a **residual** (`grep -rFc <token> … → 0`), the table looks
self-verifying and is not. Six ways the residual passes on an incomplete edit, all met in one
plan-gate chain (`docs/plans/small-model-benchmarking.md`, Passes 5–9):

1. **Sites carrying no token are invisible to the command** — a snake_case alias of a camelCase
   field, a dict-literal body under a named mapping, a callee whose name differs from the renamed
   symbol. Measured 2026-09-08 at `8fc2341`:
   `git grep -Fc armKind 8fc2341 -- model-bench/modelbench model-bench/tests` sums to **67** lines,
   `arm_kind` to **25**, of which **19 carry no `armKind` at all**.
2. **A grep pinned to a type name — or to an `isinstance()` string that also pins a variable
   name — misses every site spelling the construct differently.** Re-run at `c523a35`:
   `git grep -Fn 'isinstance(metric, BinaryMetric)'` → exactly **3** lines, all in `report.py`
   (`:211 :553 :564`); the same construct at `results.py:355` and `:584` spells the variable `m`,
   and `results.py:359`/`:385` spell the type as the string literal `"continuous"`. **Enumerate by
   the attribute the ship criterion actually reads**, not by the type name: `git grep -Fn .mean`
   returns `report.py:583`, `results.py:359`, `:584` — exactly the three bare-`else` readers.
3. **A private helper's name matches test *function names* far more often than call sites.**
   Re-run at `5878014`, `8fc2341` and `c523a35`, byte-identical at all three:
   `git grep -Fn _widen -- model-bench/modelbench model-bench/tests` → **7** lines, and the 4 in
   `tests/test_stats.py` (`:743 :773 :915 :1110`) are all `def test_…widening/widened/widens…`
   lines — **none of them calls `_widen`**. A table can enumerate 100% false positives while the
   sites its edit actually breaks (the call sites of a function gaining a required parameter)
   carry the token nowhere.
4. **A rename that keeps its token alive has no zero-residual to assert**, so the done-condition
   passes regardless of what was missed.
5. **A retired-token residual paired with a done-condition that names that token is
   self-contradictory** — the natural test asserting the refusal must spell the token, which puts
   the residual back at 1. The fix is always plan-side: restate the behaviour by key-set or
   complement ("any key outside `{id, state}`"), never by a cleverer test.
6. **Cross-table collisions survive per-table discipline.** Where several tables land as one fix
   round, sweep every line appearing in more than one: a residual can be driven to zero by a
   *different* table's edit on the same line, and two tables can prescribe incompatible forms for
   one line while each reads correct alone. Worse, a residual whose baseline is an **intermediate**
   state — after edit-set A, before edit-set B on the same line — can never be observed non-zero,
   so it is passed both by a faithful implementation and by one that skipped **both** edits. Judge
   such a residual as a conjunction with the first edit-set's own residual, never per-command.

**Two derived checks.** A residual command must be re-asked against *every* implementation the same
table authorises — an authorised literal branch can re-add the very string the residual asserts to
zero. And a required, no-default parameter added to a public function breaks that function's **call
sites**, which the defining token never reaches.

**One caution about re-deriving this class of finding.** A plan gate reads the *working tree*, so
its per-command counts are often taken against a state that was never committed. Of the six
citations above, the three re-runnable at a pinned sha reproduced exactly (2026-09-08); a seventh,
from a gate against an uncommitted `S1e` tree, could not be re-derived at any sha —
`FORBIDDEN_BY_ARM_KIND` appears in the plan and review documents at every commit in the window and
in no source file. Re-derive at a sha, or say that you could not.
