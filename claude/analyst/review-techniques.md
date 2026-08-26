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
`pip install -e '.[dev]'`-style editable installs register a `MetaPathFinder` that is consulted
before `sys.path`, so copying the tree to a scratch dir and prepending it to `PYTHONPATH` still
imports the *working-tree* version — it does not shadow it. What works instead, with zero
working-tree writes:
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
constructing any objects. `PYTHONPATH`/`sys.path.insert` does **not** reliably shadow a package
installed via `pip install -e .` (editable install) in this environment — the editable-install
`MetaPathFinder` is consulted before `sys.path`, so `import pkg` keeps resolving to the real
installed copy even with a same-named mutated tree earlier on `PYTHONPATH`. This differs from (a):
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

Origin: two independent instances. (1) A `cobb`-authored plan cited a NULL-backfill decision as
having surfaced during a specific past investigation; `grep -rn -i backfill claude/docs/` found
zero occurrences outside the plan doc itself, and the cited investigation was unrelated
(hook/permission engineering, no migrations at all) — the example was plausible-sounding but
fabricated. (2) A fix-pass plan claimed two findings were "recorded in `falkordb-quirks.md`";
grepping the file directly at the cited line ranges confirmed both were genuinely present, not
just asserted — the same check, run the other direction, separating a closed finding from an
asserted-but-undone one.

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
