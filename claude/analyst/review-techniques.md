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
