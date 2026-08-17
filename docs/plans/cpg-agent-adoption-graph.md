# CPG freshness marker — design

> **Status:** archived · **Owner:** `graph-dba` · **Tracks:** cpg-agent-adoption (M4)

Design for the FR-5/FR-6/FR-7/FR-8 slice of
[`../requirements/cpg-agent-adoption.md`](../requirements/cpg-agent-adoption.md):
a mechanism so an agent consulting a loaded CPG can tell how current it might be
and, if it looks stale, surface a suggestion to refresh — without changing the
on-demand build model or the MCP read path. Coordination:
[`cpg-agent-adoption-coordination.md`](./cpg-agent-adoption-coordination.md) (unit
U1). Downstream: `cobb`'s primary design (unit U2,
`docs/plans/cpg-agent-adoption.md`) cites the recipe in §2 by path; it owns the
agent roster and discovery wiring, neither of which this document decides.

**Scope note (AC-6).** This document does not revisit M2
(`docs/plans/m2-cpg-analysis-skill.md`) or M3
(`docs/requirements/cpg-query-access.md`) consumer-scope decisions — it adds one
new node label, one new recipe file, and a few lines to `pipeline.sh`'s already-
existing `--load` path. Nothing here changes who is wired to `cpg-analysis` or
`mcp__cpg__query`; that's `cobb`'s slice (U2).

**Implementation note.** This is the design only — no files are edited here.
Unit U4a (`graph-dba`, per the coordination doc) implements exactly what §1–§2
below specify, against `skills/joern-cpg/scripts/pipeline.sh` and a new
`skills/cpg-analysis/references/freshness.md`.

---

## 0. Grounding — what's actually in the graphs today (live-verified)

Checked via `mcp__cpg__query` against both loaded graphs, this session:

- `MATCH (n:META_DATA) RETURN n` → **0 rows** on both `cpg_falkorchat` and
  `cpg_salesperson`. `META_DATA` is documented in `cpg-model.md` as a label
  Joern *can* emit, but it is **absent from both live graphs**.
- `CALL db.labels()` on `cpg_falkorchat` → 20 labels (`CpgNode`, `METHOD`,
  `CALL`, `LOCAL`, `BLOCK`, `TYPE_DECL`, `IMPORT`, …). No `META_DATA`, `FILE`,
  `TYPE`, or `NAMESPACE` label exists at all.
- `MATCH (n:METHOD) RETURN count(n)` → **2,904** on `cpg_falkorchat` (the task
  brief cited 2,037 — already stale by the time this note was written; a fair
  illustration of the exact problem FR-5 exists to catch) and **359** on
  `cpg_salesperson`, matching the brief.

**Conclusion: don't hook `META_DATA`.** It's a Joern-side-possible label, not a
`pysrc2cpg`/current-export guarantee — this build's export simply doesn't carry
it (repo convention, `references/cpg-model.md`, says "you'll see most", never
"always"; treat this as confirmation of that hedge, not a contradiction of it).
Depending on it would mean shipping a freshness feature that silently does
nothing on every graph built the way these two were. A **dedicated node** is
the only option that's guaranteed to exist because *we* create it.

---

## 1. What gets stamped, where, and when

### 1.1 The node

One **singleton node per graph**, label `:CpgBuildInfo` — deliberately **not**
tagged with the shared `:CpgNode` label. `:CpgNode` exists so Joern-id edges can
be resolved (`cpg-model.md` §"Why the shared `:CpgNode` label"); this node has
no Joern id and no edges, and keeping it off that label means it never shows up
in an `id`-keyed traversal or perturbs the `CpgNode(id)` index's selectivity.

Properties (all **UPPER_CASE**, matching every other property key in this
schema — deliberately, not just for consistency's sake: `cpg-model.md` already
flags "a lowercase key silently returns null" as the #1 gotcha for this schema,
and a marker node with lowercase keys would be a second, self-inflicted version
of that exact trap):

| Property | Type | Always present? | Value |
|---|---|---|---|
| `BUILT_AT` | string | yes | ISO-8601 UTC, second precision: `2026-08-16T14:32:07Z` |
| `SOURCE_PATH` | string | yes | the parse root passed to the pipeline (`$SRC`) |
| `SOURCE_COMMIT` | string | only if `$SRC` is inside a git working tree at build time | short SHA (`git -C "$SRC" rev-parse --short HEAD`) |
| `SOURCE_DIRTY` | boolean | only alongside `SOURCE_COMMIT` | `true` if `git -C "$SRC" status --porcelain` is non-empty at build time |

`SOURCE_COMMIT`/`SOURCE_DIRTY` are omitted (not set to null/empty-string) when
`$SRC` isn't a git working tree — consistent with the transformer's own
"empty cells dropped" convention (`cpg-to-falkordb.py`'s `cypher_scalar`), and
it means a reader gets a real Cypher `null` on the missing property rather than
a sentinel to special-case.

**Why a source-commit hash, and why it's cheap.** `git rev-parse --short HEAD`
and `git status --porcelain` are both local, no-network, sub-10ms operations —
free relative to a Joern build that already takes minutes. The payoff is large:
raw age alone can't distinguish "old graph, code hasn't moved" from "old graph,
code has moved a lot" (see §3), and the commit reference is what lets a
consuming agent ask that question cheaply itself, from inside the same repo it
is already working in.

### 1.2 Where in `pipeline.sh`'s flow

Append one step to the **end** of the existing `if [ -n "$LOAD" ]` block in
`scripts/pipeline.sh` — after the node/edge count report, and **after** the
`--verify-prefix` loop if one runs. That ordering is deliberate: it means a
build that fails its own verification (bad parse root, `--verify-prefix` count
of 0) exits `1` **before** ever reaching the stamp, so a structurally broken
graph is left with no marker at all — which downstream reads as "unknown
freshness, treat with caution" (§2), never as "confidently fresh." A marker
therefore certifies *a load that itself passed the pipeline's own checks*, not
merely "a load was attempted."

```bash
# Freshness marker (cpg-agent-adoption M4, FR-5/FR-6) — written only after the
# load and any --verify-prefix checks have fully succeeded, so a stamped graph
# means "built successfully at this time," never "an attempt was made." One
# singleton node per graph; MERGE (no property in the pattern) keeps it that
# way across both --reset (fresh graph) and --append (existing graph) loads —
# freshness tracks "when was this graph's content last touched," not "when was
# it first created."
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="MERGE (b:CpgBuildInfo) SET b.BUILT_AT = \"$BUILT_AT\", b.SOURCE_PATH = \"$SRC\""
if command -v git >/dev/null 2>&1 && git -C "$SRC" rev-parse --short HEAD >/dev/null 2>&1; then
  SHA="$(git -C "$SRC" rev-parse --short HEAD)"
  DIRTY=false
  [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] && DIRTY=true
  STAMP="$STAMP, b.SOURCE_COMMIT = \"$SHA\", b.SOURCE_DIRTY = $DIRTY"
fi
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$STAMP" >/dev/null
echo "pipeline: stamped '$GRAPH' — BUILT_AT=$BUILT_AT SOURCE_PATH=$SRC" >&2
```

This is a `redis-cli GRAPH.QUERY` call in the same style as the script's
existing `count()` helper — no new dependency, no change to
`cpg-to-falkordb.py`. That file never sees `$SRC` (it only sees the export
directory) and stays a pure CSV→Cypher transformer; `$SRC`, `$GRAPH`,
`$HOST`/`$PORT` are all already in `pipeline.sh`'s scope, which is the natural
owner of "what source built this."

**Not a Cypher parameter.** Per this repo's established convention (neither
`redis-cli GRAPH.QUERY` nor `mcp__cpg__query` binds parameters — `cpg-analysis`
`SKILL.md` §3), the values are inlined as quoted literals, same as every other
statement `pipeline.sh`/`cpg-to-falkordb.py` already emits.

---

## 2. The read recipe (through `mcp__cpg__query`, unchanged tool — FR-8)

New file, `skills/cpg-analysis/references/freshness.md`, in the same
copy-adaptable style as `impact-analysis.md`/`rca.md`/`code-review.md`/
`test-gap.md` — **recommended as a fifth recipe file**, not a new section
bolted onto `SKILL.md` itself: freshness-checking is a distinct, self-contained
task exactly like the other four, it has its own "expected shape" and "limits"
the way they do, and keeping one file per task is the pattern already
established. The only `SKILL.md` change (for `cobb`/U4a, not executed here) is
one row added to the §4 navigation table:

```markdown
| Judge how current a loaded CPG is before trusting it | any consuming agent | [`references/freshness.md`](references/freshness.md) |
```

Proposed content for `references/freshness.md`:

````markdown
# Recipe: freshness check

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumers:** any agent consulting a loaded CPG. **Covers:** FR-5, FR-6
> (`cpg-agent-adoption`, M4).

**Purpose.** Before trusting a loaded CPG's answers, find out how current it is
relative to the source it describes — and if it looks stale, say so rather than
silently treating it as ground truth. No parameter to change; run as-is against
whichever graph you're already querying.

```cypher
MATCH (b:CpgBuildInfo)
RETURN b.BUILT_AT AS builtAt, b.SOURCE_COMMIT AS sourceCommit,
       b.SOURCE_DIRTY AS sourceDirty, b.SOURCE_PATH AS sourcePath
```

**Expected shape.** Zero or one row — this is a singleton marker node, not a
per-build history.

- **One row** → the graph was built by a pipeline run that completed the
  freshness-stamping step (`graph-dba`'s `joern-cpg` pipeline, added M4).
  `builtAt` is an ISO-8601 UTC timestamp; `sourcePath` is the parse root passed
  to the build. `sourceCommit`/`sourceDirty` are present only when the parse
  root was inside a git working tree at build time — `sourceDirty = true` means
  the tree had uncommitted changes when the CPG was built, so even a fresh
  `builtAt` is describing an unrecorded snapshot.
- **Zero rows** → either the graph predates this feature (built before M4; no
  backfill was done — see the rollout note in the graph-dba design doc) or the
  pipeline run that built it failed its own verification and never reached the
  stamping step. Treat this the same as "stale": you have no freshness signal
  at all, which is itself a reason for caution, not an error to debug.

**Judging staleness (a suggestion, not a rule).** Two escalating checks,
cheapest first — the threshold is yours to set given the task at hand:

1. **Raw age.** `now − builtAt`. There's no universal cutoff — a week-old CPG
   on a slow-moving component may be fine; an hour-old one on a component
   under active refactor might already be behind. Weigh it against how much
   the task leans on structural correctness.
2. **Actual source movement (stronger, still cheap).** If `sourceCommit` is
   present, run `git log --oneline <sourceCommit>..HEAD -- <sourcePath>` from
   the repo root; if it's absent, `git log --oneline --since=<builtAt> --
   <sourcePath>`. A **nonzero** commit count is a much stronger staleness
   signal than raw age — it means the source moved since the graph was built,
   regardless of how long ago that was. Zero commits is the converse: the
   graph may look old but the code it describes hasn't changed under it.
   (This is exactly the check `docs/plans/cpg-query-access.md` §2.3 did by
   hand — `git log --oneline --since=2026-07-18 -- falkor-chat/server` → 8
   commits — to establish an M2-era CPG was stale before its M3 rebuild. This
   recipe just hands you the two inputs to run that check yourself instead of
   guessing.)

**Surfacing the suggestion (FR-6).** When either check makes you doubt the
graph, say so in whatever you hand back — don't silently keep using it as if
current, and don't rebuild it yourself. Naming a concrete next step is enough:
*"this CPG was built at `<builtAt>` (or: has no freshness marker) and
`<sourcePath>` has moved since; consider asking `graph-dba` to rebuild
`<graph>` before trusting a broad structural claim from it."* Whether to pause
and ask, or flag it and proceed, is your call — this recipe hands you the
signal, not the threshold.

## Limits

- **One marker per graph, not a build history.** An `--append` load overwrites
  the existing marker's timestamp (by design — freshness tracks "when was this
  graph's content last touched," not "when was it first created").
- **`sourceCommit`/`sourceDirty` need the parse root to be a real git working
  tree at build time.** A build staged from a pruned scratch copy (no `.git`
  copied along — the pattern `cpg-query-access.md` §S8 used for `cpg_falkorchat`
  itself, to keep `.venv` out of the parse) has no commit reference even though
  the *real* source is tracked. `sourcePath` and raw `builtAt` age are then the
  only signal.
- **Opt-in per build.** Any CPG built before this feature shipped, or by a
  pipeline run whose own load verification failed, has no marker — that's the
  "zero rows" case above, not an error.
````

---

## 3. Staleness heuristic — summary

Two data points are handed to the consuming agent, cheapest-first:

1. **`BUILT_AT` age** — always available when a marker exists; zero cost to
   read, but a weak signal on its own (age doesn't imply drift).
2. **`SOURCE_COMMIT` + `SOURCE_PATH`**, feeding a `git log <sha>..HEAD --
   <path>` (or `--since=<builtAt>` when no commit is recorded) — a strong
   signal, still cheap (a local `git log`, no CPG query needed), and it answers
   the question that actually matters: *has the source moved since this graph
   was built*, not merely *how long ago was it built*.

This is enough for FR-5 ("some indication of how current") without FalkorDB
needing to know anything about git itself — the graph only ever records what it
was built from and when; the "is that stale" judgment happens entirely on the
consuming agent's side, against data it already has cheap access to (its own
repo checkout). No threshold is baked in anywhere in the schema or the recipe,
per FR-6's explicit "leave the judgment to the agent."

---

## 4. FR-7 / FR-8 confirmation

- **FR-7 (build model unchanged).** The only change to the build path is
  additive: one more `redis-cli GRAPH.QUERY` call at the tail of
  `pipeline.sh`'s existing `--load` branch, gated by the same conditions that
  already gate that branch. `graph-dba` still runs `pipeline.sh` deliberately,
  on request; nothing here schedules, triggers, or suggests a rebuild from
  inside the pipeline itself. The stamping step cannot fire without a human (or
  an agent acting on a human's behalf) already having asked `graph-dba` to
  build or rebuild a graph — it rides along on an action that was already
  going to happen, exactly as FR-7 requires.
- **FR-8 (MCP read path unchanged).** §2's recipe is `MATCH … RETURN` through
  the existing `mcp__cpg__query(graph, cypher)` — the same two parameters,
  same tool, same `GRAPH.RO_QUERY` execution mode, same truncation/error
  behavior as every other `cpg-analysis` recipe. `cpg/mcp/server.py` is not
  touched by this design at all; there is nothing for U4a to change there.

---

## 5. Rollout for the two live graphs — no backfill, wait for next rebuild

Confirmed live (§0): neither `cpg_falkorchat` nor `cpg_salesperson` has a
`CpgBuildInfo` node today, and none will exist until each graph is rebuilt
through the updated pipeline.

**Recommendation: do not backfill. Let each graph pick up the marker on its
next on-demand rebuild.** Reasoning:

- A backfilled `BUILT_AT` would have to be **fabricated**. There is no reliable
  source for cpg_falkorchat's true last-build timestamp: `cpg-query-access.md`
  D1 records a clean rebuild at M3 (2026-07-25) with a specific baseline
  (79,581 nodes), but the live graph today has neither that count nor the
  2,037-method count the requirements brief cited — meaning it has been
  rebuilt again since, **undocumented**. That undocumented-rebuild fact is
  itself the exact failure mode this feature exists to prevent; guessing at
  which of several silent rebuilds is "the" one to backfill from would launch
  the feature by repeating the problem.
- A wrong or approximate backfilled timestamp is **worse than no marker**. An
  agent that sees a `BUILT_AT` will reasonably trust it as ground truth (that's
  the whole point of §2); a fabricated one is silently misleading in exactly
  the way FR-6 says never to be. "Zero rows" reads honestly as "unknown, be
  careful" — a wrong date reads as false confidence.
- The wait is short and self-correcting by design. FR-7's own premise is that
  the parent feature increases how often on-demand rebuilds happen — so both
  graphs are likely to be rebuilt again reasonably soon as adoption widens,
  at which point they pick up the marker for free, honestly.

**What this means for a consuming agent today:** querying either live graph's
freshness recipe returns **zero rows** until its next rebuild. Per §2's recipe,
that reads as "no freshness signal — treat with the same caution as a stale
graph," which is the correct, honest answer for both graphs *right now*. This
is not a gap in the design; it's the intended behavior for a marker that
refuses to lie about builds that predate it.

`graph-dba` could rebuild either graph today purely to backfill the marker
sooner — but doing so proactively, outside of an actual consumer's request, is
exactly the "proactive build-out" the parent requirements doc rules out (Out of
scope: *"Proactive, wholesale CPG build-out… coverage grows only through the
existing on-demand model"*). Left as available on request, not done here.

---

## 6. Open questions / risks for the next reader

- **Staged-source builds lose `SOURCE_COMMIT`.** The one real precedent in this
  repo for building `cpg_falkorchat` (`cpg-query-access.md` §S8) staged
  `{falkorchat, tests}` into a scratch copy specifically because
  `build-cpg.sh`/`pipeline.sh` has no `--exclude` and would otherwise parse
  `.venv`. A scratch copy with no `.git` directory means `SOURCE_COMMIT`/
  `SOURCE_DIRTY` silently come back absent on exactly this repo's established
  build pattern, leaving only `BUILT_AT` + `SOURCE_PATH` (still enough for the
  raw-age check, not the stronger git-log check). Not fixed here — flagged so
  U4a's implementer doesn't discover it by surprise, and so a future
  enhancement (capture the commit from the *original* repo path before
  staging, pass it through as an extra `pipeline.sh` flag) has a clear reason
  to exist if this gap turns out to matter in practice.
- **`git` binary absence.** `pipeline.sh` already assumes a Unix-ish shell with
  `redis-cli`; the stamp step additionally soft-depends on `git` being on
  `PATH`. Guarded with `command -v git` — a missing `git` degrades to
  `BUILT_AT`/`SOURCE_PATH` only, never a hard failure of the pipeline.
  Confirm this is acceptable to `cobb`/the reviewer; the alternative (hard-fail
  the whole load without `git`) seemed disproportionate for a metadata step.
- **No cross-graph freshness comparison.** This design gives each graph its own
  marker; it does not attempt to compare `cpg_falkorchat`'s freshness against,
  say, the `falkor-chat` component's own deploy/release cadence, or expose a
  "how stale is stale" default. That's deliberately left to the consuming
  agent (FR-6) — flagging it explicitly in case a future iteration wants a
  repo-wide freshness dashboard (which would itself collide with the parent
  requirements doc's explicit "no usage-tracking / dashboards" out-of-scope
  item, so it should stay out).
- **`cobb`'s discovery step should mention this recipe exists** so a newly
  wired agent doesn't have to find `references/freshness.md` on its own — but
  the wording of that mention is `cobb`'s call (U2), not designed here.
