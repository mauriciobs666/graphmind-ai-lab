---
name: joern-cpg
description: Operate the Joern toolset to turn a source repository into a Code Property Graph (CPG) and export/load it into FalkorDB as Cypher. Use when building a CPG for a codebase, querying it with CPGQL (AST/CFG/DDG, call graphs, data-flow & taint), or exporting/ingesting a repo's code graph into FalkorDB. Carries scripts that pin JOERN_HOME/JAVA_HOME and run parse → export (neo4jcsv) → transform → FalkorDB load, plus the CPG→FalkorDB model and a CPGQL cheat-sheet. Primarily driven by `graph-dba`, on demand — CPG generation is a rare task, not a routine one.
---

# Joern CPG → FalkorDB

Turn source code into a **Code Property Graph** with [Joern](https://docs.joern.io)
and materialize it in **FalkorDB** so the code graph is traversable with Cypher.
The scripts encode the environment and the export→transform→load contract — use
them rather than hand-running `joern-*` from memory.

## Prerequisites (the scripts check these)

- **Joern** under `$HOME/joern/joern-cli` (override with `JOERN_HOME`). Verified
  build here: **v4.0.579**. **Don't assume it's installed** — the distribution has
  been observed missing on this box even after a prior session verified it (disk
  pressure, a wiped scratch dir). Run `scripts/joern-env.sh` (or `joern --version`)
  first and treat a missing binary as a blocker to report, not something to
  reinstall ad hoc — provisioning the toolchain is `devops`'s job.
- **JDK 21** (`java -version` → 21). `scripts/joern-env.sh` resolves `JAVA_HOME`
  from `java` on PATH, falling back to the system JVM — no shell config assumed.
- **FalkorDB** reachable at `localhost:6379` (override `FALKORDB_HOST`/`FALKORDB_PORT`),
  loaded via `redis-cli` (start it with `falkor-chat/scripts/start_falkordb.sh`).
  The Python `falkordb`/`redis` packages are **not** required — the loader uses
  `redis-cli GRAPH.QUERY`.

## The pipeline

One command, end to end (build → export → transform → optional load):

```bash
scripts/pipeline.sh <source> --graph cpg_myrepo --workdir ./joern-work --load \
  --verify-prefix tests/
# omit --load to stop at the .cypher artifact (./joern-work/load.cypher)
```

`pipeline.sh` is generic — the caller names the source/graph, nothing is baked in.
Useful flags: `--language <lang>` forces a frontend (**Python → `pythonsrc`**, see
Gotchas); `--reset` `GRAPH.DELETE`s the target graph before `--load` for a clean
reload (destructive, guard-gated); `--repr <r>` narrows the exported layers;
`--source-origin <dir>` names the real tracked directory a staged parse root was
copied from, so the build's provenance is recorded (see Provenance below). After
transform it asserts the CPG produced nodes (a failed frontend exits 0 but yields
an empty graph), and after `--load` it verifies node/edge counts **and**, for
every `--verify-prefix PREFIX` given (repeatable), that `MATCH (m:METHOD) WHERE
m.FILENAME STARTS WITH PREFIX RETURN count(m)` is nonzero — a healthy node/edge
count alone does **not** catch a wrong parse root (see Gotchas below), and this
does. Pass `--verify-prefix` whenever a downstream query (e.g. `cpg-analysis`'s
test-gap recipe) will filter `FILENAME` by prefix; a failing prefix exits the
pipeline non-zero with the fix (rebuild from a parse root that includes it).

### Provenance — what a `--load` stamps, and why it is captured up front

After a successful load, `pipeline.sh` writes a singleton `:CpgBuildInfo` marker
so consumers can judge the graph's freshness (`cpg-analysis`'s
[freshness recipe](../cpg-analysis/references/freshness.md) reads it, and that
file documents every field for readers). What matters about the *mechanism* when you
run a build:

- **Provenance is captured before the parse, not at stamp time**, and scoped to
  the source with a pathspec. A real build runs for hours; deriving `HEAD` after
  the load records whatever the repo moved to meanwhile, and an unscoped
  `git status` reports dirt from anywhere in the repository. Both were observed
  on the 2026-09-07 `cpg_falkorchat` build — `HEAD` moved four times in ~3h and
  the graph was stamped with a commit that was never parsed, plus a `true` dirty
  flag caused by a file outside the parse root. `scripts/git-provenance.sh`
  carries the capture and the reasoning.
- **A staged parse root has no provenance of its own — pass `--source-origin`.**
  Staging a pruned copy *inside* the repo under a gitignored path does **not**
  give the build a commit: the copy is untracked, so the pipeline refuses to
  inherit the containing repo's `HEAD` (which describes a different tree) and
  stamps `PROVENANCE=none` with a loud warning in the first seconds of the run.
  Name the real directory instead — `--source-origin falkor-chat/server` — and
  the commit, tree object (a blob, for a single-file source) and scoped dirty
  flag are derived from *it*. Object ids are stamped as **full 40-char OIDs**,
  since the consumer compares them for equality and `--short` width drifts with
  the repo's object count; the pipeline's log lines abbreviate. Stage
  immediately before invoking the pipeline, since capture happens at start, and
  confirm the copy matches its origin (`diff -rq`, modulo the paths you pruned)
  if the build is one others will lean on.
- **A rebuild erases a hand-authored marker, completely and silently.** The
  stamp is a **map assignment** — `SET b = {…the build's own fields…}` — and `=`
  replaces the node's entire property set, so *every* property the stamp did not
  write is gone afterwards: `NOTE`, `STATUS`, `MARKER_ORIGIN`, a key someone
  invents next year, all of it. There is no list of keys to keep up to date and
  no warning: a rebuild takes the annotation with it and the run reports
  success. That is deliberate — the marker describes one build and nothing else.
  So **read the marker before rebuilding a graph that has one**
  (`MATCH (b:CpgBuildInfo) RETURN b`) and re-write whatever annotation still
  applies afterwards; anything durable about the graph or its component belongs
  in `docs/`, not on this node.
  <br>Getting here took three tries, all on 2026-09-08, which is worth knowing
  if you are tempted to simplify the stamp: writing only the build's own fields
  left a hand-authored marker's keys standing over fresh values; enumerating the
  five known hand-authored keys as `= NULL` closed a *list* rather than a *set*,
  and a sixth key (`MARKER_EVIDENCE`) invented the same day sailed straight
  through it. The map form was checked by execution before being relied on.
  After every stamp the pipeline re-checks that the marker carries nothing but
  that build's own fields, and **fails the run** if anything else is there.
  **State what that catches narrowly, because the wide version was wrong:** it
  catches *the stamp failing to erase a foreign key that was already on the
  marker*. It does **not** detect "the replace semantics changed" — a `+=`, a
  reversion to `b.X = …`, or a FalkorDB treating `=` as a merge all leave
  exactly the eight stamped keys on a marker whose previous stamp was
  pipeline-clean, so none of them fires there. Leave the check in place on the
  reason that is checkable: two loaded graphs carry hand-authored markers today,
  so on each one's next rebuild it fires on exactly the defect this arc was
  about, and after that it is a cheap standing guard against a foreign key
  reintroduced by any writer other than the stamp. It is not redundant with the
  stamp, and it is not a regression test for the replace semantics.
- **A rejected stamp now fails the run.** `redis-cli` exits 0 on an error reply
  and prints it to stdout, so the stamp is checked for an error reply *and* read
  back — the run fails unless the marker in the graph carries this build's
  `PARSED_AT`. Worth knowing because the failure is late and specific: the load
  succeeded, so **nothing here ever needs a re-parse**. What it needs depends on
  which of the five failure branches you hit, and they split in two. **Two say
  the stamp did not land** (FalkorDB rejected it; the read-back did not find
  this build's `PARSED_AT`) — there, **re-sending the Cypher is the fix**, and
  the branch prints it verbatim between `--- begin stamp ---` markers, because
  it is a multi-line map literal with escaped quotes and retyping it is how you
  get a subtly wrong marker. **Three say the stamp DID land** and a later
  assertion failed (the property check could not be built, could not be run, or
  found a stray key) — there the same Cypher is printed as **evidence to
  compare against the marker, explicitly not as a fix**: re-sending it would
  reproduce exactly the state being complained about. Until 2026-09-08 the
  replay was wired into the second set only, so the branches that told you to
  re-stamp were the ones that showed you nothing, and the ones that showed you
  the Cypher told you to re-send it directly under a line saying the stamp had
  already landed. Left unchecked on an `--append` build, the previous build's
  marker would stay standing over the new content.
- **If you change the stamp — or `pipeline.sh`'s stamp block, or
  `git-provenance.sh` — run `scripts/test-stamp-wiring.sh`. It takes a second
  and needs no FalkorDB.** It lifts the real stamp block out of `pipeline.sh`
  between two anchors and drives it against a fake `redis-cli` whose reply
  shapes were measured against a live instance, covering the wiring rather than
  the Cypher. The cases: a clean pass over a hand-authored marker, the
  `provenance=none` narrowing, a planted foreign key caught under merge
  semantics, the `provenance=none` subsumption, the two did-not-land branches
  (asserting the re-send wording, not merely that *something* was printed), a
  stray read that returns a **bare** runtime error, the stray query called
  directly with an unusable allow-list, and mutations that revert the call site
  to the `$(…)` form or drop the allow-list after a populated stamp.
  **Every case asserts an exact exit code, and every case expected to fail must
  also be shown to have printed its branch's stamp block** — that pair is not
  decoration. Under the previous `rc != 0` oracle, deleting `replay_stamp`'s
  definition left the whole suite green while the run was aborting at rc 127 on
  `replay_stamp: command not found`: the test written to close "verified in
  isolation, broken in the wiring" was itself blind to a wiring defect of
  exactly that shape. The allow-list, by contrast, is **exercised** and not
  asserted — each case prints the list it built, and an empty one fails the case
  by tripping the call-site guard rather than by being compared. The `$(…)`
  case exists for the original version of the same lesson: that form *shipped*,
  a subshell silently discarded the allow-list, and every build would have
  failed after a multi-hour parse. The Cypher was executed and correct the whole
  time — it was the call path nobody ran.

Or run the stages individually:

```bash
# 1. Parse source -> CPG binary (overlays applied: call graph, control/data flow)
scripts/build-cpg.sh <source-dir> cpg.bin
#    JOERN_LANGUAGE=<lang> forces a frontend; joern-parse auto-detects otherwise.
#    For Python use pythonsrc (the pysrc2cpg frontend), NOT python — see Gotchas.

# 2. (optional) Query in the REPL before/instead of exporting — see CPGQL below
joern cpg.bin

# 3. Export the CPG to neo4jcsv (the format the transformer consumes)
scripts/export-cpg.sh cpg.bin cpg-export cpg neo4jcsv
#    --repr choices: all|ast|cdg|cfg|cpg|cpg14|ddg|pdg ; export only what you need.
#    NOTE: joern-export requires the outdir to NOT pre-exist; the script clears it.

# 4. Transform the export into FalkorDB Cypher, and load it
python3 scripts/cpg-to-falkordb.py cpg-export -o load.cypher --graph cpg_myrepo --load
#    without --load: writes load.cypher only (the "export to Cypher" artifact)
#    --load streams statements over ONE persistent socket (not a redis-cli per
#    statement) — required at scale: a batched CREATE exceeds the 128KB argv limit
#    and thousands of short-lived connections trigger a reset storm.
```

### What the export looks like

`joern-export --format neo4jcsv` writes, **nested per method** under the outdir:

- `nodes_<LABEL>_header.csv` / `_data.csv` — header `:ID,:LABEL,<PROP>[:type],…`
- `edges_<TYPE>_header.csv` / `_data.csv` — header `:START_ID,:END_ID,:TYPE`
- `*_cypher.csv` — Neo4j `LOAD CSV` scripts; **ignored** (not FalkorDB-usable).

The transformer walks the tree recursively, so the per-method nesting is handled.

## The FalkorDB model (default — `graph-dba` owns real tuning)

The transformer maps the CPG onto FalkorDB like this:

- Every node gets a **shared `:CpgNode` label** *plus* its Joern type label
  (`CREATE (n:CpgNode:CALL) …`), so edges can be matched by `id` without knowing
  the node's type label.
- The Joern `:ID` becomes an integer property **`id`**; other columns become
  properties (`:int` → integer, `:string[]` → array split on `;`, else string;
  empty cells dropped).
- **`CpgNode(id)` is indexed first** (`CREATE INDEX FOR (n:CpgNode) ON (n.id)`),
  so the edge `MATCH (a:CpgNode {id:…})` is cheap. *(Confirm this DDL against the
  pinned FalkorDB build / `graph-dba` — a wrong index only degrades load speed,
  not correctness, since the loader tolerates a failing index statement.)*
- Nodes/edges are created with **UNWIND-batched `CREATE`** (default 500/statement,
  `--batch`), deduped by `id` (nodes) and `(start,end,type)` (edges) so the
  per-method export overlap doesn't double-create.

Edge relationship types are the Joern edge types verbatim (`AST`, `CFG`, `CALL`,
`ARGUMENT`, `REACHING_DEF`, `DOMINATE`, `CONTAINS`, `RECEIVER`, …). `AST`/`CFG`/
`REACHING_DEF` dominate the edge count — a whole-repo load is large and lives in
FalkorDB's RAM; size it with `graph-dba` and export only the `--repr` you need.

### Reloading is deliberate (destructive)

The loader **refuses a non-empty graph**. For a clean reload, reset it first —
either the explicit command, or `pipeline.sh --reset` (which runs it for you,
only if the graph exists):

```bash
redis-cli GRAPH.DELETE cpg_myrepo   # destructive, shared-state — escalates via graph-dba's guard
```

This keeps the reset an explicit, guard-visible command rather than a hidden
side effect. Use `--append` to add into an existing graph instead.

## CPGQL cheat-sheet (in the `joern` REPL)

CPGQL is a Scala traversal DSL over the CPG. Common starting points:

```scala
cpg.method.name("main").l                    // methods named main
cpg.method.name("run").parameter.l           // its parameters
cpg.call.name("system|exec.*").l             // calls to risky sinks (regex)
cpg.call.name("os.system").argument.code.l   // argument source text
cpg.literal.code(".*password.*").l           // suspicious literals

// data-flow / taint: does user input reach a sink?
val src = cpg.call.name("input").argument
val sink = cpg.call.name("os.system").argument
sink.reachableBy(src).l                       // non-empty => tainted path exists

cpg.method.size ; cpg.call.size               // sanity counts after a build
```

Export a query result to a file with `... .l |> "out.txt"` or run non-interactively
with `joern --script <file.sc> --params cpgFile=cpg.bin`.

## Gotchas

- **Python frontend token is `pythonsrc`, not `python`** (Joern v4.0.579).
  `--language python` routes to a legacy generator that fails with *"CPG generator
  does not exist at: …/py2cpg.sh"* (not shipped). Use `pythonsrc` (the `pysrc2cpg`
  frontend). `joern-parse --list-languages` shows both tokens. **Worse, a failed
  frontend still exits 0** with an empty CPG — `pipeline.sh` guards this by
  asserting the export produced node data before loading.
- **Loading at scale needs one persistent connection.** `--load` streams over a
  single socket for exactly this reason: a per-statement `redis-cli` hits Linux's
  128KB per-argv limit on a batched CREATE (`Argument list too long`) and, at
  smaller batches, a connection-reset storm from thousands of short-lived TCP
  connections. Don't reintroduce a per-statement spawn.
- **Piping `pipeline.sh`'s output through `tee <file>` can read as a failed run even when
  it fully succeeded, if the tee target's parent directory doesn't exist yet at pipe-start.**
  `tee` opens its target immediately, racing `pipeline.sh`'s own `mkdir -p "$WORKDIR"` a few
  lines into the script — if it loses, you get `tee: <path>: No such file or directory` up
  front, and with no `pipefail` the overall command's exit code becomes `tee`'s (nonzero)
  rather than `pipeline.sh`'s real (successful) exit code, even though the run completed
  correctly end-to-end (verify-prefix passed, counts matched, freshness marker stamped).
  Either `mkdir -p` the tee target's directory *before* invoking `pipeline.sh`, or skip the
  tee entirely — a backgrounded run's own captured stdout/stderr already has the full log.
- **JVM startup is slow** (~30–60s per `joern-*` invocation). A full
  parse+export of a real repo takes minutes; expect it and run long jobs in the
  background.
- **`--out` must not pre-exist** for `joern-export` — `export-cpg.sh` clears it
  (with a guard against unsafe targets like `/` or `$HOME`).
- **Overlays matter:** taint/data-flow queries need the default overlays (call
  graph, control/data flow) that `joern-parse` applies — don't pass `--nooverlays`
  if you'll query flow.
- **`pysrc2cpg` ships in the `joern-cli` zip — no cold-start runtime download.** A parse
  immediately after unzip succeeds in seconds; there's no first-run stall to pre-warm.
- **A release's `.sha512` sidecar carries a build-relative path** (`<hash>  target/joern-cli.zip`),
  not the local filename — `sha512sum -c` against it fails on the path mismatch even for a valid
  download. Compare the hash column only (`awk '{print $1}'`).
- **`FILENAME` is relative to the parse root you hand `joern-parse`, not the repo
  root.** A CPG built from `<repo>/app` emits bare basenames like `services.py`,
  so any query filtering `FILENAME STARTS WITH 'app/'` silently matches nothing —
  no error, no empty-graph signal, and **node/edge counts still look healthy**
  (this is retroactively identified as the actual root cause of an earlier
  `cpg_falkorchat` build being useless before a rebuild fixed it — the missing
  test sources were a red herring). Just a wrong answer that looks like missing
  code.
  - **Scripted check:** `scripts/pipeline.sh ... --load --verify-prefix tests/`
    (repeatable) asserts `MATCH (m:METHOD) WHERE m.FILENAME STARTS WITH 'tests/'
    RETURN count(m)` is nonzero right after load and **exits the pipeline
    non-zero** if not — make this part of any load whose downstream queries
    (e.g. `cpg-analysis`'s test-gap recipe) filter `FILENAME` by prefix.
  - **Manual check** (when running the stages individually / no `--load`, or
    inspecting an already-loaded graph): `MATCH (m:METHOD) RETURN DISTINCT
    m.FILENAME LIMIT 10` — or, to directly confirm an expected prefix, `MATCH
    (m:METHOD) WHERE m.FILENAME STARTS WITH 'tests/' RETURN count(m)`. If the
    prefix you expect isn't there (or the count is 0), rebuild from a parse root
    that includes it.
  - **Live-verified** (2026-08-09, `falkordb-dev`, graph `cpg_falkorchat`, via
    `mcp__cypher__query`): happy path `MATCH (m:METHOD) WHERE m.FILENAME STARTS
    WITH "tests/" RETURN count(m)` → **1067**; failure path with the same query
    against `"nonexistent/"` → **0**, i.e. exactly the condition
    `pipeline.sh --verify-prefix` treats as a hard failure and exits 1 on.
- **No `--exclude`/ignore flag exists.** `build-cpg.sh`/`pipeline.sh` parse
  whatever directory they're pointed at verbatim — pointing at a real project
  directory also parses its `.venv`/`node_modules`/build caches. To scope a
  parse, stage the wanted subtrees into a scratch copy first (pruning
  `__pycache__`/similar) rather than expecting the tooling to filter for you.
  This is the same lever as the `FILENAME`-prefix gotcha above: the parse root
  and what's inside it are one decision. **Where you stage it doesn't recover
  the source's git identity — `--source-origin` does.** Staging inside the repo
  under a gitignored path is fine for keeping the copy near its artifacts, but a
  copy git doesn't track carries no commit either way; pass the real tracked
  directory so the build gets stamped (see Provenance above).
- **`cpg-to-falkordb.py --load` always re-transforms the export** — there's no
  "replay this `.cypher`" mode. Re-running `--load` re-reads every export CSV
  and rewrites `load.cypher` before streaming it; cheap (seconds, no re-parse)
  but not free, and there's no built-in way to just replay a saved artifact.
- **Scale:** deduped in memory by the transformer — fine for moderate repos; for
  very large codebases this is a streaming-loader concern (tracked in
  `graph-dba`'s kaizen plan, K-005). Rule of thumb: **~2,700–2,800 nodes / ~18,000–18,600
  edges per Python source file** with default overlays — consistent across two independent
  real runs (41 files → 110k nodes/735k edges; a later 60-file rebuild of the same target →
  167k nodes/1.12M edges) despite the file count itself being a moving target as the source
  tree grows (it was already 65 files a day after the 60-file measurement). **Measure your
  own repo's file count before projecting** (`find "$SRC" -name "*.py" | wc -l`) rather than
  anchoring on either worked example's total — the per-file rate is the durable number. A
  repo 10× a measured baseline projects into multi-million-edge territory, worth a
  `graph-dba` sizing conversation before a full load. Prefer a narrower `--repr` (e.g. `ast`
  or `cpg14`) when you don't need every edge layer.
- **Deeper schema & model notes:** see [`references/cpg-model.md`](./references/cpg-model.md).
