# Review — CPG provenance capture before the parse (`6012ddb`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (U38, `salesperson-ui` coordination)

## Scope & verdict

**Reviewed:** commit `6012ddb` (`fix(joern-cpg): capture CPG provenance before the parse, scoped
by pathspec`) in full — `skills/joern-cpg/scripts/git-provenance.sh` (new),
`skills/joern-cpg/scripts/pipeline.sh`, `skills/joern-cpg/SKILL.md`,
`skills/cpg-analysis/references/freshness.md`, `skills/README.md` — against its parent
`6012ddb^` (read via `git show`, never checked out).

**Method:** static reading plus non-destructive execution of the *helper in isolation*
(`git-provenance.sh` sourced against throwaway git repos under the session scratchpad), `bash -n`,
a fake-`redis-cli` argv probe, read-only `GRAPH.RO_QUERY`/`mcp__cypher__query` reads of the live
`cpg_falkorchat` and `cpg_deprecated_salesperson` markers. The Joern pipeline was **not** run; no
graph was written, deleted or rebuilt; no git command touched the working tree. `shellcheck` is not
installed in this environment (confirmed), so the shell judgment is reading plus direct execution.

**Verdict: needs changes** — one blocker (silent stamp failure), four majors.

CPG: considered, not relevant — the change is Bash and Markdown; no `cpg_skills` graph exists, and
the CPG that *is* live (`cpg_falkorchat`) is the subject of the provenance data, not of the code.

The design is right and the diagnosis is right: capture-before-parse, pathspec scoping, refusing to
inherit a containing repo's `HEAD`, and `SOURCE_TREE` as the identity check are all correct calls,
and I verified the helper does what it claims on every source shape I could construct. The blocker
is not in the captured values — it is that the one write that persists them cannot fail loudly.

### Disposition of the three risks named in the brief

| Author's risk | Finding |
|---|---|
| 1a — `redis-cli` accepts the multi-line stamp as one argument | **Closed, safe.** Verified: exactly 7 argv elements, the 301-byte multi-line stamp intact as `ARG[7]` (Appendix A2); a multi-line Cypher query executes over `redis-cli` argv against the live FalkorDB (A3). No execution test needed for *argument passing*. But see **B1** — the invocation's *error handling* is the real hole. |
| 1b — `PARSED_AT`/`PROVENANCE` survive `--reset`/`--append` | **Closed by construction.** Both are shell variables set at `pipeline.sh:89–102`; the `GRAPH.DELETE` at `:137` and the loader at `:142` touch no shell state, and the stamp at `:198` reads the variables, not the graph. No path can clear them. |
| 2 — the deliberate "absent beats plausible-but-wrong" regression | **Judgment endorsed**, consumer handling **mostly** adequate — gaps at **M3** (the pre-fix marker) and **m2** (`SOURCE_TREE` absent while `SOURCE_COMMIT` is present, a reachable shape). |
| 3 — residual stage-then-invoke race | **Accepted.** Documented in all three places a builder reads (`git-provenance.sh:22`, `pipeline.sh:43–45`, `SKILL.md` Provenance). No finding. |
| 4 — every field written, absent ones `NULL` | **Sound**, and `freshness.md:108–112` states the resulting guarantee. B1 falsifies that guarantee in the failure case. |

---

## Findings

### B1 — blocker · a failed stamp is completely silent, and on `--append` leaves the old marker over new content

`pipeline.sh:199`

```bash
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$STAMP" >/dev/null
```

Verified against the live instance: **`redis-cli` exits 0 on an error reply and prints the error to
stdout** (Appendix A1). `>/dev/null` discards it, `set -e` sees 0, and `:200–201` then print
`pipeline: stamped '$GRAPH' — …` and the pipeline exits 0. Any rejection — a Cypher parse error, a
read-only replica, an OOM, a graph-key type clash — is invisible after a 3-hour build.

Worse in the direction that matters: on a build **without** `--reset` (append into an existing
graph), a silently failed stamp leaves the *previous* build's marker standing over new content. That
is precisely the failure the write-every-field-as-`NULL` design (`git-provenance.sh:92–96`) exists to
prevent, and it falsifies `freshness.md:108–112` ("a marker never mixes two builds"). A consumer then
runs check 0 against a stale-but-plausible `SOURCE_TREE` and gets a confident wrong "unchanged".

**Fix:** capture the reply and verify, then read the marker back. Suggested:

```bash
if ! OUT="$(redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$STAMP" 2>&1)" \
   || printf '%s' "$OUT" | grep -qiE '^\(?error|^ERR |wrong number|read only'; then
  echo "pipeline: FAILED — stamp rejected by FalkorDB: $OUT" >&2; exit 1
fi
BACK="$(count "MATCH (b:CpgBuildInfo) RETURN b.PARSED_AT")"   # readback assertion
```

The same `redis-cli`-exits-0 property affects the pre-existing `count()` helper (`:148`), which is
why the readback must compare a value, not just succeed.

### M1 — major · the consumer's check-0 command is fatal for `sourceOrigin = "."`

`freshness.md:66–68` tells the reader to run `git rev-parse --short HEAD:<sourceOrigin>`. The
producer explicitly supports `sourceOrigin = "."` — `git-provenance.sh:71` normalises the repo-root
prefix to `"."`, and `:74–75` special-cases it to `HEAD^{tree}`. The consumer has no such case:

```
$ git rev-parse --short HEAD:.
fatal: Needed a single revision            # exit 128
$ git rev-parse --short HEAD:./            # works, == HEAD^{tree}
1e4fb8f
```

(Verified in this repo and in a scratch repo; `HEAD:<subdir>` is fine, only `.` breaks.) Anyone
building a CPG of a whole component/repo root — `pipeline.sh . …` or `--source-origin .` — gets a
stamp whose documented verification command errors out. Loud, not silent, but the headline new check
is unavailable for that shape.

**Fix:** in `freshness.md`, make check 0 read
`git rev-parse --short "HEAD:./<sourceOrigin>"` (works for both `.` and a subdirectory when run from
the repo root, verified), or add one clause: *"when `sourceOrigin` is `.`, use
`git rev-parse --short 'HEAD^{tree}'`"*.

### M2 — major · the pipeline dirties its own source before measuring it

`pipeline.sh:79` runs `mkdir -p "$WORKDIR"` (default `./joern-work`) **before** the capture block at
`:82–116`. `git status --porcelain -- <src>` counts untracked files, so when the workdir lands inside
the source, the pipeline's own scratch directory flips `SOURCE_DIRTY` to `true`. Reproduced:

```
workdir created under src           rc=0 origin=[src] dirty=[true]
workdir under repo root, source=.   rc=0 origin=[.]   dirty=[true]
  -> git status --porcelain -- . :  ?? joern-work/
```

Consequence: `pipeline.sh . --load` (defaults) can never produce `sourceDirty = false`, so check 0
— gated on it at `freshness.md:63–64` — is permanently unavailable for that build shape, and the
builder is told at `:112–114` that the source has uncommitted changes when it does not. This is new
harm: the old code's dirty flag was repo-wide anyway, so nobody noticed.

**Fix:** move `mkdir -p "$WORKDIR"` (`:79`) below the capture block (after `:116`). `CPG`/`EXPORT`/
`CYPHER` at `:80` are only string assignments and can stay. A pathspec exclusion would also work but
is more fragile than reordering two lines.

### M3 — major · the pre-fix-marker guidance is correct about derivation, but leaves check 2 unrunnable for the marker in use today

The *derivation* claims at `freshness.md:135–141` check out against the old code
(`6012ddb^:pipeline.sh`, tail lines 37–41): `git -C "$SRC" rev-parse --short HEAD` after the load,
and `git -C "$SRC" status --porcelain` with **no** pathspec. Correct. The `cpg_falkorchat` claim at
`:146–147` also checks out — the live node is
`{BUILT_AT:"2026-09-07T22:25:45Z", SOURCE_COMMIT:"b795f4c", SOURCE_DIRTY:False, SOURCE_PATH:"…/cpg/.cpg-artifacts/src/falkor-chat-server"}`,
i.e. `PROVENANCE`/`PARSED_AT`/`SOURCE_TREE`/`SOURCE_ORIGIN` all absent and both git values
hand-corrected away from the stamped `2624425`. So the bullet is factually right.

What it does not say is what to *do*. For this marker `sourceOrigin` is null, and `:94` instructs
**"Use `sourceOrigin`, not `sourcePath`"** — while `sourcePath` here is an absolute path into a
gitignored staged copy, the exact input `:113–118` warns returns a silent false "unchanged". Check 2
is therefore unrunnable as written for the one live marker the reader is being taught to interpret.
The workaround exists — infer the real directory from task context and verify it (`:124–132`) — but
it is scoped to the `provenance: none` bullet, not to this one.

**Fix:** append one clause to the `:135–147` bullet: *"`sourceOrigin` is absent on this shape too, so
check 2 needs the same independently-confirmed real directory as the `provenance: none` case above —
for `cpg_falkorchat` that is `falkor-chat/server`; anchor on `sourceCommit`, not on `sourcePath`."*

### M4 — major · the marker's field list has drifted in a third document

`docs/manuals/graph-ontology.md` (`Status: active`, owner `tico`) documents `CpgBuildInfo` in two
places, both now incomplete:

- `:118` — properties listed as `BUILT_AT`, `SOURCE_PATH`, `SOURCE_COMMIT`, `SOURCE_DIRTY`. Missing
  `PARSED_AT`, `PROVENANCE`, `SOURCE_ORIGIN`, `SOURCE_TREE`.
- `:486` — `MATCH (b:CpgBuildInfo) RETURN b.SOURCE_PATH, b.BUILT_AT, b.SOURCE_COMMIT`, followed at
  `:489` by *"`SOURCE_PATH` is the tree that was actually scanned — check it names the code you
  meant."* That is now contradicted head-on by `freshness.md:113–118`, which says `sourcePath` is
  frequently a pruned scratch copy and `sourceOrigin` is the field to read.

Producer ↔ `freshness.md` is exact (all eight fields, same names, same semantics — I checked each);
this is the only field-name drift I found, and it is in the document a reader reaches when asking
"is this graph the right code?".

**Fix:** routes to `tico`, not `cobb` — add the four fields at `:118` and replace the `:486`
query/`:489` gloss with `SOURCE_ORIGIN` (falling back to `SOURCE_PATH` when `PROVENANCE` is null).

### m1 — minor · check 0 overclaims for `provenance = source-origin`

`freshness.md:70–72`: *"Equal → the committed source is byte-identical to what was parsed."* True for
`parse-root`. For `source-origin` the tree describes the **tracked origin directory**, while the
parse root was a *pruned copy* of it — equality means "the origin is unchanged since capture", which
only implies "matches what was parsed" if the copy was faithful modulo pruning. Nothing enforces
that; `SKILL.md`'s Provenance bullet only *suggests* a `diff -rq`. The converse also bites: a change
to a tracked-but-pruned file moves the tree and reports a false "moved" (safe direction, but noisy).

**Fix:** split the sentence by provenance — `parse-root` → "byte-identical to what was parsed";
`source-origin` → "the directory the parse root was copied from is unchanged since capture".

### m2 — minor · `SOURCE_COMMIT` present with `SOURCE_TREE` absent makes check 2 answer "unchanged" about never-committed source

Reachable and reproduced: a source directory tracked in the index but not yet in `HEAD` (added, not
committed). `git-provenance.sh:77`'s `rev-parse HEAD:<origin>` fails, `|| true` swallows it, and the
stamp is `PROVENANCE="parse-root", SOURCE_COMMIT="c61947b", SOURCE_TREE=NULL, SOURCE_DIRTY=true`.
Check 0 correctly skips (no `sourceTree`); check 2 then runs `git log <commit>..HEAD -- <origin>` →
**0 commits**, which reads as "the code it describes hasn't changed under it" for source that has
never been committed at all. `sourceDirty = true` is the only tell, and check 2 (`:83–95`) never
mentions consulting it — the caveat lives in check 0's parenthetical only.

**Fix:** one clause on check 2: *"a zero result means nothing when `sourceDirty = true`, or when
`sourceTree` is null while `sourceCommit` is present — the source was not fully committed at capture."*

### m3 — minor · bullet ordering mis-classifies a hand-written marker

`freshness.md:37–39` — *"One row, `provenance` null → a **pre-2026-09-07 stamp**"* — matches the
hand-written marker too: `cpg_deprecated_salesperson` live is `{BUILT_AT:"unknown", MARKER_ORIGIN:…}`
with **no** `PROVENANCE` (verified). A reader working down the list stops there, is told
`sourceCommit`/`sourceDirty` are merely "weaker than they look", finds neither present, and reaches
for check 2's `--since` fallback with `builtAt = "unknown"` — the silent zero-commit trap the file
itself documents at `:148–158`. The old text carried an explicit cross-reference ("the scratch-build
escape hatch does not apply either, because there is no `builtAt` to anchor a `--since` on"); the
rewrite dropped it while the ordering hazard remained.

**Fix:** gate bullet 2 — *"One row, `provenance` null **and `builtAt` parses as a timestamp**"* —
and/or restore the one-line cross-reference in the hand-written bullet.

### m4 — minor · abbreviated OIDs make check 0 a length-sensitive string comparison

Both producer (`git-provenance.sh:73–78`) and consumer (`freshness.md:67`) use `git rev-parse
--short`, whose width is `core.abbrev=auto` — derived from the repository's object count at the time
it runs. A stamp taken at 7 chars, re-checked months later at 8, fails equality on width alone, and
check 0's contract is a bare "Equal / Different". Fail direction is safe (false "moved"), but it
erodes the one yes/no answer the fix was built to give.

**Fix:** store full 40-char OIDs (drop `--short` at `:73`/`:75`/`:77`; keep the short form only for
the human-readable log lines at `pipeline.sh:110–111`, `:201`), and have `freshness.md` compare full
OIDs. Cheap and permanent. If short values are kept for readability, say explicitly in check 0 that
one being a prefix of the other counts as equal.

### m5 — minor · `SOURCE_TREE` is a blob, not a tree, when the source is a single file

`pipeline.sh` accepts a file as `<source>`, and `git-provenance.sh:58` handles it. For
`src/a.py` the capture produced `origin=[src/a.py] tree=[bafc5d9]` — that is the **blob** id.
Harmless for the check (the consumer's `HEAD:<sourceOrigin>` resolves the same blob), but
`git-provenance.sh:35`, `freshness.md:31` and `SKILL.md` all say "tree object", which will read as a
bug to the next person who inspects one.

**Fix:** doc-only — "the tree (or blob, for a single-file source) object of `sourceOrigin`".

### n1 — nit · `_cpg_str` leaks into the caller's global namespace

`git-provenance.sh:106–112` defines `_cpg_str` **inside** `cpg_provenance_stamp`; bash has no
function-scoped functions, so it persists after the call (verified: `type -t _cpg_str` → `function`).
Harmless today given the `_cpg_` prefix and that the file is sourced by one script.
**Fix:** define it at file scope alongside the two public functions, or `unset -f _cpg_str` before
returning.

### n2 — nit · `--source-origin ""` is silently ignored

`pipeline.sh:91` gates on `[ -n "$SOURCE_ORIGIN_ARG" ]`, so an empty value falls through to the
parse-root branch instead of hitting the fail-fast at `:95–98`. **Fix:** track a separate
`SOURCE_ORIGIN_SET=1` in the arg loop and branch on that.

### n3 — nit · `$name` is passed to git as a glob pathspec

`git-provenance.sh:64` and `:80` pass `"$name"` after `--`, which git still interprets as a wildmatch
pathspec: a basename beginning with `:` is pathspec magic, and `[`…`]` is a character class. Git's
exact-path match rescues the common cases (a directory literally named `glob[1]` captured correctly
in testing), but the class of bug is removable for free.
**Fix:** `git -C "$dir" --literal-pathspecs ls-files -- "$name"` (and the same on `status`), or
`-- ":(literal)$name"`.

### n4 — nit · `teco`'s brief template still quotes `builtAt`

`claude/teco/teco.md:102` prescribes `CPG: <graph>, built <builtAt>, fresh`. The recipe now says to
anchor freshness on `parsedAt` (`freshness.md:27`, `:78`). Cosmetic, but `teco` is the recipe's only
dispatch-time consumer. **Fix:** `built <builtAt>, source snapshot <parsedAt>` — routes to `cobb`.

---

## What's solid

- **The diagnosis and the ordering fix are correct**, and the tracked-ness gate does exactly what it
  claims: a gitignored staged copy inside a work tree returns `rc=1` with all four variables cleared,
  rather than inheriting the containing repo's `HEAD` (reproduced, Appendix A4).
- **Pathspec scoping works.** With a modification outside the source, `capture src` → `dirty=false`
  while `capture other` → `dirty=true`. Untracked files count; gitignored files correctly do not.
- **The Cypher escaping is sufficient** for everything reachable. A source named `we"ird\dir`
  rendered as `"we\"ird\\dir"` — backslash-then-quote order is right, and `printf '%s'` keeps the
  format string safe. The only unescaped hazard is a literal newline in `<source>`/`--source-origin`,
  which a caller would have to type deliberately.
- **No `set -e`/`set -u` traps.** Both functions are called only in condition position (`:92`,
  `:100`), so `set -e` is correctly suspended for their bodies; every optional expansion uses `:-`;
  the sourced file executes no top-level code and changes no shell option. Variables are `local`
  except the four intentional `CPG_SOURCE_*` globals, which are reset at `:53` before any early
  return, so a failed capture cannot leave a previous call's values standing.
- **The `head -1`/`pipefail` comment at `:61–63` is correct reasoning**, not folklore.
- **Fail-fast on a bad `--source-origin`** (`pipeline.sh:95–98`) is the right call — exit 2 before the
  parse rather than after three hours — and the `PROVENANCE=none` warning names the exact remedy.
- `skills/README.md`'s two rows are accurate: `references/` does hold five recipes (verified).

## Recommended verification before the next real build

The entire "never run end to end" surface is testable in **seconds**, without Joern: source
`git-provenance.sh`, render a stamp, fire it at a throwaway graph key, read it back, and
`GRAPH.DELETE` the throwaway. That exercises argument passing, FalkorDB's acceptance of the
multi-line query, the `SET prop = NULL` removal semantics, and — once **B1** is fixed — the new error
path. I did not run it myself, since it writes to the FalkorDB instance and the brief scopes me to
read-only. Worth handing to `qa-engineer` (or doing inline with the B1 fix) as the gate on U38.

## Open questions

1. **Backfill of `cpg_falkorchat`?** Its marker was hand-corrected but still lacks `PARSED_AT`,
   `PROVENANCE`, `SOURCE_ORIGIN` and `SOURCE_TREE` — the very fields the new recipe leans on. Now
   that the truth is known (`b795f4c`, `falkor-chat/server`, tree `85ddeed`, dirty `false`, and an
   approximate parse start), a hand-written top-up would make check 0 available today and remove the
   need for M3's guidance in practice. That is a `graph-dba` write, and a policy call — the current
   text says "no backfill was done" (`freshness.md:41`). Your call whether it stays that way.
2. **Should `--source-origin` be sanity-checked against the parse root?** Nothing verifies the named
   directory has anything to do with what Joern parsed; a typo yields authoritative-looking wrong
   provenance. A cheap heuristic (compare tracked-file basenames against the copy) would catch gross
   errors but adds a false-failure surface. I lean *no* — document-and-trust, as now — but it is the
   one remaining way to get a confidently wrong marker.

---

## Appendix

### A1 — `redis-cli` exits 0 on an error reply, and prints it to stdout

```
$ redis-cli -h localhost -p 6379 NOTACOMMAND; echo "exit=$?"
ERR unknown command 'NOTACOMMAND'
exit=0
$ out=$(redis-cli … NOTACOMMAND 2>/dev/null); echo "stdout=[$out]"
stdout=[ERR unknown command 'NOTACOMMAND']
$ err=$(redis-cli … NOTACOMMAND 2>&1 >/dev/null); echo "stderr=[$err]"
stderr=[]
```

Reproduced through the module too, read-only, against the live graph:

```
$ redis-cli … GRAPH.RO_QUERY cpg_falkorchat 'MERGE (b:CpgBuildInfo) SET b.X = ' >/dev/null
$ echo $?
0                                     # syntax error, output discarded, exit 0
$ redis-cli … GRAPH.RO_QUERY cpg_falkorchat 'MERGE (b:CpgBuildInfo) SET b.X = 1'
graph.RO_QUERY is to be executed only on read-only queries
$ echo $?
0                                     # rejected write, exit 0
```

`pipeline.sh:199` is the first form: `>/dev/null` + exit 0 + `set -e` sees success.

### A2 — argv integrity of the stamp invocation (fake `redis-cli` on `PATH`)

```
ARGC=7
ARG[1] (2 bytes)   <<-h>>
ARG[2] (9 bytes)   <<localhost>>
ARG[3] (2 bytes)   <<-p>>
ARG[4] (4 bytes)   <<6379>>
ARG[5] (11 bytes)  <<GRAPH.QUERY>>
ARG[6] (8 bytes)   <<cpg_test>>
ARG[7] (301 bytes) <<MERGE (b:CpgBuildInfo)
SET b.BUILT_AT = "2026-09-07T23:00:00Z",
    b.PARSED_AT = "2026-09-07T20:00:00Z",
    b.SOURCE_PATH = "/tmp/staged copy",
    b.PROVENANCE = "source-origin",
    b.SOURCE_ORIGIN = "src",
    b.SOURCE_COMMIT = "c61947b",
    b.SOURCE_TREE = "ef02eb2",
    b.SOURCE_DIRTY = false>>
```

Newlines, embedded double quotes and a space in `SOURCE_PATH` all survive as one argument.

### A3 — FalkorDB accepts a multi-line query over `redis-cli` argv (read-only)

```
$ Q='MATCH (b:CpgBuildInfo)
RETURN b.BUILT_AT AS builtAt,
       b.SOURCE_COMMIT AS c'
$ redis-cli … GRAPH.RO_QUERY cpg_falkorchat "$Q"
builtAt / c
2026-09-07T22:25:45Z
b795f4c
```

### A4 — `cpg_provenance_capture` across source shapes (scratch repo, `staged/` gitignored)

```
.                  rc=0 origin=[.]        commit=[c601c77] tree=[1d500d6] dirty=[false]
src                rc=0 origin=[src]      commit=[c601c77] tree=[ef02eb2] dirty=[false]
src/               rc=0 origin=[src]      commit=[c601c77] tree=[ef02eb2] dirty=[false]
src/a.py           rc=0 origin=[src/a.py] commit=[c601c77] tree=[bafc5d9] dirty=[false]   # blob (m5)
staged             rc=1 (no provenance)   origin=[] commit=[]                             # the gate
./staged           rc=1 (no provenance)   origin=[] commit=[]
nonexistent        rc=1 (no provenance)   origin=[] commit=[]
""                 rc=1 (no provenance)   origin=[] commit=[]
we"ird\dir         rc=0 origin=[we"ird\dir] tree=[2e3a786] dirty=[false]                   # escaping
glob[1]            rc=0 origin=[glob[1]]   tree=[1f4458c] dirty=[false]                    # n3
```

Dirty scoping, with `other/z.txt` modified:

```
src                rc=0 origin=[src]   dirty=[false]     # scoped: ignores other/
other              rc=0 origin=[other] dirty=[true]
```

Dirty inputs (M2, m2):

```
untracked file under src            dirty=[true]
gitignored file under src           dirty=[false]
workdir created under src           dirty=[true]         # M2
workdir under repo root, source=.   dirty=[true]         # M2 — `?? joern-work/`
added-not-committed dir             rc=0 commit=[c61947b] tree=[] dirty=[true]   # m2
  └─ check 2 against it: `git log --oneline <commit>..HEAD -- brandnew` → 0 commits
```

### A5 — live markers read (read-only)

```
cpg_falkorchat:
(:CpgBuildInfo{BUILT_AT:"2026-09-07T22:25:45Z", SOURCE_COMMIT:"b795f4c",
               SOURCE_DIRTY:False,
               SOURCE_PATH:"/home/mauricio/prg/graphmind-ai-lab/cpg/.cpg-artifacts/src/falkor-chat-server"})
  → no PROVENANCE / PARSED_AT / SOURCE_ORIGIN / SOURCE_TREE  (M3)

cpg_deprecated_salesperson:
(:CpgBuildInfo{BUILT_AT:"unknown", MARKER_ORIGIN:"hand-written by graph-dba, NOT a pipeline stamp",
               MARKER_WRITTEN_AT:"2026-09-02T14:54:12Z", NOTE:"…"})
  → no PROVENANCE either  (m3)
```

### A6 — old-code baseline for the M3 derivation claims

`git show 6012ddb^:skills/joern-cpg/scripts/pipeline.sh`, tail:

```bash
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="MERGE (b:CpgBuildInfo) SET b.BUILT_AT = \"$BUILT_AT\", b.SOURCE_PATH = \"$SRC\""
if command -v git >/dev/null 2>&1 && git -C "$SRC" rev-parse --short HEAD >/dev/null 2>&1; then
  SHA="$(git -C "$SRC" rev-parse --short HEAD)"
  DIRTY=false
  [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] && DIRTY=true
  STAMP="$STAMP, b.SOURCE_COMMIT = \"$SHA\", b.SOURCE_DIRTY = $DIRTY"
fi
```

Confirms `freshness.md:136–138`: derived after the load, from the parse root's containing repo, with
no pathspec.
