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

## Pass 2 — 2026-09-07 · gating M4 (`docs/manuals/graph-ontology.md`, commit `c92f35d`)

**Scope:** the manual only. Field names and meanings re-derived from `git-provenance.sh:113-121`
and `pipeline.sh:89-102,197-198` (not from the Pass 1 summary), both live markers re-read with
`keys(b)` via `mcp__cypher__query`, and the `git log`/`git rev-parse` claims run. No other artifact
was re-reviewed; Pass 1's blocker and majors M1-M3 and minors m1-m5 stand except where noted.

**Verdict: approve with suggestions.** The fix is real and went further than the finding asked.
One item (**P2-1**) must land with `cobb`'s in-flight round rather than after it.

**M4 — fixed.** All eight producer fields are now listed at `:121`, with names and meanings matching
the producer exactly (`BUILT_AT`, `PARSED_AT`, `SOURCE_PATH`, `SOURCE_ORIGIN`, `SOURCE_COMMIT`,
`SOURCE_TREE`, `SOURCE_DIRTY`, `PROVENANCE`); the `PROVENANCE` values are the producer's three
literals; `PARSED_AT` vs `BUILT_AT`, the scoping of `SOURCE_DIRTY`, and the "`SOURCE_PATH` is a parse
root, not an identity" correction are all accurate. The fourth spot tico found (the Overview
blockquote, now `:57-61`) was a real miss on my part — I cited three line numbers and it swept the
file, which is the right instinct.

### Answers to the four questions asked

1. **Field accuracy** — clean. Every name and gloss checks out against the producer. One nuance,
   not a finding: `SOURCE_ORIGIN` is a *file* path when the parse root is a single file (Pass 1 m5);
   at manual altitude "directory" is the right word.
2. **The pre-fix-marker claim** — **confirmed independently.** `keys(b)` on the live
   `cpg_falkorchat` returns exactly `['BUILT_AT','SOURCE_PATH','SOURCE_COMMIT','SOURCE_DIRTY']`, no
   `PROVENANCE` (Appendix A7). tico's claim is precisely right.
3. **The m1 dodge** — **it works, and the manual's phrasing is the better one.** See below.
4. **The absent cases** — the four taught are the right four and none is wrong. Two gaps: **P2-2**
   and **P2-3**.

### On question 3 — adopt the manual's wording in `freshness.md`, not the reverse

"Equal means **the source is unchanged since it was captured**" is a claim about *change over time*,
and it is true under both `parse-root` and `source-origin`: `SOURCE_TREE` is the tree of
`SOURCE_ORIGIN` at capture, so equality today means that directory has not moved. `freshness.md:70`'s
"byte-identical to what was parsed" is a claim about *identity of content*, and that is the one that
breaks under `source-origin`, where the parsed thing was a pruned copy. tico's version dodges m1
cleanly because it never asserts the copy and the origin are the same bytes.

The trailing clause — "so the graph is current however old the build is" — does re-cross into
inference, but the inference is sound for a *freshness* question: if the origin has not moved,
re-staging it yields the same copy, so the graph is as current as it was at build time. (What it
cannot tell you is whether the copy was faithful *when staged* — that is a build defect, not
staleness, and `SKILL.md`'s `diff -rq` note is its right home.) The `SOURCE_DIRTY` parenthetical
mirrors `freshness.md:73-77` correctly.

**Recommendation for `cobb`'s round:** replace `freshness.md:70-72`'s "byte-identical to what was
parsed" with the manual's formulation, rather than editing the manual toward the reference. That
resolves m1 with no per-provenance split at all.

### P2-1 — major · the manual copied the check-0 command, including its bug and its pending change

`docs/manuals/graph-ontology.md:492` (question-to-field table, *Is it the revision I mean?*) restates
`git rev-parse --short HEAD:<SOURCE_ORIGIN>` verbatim. That inherits **M1** — fatal for
`SOURCE_ORIGIN = "."` (`fatal: Needed a single revision`, exit 128; `HEAD:./` is the working form) —
and **m4**, the `--short` abbreviation-width drift, into a second document. Both are open against
`freshness.md` in the round `cobb` is running now, so the manual will be silently wrong the moment
that round lands unless it moves in the same change.

This is also the duplication most likely to drift: the file otherwise cites the reference for
procedure, and this one row is the exception.

**Fix (routes to `tico`, sequenced with `cobb`):** either drop the command and let the row read
*"compare it with the tree `git` reports for `SOURCE_ORIGIN` at `HEAD` — the exact command is in
`skills/cpg-analysis/references/freshness.md`"*, or keep it and update both files in the same commit
as M1/m4. I lean to the citation: the manual already defers procedure, and this row is procedure.

### P2-2 — minor · the "no `PROVENANCE`" bullet mis-describes the repo's *other* live marker

`:512-517` says a marker with no `PROVENANCE` is a pre-2026-09-07 stamp "carrying only the four
original fields", and tells the reader to treat its `SOURCE_COMMIT`/`SOURCE_DIRTY` as approximate.
`cpg_deprecated_salesperson` — one of only two `cpg_*` graphs loaded — also has no `PROVENANCE`, and
its `keys(b)` is `['BUILT_AT','SOURCE_PATH','STATUS','MARKER_ORIGIN','MARKER_WRITTEN_AT',
'RENAMED_FROM','NOTE']`: no `SOURCE_COMMIT`, no `SOURCE_DIRTY` (A7). A reader working down the four
bullets matches it here and hunts for two fields that do not exist before reaching the
`BUILT_AT: unknown` bullet at `:522` that actually describes it.

This is Pass 1 **m3** reproduced in a second document — with the sting drawn, since the manual never
teaches the `--since=unknown` trap and defers instead. **Fix:** gate the bullet — "No `PROVENANCE`
property **and a real timestamp in `BUILT_AT`**" — and land it consistently with m3's fix in
`freshness.md`.

### P2-3 — minor · the FAQ header promises a case none of its bullets covers

`:509-510` heads the section *"The marker is missing `PROVENANCE`, or **`SOURCE_TREE`**, or isn't
there at all"*. No bullet covers a marker that **has** `PROVENANCE` but no `SOURCE_TREE` — Pass 1's
**m2**, reproduced in testing: a source tracked in the index but not yet in `HEAD` stamps
`PROVENANCE=parse-root, SOURCE_COMMIT=<sha>, SOURCE_TREE=NULL, SOURCE_DIRTY=true`. The `'none'`
bullet describes all three fields absent together, which is a different state.

Low reader impact (no such graph is loaded, and the `PROVENANCE` table row still steers correctly),
but the header is the manual's own promise. **Fix:** drop "or `SOURCE_TREE`" from the header, or add
a clause to the `'none'` bullet: *"a tree can also be absent on its own, when the source was not
committed at capture — `SOURCE_DIRTY` will be true; fall back to `PARSED_AT` age."*

### P2-4 — minor · the Overview's "settles it in one query" is false on the only CPG it can be run against

`:57-61` now tells a reader to read `SOURCE_ORIGIN` and `SOURCE_TREE` to confirm a graph covers the
code they mean. On `cpg_falkorchat` both are **null** (A7) — and that is the graph nearly every
reader of this manual will open. The FAQ handles it 430 lines later; the Overview presents it as
settled. **Fix:** half a clause — *"…before you analyse the wrong codebase. (Both are absent on a
marker stamped before 2026-09-07, such as `cpg_falkorchat`'s — the FAQ says what to read then.)"*

### P2-5 — minor · the reason given for preferring `SOURCE_TREE` is a pre-fix fact stated as a general one

`:504-506`: *"`SOURCE_COMMIT` on its own is weaker than `SOURCE_TREE`. A commit can be recorded for a
tree that was never parsed."* True of a pre-2026-09-07 marker — that is the defect `6012ddb` fixed.
On a **current** marker both values come from the same pre-parse capture of the same `HEAD`
(`git-provenance.sh:73-78`), so the commit is not the less trustworthy of the two. The tree is
stronger for a different reason: it answers "is this still the same content?" by identity, instead of
by counting commits that may have touched the path and reverted.

As written a reader may distrust a current `SOURCE_COMMIT` for the wrong reason. **Fix:** *"a commit
tells you where `HEAD` was; the tree tells you what the content was, so comparing trees answers by
identity rather than by counting commits — and on a pre-2026-09-07 marker the commit may name a tree
that was never parsed at all."*

### Solid in the manual

- **Altitude is right.** It teaches which field answers which question and hands the procedure to
  `skills/cpg-analysis/references/freshness.md` in two places (`:539-541`, `:562-563`) instead of
  restating the escalating checks. P2-1 is the single exception, which is why it is the one to fix.
- **The `SOURCE_PATH` correction is verified true**, including the detail that makes it dangerous:
  `git log --oneline -- cpg/.cpg-artifacts/src/falkor-chat-server` returns **zero commits, exit 0,
  no warning** — in both the relative and the absolute form the marker actually stores (A7).
- The absent cases are framed as legitimate states rather than faults, which is the correct reading
  and the one Pass 1 endorsed as the author's deliberate design call.
- The "all eight rewritten on every stamp, an absent one removed" guarantee at `:121` matches the
  producer's intent. Note it is falsified in the failure case by Pass 1's **blocker B1** — a stamp
  that fails silently leaves the whole previous marker standing. Inherited, not the manual's defect;
  no manual change needed if B1 is fixed.

---

## Pass 3 — 2026-09-08 · closing the fix round (`9124a1f` scripts+reference, `8779ee8` manual)

**Scope:** both fix commits, read directly (`git show`) — all four files confirmed unchanged between
their fix commit and current `HEAD`, so the working tree is what I reviewed. Re-ran the Pass 1 shape
matrix against the new helper in throwaway repos, exercised the new stamp gate and read-back through
a fake `redis-cli` across six failure scenarios, probed the live FalkorDB's three error-reply shapes
read-only, and re-read both documents end to end for the cross-document check. I did **not** re-run
`cobb`'s two suites — the coordinator did (31/31, 40/40) — and did not run the pipeline.

**Verdicts.** `skills/cpg-analysis/references/freshness.md`: **approve.**
`docs/manuals/graph-ontology.md`: **approve with suggestions** (P3-2).
`git-provenance.sh` / `pipeline.sh`: **approve with suggestions** (P3-1).

**Dispositions.** All verified against the code/docs, not the report. **B1** fixed (six-scenario
harness, A8) · **M1** fixed after a second-generation defect, see below · **M2** fixed (`mkdir` moved
below the capture block, plus an unasked-for NOTE when a *surviving* workdir is what made the source
dirty — the case reordering cannot fix) · **M3** fixed (`freshness.md:177-185`; its stated keys for
`cpg_falkorchat` match the live node exactly) · **m1** fixed, adopting the manual's wording and
splitting build fidelity out to `SKILL.md` · **m2** fixed both sides (`:31`, `:115-120`; producer now
emits a genuine `NULL`, verified) · **m3** fixed (`:37-42` gated on a real timestamp) · **m4** fixed
(full 40-char OIDs, verified `len=40` on every shape) · **m5** fixed · **n1**, **n2**, **n3** fixed
(`:(literal).` verified to match, 733 files) · **P2-1 … P2-5** fixed in the manual; no copied
procedure remains. **n4 declined** — correct call, `claude/teco/teco.md` is outside `cobb`'s write
scope; no objection to routing it separately.

### Question 2 — is the corrected M1 form right? Yes, and I satisfied myself independently

The new form is `top="$(git -C "$dir" rev-parse --show-toplevel)"` then
`git -C "$top" rev-parse --verify --quiet "HEAD:./$CPG_SOURCE_ORIGIN"`. Both halves are load-bearing
and I confirmed each in isolation:

- `--show-toplevel` is what makes the CWD-relative `./` safe. I re-ran the Pass 1 matrix from three
  different working directories — repo root, a subdirectory of the repo, and `/tmp` (outside it
  entirely) — and every shape resolved to the same correct OID (A8). The old form's failure was
  invisible precisely because it depended on CWD; this one does not.
- `--verify --quiet` is what stops the echo-back. Confirmed both directions:
  `git rev-parse 'HEAD:./nope'` captures the literal string `HEAD:./nope`, while
  `git rev-parse --verify --quiet 'HEAD:./nope'` captures the empty string. That is the third
  generation of this defect class closed at the source rather than filtered downstream.
- **`HEAD:./.` resolves** at the repo root and equals `HEAD^{tree}` — I tested this specifically,
  since a fourth generation would have been the `.` case silently returning empty instead of wrong.
- **The m2 null case is now genuinely null**: a source tracked in the index but absent from `HEAD`
  gives `tree=[]` → `b.SOURCE_TREE = NULL` in the rendered stamp. `cobb`'s account of why that case
  "never produced a NULL" is accurate and scoped to the interim M1 fix, not to the original.

Eleven shapes, all correct, all full 40-char OIDs, each cross-checked against what a consumer running
the documented command would compute (A8). The consumer's form differs by one flag — `--verify`
without `--quiet` — which is right: a human wants the error message, and I verified it still prints
nothing to stdout, so the echo-back trap does not reach the recipe's reader either.

### Question 3 — B1's read-back discriminates; the pattern gate is diagnostics, not detection

Six scenarios through a fake `redis-cli` (A8). The two that matter: a stamp that "succeeds" while a
**stale marker survives** — the exact `--append` failure B1 named — is caught by the read-back, and
so is an **error shape no pattern covers**. That is the proof the read-back is load-bearing.

`PARSED_AT` is an adequate discriminator. It is set once (`pipeline.sh:96`) and never re-derived, and
it cannot be empty: `set -e` aborts the run if `date` fails, so the `*"$PARSED_AT"*` match can never
degrade to a vacuous `**`. Its granularity is one second, so the theoretical collision is *two builds
of the same graph starting in the same UTC second* — and a collision only matters if the write also
failed. Adequate; no change wanted.

**The pattern gate adds no detection power** — scenario E shows the read-back catches what the
patterns miss, and scenario F's connection failure is caught by `rq`'s nonzero-exit branch, not by
the `case`. It earns its place on **diagnostics**: it turns "the stamp did not land" into the actual
rejection text at the point of failure, which for the likeliest failure is
`errMsg: Invalid input at end of input: … column: 33`. Keep it, and the code comment already
characterizes it exactly right ("only the first gate; the READ-BACK below is the load-bearing one").

`cobb` is right and I was wrong: my Pass 1 `grep -qiE '^\(?error|^ERR |wrong number|read only'`
**misses** a FalkorDB Cypher rejection. I ran both patterns against the live reply — mine misses,
`cobb`'s matches (A8).

### Question 1 — the cross-document check: the two documents describe the same set of shapes, with one gap

I enumerated every marker shape each document teaches and matched them pairwise:

| Shape | `freshness.md` | `graph-ontology.md` |
|---|---|---|
| `provenance` present, full stamp | bullet 1 + field table | question-to-field table |
| `provenance: 'none'` | bullet 1 + Limits `:149-164` | FAQ bullet 1 |
| `provenance` present, `sourceTree` absent | table `:31` + check 2 `:115-120` | FAQ bullet 3 |
| `provenance` null, real `builtAt` (pre-fix) | bullet 2 `:37-42` + Limits `:165-185` | FAQ bullet 2 |
| zero rows | bullet 3 `:43-47` + Limits `:203-205` | FAQ bullet 4 |
| hand-written, `builtAt: unknown` | bullet 4 `:48-60` + Limits `:186-202` | FAQ bullet 5 |
| `sourceDirty = true` | check 0 gate + check 2 caveat | `SOURCE_DIRTY` row + `SOURCE_TREE` caveat |

**Six shapes each, no omission in either direction, and no contradiction** — including the two that
were the point of the exercise: the `sourceTree`-absent shape (m2/P2-3) is now taught by both, and
the pre-fix/hand-written disambiguation (m3/P2-2) is gated on a real `BUILT_AT` timestamp in both,
in the same direction. The one shape where the two could have drifted apart on *substance* rather
than wording — "is `SOURCE_DIRTY` will be true" for the tree-absent case — is correct in both, and I
verified it holds by construction: a path in the index but not in `HEAD` always shows as added in
`git status`, so that shape cannot occur with a clean flag.

Phrasing differences are all within the altitude split and none of them changes a reader's decision:
the reference gives the command and says "skip check 0" when dirty, the manual gives the decision and
says equality "only covers the committed part". That is the intended division, not drift.

**The one gap — P3-2 below.** The reference gained a *seventh* shape the manual does not carry.

### P3-1 — minor · the failure path tells the operator to re-stamp by hand but withholds the stamp

`SKILL.md` (Provenance, new bullet) correctly says the fix for a rejected stamp is "re-stamping by
hand … not re-parsing". But neither failure branch in `pipeline.sh` prints `$STAMP`, and every OID
the pipeline logs is truncated to 12 chars (`:117-119`, and the final line is never reached on
failure). The operator can recover — a 12-char tree OID expands via `git rev-parse` (verified) — but
must then hand-write the Cypher, and **a hand re-stamp that writes only the fields it has silently
breaks the "absent one is removed, not left stale" guarantee** that both documents now rest on: the
`= NULL` assignments are exactly what a human reconstructing the query would drop.

**Fix:** on both failure branches, emit the query verbatim so it can be replayed —
`printf 'pipeline: replay this stamp verbatim:\n%s\n' "$STAMP" >&2`. One line, and it removes the
only step of the recovery path that can silently reintroduce a stale field.

### P3-2 — minor · the reference's seventh shape, and an unconditional guarantee in the manual

`freshness.md:133-142` now teaches a shape the manual does not: **a marker that survived a silently
failed stamp** on the pre-2026-09-07 pipeline, whose tell is "a marker whose `builtAt` predates
content you can see in the graph". Meanwhile the manual's `§1` cell states the guarantee
unconditionally — *"all eight rewritten on every stamp (an absent one is removed, not left stale)"* —
while the reference now scopes it (*"since 2026-09-07 the pipeline reads the marker back… so a marker
you find is one that actually landed"*). The manual says elsewhere that pre-2026-09-07 markers exist
and that `cpg_falkorchat` is one, so as written it asserts a guarantee about markers that predate the
mechanism providing it.

My read: the shape itself is a historical diagnostic, not a reader-facing state, and omitting it from
a manual that defers procedure is defensible — **the fix is the overclaim, not the missing shape.**

**Fix (routes to `tico`):** one clause in the `§1` cell — *"…an absent one is removed, not left
stale (guaranteed for stamps from 2026-09-07 on, when the pipeline began verifying its own write)"*.

### P3-3 — minor · the pending `cpg_falkorchat` backfill touches four statements, not one line

The coordinator asked whether the document depends on "no backfill was done" beyond that line. It
does, and so does the manual. Dispatching the backfill invalidates:

1. `freshness.md:43-47` — the "no backfill was done" clause itself.
2. `freshness.md:182-185` — *"`cpg_falkorchat` carries exactly this marker — keys `BUILT_AT`,
   `SOURCE_PATH`, `SOURCE_COMMIT`, `SOURCE_DIRTY` and nothing else"*. Becomes false, and it is the
   pre-fix shape's only live example.
3. `graph-ontology.md` FAQ bullet 2 — *"The live `cpg_falkorchat` is one of these today, so you will
   meet one."*
4. `graph-ontology.md` Overview — *"Both are absent on a marker stamped before 2026-09-07 —
   including `cpg_falkorchat`'s, the graph most readers of this manual open."*

After the backfill **no loaded graph exemplifies the pre-fix shape**, so both documents would teach it
with a dead example. Worth folding into the backfill unit's done-condition.

**And a design question for that unit, not a defect here:** a hand-backfilled marker carrying
`PROVENANCE` would be an eighth shape — pipeline-shaped fields with a hand-supplied origin — and
`PROVENANCE: source-origin` would assert a capture that never happened, which is the plausible-but-
wrong failure this whole chain exists to prevent. `graph-dba` writing an explicit marker property
(the `MARKER_ORIGIN` convention `cpg_deprecated_salesperson` already uses) would keep it honest. I'd
want that decided before the write, not after.

### Nits

- **`rq`'s last two patterns are unanchored** (`*"read only"*`, `*"read-only"*`) and match anywhere
  in a reply, unlike the three anchored ones. Harmless for the two fixed queries it serves; a comment
  saying so would stop a future reuse from inheriting a false-failure mode.
- **`count()` was left as-is** and still swallows error replies. The direction is safe (an error
  yields an empty `PCOUNT`, which `--verify-prefix` already treats as failure); only the
  `nodes=/edges=` log line can silently print blank. Reusing `rq` would make it uniform.
- **`pipeline.sh` has no trailing newline** (`\ No newline at end of file`).
- The workdir NOTE's `case "$_wd/" in "$_sr"/*)` prints a spurious note if `realpath -m "$SRC"`
  returns empty (pattern degrades to `/*`). `realpath -m` effectively cannot fail on a syntactically
  valid path, so this is theoretical; a `[ -n "$_sr" ]` guard would close it.

### What's solid

- **Every fix was verified at the level the finding was made at**, and three went further than asked:
  the M2 fix added the surviving-workdir NOTE (the half reordering cannot solve), the m1 fix split
  build fidelity from staleness rather than just softening a sentence, and the B1 fix chose a
  read-back over the enumeration I suggested — which is the better design and which my own suggested
  pattern would have failed at.
- **The M1 regression was caught by its own regression test**, which is the outcome the test suite
  exists for; a fix round that surfaces a defect in the previous fix round is working as intended.
- **The manual's decision to cite rather than copy** (P2-1) has already paid: `9124a1f` changed the
  check-0 command materially — `--short` → full OIDs, `HEAD:<origin>` → `--verify "HEAD:./<origin>"`
  — and the manual needed no edit to stay correct. That is the drift this pass was held open to catch,
  caught structurally instead.

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

### A7 — Pass 2 verification (read-only)

Both loaded `cpg_*` markers, by `keys(b)`:

```
cpg_falkorchat:
  keys = ['BUILT_AT', 'SOURCE_PATH', 'SOURCE_COMMIT', 'SOURCE_DIRTY']
  builtAt=2026-09-07T22:25:45Z  commit=b795f4c  dirty=false
  parsedAt=null  provenance=null  origin=null  tree=null
  path=/home/mauricio/prg/graphmind-ai-lab/cpg/.cpg-artifacts/src/falkor-chat-server
  -> exactly the four fields the manual names.  (question 2: confirmed)

cpg_deprecated_salesperson:
  keys = ['BUILT_AT', 'SOURCE_PATH', 'STATUS', 'MARKER_ORIGIN',
          'MARKER_WRITTEN_AT', 'RENAMED_FROM', 'NOTE']
  builtAt=unknown  provenance=null  commit=null
  -> no PROVENANCE *and* no SOURCE_COMMIT/SOURCE_DIRTY.  (P2-2)
```

Loaded graph list at review time contains exactly two `cpg_*` keys (`cpg_falkorchat`,
`cpg_deprecated_salesperson`), which bounds "cases a reader will actually meet" to the pre-fix
marker and the hand-written one.

The `SOURCE_PATH` silent-zero claim, both forms:

```
$ git check-ignore -v cpg/.cpg-artifacts
cpg/.gitignore:3:.cpg-artifacts/	cpg/.cpg-artifacts
$ git log --oneline -- cpg/.cpg-artifacts/src/falkor-chat-server
$ echo "exit=$? commits=0"
exit=0 commits=0                                  # no output, no warning
$ git log --oneline -- /home/mauricio/prg/graphmind-ai-lab/cpg/.cpg-artifacts/src/falkor-chat-server
$ echo $?
0                                                 # same for the absolute form stored in SOURCE_PATH
$ git log --oneline -- /tmp/cpg-src/falkor-chat-server
fatal: Invalid path '/tmp/cpg-src': No such file or directory   # only an out-of-repo path errors
```

The check-0 command the manual copied (P2-1), re-confirmed from Pass 1:

```
$ git rev-parse --short HEAD:.        # SOURCE_ORIGIN == "." at a repo root
fatal: Needed a single revision                   # exit 128
$ git rev-parse --short HEAD:./
1e4fb8f                                           # == git rev-parse --short "HEAD^{tree}"
```

### A8 — Pass 3 verification

**Corrected tree resolution, 11 shapes × 3 working directories** (throwaway repos; every value
cross-checked against what the consumer's documented command computes):

```
repo root, cwd=root            origin=[.]          tree=8bc2ab58…799e0c len=40  MATCHES HEAD^{tree}
repo root, cwd=SUBDIR          origin=[.]          tree=8bc2ab58…799e0c len=40  MATCHES HEAD^{tree}
subdir rel, cwd=root           origin=[src]        tree=379c8bdb…f1d3f6 len=40  MATCHES HEAD:src
subdir abs, cwd=/tmp           origin=[src]        tree=379c8bdb…f1d3f6 len=40  MATCHES HEAD:src
nested subdir, cwd=root        origin=[src/deep]   tree=bd269627…17a26e len=40  MATCHES HEAD:src/deep
nested, cwd=INSIDE src         origin=[src/deep]   tree=bd269627…17a26e len=40  MATCHES HEAD:src/deep
single file                    origin=[src/a.py]   tree=bafc5d9a…cd997a len=40  MATCHES HEAD:src/a.py  (blob)
trailing slash                 origin=[src]        tree=379c8bdb…f1d3f6 len=40  MATCHES HEAD:src
quotes+backslash in name       origin=[we"ird\dir] tree=2e3a786d…da8fe2 len=40  MATCHES HEAD:we"ird\dir
gitignored staged copy         rc=1 (no provenance)
nonexistent                    rc=1 (no provenance)
added-not-committed (m2)       origin=[brandnew]   tree=[] dirty=true  ->  b.SOURCE_TREE = NULL
```

The two mechanics the fix rests on, isolated:

```
$ git rev-parse --verify --quiet 'HEAD:./.'      # the "." case, at the repo root
db7ac6c5fd07aa062c30fcdbe5392c74fc476b2c
$ git rev-parse 'HEAD^{tree}'
db7ac6c5fd07aa062c30fcdbe5392c74fc476b2c         # identical

$ out=$(git rev-parse --verify --quiet 'HEAD:./nope' || true); echo "[$out]"
[]                                               # echo-back suppressed
$ out=$(git rev-parse 'HEAD:./nope' 2>/dev/null || true); echo "[$out]"
[HEAD:./nope]                                    # the generation-2 defect, reproduced

$ out=$(git rev-parse --verify "HEAD:./no-such-path" 2>/dev/null || true); echo "[$out]"
[]                                               # the CONSUMER's documented form is safe too
                                                 # (stderr: "fatal: Needed a single revision")

$ git ls-files -- ':(literal).' | wc -l
733                                              # n3's literal pathspec still matches a directory
```

**Live FalkorDB error-reply shapes, and the two patterns against them** (read-only):

```
$ redis-cli … GRAPH.RO_QUERY cpg_falkorchat 'MATCH (b:CpgBuildInfo) SET b.X = '
errMsg: Invalid input at end of input: expected NOT, '+', … line: 1, column: 33, offset: 32
  errCtx: MATCH (b:CpgBuildInfo) SET b.X =  errCtxOffset: 32
  first12 = "errMsg: Inva"
  Pass-1 pattern /^\(?error|^ERR |wrong number|read only/i  -> *** MISSED ***
  cobb's case  errMsg:*|ERR\ *|WRONGTYPE*|*read only*|…     -> matched

$ redis-cli … GRAPH.RO_QUERY                     -> ERR wrong number of arguments for 'graph.RO_QUERY' command
$ redis-cli … GRAPH.RO_QUERY <g> '<a write>'     -> graph.RO_QUERY is to be executed only on read-only queries
                                                    (matched via the unanchored *"read-only"* arm)
```

**Stamp gate + read-back, six scenarios through a fake `redis-cli`** (the real `rq` body, verbatim):

```
A  stamp ok, read-back carries this run's PARSED_AT   -> PASS (read-back matched)
B  Cypher rejection "errMsg: …"                       -> caught by pattern gate  [errMsg: Invalid input…]
C  stamp no-ops, STALE marker survives (--append)     -> caught by read-back     [2026-09-01T08:00:00Z]
D  stamp "succeeds", graph has NO marker              -> caught by read-back     [b.PARSED_AT]
E  unenumerated error shape, stale marker survives    -> caught by read-back     <- read-back is load-bearing
F  redis-cli unreachable (nonzero exit)               -> caught by rq's exit branch
```

C and E are the two that settle the question: the read-back catches the exact `--append` failure B1
named, and catches it even when no pattern matches the reply.

**Recoverability of a truncated OID** (P3-1):

```
$ T=$(git rev-parse 'HEAD^{tree}'); git rev-parse "${T:0:12}"
8c2a1384af9dea0948d34f6144392a92a58e7cc1        # 12 chars expands, so recovery is possible…
```

…but the operator must then hand-write the Cypher, including the `= NULL` assignments — which is the
step P3-1 asks the pipeline to remove by printing `$STAMP` on the failure path.

**File drift check** — all four reviewed files are unchanged between their fix commit and current
`HEAD` (`git diff --quiet 9124a1f HEAD -- <path>`, and `8779ee8` for the manual), so this pass
reviewed the live text.
