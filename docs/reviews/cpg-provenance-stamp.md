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

## Pass 4 — 2026-09-08 · the pairwise cross-document gate (`81b43cd` reference, `b47c84a` manual)

**Scope:** the two documents as committed, read with `git show` (both confirmed identical to `HEAD`);
the live backfilled `cpg_falkorchat` marker and its `NOTE`; `cpg/.cpg-artifacts/MANIFEST.txt`; and
the stamp body in `git-provenance.sh` (unchanged since `9124a1f`). No suite run, no graph write, no
tree-mutating git. The working tree's `falkor-chat/server/` and `model-bench/` changes were ignored.

**Verdicts.** `skills/cpg-analysis/references/freshness.md`: **needs changes** — one major (**P4-1**),
a false statement about system behaviour in the direction that produces a misleading marker;
everything else in the sweep is right. `docs/manuals/graph-ontology.md`: **approve with suggestions**
(**P4-3**).

**Dispositions.** **P3-3 fixed** — all four backfill-dependent statements updated, and the design
question I flagged was answered exactly as recommended: the marker carries `PROVENANCE =
hand-backfilled` plus `MARKER_ORIGIN`, rather than a `source-origin` that would have asserted a
capture that never happened. **P3-2 not fixed** — the manual's `§1` cell still states "all rewritten
on every stamp (an absent one is removed, not left stale)" unconditionally; now entangled with P4-1,
so settle P4-1 first and let both files state the same true thing. **P3-1 and the Pass 3 nits not
fixed** — `git-provenance.sh`, `pipeline.sh` and `SKILL.md` are untouched since `9124a1f`; correctly
out of scope for two docs-only units, still open.

**Coordinator's facts, re-derived independently and all confirmed:**
`git rev-parse --verify "HEAD:./falkor-chat/server"` → `515ee7e9…`, marker `sourceTree` → `85ddeed0…`
(different — check 0 says *stale*); check 2 → exactly 3 commits (`e6fa20c`, `3fe3d8f`, `c708423`),
still 3 at today's `HEAD`; `MANIFEST.txt:19-21` records `diff -rq`, exit 0. Nothing to challenge.

### The central question — two partitions of one set, not an omission

I enumerated each document's **classification surface** — the list a reader actually walks to decide
what they are holding — and matched them:

| Marker state | `freshness.md` | `graph-ontology.md` |
|---|---|---|
| pipeline stamp, full | bullet 1 (gated on a pipeline `provenance` **and** null `markerOrigin`) | question-to-field table |
| … sub-state `provenance: 'none'` | `provenance` row + dedicated Limits bullet | FAQ bullet 1 |
| … sub-state `sourceTree` absent | `sourceTree` row (`:31`) + check 2's caveat | FAQ bullet 4 |
| pre-2026-09-07 stamp | bullet 2 | FAQ bullet 2 |
| hand-backfilled | bullet 5 | FAQ bullet 3 |
| zero rows | bullet 3 | FAQ bullet 5 |
| hand-written, `builtAt: unknown` | bullet 4 | FAQ bullet 6 |

**Five and six are the same set.** The whole difference is that `cobb` treats `none` and
`sourceTree`-absent as *sub-states of a pipeline stamp* — correctly, since both are things the
pipeline itself writes — while `tico` promotes them to top-level reader-facing states. Neither is
omitted anywhere: a reader of `freshness.md` alone can classify and act on a `none` marker (field
table plus a dedicated Limits bullet with the full workaround) and on a tree-absent one (`:31`,
check 0's gate, check 2's two-cases caveat); a reader of `graph-ontology.md` alone can classify a
full pipeline stamp (the question-to-field table, whose `PROVENANCE` row now enumerates all four
literals). I also walked both orderings against the two live markers: `cpg_falkorchat` lands on
hand-backfilled in both, `cpg_deprecated_salesperson` on `builtAt: unknown` in both, and no earlier
bullet in either list captures either of them first.

**What does differ is the gates, not the sets** — see P4-3. That is the honest residue of the two
authors not seeing each other's file, and it is one sentence wide.

### The three specific weights

**1. `cobb`'s per-marker check-0 gate — endorsed, and operable.** The reasoning is right and it is
this chain's own discipline applied one level up: `source-origin` can carry a blanket guarantee
because a machine produced it under fixed rules; `hand-backfilled` only asserts a human was present,
so the guarantee has to be re-earned per marker. It is operable by the recipe's actual consumer —
`teco`, an agent that can run `MATCH (b:CpgBuildInfo) RETURN b` and judge a note — and
`cpg_falkorchat`'s note does establish the derivation (I read it and checked its `MANIFEST.txt`
citation resolves). Two things worth saying: a **scripted** consumer's safe default exists but is
implicit (**P4-4**), and the gate's first real use produces *stale*, so nobody is currently trusting
a graph on hand-derived evidence — the admission's practical effect today is that check 0 runs and
says no.

**2. The `markerOrigin` re-gating — complete on `provenance`, one residual on a word.** I swept every
`provenance`-keyed statement in the file (23 sites). Bullets 2, 3 and 4 are keyed on null
`provenance`, zero rows and an unparseable `builtAt` respectively, so the fourth literal cannot reach
any of them; checks 0 and 1 were both updated; the Limits bullets are keyed on literals or on the
field's absence. Nothing is still keyed on `provenance` alone. The one residual is lexical, not
logical: **P4-2**.

**3. The dead-example claim — both files now point the same graph at the same shape.** Verified
against the live node (ten keys, `PROVENANCE = hand-backfilled`, `SOURCE_ORIGIN = falkor-chat/server`,
`SOURCE_TREE = 85ddeed0…`, no `PARSED_AT`): `freshness.md`'s pre-fix bullet says no loaded graph
carries that shape and explicitly redirects `cpg_falkorchat` to the fifth bullet; its fifth bullet
names it as the live hand-backfilled example. The manual says the same in both places, and its
Overview, `§1` cell and "how current" row all now describe the backfilled marker accurately. **Neither
file teaches `cpg_falkorchat` as an example of the old shape anywhere.** `cobb`'s choice to keep the
pre-fix shape documented without a live example is right — reloading a pre-fix export re-creates it.

### P4-1 — major · "the next `--load` overwrites it wholesale" is false without `--reset`

`freshness.md` (one-marker-per-graph limit, new text): *"**A hand-authored marker is subject to the
same rule**, and nothing exempts it: the next successful `--load` overwrites it wholesale, `NOTE` and
`MARKER_ORIGIN` included."*

The stamp is `MERGE (b:CpgBuildInfo) SET` **eight named properties** (`git-provenance.sh`,
`cpg_provenance_stamp`). Nothing in `pipeline.sh`, `git-provenance.sh` or `cpg-to-falkordb.py`
references `MARKER_ORIGIN`, `MARKER_WRITTEN_AT`, `NOTE`, `STATUS` or `RENAMED_FROM` (grepped). So the
claim holds only for a `--reset` load, which `GRAPH.DELETE`s the graph first — and `--reset` is
optional (`pipeline.sh:161` gates on it). On an **`--append`** load the `MERGE` matches the existing
hand-authored node, rewrites the eight, and **leaves the three hand-authored properties standing.**

That is this chain's defect class one level up again — a stale value that looks authoritative. The
resulting marker carries freshly captured `source*` fields *plus* a `MARKER_ORIGIN` saying
"hand-backfilled by graph-dba, NOT a pipeline stamp" and a `NOTE` describing a build that no longer
exists. Under `cobb`'s own new bullet 1 (which requires `markerOrigin` null) it is classified as *not*
a pipeline stamp, and check 0's per-marker gate then sends the reader to that stale note as evidence.
Traced statically, not executed — writing to a graph was out of scope.

**Fix — prefer (b).** (a) Scope the sentence: *"a `--reset` load deletes the node outright; an
`--append` load rewrites the eight pipeline fields and leaves `MARKER_ORIGIN`/`MARKER_WRITTEN_AT`/
`NOTE` standing, so a rebuilt graph can carry a stale hand-authored note — check `MARKER_WRITTEN_AT`
against `BUILT_AT`."* (b) Make the sentence true: add
`b.MARKER_ORIGIN = NULL, b.MARKER_WRITTEN_AT = NULL, b.NOTE = NULL, b.STATUS = NULL,
b.RENAMED_FROM = NULL` to the stamp. That is exactly the existing write-every-field-as-`NULL`
discipline extended to the hand-authored keys, it restores the "wholesale" guarantee both documents
want to rest on, and it is correct on the merits: a note describing the previous build has no
business surviving the rebuild. Routes to `cobb` (code + the sentence).

### P4-2 — minor · check 2 still says "hand-written" where the document now means one specific shape

`81b43cd` disambiguated the term everywhere else — the Limits bullet is now *"A hand-written marker —
the `builtAt = unknown` shape, not the hand-backfilled one"* — but check 2's closing line was not
updated: *"skip this check entirely for a hand-written marker."*

A reader holding `cpg_falkorchat` has just been told by bullet 5 that "checks 0 and 2 are both
available", and is looking at a marker whose own `MARKER_ORIGIN` reads *"hand-backfilled by graph-dba,
NOT a pipeline stamp"*. Reading "hand-written" in its plain sense, they skip the one check that
returns an actionable answer today (3 commits — the *only* check currently telling them the graph is
behind). **Fix:** *"skip this check entirely for a hand-written marker (the `builtAt = unknown`
shape — a hand-backfilled marker has a real `sourceCommit` and check 2 does apply)."*

### P4-3 — minor · the manual states `MARKER_ORIGIN` as a fact but never as a classification rule

This is the one place the two authors' gates diverge. `cobb` made `markerOrigin` a documented query
field and a **rule**: bullet 1 requires it null, and the field table says non-null means "a human
wrote this marker … read the whole node before acting on any other field" — global, not bullet-scoped.
`tico`'s manual carries the *fact* (the `§1` cell: "one written or repaired by hand adds
`MARKER_ORIGIN`, `MARKER_WRITTEN_AT` and a `NOTE`") but its FAQ decision list — the surface a reader
actually classifies on — keys entirely on the `PROVENANCE` literal and never mentions it.

So a hand-authored marker carrying a *pipeline* `provenance` literal reads as a pipeline stamp to a
manual-only reader. By convention that marker should not exist — which is why this is minor, not
major — but it is precisely the case `cobb` added the field to bullet 1 to catch, and P4-1 shows a
routine `--append` rebuild can manufacture something close to it. **Fix (routes to `tico`):** one
sentence at the head of the FAQ absent-cases list — *"whatever `PROVENANCE` says, a marker with a
`MARKER_ORIGIN` property was written or repaired by a human: read the whole node before trusting any
other field."*

### Nits

- **P4-4** — check 0's per-marker gate has no stated fallback for a consumer that cannot read and
  judge a note. The safe default does exist (implement the gate as the literal pair `parse-root` /
  `source-origin` and fall through to checks 1-2), and the `provenance` row already warns against a
  scripted `startswith`. One clause would make it explicit rather than inferable.
- **P4-5** — the live marker's own `NOTE` says check 0 is safe to run against it *"even though its
  literal gate names only `parse-root` and `source-origin`"*. `81b43cd` changed that gate, so the
  note now describes a superseded version of the document that admits it. Harmless today, but the
  note is load-bearing evidence under the per-marker gate, so it should not contradict the recipe.
  Routes to `graph-dba` with the next marker touch, not urgently.

### What's solid

- **Neither author's independent sweep missed a site the other's brief named**, and each found sites
  outside it: `cobb` found three (the hand-written bullet's blanket "has no date", the newly
  load-bearing one-marker limit, check 1's fallback), `tico` found two (the `PROVENANCE` row's
  three-literal enumeration — the most-read place a fourth literal would have looked like corruption
  — and the "how current" row pointing at a `PARSED_AT` that no longer exists on the graph readers
  open). Both are the sweep-don't-patch instinct that Pass 2 credited `tico` with.
- **`hand-backfilled` as a distinct literal, in `PROVENANCE` rather than hidden in a side field, is
  the right call** and both files give the same reason: a caveat parked in a field the read path
  skips would let a reader see `source-origin` and trust a capture that never happened.
- **The manual keeps deferring procedure** and gained no copied command in this round; the reference
  keeps owning the checks. Two rounds of upstream change have now landed without the manual needing
  a correction to stay true — the structural fix from Pass 2 continuing to pay.

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

### A9 — Pass 4 verification

**The live `cpg_falkorchat` marker after the backfill** (read-only, `keys(b)` + field read):

```
keys = ['BUILT_AT','SOURCE_PATH','SOURCE_COMMIT','SOURCE_DIRTY','PROVENANCE',
        'SOURCE_ORIGIN','SOURCE_TREE','MARKER_ORIGIN','MARKER_WRITTEN_AT','NOTE']   # ten
PROVENANCE    = hand-backfilled          SOURCE_ORIGIN = falkor-chat/server
SOURCE_TREE   = 85ddeed09479091a69b66d0301ed0d3399cc8387
SOURCE_COMMIT = b795f4c23e066278ba8582d8cdd213d03b73e9df     SOURCE_DIRTY = false
PARSED_AT     = null                     MARKER_ORIGIN = "hand-backfilled by graph-dba,
                                                          NOT a pipeline stamp"
```

Matches every claim both files make about it, including "`PARSED_AT` is the one it does *not* have".

**Coordinator's three facts, re-derived:**

```
$ git rev-parse --verify "HEAD:./falkor-chat/server"
515ee7e99cf884bd053ea6564cbf28fe1989375d      # vs sourceTree 85ddeed0… -> DIFFERENT (stale)
$ git log --oneline b795f4c23e…..HEAD -- falkor-chat/server
e6fa20c  3fe3d8f  c708423                     # exactly 3, still 3 at HEAD de0257d
$ sed -n '19,21p' cpg/.cpg-artifacts/MANIFEST.txt
  Source commit : b795f4c — falkor-chat/server tree 85ddeed, working tree CLEAN
                  under falkor-chat/server/ at staging time. The staged copy was
                  verified byte-identical to the committed tree (`diff -rq`, exit 0).
```

**P4-1 — what the stamp actually writes** (`git-provenance.sh`, `cpg_provenance_stamp`):

```
MERGE (b:CpgBuildInfo)
SET b.BUILT_AT = …, b.PARSED_AT = …, b.SOURCE_PATH = …, b.PROVENANCE = …,
    b.SOURCE_ORIGIN = …, b.SOURCE_COMMIT = …, b.SOURCE_TREE = …, b.SOURCE_DIRTY = …
```

Eight named properties. `grep -n 'MARKER_ORIGIN\|NOTE\|STATUS\|RENAMED_FROM'` over
`skills/joern-cpg/scripts/*.sh` and `*.py` returns only unrelated shell comments — nothing clears the
hand-authored keys. `--reset` is optional (`pipeline.sh:161` gates the `GRAPH.DELETE` on it), so the
`--append` path leaves `MARKER_ORIGIN`, `MARKER_WRITTEN_AT` and `NOTE` standing over a fresh stamp.
Traced statically; not executed, since verifying it live would mean writing to a graph.

**P4-2 — the un-updated line** (`freshness.md`, check 2, closing):

```
**Use `sourceOrigin`, not `sourcePath`** … **Both forms need a real `parsedAt`/`sourceCommit`**;
skip this check entirely for a hand-written marker.
```

while the Limits bullet in the same commit now reads *"A hand-written marker — the
`builtAt = unknown` shape, not the hand-backfilled one — …"*.

**Re-gating sweep** — 23 `provenance`/`markerOrigin`/`hand-*` sites in `freshness.md` read; bullets
2/3/4 are keyed on null `provenance`, zero rows and an unparseable `builtAt`, none of which the
fourth literal can satisfy; checks 0 and 1 updated; Limits bullets keyed on literals or absence.
No site still keyed on `provenance` alone.

**File drift** — `freshness.md` identical to `81b43cd`, `graph-ontology.md` identical to `b47c84a`,
and `git-provenance.sh`/`pipeline.sh`/`SKILL.md` identical to `9124a1f` (so P3-1 and the Pass 3
script nits are untouched, as expected for two docs-only units).

---

## Pass 5 — 2026-09-08 · gating the arc that closed the stamp's property set (`29538d6`, `0da3eb9`, `5417f0e`)

**Scope:** the three commits as an arc, read with `git show` against their parents — never the
working tree, which carries a second session's unrelated changes under `claude/`, `model-bench/`,
`skills/agent-standards/`, `skills/python-web-quirks/`. Files judged at `HEAD`:
`skills/joern-cpg/scripts/git-provenance.sh`, `skills/joern-cpg/scripts/pipeline.sh`,
`skills/joern-cpg/SKILL.md`, `skills/cpg-analysis/references/freshness.md`, `skills/README.md`,
`claude/cobb/kaizen/{plan,history}.md`. Plus the live `cpg_falkorchat` marker (read-only) and
`docs/plans/cpg-agent-adoption-graph.md` §1.1. Nothing under `falkor-chat/` was opened. No graph
write, no `GRAPH.DELETE`, no `GRAPH.QUERY` against a possibly-absent graph, no tree-mutating git,
nothing staged or committed. `pipeline.sh` was not run; its stamp block was extracted and executed
against a fake `redis-cli` that models a *correct* map replace (A10).

**Verdict: needs changes** — one blocker, three majors. The design decision is right and the map
form is the right mechanism; the wiring that carries it is broken.

CPG: considered, not relevant — the arc is Bash and Markdown, and the two loaded `cpg_*` graphs are
the *subject* of the marker data, not graphs of this code; no `cpg_skills` graph exists.

### P5-1 — blocker · the stray assertion's allow-list is always empty, so every `--load` build fails

`pipeline.sh:228` builds the stamp in a command substitution:
`STAMP="$(cpg_provenance_stamp …)"`. That subshell is the only place `CPG_STAMPED_KEYS` is ever
assigned (`git-provenance.sh:214,221` — the only setters in the repo). In the parent shell the
variable is **unset**, so `cpg_provenance_stray_query` renders `NOT k IN []` and every property on
the marker is a stray. `_cpg_prop`'s own comment (`git-provenance.sh:204-210`) names this exact
trap — *"CALL IT AS A STATEMENT, NEVER INSIDE `$(…)`"* — one level below the call site that
commits it, and dismisses the consequence as *"at least the safe direction"*. It is not a
direction; it is the shipped state.

Verified three ways (A10): the repro prints `CPG_STAMPED_KEYS=<UNSET>`; the empty-allow-list query
run read-only against `cpg_falkorchat` returns all 10 keys; and the stamp block executed verbatim
against a *correctly replaced* 8-key marker exits 1 listing all eight of its own fields as strays.
Impact: the marker has already been replaced when this fires, so a rebuild destroys the annotation
**and** reports `FAILED … THIS SHOULD BE IMPOSSIBLE`, sending the operator to hunt a FalkorDB
semantics change after a multi-hour parse. It also falsifies `freshness.md`'s "with no warning
either way" and every "fails the run only if the replace stopped working" sentence in the arc.

**Fix:** call the stamp as a statement and read the rendered Cypher from a variable —
`cpg_provenance_stamp … >/dev/null` then `printf` into `STAMP`, or have the function assign
`CPG_STAMP_CYPHER` instead of echoing. Then assert the wiring, not just the query: `[ -n
"${CPG_STAMPED_KEYS:-}" ] || { echo "pipeline: internal — allow-list empty"; exit 1; }`. An empty
allow-list is a bug in the pipeline, not a finding about the graph, and must not render as one.

### P5-2 — major · the stray check's three named triggers are one wrong and two conditional

`git-provenance.sh:244-256` and `pipeline.sh:283-296` (same wording, echoed at `SKILL.md:96-103`)
justify keeping the check with three triggers: a FalkorDB treating `=` as a merge, a reversion to
`b.X = …`/`+=`, and *"an edit that drops a property out of the map"*. The third cannot fire: a key
is added to the map and to `CPG_STAMPED_KEYS` by the same `_cpg_prop` call, so deleting the line
removes it from both and the node simply lacks it — zero rows. The first two fire **only over a
marker that already carries a foreign key**: on a graph whose previous marker is pipeline-clean,
`+=`, `b.X = …` and a merge-semantics `=` all leave exactly the eight stamped keys. So the claim
"each is caught here, on every build" is false, and with it the "standing regression test for the
property the whole design rests on" framing that the DO-NOT-DELETE argument rests on.

**Ruling on the check:** keep it — it is not dead code and not theatre — but it detects *the stamp
failed to erase a pre-existing foreign key*, not *the replace semantics changed*. Today that makes
it a real, firing check on exactly two graphs (`cpg_falkorchat`, `cpg_deprecated_salesperson`);
after each rebuilds once it can no longer fire under any named trigger. **Fix:** replace the
three-trigger list with that one sentence, and keep the DO-NOT-DELETE line — the honest reason is
stronger than the overclaimed one, because it is checkable.

### P5-3 — major · the sweep missed the fifth carrier: the live marker's own `NOTE`

`0da3eb9` "corrects a false universal in all three places that carried it"; `5417f0e` swept those
plus `skills/README.md`. A fifth copy sits in the graph. `cpg_falkorchat`'s `NOTE`
(`MARKER_WRITTEN_AT` `2026-09-08T10:39:56Z`, 2,245 chars, read read-only) says: *"Since 2026-09-08
the stamp writes every property on this node, so MARKER_ORIGIN, MARKER_WRITTEN_AT, NOTE, STATUS and
RENAMED_FROM are all cleared to NULL on any rebuild."* That is mechanism 1's false universal
verbatim — the sentence the second tombstone exists to retract — attached to a conclusion that
happens to be true under mechanism 3. Correct conclusion, retracted reason: the arc's own defect
class, in the artifact `freshness.md`'s check-0 per-marker gate treats as evidence.

**Fix (routes to `graph-dba`; a marker write is not mine):** replace that sentence with the
mechanism that is actually there — *"the stamp is `SET b = {…}`, a map assignment, which replaces
this node's whole property set, so everything the stamp did not write is gone after any rebuild."*
Same conclusion, one fewer false universal, and it survives the next re-check.

### P5-4 — major · `K-024` still routes the superseded shape to `architect`

`claude/cobb/kaizen/plan.md:23,155-162` is untouched since `29538d6` and still frames the debt in
mechanism 2's terms: *"§1.1's property table … never listed the five hand-authored keys and now
understates what a stamp writes."* Under the map form the stamp writes **eight** properties and
never writes those five at all — they were `= NULL` assignments for one commit and are now deleted.
An `architect` executing K-024 as written would add five never-written keys to the schema table of
the document that this arc cited as the ownership argument for erasing them.

**Fix:** rewrite K-024's rationale to the current mechanism — §1.1 owes (a) the eight properties
the stamp writes, (b) full 40-char OIDs rather than "short SHA", and (c) the new schema-level fact,
that the property set is closed by construction because the stamp is a map assignment. Add the
same erasure consequence to the `tico` half; the manual's `§1` cell (`graph-ontology.md:125`) tells
a reader a marker "can carry more than eight" and never says a rebuild takes them.

### Minor

- **P5-5** — `freshness.md:206-217`, the **second** tombstone, still says in the present tense
  *"What makes the rule hold **is now** the post-stamp assertion described above"*, and closes with
  *"Verified by execution … not inferred."* The third tombstone immediately below contradicts it.
  A tombstone is a record; write it in the past tense — *"What made the rule hold at that point
  was…"* — so a reader who stops after two isn't handed the superseded mechanism under an
  executed-not-inferred credential.
- **P5-6** — the counter prohibition's *reason* is itself unverified. `freshness.md:229-231` and
  `git-provenance.sh:127` say counters *"conflate properties set with properties removed"*. The two
  observations behind it (13 reported against 5 actual; 4 reported against 0) establish only that
  `Properties removed` does not track actual removals. State the observation, not the internals —
  in the paragraph whose subject is stating mechanisms you have not checked.
- **P5-7** — **P3-1 is not fixed and this arc widened it.** No failure branch in `pipeline.sh`
  prints `$STAMP`, and there is now a *third* branch (`:311-326`) telling the operator
  "re-stamping by hand is enough" — for a multi-line map literal with escaped quotes, which is
  strictly harder to reconstruct than the old flat SET clause, and (per P5-1) is the branch every
  build now takes. One line on all three: `printf 'pipeline: replay this stamp verbatim:\n%s\n'
  "$STAMP" >&2`.

### Nits

- **n1** — `SKILL.md:58` still says *"**Two** properties of the *mechanism* matter when you run a
  build"* over what is now a **four**-bullet list. Wrong since `6012ddb`; this arc added the fourth.
- **n2** — the stray read goes through `rq`, i.e. `GRAPH.QUERY`, while its own failure message
  (`pipeline.sh:325`) tells the operator to use `GRAPH.RO_QUERY` for the identical read, for the
  reason `5417f0e` gives (a `GRAPH.QUERY` against an absent graph materializes it). Use
  `GRAPH.RO_QUERY` for the two read-only calls, or say why the pipeline is exempt.
- **n3** — `claude/cobb/kaizen/history.md`'s U47 entry does not record the reversal its own commit
  message leads with. The durable copy lives only in the commit message and in
  `docs/plans/salesperson-ui-coordination.md`, which will be archived.

### Pass 4 dispositions

- **P4-1 — fixed**, twice. Rechecked at `HEAD`: `cpg_provenance_stamp` emits `MERGE
  (b:CpgBuildInfo) SET b = {` over exactly eight entries (`git-provenance.sh:219-234`); no
  hand-authored key is named anywhere in the stamp. The hybrid marker is unrepresentable. I did not
  re-execute the replace — a graph write is not mine — but `graph-dba`'s four probes are recorded
  with `keys(b)` read-backs and re-derived by the coordinator
  (`docs/plans/salesperson-ui-coordination.md:167`, `4380-4404`).
- **P4-2 — fixed** at `29538d6`, in the suggested wording (`freshness.md:150-153`).
- **P4-3 — not fixed**, correctly out of `cobb`'s remit; filed as K-024 → `tico`. See **P5-4** for
  what that ticket now says wrongly.
- **P4-4 — not fixed.** `freshness.md:34` still warns against a scripted `startswith` without
  stating the safe fallback. Three passes over that file have left it.
- **P4-5 — fixed**, though not by this arc: `graph-dba`'s rewritten `NOTE` now argues the per-marker
  gate explicitly instead of contradicting it. It introduced **P5-3** in the same write.
- **P3-1 — not fixed and widened** (P5-7). **P3-2 — not fixed**; `tico`'s, and unchanged by this arc.

### On the three things the brief asked me to weigh

**The third tombstone, sentence by sentence.** Its empirical claims hold as far as I can check them
without writing to a graph: 8+3 keys → `keys(b)` of 8 with `MARKER_EVIDENCE` gone, `count(b)` 1,
`labels(b)` `[CpgBuildInfo]` — corroborated in `graph-dba`'s U56b report and the coordinator's
re-derivation, and consistent across the three places the arc restates them. Its **differentiator**
is what fails. *"What separates the third is not that it is more plausible; it is that it was
executed before it was written down"* is true of the **Cypher construct** and false of the
**shipped mechanism**: the map literal, the `CPG_STAMPED_KEYS` accumulator and the pipeline gate
were never run together, and P5-1 is what that gap was hiding. The second tombstone made the same
kind of claim — *"Verified by execution in both directions against this instance"* — about a
mechanism that was also executed in isolation and also broken in its wiring. An execution
credential that covers the construct and not the call path does not distinguish mechanism 3 from
mechanism 2, and is exactly the sentence a re-checker will not re-check.

**The reversal sweep — complete in the shipped artifacts, incomplete outside them.** I swept the
repo for both the mechanism ("stops the rebuild by name", "closed list", "clear the key and add it
to the list") and the consequence ("fail its own next rebuild", "stops on purpose until those keys
are cleared"). `skills/README.md`, `freshness.md`, `SKILL.md`, `git-provenance.sh` and
`pipeline.sh` are clean; every surviving mention is inside a tombstone or a dated history entry
where it belongs. Two carriers were missed, both outside the four files `cobb` swept: the live
`NOTE` (**P5-3**) and `K-024` (**P5-4**) — and one present-tense mechanism sentence inside the
sweep, in the second tombstone (**P5-5**).

**The `keys(m)` vs `keys(b)` nuance — right, and adequately placed.** I reproduced the map half
read-only: the exact literal the stamp emits for `provenance=none` returns all eight keys from
`keys(m)`, four of them `NULL` (A10). The explanation in `git-provenance.sh:139-149` is correct,
and it sits in the docstring of the function that emits the literal — where the only reader who can
meet the trap is standing. `freshness.md:229` tells a consumer-side re-checker "`keys(b)` is the
discriminator" without the map-literal reason, which is fine: a consumer never sees the map.

**Operational consequence, and whether it is warned.** Adequately, in three of four places and by
accident in the fourth. `SKILL.md:82-92` tells a builder plainly that a rebuild erases the
annotation "completely and silently" and to read the marker first; `freshness.md:181-198` names
`cpg_falkorchat`'s ten keys and says they become eight; and the marker's own `NOTE` carries "THIS
NOTE IS BUILD-SCOPED AND DIES ON THE NEXT REBUILD", which is the one a rebuilder is actually
standing in front of. What is missing is the runtime: `pipeline.sh` reads no marker before
stamping and prints nothing about what it replaced, so the whole warning is documentation the
operator has to have read. Worth one pre-stamp read that echoes any non-stamped key it is about to
destroy — which, unlike the post-stamp assertion, fires *before* the annotation is gone.

**What §1.1 owes, given this arc** (not mine to fix; K-024 → `architect`): the debt changed kind,
not just size. Before, `docs/plans/cpg-agent-adoption-graph.md` §1.1 understated a stamp by four
properties. Now it also documents the wrong *write semantics* — its code block is the original
`SET b.X = …` concatenation, and its "omitted, not set to null/empty-string" rationale describes an
omission that is now achieved by a `NULL` entry inside a replacing map. The one genuinely new thing
it owes is the schema-level fact this arc established: the marker's property set is **closed by
construction**, so `:CpgBuildInfo` cannot be extended by any writer other than the stamp.

### What's solid

- **The design call is right, and the refusal that produced it is the best thing in the arc.**
  `0da3eb9` declining to ship `SET b = {map}` on a documentation sentence, routing it for execution,
  and shipping it at `5417f0e` on `graph-dba`'s executed evidence is the correct handling of a
  doc-sourced mechanism in a chain with this history. Deleting the five `= NULL` lines rather than
  relocating them is also right: a list that no longer protects anything teaches the wrong lesson.
- **The evidence is written as evidence, not as a citation.** `git-provenance.sh:126-158` carries
  four probes with their `keys(b)` read-backs inline. That is the form that survives a re-check.
- **The counter prohibition is a real finding, honestly propagated** — three independent
  observations that `Properties removed` cannot be cited, and `keys(b)` named as the discriminator
  at every site. Only the stated reason for it needs softening (P5-6).
- **`0da3eb9`'s byte-identity check on the refactor** was the right gate for a pure restructure, and
  the `_cpg_prop` accumulator is the right shape — a derived allow-list rather than a second list.
  P5-1 is a wiring defect in how it is called, not a defect in the idea.

### Open questions

1. **P5-1's fix touches `pipeline.sh` and `git-provenance.sh` — `cobb`'s files, and the fix is a
   behaviour change to a script nobody can run end-to-end here** (a real build is a multi-hour
   parse). I would gate the fix on the fake-`redis-cli` harness in A10 rather than on reading:
   assert that a correct replace yields **zero** stray rows and that a planted foreign key yields
   exactly one. That harness is the missing regression test for the whole arc, and it is cheap.
2. **P5-3 needs a marker write.** Route to `graph-dba` with the replacement sentence; the rest of
   the 2,245-char `NOTE` is accurate and should not be re-derived.

---

## Appendix

### A10 — Pass 5 verification

**The empty allow-list, at the shipped call site** (`git-provenance.sh` sourced; `pipeline.sh:228`'s
call reproduced verbatim):

```
$ STAMP="$(cpg_provenance_stamp 2026-09-08T00:00:00Z … /src parse-root)"
$ echo "CPG_STAMPED_KEYS in parent: [${CPG_STAMPED_KEYS:-<UNSET>}]"
CPG_STAMPED_KEYS in parent: [<UNSET>]
$ cpg_provenance_stray_query
MATCH (b:CpgBuildInfo)
UNWIND keys(b) AS k
WITH k WHERE NOT k IN []                      <- allow-list empty
RETURN 'STRAY_KEY=' + k AS stray
```

Called as a **statement** it is correct — `[BUILT_AT PARSED_AT SOURCE_PATH PROVENANCE
SOURCE_ORIGIN SOURCE_COMMIT SOURCE_TREE SOURCE_DIRTY]`, and `[BUILT_AT PARSED_AT SOURCE_PATH
PROVENANCE]` under `provenance=none`. The defect is the call site, not the accumulator.

**That query against the live marker** (read-only, `mcp__cypher__query`, `cpg_falkorchat`):

```
rows=10 — STRAY_KEY=BUILT_AT / SOURCE_PATH / SOURCE_COMMIT / SOURCE_DIRTY / PROVENANCE /
          SOURCE_ORIGIN / SOURCE_TREE / MARKER_ORIGIN / MARKER_WRITTEN_AT / NOTE
```

**End-to-end, stamp block extracted verbatim, fake `redis-cli` modelling a CORRECT map replace**
(marker = exactly the eight pipeline keys afterwards):

```
read-back: PASS
pipeline: FAILED — the marker in "cpg_sim" carries properties this build did not write:
pipeline:   BUILT_AT
pipeline:   PARSED_AT
pipeline:   SOURCE_PATH
pipeline:   PROVENANCE
pipeline:   SOURCE_ORIGIN
pipeline:   SOURCE_COMMIT
pipeline:   SOURCE_TREE
pipeline:   SOURCE_DIRTY
EXIT=1
```

A build that did everything right fails, naming its own eight fields, under a message that says the
condition is impossible.

**`keys()` on the map literal** (read-only, the exact `provenance=none` literal the stamp emits):

```
k = ['BUILT_AT','PARSED_AT','SOURCE_PATH','PROVENANCE','SOURCE_ORIGIN','SOURCE_COMMIT',
     'SOURCE_TREE','SOURCE_DIRTY']        n = 8
```

Eight, four of them `NULL` — confirming the docstring's assignment-vs-literal nuance from the map
side. The node side (`keys(b)` = 4) is `graph-dba`'s probe 2b; not re-executed here.

**Live marker, undisturbed** (read-only): 10 keys, `PROVENANCE = hand-backfilled`, `SOURCE_TREE =
85ddeed09479091a69b66d0301ed0d3399cc8387`, `MARKER_WRITTEN_AT = 2026-09-08T10:39:56Z`, `NOTE`
2,245 chars. `GRAPH.LIST` was **24** keys at review time (the brief expected 25); I created none —
every read went through `mcp__cypher__query` or `GRAPH.RO_QUERY` against an existing key.

**Syntax and sweep:**

```
$ bash -n skills/joern-cpg/scripts/pipeline.sh        -> OK
$ bash -n skills/joern-cpg/scripts/git-provenance.sh  -> OK
$ git grep -n 'CPG_STAMPED_KEYS' -- skills/
  git-provenance.sh:214  (set, inside _cpg_prop)   :221 (reset)   :271 (read)
  pipeline.sh:297        (a comment only)
  -> the only assignments are inside cpg_provenance_stamp, which pipeline.sh calls in a subshell
```

Reversal sweep, whole repo minus `.git/`, `falkor-chat/` and this review: `stops the rebuild|fail
its own next rebuild|until those keys are cleared|stops on purpose|clear the key|closed list|add it
to the list` — every hit is a tombstone, a dated history entry, or an unrelated component, except
`claude/cobb/kaizen/plan.md:23,159` (**P5-4**). The `NOTE` (**P5-3**) is not reachable by grep; it
lives in the graph.

---

## Pass 6 — 2026-09-08 · gating the Pass 5 fixes (`049f063`, `271c899`) and the rewritten live `NOTE`

**Scope:** the two commits read with `git show` against their parents (the working tree was in fact
clean for `skills/` this run — verified by `git status --porcelain -- skills/` and by md5 against
`271c899` for all three scripts), plus the live `cpg_falkorchat` marker read read-only. Files
judged: `skills/joern-cpg/scripts/{pipeline.sh,git-provenance.sh,test-stamp-wiring.sh}`,
`skills/cpg-analysis/references/freshness.md`, `skills/joern-cpg/SKILL.md`,
`claude/cobb/kaizen/{plan,history}.md`. **Method:** ran the shipped wiring test; then ran it against
**nine byte-copy mutants** in the session scratchpad (repo never touched — the three scripts still
md5-match `271c899`); probed the real `redis-cli`/FalkorDB reply shapes read-only via
`GRAPH.RO_QUERY`; re-derived `keys(b)`, `size(b.NOTE)` and the P4-4 predicate on the live marker.
`pipeline.sh` was not run. No graph write, no `GRAPH.DELETE`, no `GRAPH.QUERY` against a
possibly-absent graph, no tree-mutating git, nothing staged or committed. **One disclosure:** to
reproduce the `WRONGTYPE` row `pipeline.sh:268` claims, I created and immediately deleted a Redis
**list** key `__probe_key_analyst_tmp` (`DEL` → 1, `EXISTS` → 0 confirmed). No graph key was created
or removed; `GRAPH.LIST` is recorded as a listing in **A11.5**, not a count.

**Verdict: needs changes** — no blocker; four majors. P5-1 is genuinely closed and the wiring test
is a **real guard**, not theatre. But its oracle is soft in one specific way that let the arc's
newest defect class through again, the stray assertion still cannot fail on a real error reply, the
P3-1 fix landed in the three branches that don't need it and not in the two that do, and the third
credential contains one sentence that is false.

CPG: considered, not relevant — the artefacts are Bash and Markdown; the two loaded `cpg_*` graphs
are the *subject* of the marker data, not graphs of this code, and no `cpg_skills` graph exists.

### Ruling on the wiring test, as a guard

**Anchors: robust.** Four mutation modes, all fail loudly, none passes vacuously (A11.1). START
reworded → `BLOCK` empty → the `-z` gate at `:72-74` prints "anchors moved" and exits 1. END
reworded (with and without a new trailing step in the `--load` branch) → extraction runs on to the
file-final `fi` → `bash -c` syntax error → every case FAILs. END string duplicated earlier →
truncated block → the `:75-78` `cpg_provenance_stray_query` gate catches it. The END-loss diagnosis
is *accidental* — it works because `fi` is the next line — but it is structurally stable, since any
loss of that anchor pulls in the block-closing `fi`.

**The P5-1 mutation discriminates.** Confirmed both directions: the shipped suite refuses; with the
call-site guard deleted the mutation case **FAILs** ("refused, but not with the wiring message"),
and with the *whole rejected design* restored — `cpg_provenance_stamp` back to `printf`, call site
back to `$(…)` — the suite FAILs on four cases. That is the mutant the chain's rule asks for, and it
is caught.

**The fake is faithful where it matters, and hides one thing.** I verified against the live
instance that real `redis-cli` piped output is **bare** `STRAY_KEY=<NAME>` lines at column 0, so
`pipeline.sh:361`'s `sed -n 's/^STRAY_KEY=/…/p'` really does name the offending keys — the fake
models that correctly. Where it diverges is error shape: it emits only `errMsg: …`, and real
FalkorDB does not always. See **P6-2**.

### P6-1 — major · the oracle ignores the exit code, so deleting `replay_stamp` again passes green

`test-stamp-wiring.sh:100-105` decides a case by `rc == 0 → PASS`, else by scraping stray key names
out of `^pipeline:   NAME$`. It never checks *which* non-zero, and never checks that the failure
branch finished. So: delete the `replay_stamp` definition from `pipeline.sh` — the exact defect
`271c899` is titled for — and **all six cases still report PASS** (A11.2). Under the hood case 3 is
now aborting at **rc 127** with `replay_stamp: command not found`, and the scrape still finds
`MARKER_EVIDENCE` from the lines printed before the abort. The test built to close "verified in
isolation, broken in the wiring" does not cover the wiring defect that produced it.

**Fix:** assert the exit code exactly (`expect_rc`, 1 for every modelled failure), and add a
positive assertion on the replay block — each failing case must print `--- begin stamp ---` and a
line matching `^MERGE (b:CpgBuildInfo)$`. Both are one line each in `run_case`.

### P6-2 — major · the stray assertion is not fail-closed against a real FalkorDB error reply

`pipeline.sh:343` says the negative check is closed by "the explicit status checks". It is not:
`rq`'s classifier (`:268`) tests `errMsg:*|ERR\ *|WRONGTYPE*|*"read only"*|*"read-only"*`, and real
FalkorDB returns runtime errors **bare**, with `redis-cli` exiting 0 — verified live (A11.3):
`Unknown function 'notafunc'`, `Type mismatch: expected Map, Node, Edge, or Null but was String`.
Driving the shipped `rq` at one of those returns **rc 0**, and the reply carries no `STRAY_KEY=`, so
the run prints "stamp verified by read-back" over a marker that was never checked. The file's own
comment (`:255-256`) already concedes "pattern-matching that list is inherently incomplete" for the
stamp — and then two branches later relies on the same list as load-bearing. The fake never
surfaces it because its only error shape is `errMsg:`.

**Fix:** make the stray read positive rather than prefix-blind — the reply of a successful
`GRAPH.RO_QUERY` always opens with the column header `stray`; require it (`case "$STRAY_BACK" in
stray*) ;; *) fail`), or require the `Query internal execution time:` trailer. Add the same as a
seventh test case: fake reply `Type mismatch: …` must FAIL the block.

### P6-3 — major · `replay_stamp` is wired into the three branches where the stamp landed, not the two where it didn't

Pass 5's P5-7 named the branches that tell the operator "re-stamping by hand is enough". The fix
went to the other set. `replay_stamp` is called at `:348`, `:357` and `:373` — all three *after* the
read-back proved the stamp landed — and is **absent** from `:291-298` (FalkorDB rejected the stamp:
"only the provenance marker is missing") and `:308-312` ("the freshness stamp did not land"), which
are the only two branches where re-sending the Cypher is the fix. The result is a self-contradiction
inside one branch: `:346-347` prints *"The stamp DID land (it was read back above)"* and then
`replay_stamp` prints *"the load succeeded and does NOT need repeating — only the stamp does."*
`SKILL.md:109-110`'s claim that **every** stamp failure branch now prints the Cypher verbatim is
false — three of five do, and they are the wrong three.

**Fix:** call `replay_stamp` from `:291-298` and `:308-312`; on the three stray branches either drop
it or give it a second wording that does not claim the stamp needs re-sending.

### P6-4 — major · the third credential's generalisation is false of mechanism one

`freshness.md:228-231` (echoed in `claude/cobb/kaizen/history.md`'s U48 entry): *"since mechanisms
one and two both carried **execution** credentials too. An execution credential is only worth the
level it covers, and **both earlier ones covered the primitive and not the call path**."* Checked
against the file three paragraphs up and against `29538d6`: mechanism one's credential is
*"Re-checked there, not inferred"* (`freshness.md:203-204`) — a **re-reading** credential, not an
execution one — and its defect was not a level gap at all but an incomplete enumeration (a sixth
key, `MARKER_EVIDENCE`), which the second tombstone states in those words at `:206-210`. So a
"both" is carrying one supporting instance. Everything the sentence says about **mechanism two** I
confirmed independently: `0da3eb9` introduced `CPG_STAMPED_KEYS` while `pipeline.sh:225` still read
`STAMP="$(cpg_provenance_stamp …)"`, so its shipped allow-list really was empty.

**Fix:** narrow it, or generalise it correctly — the shared factor across all three is *the
credential named a narrower level than the claim it licensed*: mechanism one's covered the code but
not the space of keys it had to close; mechanism two's covered the query but not the call path.
That version is true of both and stays checkable.

### Minor

- **P6-5** — two more neighbouring wrong implementations pass the suite clean (A11.2).
  (a) Deleting `cpg_provenance_stray_query`'s empty-allow-list refusal (`git-provenance.sh:301-305`)
  — one of the *"two mechanisms"* `:222-226` claims — is invisible, because the call-site guard
  always fires first; a seventh case calling the function directly with `CPG_STAMPED_KEYS` unset and
  requiring rc 1 closes it. (b) Changing the emitted Cypher from `SET b = {` to `SET b += {` is
  invisible, because the fake's `MODE` is set by the harness rather than derived from the query, so
  case 3 asserts "merge caught" without depending on the client emitting a *replacing* map — while
  `pipeline.sh:365-366` tells the operator that a `+=` reversion is one of the two things to check.
  Deriving `MODE` from the query text (`SET b = {` ⇒ replace, else merge) makes case 3 catch it and
  costs two lines.
- **P6-6** — the call-site guard's own diagnostic is wrong in the one case it uniquely handles.
  `pipeline.sh:239` renders `${CPG_STAMP_CYPHER:+<set>}${CPG_STAMP_CYPHER:-<empty>}`, which for a
  *set* variable prints `<set>` **followed by the entire multi-line map literal** — observed
  verbatim when I broke the accumulator (A11.2, M10). The branch was never executed. Use
  `${CPG_STAMP_CYPHER:+<set>}${CPG_STAMP_CYPHER:-<empty>}` → `$([ -n "${CPG_STAMP_CYPHER:-}" ] &&
  echo '<set>' || echo '<empty>')`, or just report the length.
- **P6-7** — *"closure is by construction and there is no list at any layer"* (`freshness.md:223-224`,
  and the same words in the U47 history entry). `CPG_STAMPED_KEYS` **is** a list, at the assertion
  layer — the point being made is that it is *derived* rather than hand-maintained, which is the
  stronger claim and the one that survives a re-check. Same false-universal shape the tombstones
  exist to retract, one clause wide.
- **P6-8** — *(routes to `graph-dba`; a marker write is not mine)* the `NOTE` was rewritten
  (2,245 → **2,267** chars) while `MARKER_WRITTEN_AT` stayed `2026-09-08T10:39:56Z`, the value Pass
  5 read before the rewrite. The marker's own timestamp no longer dates its own content, on the one
  node whose `NOTE` `freshness.md`'s check-0 gate treats as evidence. Advance it on the next write.

### Nits

- **n4** — *"it asserts a populated allow-list"* (`freshness.md:244-245`, `SKILL.md:117`) overstates
  the mechanism: `ALLOWLIST=[…]` is `echo`ed and `grep`ed for display only (`:98`, `:111`), never
  compared. The property *is* covered — transitively, since an empty list trips the call-site guard
  and case 1 then FAILs (confirmed, A11.2 M10) — so this is wording, not a hole. "exercises" rather
  than "asserts".
- **n5** — `history.md`'s U48 lists "Six cases: allow-list populated, clean pass…, subsumption, and
  a mutation case" — that is five `run_case`s plus the mutation, with "allow-list populated" counted
  as a case it is not.
- **n6** — losing the END anchor reports `syntax error near unexpected token 'fi'` rather than
  "anchors moved". Correct outcome, misleading diagnosis; a `case "$BLOCK" in *"$END"*) ;; *) echo
  "FAIL: END anchor not found"` before the run costs one line.

### Dispositions

- **P5-1 — fixed.** `git-provenance.sh:260` assigns `CPG_STAMP_CYPHER`; `pipeline.sh:235-236` calls
  as a statement; the non-empty guard is at `:237-244`. Ran the suite: `CPG_STAMPED_KEYS` reaches
  the parent with all 8 keys (4 under `provenance=none`). Guard deleted ⇒ mutation case FAILs.
- **P5-2 — fixed.** Both prose sites carry the single narrow sentence plus DO-NOT-DELETE
  (`git-provenance.sh:273-291`, `pipeline.sh:329-341`). Case 4's "passes by design" assertion is
  **correct** and pinned at the right altitude — I re-derived it: a merge over a pipeline-clean
  marker with a `parse-root` stamp yields exactly the eight stamped keys, zero strays.
- **P5-3 — fixed** (wording, by `graph-dba`). The retracted universal is gone; the replacement
  sentence is **correct about the shipped mechanism** — the rendered Cypher really is `MERGE
  (b:CpgBuildInfo)\nSET b = {…}` (A11.4). The rest of the `NOTE` survives the last three commits: the
  `MANIFEST.txt:19` chain still reads as cited, check 0's per-marker gate still exists
  (`freshness.md:80`), and P4-4's new scripted advice agrees with it. See **P6-8** for the timestamp.
- **P5-4 — fixed.** Both K-024 rows rewritten, leading with the do-not-do warning; (a)–(d) match
  what §1.1 owes.
- **P5-5 — fixed.** Second tombstone in the past tense and self-labelled as the cautionary one.
- **P5-6 — fixed.** `git-provenance.sh:130-137` and the third tombstone both state the observation
  (13 vs 5, 4 vs 0) and explicitly decline the mechanism.
- **P5-7 / P3-1 — partially fixed, wrong branches.** See **P6-3**.
- **P4-4 — fixed**, after three passes. I ran the prescribed predicate live against
  `cpg_falkorchat`: `b.PROVENANCE IN ['parse-root','source-origin','none']` → `false`,
  `b.MARKER_ORIGIN IS NULL` → `false`, and the recipe's aliases (`AS provenance`, `AS markerOrigin`)
  match the names the advice uses.
- **P3-2 — not fixed**; `tico`'s, untouched by these commits, carried by K-024.
- **n1 — fixed** (count removed rather than incremented). **n2 — fixed**: both reads use
  `GRAPH.RO_QUERY` (`:307`, `:351`), the stamp still `GRAPH.QUERY`, correct. **n3 — fixed**: the U47
  entry now carries the behavioural reversal.

### What's solid

- **The extraction idea is the right one and it holds up.** Testing the shipped lines rather than a
  retyped copy is what makes P6-1's fix cheap: the oracle is soft, the harness is not.
- **The self-report in `271c899`'s message is accurate and complete about what it found** — the
  `bash -n` non-evidence admission is exactly right, and `bash -n` on the pre-fix file does pass
  (re-confirmed). That honesty is what made this pass tractable.
- **P5-2's ruling is now empirical rather than argued.** Case 4 asserting a known blind spot passes
  *by design* is the right shape for pinning a limit, and it is correct.
- **The counter prohibition (P5-6) is now stated as observation.** It survives re-checking, which the
  earlier mechanism claim did not.

### Open questions

1. **P6-8 needs a marker write** — `MARKER_WRITTEN_AT` on `cpg_falkorchat`. Route to `graph-dba`;
   the `NOTE` text itself is correct and should not be re-derived.
2. **P6-2's fix changes the shape of `rq`'s contract** (positive header match instead of an error
   blacklist) and touches the branch every build takes. Worth deciding whether the same positive
   form should replace the blacklist for the stamp write too, or only for the negative assertion.

---

## Appendix

### A11 — Pass 6 verification

**A11.1 — anchor mutations.** Nine byte-copy mutants under the session scratchpad; the repo copies
still md5-match `271c899` (`bda6a6c…` / `808a114…` / `c8c5698…`).

| # | Mutation | Result |
|---|---|---|
| M1 | START anchor reworded (`date -u "+…"`) | `FAIL: could not extract the stamp block … (anchors moved)`, exit 1 |
| M2 | END anchor reworded | 3 cases FAIL — block runs to the file-closing `fi`, `syntax error near unexpected token 'fi'` |
| M3 | END reworded **+** a new step appended to the `--load` branch | same; STAMP WIRING TEST FAILED |
| M4 | END anchor string duplicated after `STAMP="$CPG_STAMP_CYPHER"` | `FAIL: extracted block does not contain the stray assertion` |

**A11.2 — implementation mutations.**

| # | Mutation | Suite |
|---|---|---|
| M6 | call-site non-empty guard deleted (`pipeline.sh:237-244`) | **FAILS** — "refused, but not with the wiring message (rc=1)" |
| M8 | **full revert to the rejected design** — `printf` in `cpg_provenance_stamp` + `$(…)` at the call site | **FAILS** — 4 cases |
| M10 | `_cpg_prop`'s `CPG_STAMPED_KEYS` accumulation removed | **FAILS** — case 1 `FAILED(rc=1)`; output shows P6-6 (`CPG_STAMP_CYPHER=<set>MERGE (b:CpgBuildInfo)\nSET b = {…`) |
| M9 | **`replay_stamp` definition deleted** | **PASSES GREEN** — direct probe of case 3 shows `rc=127`, `bash: line 145: replay_stamp: command not found` (shipped copy: `rc=1`, stamp printed) → **P6-1** |
| M7 | `cpg_provenance_stray_query`'s empty-list refusal deleted | **PASSES GREEN** → P6-5(a) |
| M12 | emitted Cypher `SET b = {` → `SET b += {` | **PASSES GREEN** → P6-5(b) |

**A11.3 — real `redis-cli`/FalkorDB reply shapes** (`GRAPH.RO_QUERY`, read-only, `redis-cli` exit 0
in every row):

| Query | Reply | `rq` verdict |
|---|---|---|
| `… RETURN nosuchfunc(b)` | `Unknown function 'nosuchfunc'` | **rc 0 — treated as success** |
| `… RETURN keys(b.NOTE)` | `Type mismatch: expected Map, Node, Edge, or Null but was String` | **rc 0 — treated as success** |
| `THIS IS NOT CYPHER` | `errMsg: Invalid input 'T': …` | rc 1 |
| `… SET b.X = 1` via `GRAPH.RO_QUERY` | `graph.RO_QUERY is to be executed only on read-only queries` | rc 1 (`*"read-only"*`) |
| `GRAPH.RO_QUERY <absent graph>` | `ERR Invalid graph operation on empty key` | rc 1 |
| `GRAPH.RO_QUERY <list key>` | `WRONGTYPE Operation against a key holding the wrong kind of value` | rc 1 |

Also confirmed here: a successful stray read prints bare `STRAY_KEY=<NAME>` lines at column 0 (plus
a `stray` header and a `Query internal execution time:` trailer), so `pipeline.sh:361`'s anchored
`sed` works against the real client — the fake is faithful on the shape that matters.

**A11.4 — live `cpg_falkorchat` marker** (read-only): `size(keys(b))` = **10**, `size(b.NOTE)` =
**2267**, `MARKER_WRITTEN_AT` = `2026-09-08T10:39:56Z` (unchanged from Pass 5's read at 2,245 chars
— P6-8). Keys: `BUILT_AT, SOURCE_PATH, SOURCE_COMMIT, SOURCE_DIRTY, PROVENANCE, SOURCE_ORIGIN,
SOURCE_TREE, MARKER_ORIGIN, MARKER_WRITTEN_AT, NOTE`. The `NOTE`'s replacement sentence reads
*"Since 2026-09-08 the stamp is `SET b = {…}`, a map assignment, which replaces this node's whole
property set — so everything the stamp did not write is gone after any rebuild, whatever it is
called."* — which matches the rendered Cypher emitted by `git-provenance.sh:260`.

**A11.5 — `GRAPH.LIST` listing** (a listing, never a count to reconcile — membership churns):
`cpg_deprecated_salesperson, cpg_falkorchat, kaizen_team, probe_u8_rename_dst, reference, test,
ws:acme, ws:eval, ws:nlq-eval, ws:probe-s0-reset, ws:probe-s0r2, ws:probe-s0r3, ws:probe-s4b,
ws:qa-cart-totals, ws:qa-cart-totals2, ws:qa-catalog-lookup, ws:qa-catalog-lookup2,
ws:qa-durable-profile, ws:qa-salesperson-demo, ws:qa-tico-workflows-manual, ws:qa028, ws:s1v6,
ws:s1v7, ws:test`.
