# Recipe: freshness check

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumer:** `teco`, at dispatch time, for a unit whose specialist will consult a CPG (2026-08-19: centralized — see `docs/plans/cpg-agent-adoption2.md`; a specialist invoked standalone no longer runs this check). **Covers:** FR-5, FR-6
> (`cpg-agent-adoption`, M4).

**Purpose.** Before trusting a loaded CPG's answers, find out how current it is
relative to the source it describes — and if it looks stale, say so rather than
silently treating it as ground truth. No parameter to change; run as-is against
whichever graph you're already querying.

```cypher
MATCH (b:CpgBuildInfo)
RETURN b.BUILT_AT AS builtAt, b.PARSED_AT AS parsedAt,
       b.PROVENANCE AS provenance, b.SOURCE_ORIGIN AS sourceOrigin,
       b.SOURCE_COMMIT AS sourceCommit, b.SOURCE_TREE AS sourceTree,
       b.SOURCE_DIRTY AS sourceDirty, b.SOURCE_PATH AS sourcePath,
       b.MARKER_ORIGIN AS markerOrigin
```

**Expected shape.** Zero or one row — this is a singleton marker node, not a
per-build history.

| Field | What it is |
|---|---|
| `builtAt` | When the **load finished** — "when this graph's content was last touched". |
| `parsedAt` | When the **source snapshot was taken**, i.e. what the graph actually describes. On a multi-hour build these differ by hours: anchor any `--since` on `parsedAt`, never `builtAt`. |
| `sourcePath` | The parse root handed to Joern. Often a pruned scratch copy, so **not** necessarily a path `git` understands. |
| `sourceOrigin` | The repo-relative directory the provenance below describes — **this** is the path to hand `git log` / `git rev-parse`. |
| `sourceCommit` | The repo's `HEAD` when the source was captured, *before* the parse. A **full 40-char OID**. |
| `sourceTree` | The tree object of `sourceOrigin` at `sourceCommit` — a **blob** when the source is a single file — i.e. the exact identity of that content. Also a full OID. Can be absent while `sourceCommit` is present: see check 2. |
| `sourceDirty` | `git status --porcelain -- <sourceOrigin>` was non-empty: modified **or untracked** files under the source. Scoped — it says nothing about the rest of the repo. |
| `provenance` | How the four `source*` values were obtained. Three values are **pipeline stamps**: `parse-root` (the parse root is itself tracked) · `source-origin` (the parse root is a staged copy; the builder named the real tracked directory) · `none` (no git identity — the three commit/tree/dirty fields are deliberately absent, not missing). A fourth, **`hand-backfilled`**, is *not* a pipeline stamp: a human derived the `source*` values after the build and `graph-dba` wrote them in. Spelled as its own word, not a variant of the other three, so a skimmer — or a scripted `startswith("source-origin")` — cannot quietly treat it as a pre-parse capture. **If you are scripting this, the safe form is exact equality against the closed set** — `provenance IN ['parse-root','source-origin','none']` for "a pipeline stamped it" — never a prefix or substring test, and never a default-to-trusted `else`. For "was this written by the pipeline at all", test `markerOrigin IS NULL` instead: it is the reliable tell, where `provenance` is only a label the writer chose. An unrecognised value means a shape postdating this recipe, so fail closed and read the whole node. See the fifth bullet below. |
| `markerOrigin` | Non-null **exactly when a human wrote this marker** rather than the pipeline; the pipeline never sets it. This — not `builtAt`, not `provenance` — is the reliable hand-authored tell, because a hand-authored marker may carry a real timestamp *and* a real provenance value. Non-null → read the whole node (`MATCH (b:CpgBuildInfo) RETURN b`) for `NOTE`/`STATUS` before acting on any other field. |
| `MARKER_WRITTEN_AT` | Appears only alongside `markerOrigin`, and means **when a human last wrote or re-affirmed this marker — not when its text was first authored.** Compare it against `builtAt`: earlier than the build means the annotation predates the content it sits on and is suspect; later means a human has looked at the marker since the build. **Defined only on 2026-09-08**, having been carried in key lists until then with no gloss anywhere in the repo — so a value older than that date was written under no agreed meaning and carries none of this. "Last re-affirmed" is the reading that survives an edit-in-place, and it is exactly true rather than approximately, because it is set in the same act as a re-read of the marker against the graph as it then stands. **That obligation is what the field rests on — advancing this timestamp without actually re-reading the `NOTE` converts it into the false-freshness signal it exists to prevent.** If you cannot honestly re-affirm the content, leave the value alone; a stale-looking annotation is a working signal, a falsely fresh one is not. |

- **One row, `provenance` a pipeline value (`parse-root`, `source-origin`,
  `none`) and `markerOrigin` null** → a stamp from the current pipeline; read it
  with the table above. A non-null `provenance` does **not** on its own mean
  "pipeline" — check `markerOrigin` too, and see the fifth bullet.
- **One row, `provenance` null *and* `builtAt` a real timestamp** → a
  **pre-2026-09-07 stamp**. Still usable, but its `sourceCommit`/`sourceDirty`
  were derived *after* the load, repo-wide — see Limits before acting on either.
  (`provenance` is null on a hand-written marker too, which is why this bullet
  is gated on the timestamp; that shape is the fourth bullet below, and it
  carries no `sourceCommit`/`sourceDirty` at all.)
- **Zero rows** → either the graph predates this feature (built before M4; no
  marker was ever back-filled *into a graph that had none* — see the rollout
  note in the graph-dba design doc; that is a different act from the
  `hand-backfilled` provenance in the fifth bullet, which fills in the *fields*
  of a marker that already existed) or the
  pipeline run that built it failed its own verification and never reached the
  stamping step. Treat this the same as "stale": you have no freshness signal
  at all, which is itself a reason for caution, not an error to debug.
- **One row, but `builtAt` is not a parseable timestamp** → a **hand-written
  marker**, not a pipeline stamp. `graph-dba` writes one when a graph's
  provenance is genuinely unrecoverable but the graph is still worth keeping —
  a pre-M4 graph renamed rather than rebuilt, say. The tell is `BUILT_AT`
  holding the literal string `unknown`, chosen so it fails ISO parsing
  **loudly** rather than being coalesced into a plausible date. `markerOrigin`
  is non-null, as it is on **every** hand-authored marker — there are two such
  shapes now and the unparseable `builtAt` distinguishes only this one, so read
  `markerOrigin` whenever it exists: a marker can be hand-authored and still
  carry a real `BUILT_AT`. Such a marker explains itself in properties the query
  above doesn't return, so read the whole node
  (`MATCH (b:CpgBuildInfo) RETURN b`) — expect `STATUS`, `RENAMED_FROM` and a
  `NOTE`. Treat it as **"stale, and not
  rebuildable on demand"**: you have provenance but no date, so checks 0 and 1
  are unavailable and **check 2 must not be run** (see Limits). Live example:
  `cpg_deprecated_salesperson`, the CPG of the retired Streamlit `salesperson/`
  app whose source now sits at `deprecated/salesperson/`.
- **One row, `provenance` = `hand-backfilled`** → a **hand-backfilled marker**:
  a real `builtAt` and real `source*` values, but derived *after* the build by a
  human and written in by `graph-dba` — not captured by the pipeline before the
  parse. `markerOrigin` is non-null; `parsedAt` is **absent**, because it is
  genuinely unrecoverable after the fact and was not invented. The literal says
  a human filled the fields in, not that they filled them in well, so trust it
  only as far as its own `NOTE` earns: read the whole node and check what the
  note says each value was derived from and how it was verified. When the note
  establishes that, checks 0 and 2 are both available (see check 0's gate);
  check 1 falls back to `builtAt`. **Live example:** `cpg_falkorchat`,
  backfilled 2026-09-08, whose `NOTE` cites `cpg/.cpg-artifacts/MANIFEST.txt`
  and a `diff -rq` of the staged parse root against the committed tree.

**Judging staleness (a suggestion, not a rule).** Three escalating checks,
strongest first — the threshold is yours to set given the task at hand:

0. **Content identity — a yes/no, when you can get it.** Requires a
   `sourceTree`, `sourceDirty = false`, a `sourceOrigin` that `git` understands,
   and a `provenance` you can stand behind: `parse-root` or `source-origin`
   unconditionally, or **`hand-backfilled` once you have read that marker's
   `NOTE`** and it records where `sourceTree` came from and how it was checked.
   That last clause is **per-marker, not per-literal** — `hand-backfilled` only
   asserts that a human filled the fields in, so a backfilled marker whose note
   doesn't establish the derivation stays out of this check. `cpg_falkorchat`'s
   note does establish it, and the evidence is *stronger* than a routine
   `source-origin` stamp's: its staged parse root was verified byte-identical to
   the committed tree (`diff -rq`, exit 0, recorded in
   `cpg/.cpg-artifacts/MANIFEST.txt`), where a `source-origin` stamp only implies
   as much. Run from the repo root:

   ```bash
   git rev-parse --verify "HEAD:./<sourceOrigin>"   # compare to sourceTree
   ```

   **Use exactly that form, from the repo root.** `HEAD:./<origin>` resolves
   both a subdirectory and the repo root itself, where a bare `HEAD:.` is fatal
   (`Needed a single revision`, exit 128) — and `sourceOrigin` really is `.` for
   a whole-repo build. The `./` makes it relative to your working directory, so
   run it at the top level. `--verify` matters too: without it an unresolvable
   path makes `git rev-parse` **echo your argument back on stdout**, which in a
   script compares unequal and reads as "the source moved" rather than "you
   typed the path wrong". Both sides are full 40-char OIDs; don't `--short`
   either, since
   abbreviation width follows the repo's object count at the moment it runs, so
   the same tree can render 7 chars today and 8 next month and fail a string
   comparison on width alone.

   **Equal** → **the source is unchanged since it was captured**, so the graph
   is as current as it was at build time, however old `builtAt` is — you are
   done: no commit counting, and no false alarm from commits that touched the
   path and reverted. **Different** → the source moved; check 2 tells you by how
   much. *(This answers staleness, not build fidelity: under
   `provenance: source-origin` the parse root was a pruned copy of
   `sourceOrigin`, and whether that copy was faithful when staged is a build
   question — `joern-cpg`'s `SKILL.md` owns it — not one this marker can settle.
   A `hand-backfilled` marker admitted by the clause above inverts that: it is
   admitted precisely because its note settles the staging question with
   evidence a stamp doesn't carry.
   And when `sourceDirty = true` the parse also swallowed uncommitted work, so
   `sourceTree` describes only the committed part: skip check 0 and treat the
   graph as matching no commit exactly.)*
1. **Raw age.** `now − parsedAt` (fall back to `builtAt` on an older marker, and
   on a hand-backfilled one, which has no `parsedAt`).
   There's no universal cutoff — a week-old CPG on a slow-moving component may
   be fine; an hour-old one on a component under active refactor might already
   be behind. Weigh it against how much the task leans on structural
   correctness.
2. **Actual source movement.** If `sourceCommit` is present, run
   `git log --oneline <sourceCommit>..HEAD -- <sourceOrigin>` from the repo
   root; if it's absent, `git log --oneline --since=<parsedAt> -- <the real
   source dir>`. A **nonzero** commit count is a much stronger staleness signal
   than raw age — it means the source moved since the graph was built,
   regardless of how long ago that was. Zero commits is the converse: the graph
   may look old but the code it describes hasn't changed under it.
   (This is exactly the check `docs/plans/cpg-query-access.md` §2.3 did by
   hand — `git log --oneline --since=2026-07-18 -- falkor-chat/server` → 8
   commits — to establish an M2-era CPG was stale before its M3 rebuild.)
   **Use `sourceOrigin`, not `sourcePath`** — see Limits. **Both forms need a
   real `parsedAt`/`sourceCommit`**; skip this check entirely for a
   hand-written marker — the `builtAt = unknown` shape, *not* a
   **hand-backfilled** one, which carries a real `sourceCommit` and to which
   check 2 does apply.
   **A zero result means nothing in two cases**: when `sourceDirty = true`, and
   when `sourceTree` is null while `sourceCommit` is present. The second is a
   source tracked in the index but never committed at capture — `git log
   <commit>..HEAD -- <origin>` then reports 0 commits about code that has no
   commit history at all, which reads as "unchanged" when the truth is "never
   recorded". Check `sourceDirty` before believing a zero.

**Surfacing the suggestion (FR-6).** When any check makes you doubt the
graph, say so in whatever you hand back — don't silently keep using it as if
current, and don't rebuild it yourself. Naming a concrete next step is enough:
*"this CPG was built at `<builtAt>` (or: has no freshness marker) and
`<sourceOrigin>` has moved since; consider asking `graph-dba` to rebuild
`<graph>` before trusting a broad structural claim from it."* Whether to pause
and ask, or flag it and proceed, is your call — this recipe hands you the
signal, not the threshold.

## Limits

- **One marker per graph, not a build history.** An `--append` load overwrites
  the existing marker (by design — freshness tracks "when was this graph's
  content last touched," not "when was it first created"). Every field is
  rewritten on each stamp, absent ones removed, so a marker never mixes two
  builds — and since 2026-09-07 the pipeline reads the marker back and fails the
  run unless this build's `parsedAt` is in it, so a marker you find is one that
  actually landed. (`redis-cli` exits 0 on an error reply, so before that a
  rejected stamp was silent and an `--append` build could leave the *previous*
  marker standing over new content. A marker whose `builtAt` predates content
  you can see in the graph is that shape. Since 2026-09-09 the read-back's own
  failure is a distinct branch from an absent marker, so that guarantee no
  longer rests on a check that might not have run — the pipeline says "could not
  read the stamp back" rather than asserting the stamp did not land.)
  **A hand-authored marker is subject to the same rule**, and nothing exempts
  it: since 2026-09-08 the stamp is a
  **map assignment** (`SET b = {…}`), and `=` replaces the node's whole property
  set, so every property that stamp did not write is gone afterwards —
  `markerOrigin`, `NOTE`, `STATUS`, a key invented next year, all of it. There
  is no list of keys involved and nothing to keep in sync. **What this buys you
  as a reader: a marker is the product of exactly one act** — one build's stamp,
  or one hand-authoring — and never a mixture of both, which was a real shape
  found in the wild the day before. A hand-authored marker is therefore
  provisional: it stands exactly until the graph is rebuilt, it dies **silently**
  when that happens (the rebuild reports success and says nothing about what it
  erased), and whoever rebuilds inherits none of its reasoning. So
  `cpg_falkorchat`'s ten keys today are ten keys until someone rebuilds it and
  then they are eight, with no warning either way. **If you rebuild a hand-authored graph, re-write whatever
  annotation still applies** — the stamp will have cleared it, and the marker
  is build-scoped by design, so anything durable about the graph or its
  component belongs in `docs/` rather than on this node.
  *(Tombstone, 2026-09-08: this rule was documented one commit before it was
  true. The stamp then wrote only its own eight fields, so an `--append`
  rebuild left the hand-authored keys standing over freshly captured pipeline
  values — a marker announcing itself as "NOT a pipeline stamp" while carrying
  one. The rule was right and the mechanism given for it was not; the mechanism
  was made true rather than the rule weakened, by closing the stamp's property
  list in `skills/joern-cpg/scripts/git-provenance.sh`. Re-checked there, not
  inferred. Don't delete the rule on rediscovering the history.)*
  *(Second tombstone, same day: closing the list did not close the hole. A sixth
  hand-authored key, `MARKER_EVIDENCE`, was written onto a marker and survived a
  full `parse-root` stamp — the same defect one key over, because a closed list
  cannot enforce its own completeness and whoever hand-writes a marker is
  editing a different file from the one that would need updating. So the rule
  survived a **second** wrong mechanism: "the stamp writes every property on the
  node" was never true, only "the stamp writes the properties it names". What
  made the rule hold **at that point** was a post-stamp assertion whose
  allow-list was generated from the stamp's own assignments rather than
  hand-copied. That mechanism was itself superseded within the day — see the
  third tombstone, which is the one in force. And read this sentence as the
  cautionary part: it was written under a credential of "verified by execution
  in both directions", which was true of the *query* and untrue of the *wiring
  that called it* — the allow-list was empty in the shipped pipeline. The
  credential was real and covered the wrong level.)*
  *(Third tombstone, same day, and this one is the mechanism that is actually
  here. The clearing enumeration above is gone: the stamp replaces the marker's
  whole property set with `SET b = {…}`, so **closure of the marker is by
  construction** — nothing has to be listed for a property to be erased. State
  that precisely, because the first version of this sentence said "there is no
  list at any layer" and that was false one clause wide: `CPG_STAMPED_KEYS`
  **is** a list, at the assertion layer, and the pipeline's stray check reads
  it. The claim that survives is the stronger one — that list is **derived from
  the stamp's own map, in the call that builds it** (`_cpg_prop`), so it cannot
  drift from what was written the way a hand-copied second list can. A generated
  list is not the absence of a list, and saying so was a false universal of
  exactly the shape these tombstones exist to retract.
  <br>**Read the sequence rather than only the answer.** The rule "a rebuild
  erases a hand-authored marker" has now been stated three times with three
  different mechanisms, and the first two were both wrong while sounding
  checkable. What separates the third is not that it is more plausible, and —
  the correction that matters — not merely that it was "executed", since
  mechanisms one and two carried credentials too. **The shared defect is that
  the credential named a narrower level than the claim it licensed.** Mechanism
  one's credential was *"Re-checked there, not inferred"* — a re-reading of the
  code, which is what it covered; the claim it licensed was about the *space of
  keys* that code had to close, and a sixth key existed. Mechanism two's
  credential was an executed query, which is what it covered; the claim it
  licensed was about the call path, and the call path passed the allow-list
  through a `$(…)` subshell, so the shipped check saw an empty list. Neither
  credential was fake, and neither was an execution credential covering a
  primitive — that generalisation was written here and is untrue of mechanism
  one. Each credential was one level below its claim. That is the thing to
  check on the next one.
  <br>**And this part is about the tombstones rather than about the stamp.**
  Each of these three was written in the same sitting as the fix it certifies,
  by whoever had just made it — and each has since had its own certifying
  sentence corrected on review, this one included, twice. **The form is
  therefore a hazard.** The retraction half is trustworthy: it reports a failure
  that already happened. The certification half is not, and it inherits
  credibility from the retraction it is bolted to. So, for whoever edits this
  passage next: a sentence of the shape *"the earlier ones were wrong and here
  is why this one is different"* has the worst record of any sentence in this
  file. State the mechanism and the level its evidence covers, and stop. **Do
  not write a fourth tombstone certifying the third** — a further revision is
  one dated line, not a new narrative.
  <br>**What is checked, at both levels.** *The construct:* executed on
  throwaway graphs, with `keys(b)` read back every time — a marker carrying the
  eight pipeline keys plus three hand-authored ones, `MARKER_EVIDENCE` among
  them, the key that had just defeated mechanism two — came back with `keys(b)`
  of size 8 and all three gone, while `count(b)` stayed 1 and `labels(b)` stayed
  `[CpgBuildInfo]`, so `MATCH (b:CpgBuildInfo)` still finds it. *The call path:*
  `skills/joern-cpg/scripts/test-stamp-wiring.sh`, which extracts the real stamp
  block out of `pipeline.sh` between two anchors — never a retyped copy — and
  drives it against a fake `redis-cli` whose reply shapes were measured against
  the live instance rather than imagined. It **exercises** a populated
  allow-list rather than asserting one (each case prints the list it built; an
  empty list fails the case by tripping the call-site guard), and it asserts: a
  clean pass over a hand-authored marker, the `provenance=none` narrowing, a
  planted foreign key caught under merge semantics, **the same merge against a
  pipeline-clean marker, which must still pass**, the `provenance=none`
  subsumption, the two branches where the stamp did **not** land, a stray read
  that returns a bare runtime error, a stamp write rejected with **no error
  prefix**, a **read-back that itself errors**, the stray query called directly
  with an unusable allow-list, a **static check that every `rq` call site in
  `pipeline.sh` passes `GRAPH.QUERY` or `GRAPH.RO_QUERY`** (no other command's
  reply carries the trailer `rq` gates on), and two call-site mutations. **The prefix-less stamp
  rejection and the erroring read-back** were added 2026-09-09 with the `rq()`
  fix, and neither is judged on the exit code: both already exited 1 beforehand,
  by falling through to a *later* assertion and reporting that one's finding, so
  each pins the wording of the branch that is supposed to fire. Every case pins an **exact
  exit code**, and every case expected to fail must also be shown to have
  reached the end of its branch — both added 2026-09-08, after the previous
  oracle reported all six of the cases the suite then had as green against a
  `pipeline.sh` whose `replay_stamp` definition had been deleted and which was
  aborting at rc 127.
  <br>**Neither level is worth anything alone, and neither is worth anything
  unmutated.** Every assertion named above was re-run against a byte-copy mutant
  that removes or reverses the thing it claims to protect, and every such mutant
  fails the suite. The ones worth naming are the designs that were *rejected*,
  not merely the absence of the ones chosen: the `$(…)` call site, an `rq` that
  keeps only its error blacklist, an emitted `SET b += {`, and the replay wired
  into the three branches that had already read the stamp back instead of the
  two where it never landed. Reply *counters* were not used as evidence anywhere
  in this, and the reason is stated as what was observed rather than as a
  mechanism nobody checked: `Properties removed` does not track actual removals
  — one probe reported 13 against 5 real ones, another 4 against none. Why it
  diverges is not known here and is not asserted. If you are re-checking this,
  `keys(b)` is the discriminator. Don't
  delete the rule on rediscovering the history.)*
- **`sourcePath` is a parse root, not a git path.** It is what Joern was
  pointed at — frequently a pruned scratch copy staged to keep `.venv` and
  friends out of the parse. Running it straight through `git log` doesn't
  error; it silently returns zero commits, which reads as false "unchanged"
  confidence rather than "no signal available". `sourceOrigin` exists precisely
  so you never have to guess: use it, and if it is null, see the next bullet.
- **`provenance: none` means the builder had no git identity for the source,
  and the pipeline stamped nothing rather than something plausible.** A staged
  copy sitting *inside* a repo (under a gitignored path) is the common case:
  that repo's `HEAD` describes a different tree, so inheriting it would hand you
  a commit that answers "unchanged" about code it never saw. You then have
  `parsedAt` age as the direct signal — **but not necessarily the only signal
  you can act on.** If you can independently confirm, from task context (not
  from any field on the marker), the real repo-relative directory the copy was
  staged from, `git log --since=<parsedAt> -- <realSourceDir>` is still valid:
  one live dispatch correctly inferred `falkor-chat/server` as the real
  counterpart of a `.git`-less `/tmp/cpg-src/falkor-chat-server` scratch build
  and ran the stronger check on it, independently confirmed correct
  (`docs/test-reports/cpg-agent-adoption2-report.md` TP-002). Valid signal, but
  only once the real path is verified. **Prevention, for whoever rebuilds:**
  `pipeline.sh --source-origin <the tracked dir it was staged from>` records it
  so the next reader doesn't have to infer anything.
- **A marker with no `provenance` field predates the 2026-09-07 fix, and both
  its git fields are weaker than they look.** They were derived after the load
  finished, from the parse root's containing repo, with no pathspec — so
  `sourceCommit` is whatever `HEAD` happened to be *then*, not what was parsed
  (on the ~3h `cpg_falkorchat` build of 2026-09-07, `HEAD` moved four times and
  the graph was stamped with a commit that was never parsed), and
  `sourceDirty = true` may come from a file nowhere near the source (the same
  build stamped `true` because of a modified file under `claude/`, while the
  parsed tree was verified byte-identical to its commit). So: check 0 is
  unavailable (no `sourceTree`), check 2's commit is approximate — a *zero*
  result from it is the untrustworthy direction — and `sourceDirty` reads only
  as "the repo had changes somewhere".
  **`sourceOrigin` is absent on this shape too**, so check 2 needs the same
  independently-confirmed real directory as the `provenance: none` case above —
  never `sourcePath`, which on these markers is an absolute path into a
  gitignored staged copy and returns a silent zero. Anchor on `sourceCommit`
  (`<sourceCommit>..HEAD -- <the real dir>`), not on `parsedAt`, which these
  markers also lack. **No loaded graph carries this shape any more** (checked
  2026-09-08). `cpg_falkorchat` did — keys `BUILT_AT`, `SOURCE_PATH`,
  `SOURCE_COMMIT`, `SOURCE_DIRTY` and nothing else, with both git values
  hand-corrected afterwards to the parsed truth — until it was hand-backfilled
  on 2026-09-08; it now carries ten keys and `PROVENANCE = hand-backfilled`, so
  it is the fifth bullet's shape, not this one. The shape stays documented
  because reloading a pre-fix export re-creates it; when you meet one, the real
  directory is the one thing you must establish yourself (for `cpg_falkorchat`
  it was `falkor-chat/server`).
- **A hand-written marker — the `builtAt = unknown` shape, not the
  hand-backfilled one — has no date, and check 2 fails silently against it.**
  When `builtAt` is the literal `unknown`, `git log --oneline --since=unknown --
  <path>` **does not error**: git accepts the unparseable approxidate and
  returns **zero commits with exit 0** — verified 2026-09-02, and reproduced
  against a path with heavy recent history (`--since=unknown -- skills/` → no
  commits, while `--since=2026-08-01 -- skills/` → many). So don't run check 2
  for this shape at all — **and note the `provenance: none` escape hatch does
  not apply either**, because there is no `parsedAt` or `builtAt` to anchor a
  `--since` on even once you have confirmed the real directory. This shape also
  carries **no `sourceCommit`/`sourceDirty` whatsoever** — don't go hunting for
  them; `cpg_deprecated_salesperson`'s keys are `BUILT_AT`, `SOURCE_PATH`,
  `STATUS`, `MARKER_ORIGIN`, `MARKER_WRITTEN_AT`, `RENAMED_FROM`, `NOTE`.
  What you can still do is read the marker's
  `NOTE`/`STATUS`, and treat the graph as a frozen snapshot: for a retired
  component that is the correct reading, not a gap to close, and asking
  `graph-dba` for a rebuild is usually the wrong next step (the source may no
  longer be maintained, or may not be where the graph name suggests).
- **Opt-in per build.** Any CPG built before this feature shipped, or by a
  pipeline run whose own load verification failed, has no marker — that's the
  "zero rows" case above, not an error.
