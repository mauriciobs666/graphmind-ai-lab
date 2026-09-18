# Kaizen-team distillation — pass 2, gate reviews

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (`docs/plans/kaizen-team-distillation2-coordination.md`)

Family review document for the 2026-09-18 `cobb` sweep of the shared `kaizen_team` graph. One
`## U<n>` section per unit; each is a diff-scoped gate on that unit's uncommitted working tree,
baseline `1022c20` (never `HEAD`), in the shape of the 09-16 sweep's
`docs/reviews/kaizen-team-distillation-u1.md` … `-u7.md`.

## U1 — `analyst` chunk A (2026-09-18)

### Scope & verdict

Reviewed `cobb`'s uncommitted diff (`git diff 1022c20 -- <path>`) for U1: the distillation of 8
`analyst`-produced meta-lessons (`b4f6c1a2` `a1e2c3d4` `f4b8c2d1` `7f3c9a2e` `f3d9a1c2` `b7e4a1f2`
`e2f1a8c3` `f0e8b2a4`) captured while gating the 09-16 sweep. Five files, all read whole around
the hunks: `claude/analyst/review-techniques.md`, `skills/agent-maintenance/SKILL.md` (§5 read
whole after the diff), `claude/analyst/kaizen/history.md`, `claude/analyst/kaizen/plan.md`,
`claude/cobb/kaizen/history.md`. Not reviewed (concurrent session's work): `claude/graph-dba/
falkordb-quirks.md`, the untracked `falkor-chat/` and `claude/docs/plans/` files, and `teco`'s
two coordination documents. The 8 source nodes are already cleared, so promotions were judged
against the current tree and git history, not against the deleted entries.

**Verdict: needs changes** — one Major, a wording-level fix in one paragraph of `SKILL.md` §5
step 1 (plus the matching sentence in the history entry); disjoint from everything else, no Pass 2
re-gate needed once applied. Everything else re-derives cleanly: every git-history figure the
promotions and the disposition record rest on was re-run and matches; no figure from any entry's
`evidence` travelled into a promotion (the 09-16 U4 defect class — clean here); the graph state
is exactly as reported; the bookkeeping mechanics all check.

**CPG: not applicable — documentation-only diff (an agent knowledge base, a skill's procedure
text, kaizen bookkeeping) with no code-level component of its own; every source claim inside the
diff was verified by re-running the cited git commands, direct source reads (the `model-bench`
`hostinfo`→`runner`→`cli` chain), and live read-only `kaizen_team` queries.**

### Findings

#### Major — §5 step 1's new attribution instrument states an invariant the sweep's own ordering violates, and the disposition's "verified" claim is day-granular against a timestamp-level rule

`skills/agent-maintenance/SKILL.md` §5 step 1 (new text, working tree ~L463-468): *"An agent
whose unit cleared it to zero and that then ran further sessions in the same coordination (a gate
review is one) legitimately reads non-zero again, and **every such entry post-dates its clearing
commit** — one dated *before* it is the real miscount lead."* The sweep's ordering is clear →
gate → commit: `cobb` runs the `DETACH DELETE`s inside its unit, `analyst` gates, *then* `teco`
commits. Verified: `git show --stat 1896a1f` contains `docs/reviews/kaizen-team-distillation-u1.md`
(117 lines) — the gate's own deliverable is *in* the clearing commit, so the gate ran before it.
Any entry that gate writes (this gate is one; its entry, if written, sits in this exact window)
carries a `createdAt` before the clearing commit's timestamp while being entirely legitimate — the
instrument as written flags it as "the real miscount lead". Compounding it, the disposition
record's verification (`claude/analyst/kaizen/history.md`, the `f0e8b2a4` bullet: *"all 8 entries
here are dated 2026-09-16/17"*) compares the day-granular `date` field against `1896a1f`'s
`20:32:35` timestamp — a same-day `date` cannot establish post-dating, and the 09-16 U7 review the
claim inherits from (`docs/reviews/kaizen-team-distillation-u7.md:74-80`) made the same
day-granular comparison. **Fix (SKILL.md §5 step 1, same paragraph):** state the commit timestamp
as an *upper bound* on the clear — "an entry dated before the clearing commit is a lead only if its
`context` names nothing in that unit (the unit's own gate runs between the clear and the commit,
so a gate-written entry legitimately pre-dates it); compare `createdAt`, never the day-granular
`date`". **And in the history entry:** restate the `f0e8b2a4` verification honestly at the level
it was run ("dated on/after the clearing day; `context` names the U1–U7 gates") rather than as
confirmation of post-dating.

#### Minor — the `f4b8c2d1` paragraph is inserted mid-section, splitting the worked example it now interrupts; its recount half re-instances two existing rules

`claude/analyst/review-techniques.md:1323-1327` (new) sits between the section's anchor advice
("Anchor the range on text that exists in *every* revision…") and its measurement paragraph
("Measured 2026-09-09 over the five committed revisions…", L1329), so the measurement now reads as
if it follows the header-promise paragraph rather than the hashing technique it measures. Read at
`1022c20`: those two paragraphs were adjacent by design. The header-promise half ("no mechanism
enforcing it — check the promise against every entry body") is genuinely new — nothing at
baseline says it (`grep -i 'blanket\|per entry\|recount'` finds only the unrelated L278/L449
hits). The recount half ("recount any enumerated instance list … against a fresh grep") is a
re-instance: baseline L447-452 item (4) already says a copied count is "the same kind of claim as
a pasted grep result — re-run it, never cite it from a doc", and *"Reconciling a kaizen-graph
distillation's claimed dispositions"* (L322) says "verify the aggregate claim against itemized
ground truth". **Fix:** move the paragraph to after "Compare the AST line-range hash technique…"
(L1338-1340), immediately before the `Origin:` line, and shorten the recount clause to a
cross-reference ("…and recount an enumerated list the way item (4) above treats a copied count").

#### Minor — the promoted "duplicate-heading scan" is adjacent-only, and the text does not say so

`skills/agent-maintenance/SKILL.md` §5 step 4 item 4 (new): `awk '/^##/{if($0==p)print
FILENAME": "NR; p=$0}' <file>`. Ran it on a synthetic file: it fires on a heading duplicated
immediately under itself (line 8 reported — the exact 09-16 U1 Major shape) and is silent on the
same heading repeated three sections apart. The sentence names the defect it targets ("inserting
a dated entry above an existing one can duplicate the older heading"), so the scope is implied,
but "a duplicate-heading scan" promises more than it delivers. **Fix (same sentence):** either
call it an *adjacent*-duplicate scan, or use `grep '^## ' <file> | sort | uniq -d` (tested: fires
on both shapes; clean on all five reviewed files).

#### Info — the "`sessionId: null` … would have attributed nothing" parenthetical is at odds with the survivors' arithmetic

Both history entries (`claude/analyst/kaizen/history.md` `f0e8b2a4` bullet; `claude/cobb/kaizen/
history.md` "Observed" bullet) say 8 of `analyst`'s 14 `PRODUCED` edges carried `sessionId: null`,
"so the documented instrument alone would have attributed nothing". Live: 4 of the 6 surviving
edges are null, so 4 of the 8 cleared edges carried a `sessionId` — the instrument would have
covered half of them. Defensible only under the reading that no reference `sessionId` for the
09-16 coordination was known to compare against. Inferred (the cleared edges cannot be re-read);
low stakes, history text only. Suggest "would have covered at most half of them" or the explicit
"no reference `sessionId` to compare against" reading.

#### Info — "both ends" now carries two meanings in one paragraph

`claude/analyst/review-techniques.md:1525-1528`: the enclosing section's "re-run it at **both**
ends" means *the sha you review vs. the sha the artifact claims*; the appended word-count clause's
ends are `<sha>^` vs. `<sha>`. The explicit commands make the clause unambiguous on its own; a
two-word tweak ("both ends *of the delta*") would keep the two from reading as the same pair. Nit.

### Gate questions, with what was run

1. **Every promoted statement re-derived against the current tree.** `git show 320f682^:claude/
   AGENTS.md | wc -w` = 2783, `320f682:` = 3074 (+291); `git show 320f682:claude/cobb/kaizen/
   plan.md` L29 carries "+33 net", and `1022c20:` L29 carries the 2026-09-16 correction. `git show
   -s --format=%ci`: `1896a1f` = 2026-09-16 20:32:35 -0300, `b6e1454` = 21:51:27 -0300.
   `frontend-quirks.md` header at `1022c20` scopes its version promise ("Each entry tied to a
   specific npm package names the pinned version… the two entries that aren't"), the TS2312 entry
   says "10 interfaces converted", `grep -c '^export type Use.*Result' salesperson/src/api/
   hooks.ts` = 10 at both `1022c20` and the working tree. `git grep -n S16 1022c20 -- docs/plans/
   salesperson-ui2-coordination.md falkor-chat/docs/HISTORY.md`: both document the root
   `docs/HISTORY.md` CPG-only scope on 2026-09-16; root `AGENTS.md` (unchanged vs `1022c20`)
   carries it in the `cpg/` bullet. `git show 1022c20:claude/architect/kaizen/history.md` L29
   header: 5 promoted + 3 discarded = 8. The 09-16 U1 Major (duplicate heading, `-u1.md:28`) and
   U3 Minor (header counts, `-u3.md:33`) are real. Word deltas match exactly: `review-techniques.md`
   18,881→19,039; `SKILL.md` 6,366→6,603; `analyst.md` 2,699→2,699.
2. **No untraceable figure travelled into a promotion.** The two `review-techniques.md` clauses
   and the four §5 sharpenings contain no numbers from any entry's `evidence`; every number sits in
   the disposition record and each was re-run above. The `1418` figure: `git log --all --oneline
   --stat | grep -c 1418` = 0, `git log --all --format=%s | grep -c 1418` = 0, tracked-`.md` hits
   are only the 09-16 U4 review, `tdd-engineer`'s history, `cobb`/`analyst` histories quoting it,
   and the unrelated `model-bench` `stats.py` line count — exactly as the history entry says.
3. **The two `review-techniques.md` sharpenings.** `a1e2c3d4`: genuine — `wc -w` appears nowhere
   in the file at `1022c20`, and the mechanism it adds (an unrelated uncommitted hunk in the same
   file shifting a live-tree count) is distinct from the section's own (a concurrent implementer
   moving `HEAD`); the episode is real (`docs/reviews/commit-granularity.md:79-87`). `f4b8c2d1`:
   half genuine, half re-instance — see the Minor above.
4. **Routing.** The `analyst`-side halves are where the history entry says: `analyst.md:80` ("The
   unit of verification is the citation's *scope* … read every function the claim spans"),
   `analyst.md:86` (grep sibling kaizen-history files), `review-techniques.md:322` (Reconciling),
   `:397` (a grep away from confirmation), `:1220` ("Verified by execution" names a level).
   Nothing `analyst`-side covers the duplicate-heading scan or the date-vs-commit instrument —
   acceptable, since the gate re-applies §5 (this gate ran both), but worth knowing. §5 altitude:
   the four sharpenings are procedure-level and read as instructions to the distiller, which is
   the skill's reader; step 1's census paragraph is now long (21 lines) but not incoherent.
5. **§5 read whole after the diff.** Steps 1-4 numbering intact; step 4's inner items 1-5 intact.
   No contradiction: step 2's two grep instructions are distinct (an exact figure vs. the fact's
   keywords) and the "Unverifiable ≠ discard" sentence between them still scopes to the entry,
   not the figure. The only coherence defect is the Major above (a stated invariant the ordering
   in step 4 itself — log, then clear — combined with the gate-before-commit convention breaks).
6. **Bookkeeping.** History header 2 + 5 + 1 = 8 and matches the body (2 bullets under
   "Promoted (2)", 4 bullets covering 5 ids under "Promoted (5)" with the merge named, 1 under
   "Discarded (1)"); each of the 8 ids has exactly one disposition. Adjacent-duplicate `awk` scan
   and `sort | uniq -d` scan: clean on all five files; also clean on `claude/analyst/kaizen/
   history.md` at `1896a1f` and `1022c20`. `plan.md`: none of the 8 ids present; `Last reviewed`
   bumped; one parking-lot line, consistent with the U37 precedent below it. `cobb`'s history
   entry's file list and word deltas match the actual diff.
7. **The discard `f3d9a1c2`.** Both cited publications confirmed verbatim (`analyst.md:80`,
   `review-techniques.md:1220-1224`). Chain re-traced: `model-bench/modelbench/hostinfo.py:274`
   `def check_attestation_staleness` → `runner.py:850` call, `:858` `raise RunRefused(…,
   exitCode=5)` → `cli.py:406` `except RunRefused` → stderr + `return exc.exitCode`;
   `claude/qa-engineer/qa-testing-techniques.md:234` ("`run` refuses outright … no traceback")
   holds end-to-end.

**Graph state (live, read-only):** `analyst` holds exactly 6 `PRODUCED` entries — `a1f3c9e2-6b7d`,
`a1e6c9d4`, `b3f1b8b4`, `c7e2a814`, `c7f2a815`, `a1f3c9e2-7b4d`, the six U2 ids; a prefix match
on all 8 U1 ids returns 0 nodes; `e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e` exists with `MENTIONS`→
`tico` and no producer. (If this gate's own learning-capture entry is written after this review,
`analyst` reads 7 — the seventh is dated after the clear and is the Major's window in the flesh.)

### What's solid

- The verification standard the 09-16 Close-out set is met: every figure in the disposition
  record re-derives, the U4 defect class is absent, and the promoted forms are rules only.
- The §5 step 2 "figure inside `evidence` is a claim of its own — trace or drop" clause and the
  generalised sibling-doc grep are the right altitude and the right home; the corroboration
  direction (a sibling doc proving the gap is still open) is a real addition the old clause lacked.
- Merging `b4f6c1a2`/`b7e4a1f2` into one producer-side self-check sentence is proportionate; the
  header-count check is exactly the 09-16 U3 defect, placed where the entry is written.
- The discard is correct and its chain was re-traced anyway rather than taken on the two citations.
- The history entry is a complete disposition record: it states the bar applied, names the
  section each promotion sharpens, and records the instance-vs-technique split for `f4b8c2d1`.

### Open questions

None blocking. For `teco`: after `cobb` applies the Major, no re-gate is needed — a diff of the
one §5 paragraph and the one history bullet is enough to confirm; the two Minors and two Infos are
optional in the same fix pass.

## U2 — `analyst` chunk B (2026-09-18)

### Scope & verdict

Reviewed `cobb`'s uncommitted diff (`git diff 48e8a84 -- <path>`, baseline `48e8a84`, never `HEAD`)
for U2: the distillation of the 6 `analyst`-produced code facts `b3f1b8b4` `a1f3c9e2-6b7d` `a1e6c9d4`
`c7f2a815` (model-bench) · `c7e2a814` `a1f3c9e2-7b4d` (falkor-chat `CallContext`). Seven files, each
read whole around the hunks: `claude/analyst/review-techniques.md` (new final section),
`model-bench/AGENTS.md` (one "Load-bearing invariants" paragraph), `model-bench/docs/BACKLOG.md`
(one open item), `falkor-chat/docs/SERVER.md` §1.3 (read whole, two paragraphs merged),
`claude/analyst/kaizen/history.md`, `claude/analyst/kaizen/plan.md`, `claude/cobb/kaizen/history.md`.
Every code claim was checked at `git show 48e8a84:<path>`; `falkor-chat/server/**` was never read
from the working tree (a concurrent session's mid-edit), except the one read-only look at the
`mcp.py` diff the brief asked for (the U3-sequencing Info below). Not reviewed: `falkor-chat/AGENTS.md`,
`claude/graph-dba/falkordb-quirks.md`, the untracked `falkor-chat/` and `claude/docs/plans/` files,
`teco`'s coordination documents, and the concurrent session's dirty `server/` files. The 6 source
nodes are already cleared; promotions were judged against the tree, git history, and the
disposition record.

**Verdict: approve with suggestions** — no Blocker, no Major. Three Minors, all wording-level and
disjoint (a false axis count in the history record; a sentence in the merged SERVER.md paragraph
that now contradicts its own next sentence; a quoted "contract" in the BACKLOG item that is a
review paraphrase, not source text), plus five Infos, two of them sequencing notes for `teco` (U3,
U6). Everything promoted re-derives: the `CallContext` facts hold end-to-end at `48e8a84`; the
sentence-boundary regex, its comment, and all five named tests exist and pass; the one remaining
unguarded `validate_pack` axis is real and the crash reproduced with the exact frame chain the
record cites; `aa29fc2`, U170, U175 and U177 are all confirmed in git, not on report; no figure from
any entry's `evidence` reached a promotion (the 09-16 U4 class — clean); the graph is exactly as
reported; the bookkeeping mechanics all check. No Pass-2 re-gate needed once the Minors are applied.

**CPG: considered, not relevant — the diff is documentation-only but every promotion asserts a
code fact; `teco` flagged `cpg_model-bench`/`cpg_falkor-chat` freshness as unverified, so each
claim was verified by direct source reads pinned to `48e8a84` plus executing the named tests and
the crash reproduction — evidence (exact line numbers, a `try/except` guard's presence, a runtime
`FileNotFoundError`) a possibly-stale CPG could not have supplied.**

### Findings

#### Minor — the history record says "exactly two axes read a data file"; at `48e8a84` it is three

`claude/analyst/kaizen/history.md`, the `b3f1b8b4…` bullet: *"exactly two axes read a data file,
and one is still unguarded"*. Enumerated from `validate_pack` (`packs.py:986-993`) and every
`data_path(`/`read_text(` site in the file: **three** axes read a `data.*` file —
`_row_count_identity_problems` (`:670-678`, `data.conversations` via `rows_path.read_text`, guarded
`try/except (OSError, json.JSONDecodeError)`), `_clean_through_turn_h_problems` (`:929-935`,
`data.conversations` via `iter_scripts()`, guarded since `aa29fc2`), and
`_answerability_stamp_problems` (`:890`, `data.items` via `iter_items()`, **unguarded**). The
promoted BACKLOG text is correct ("one axis that crashes", "the sibling axis had the identical
defect") — only the record's count is wrong, and it is the record's own re-derivation claim. Also
verified for the record: U162's gate row (`docs/plans/small-model-benchmarking-coordination.md:2587`)
already named `_row_count_identity_problems` as the guarded pattern the fix mirrored. **Fix (history
bullet only):** "three axes read a `data.*` file; two are guarded (`_row_count_identity_problems`,
`_clean_through_turn_h_problems`), one is not — `_answerability_stamp_problems`". Optional, same
shape, in the BACKLOG item: "the two sibling axes … are guarded" instead of "the sibling axis".

#### Minor — the merged §1.3 paragraph keeps "every REST and MCP call resolves to the same actor" two lines above the sentence that names the exception

`falkor-chat/docs/SERVER.md` §1.3, merged paragraph (working tree L86-91): *"every REST and MCP call
resolves to the same actor **and the same workspace** … the storefront's own per-caller path (below)
varies only `actor`"*. At baseline the "every REST call" sentence and the storefront exception sat
~60 lines apart, with the later paragraph ("no `/shop/api` request resolves through it") doing the
qualifying; the merge pulls the exception into the same paragraph, so the two sentences now read as
a contradiction on their face. The sentence is true at `48e8a84` only when "REST" means the legacy
`api.py` router (44× `Depends(get_context)`, verified) — the `/shop/api` REST routes resolve
`actor` per participant (`storefront.py:685`). The "and the same workspace" addition is true for
both paths (`app.py:374` `ws=provider().ws`). This is also the sentence U3's `architect` clause is
slated to land on, so a loose antecedent now becomes a stacked one. **Fix (that sentence):** "every
call that resolves through `get_context` — the legacy `api.py` router and `/mcp` — gets the same
actor and the same workspace; the storefront's per-caller path (below) varies only `actor`…".

#### Minor — the BACKLOG item quotes a "documented 'always returns a list' contract" that is not in the source

`model-bench/docs/BACKLOG.md` L88: *"every caller relying on its documented 'always returns a
list' contract"*. `validate_pack`'s docstring (`packs.py:950`) says *"`[]` means valid, matching
`Fingerprint.validate()`'s shape"*; `git grep -i 'always return' 48e8a84 -- model-bench/modelbench/`
finds nothing. The quoted phrase is U162's gate-row paraphrase (coordination doc L2587), carried
into a component backlog as if it were the code's own words — the 09-16 U4 class in miniature (a
citation that reads as verbatim and cannot be found). The contract it points at is real (a raise
plainly violates "`[]` means valid"), and "every caller" is exactly two, both `cli.py` (`:350`
`validate`, `:383`). **Fix (same sentence):** drop the quotation marks and cite the docstring's
actual wording, e.g. "…from `validate` and the one other `cli.py` caller, breaking the docstring's
'`[]` means valid' shape".

#### Info — `guard-testing-techniques.md` is `tdd-engineer`'s KB, cited bare as if it were `analyst`'s

`claude/analyst/kaizen/history.md`, the `a1f3c9e2-6b7d…` bullet: *"in neither `review-techniques.md`
nor `guard-testing-techniques.md`"*. `git ls-files | grep guard-testing` →
`claude/tdd-engineer/guard-testing-techniques.md` only; `claude/analyst/` has no such file. The check
itself was the right one (the mutation-technique KB that could already hold this lives there);
path-qualify the citation so the next distiller does not go looking under `analyst/`.

#### Info — the `plan.md` dedup sentence is true pre-write and false post-write, by the same phrasing U1 used

`claude/analyst/kaizen/plan.md` parking-lot line: *"Dedup check run on all six `entryId`s — none
appears in this file"* — the line that says so carries `b3f1b8b4-6f3c-4b6a-9a1a-2f0f9a2b6d31` in full
(`grep -c b3f1b8b4 plan.md` = 1, 0 at baseline). Reads fine as a pre-write check and matches the U1
precedent line below it; "none appeared before this line" would make it exact. Nit.

#### Info (for `teco`, U3 sequencing) — the concurrent `mcp.py` diff makes the retained sentence stale in implicature, not in letter

The concurrent session's uncommitted `falkor-chat/server/falkorchat/mcp.py` adds `produced_by` to
`ingest_document`/`ingest_documents`: `ctx = _get_context()` is unchanged (the `CallContext` actor
stays process-constant), but its new docstring says attribution *"resolves ONLY against an existing
`Agent` (never the `get_context()` actor)"*. So "every … MCP call resolves to the same actor" stays
literally true of the *context* while what an MCP call *attributes* becomes per-item — which is
exactly the per-caller-attribution caveat U3's `architect` entry (`a1f3d9c2…`) promotes onto this
sentence. Sequence U3 after that `mcp.py` change is committed (or have U3 word the clause
conditionally), and apply the Minor above first so the clause lands on a precise antecedent.

#### Info (for `teco`, U6) — the `tdd-engineer` entry the AGENTS.md paragraph is written to absorb recommends the very split the paragraph forbids

`a1f3c2e4-…` (live, `tdd-engineer`): *"canonicalized reply text can still be split into sentences
with a plain rfind/find on `.!?`"*. The new `model-bench/AGENTS.md` paragraph forbids exactly that
(`rfind(".")`, `split(".")`) and is right to — the fold in U6 must promote only the `_canon_str`
preserves-punctuation precondition and discard the `rfind` advice as superseded by U177. Not a U2
defect; the paragraph is correctly shaped for that fold.

#### Info (out of U2's scope, pre-existing) — a delivered item is still in the forward-looking backlog

`model-bench/docs/BACKLOG.md` L34-39 ("`validate_pack` doesn't check that a pack's declared
`"scorer"` name resolves…") is delivered: `_scorer_problems` (`packs.py:846` at `48e8a84`) does
precisely that and is wired into `validate_pack` (`:991`). Not in `cobb`'s diff and not a distiller's
job; noted because root `AGENTS.md`'s "a delivered item does not stay in it" rule is breached one
bullet above the new item. Route to the model-bench closeout list.

### Gate questions, with what was run

1. **Truth, re-derived at `48e8a84`.** `config.py:16-17` `WS_ID`/`USER_ID`, `:276-284`
   `get_context() → CallContext(ws=WS_ID, actor=USER_ID)`; `api.py` `grep -c 'Depends(get_context)'`
   = 44; `mcp.py:41` `_get_context = config.get_context`, replaceable only via `context_provider`
   (`:147-169`), read at `:210-260`; `storefront.py:570` `self._ws = config.WS_ID if ws is None else
   ws`, `:678-685` `context_for → CallContext(ws=self._ws, actor=participant_id)`, `:1494`
   `_catalog_ctx` same `ws`; `app.py:364-374` constructs the one `Storefront` with `ws=provider().ws`;
   `modelconfig.py:602-607` docstring "resolving per call (not at construction)", `.resolve()/.llm()/
   .embedder()` all take `ws=` (`:729-787`); `embedding.py:127` `_index_dim_cache`, `:138` `key =
   (ws, label)`, `:216/:230` `embed_message(ws, …)`/`embed_chunk(ws, …)`; `ingestion.py:100-123`
   `extract_chunk … self._models.llm("extraction", ws=ws)`; `db.py:78` `workspace_graph(db, ws)`;
   `repository.py:193-194` `_graph(self, ws) → db.workspace_graph(self._conn, ws)`;
   `scripts/start_demo.sh:103` `FALKORCHAT_WS_ID="${FALKORCHAT_WS_ID:-demo}"`, `falkor-chat/AGENTS.md`
   Key-scripts row at `48e8a84` L84. Grounding: `grounding.py:67` `_SENTENCE_BOUNDARY_RE =
   re.compile(r"(?<!\d)\.(?!\d)|[!?]")` with the 8-line comment naming the decimal mechanism; `git
   grep` for `rfind(".")`/`split(".")` in `scoring/` at `48e8a84` → none; `looks_like_abstention`
   docstring (`:87-103`) documents "direction-*insensitive*" and U174 findings 3/5. `validate_pack`:
   axis enumeration above (Minor 1); `aa29fc2` exists (2026-09-17, "S6 Steps 0-1 code gate … (U162-
   163)"), touches `packs.py` +`tests/test_packs.py` and adds
   `test_clean_through_turn_h_problems_reports_missing_conversations_file`.
2. **"Already fixed" dispositions, in git.** U170: `9ef89d7` ("Steps 0-2 … U165-U168,U170") adds
   both `test_format_directive_*` defs (`grep -c` = 2); `model-bench/docs/HISTORY.md:132-133` records
   "two `_format_directive` coverage tests (U170)". U175: `07cccb1` adds `_sentence_span` and the
   "direction-*insensitive*" docstring; coordination rows U174/U175/U176 (`:2702/:2722/:2723`) carry
   the order-insensitive mutation finding and its redesign. U177: `fd515a2` adds
   `_SENTENCE_BOUNDARY_RE`; `HISTORY.md:167-169`. Ran from `model-bench/` with its venv: `pytest -q
   -k "format_directive or decimal_number_between or percentage_decimal or unrelated_later_sentence
   or clean_through_turn_h_problems_reports_missing"` → **6 passed**, 1707 deselected. Reproduced
   the open crash on a scratchpad copy of `packs/nlq-structured-query` with `items.jsonl` removed:
   `load_pack` ok, `validate_pack` → `FileNotFoundError`, frames `validate_pack:992 →
   _answerability_stamp_problems:890 → iter_items:299` — byte-for-byte the record's chain. Scratch
   removed; `git status model-bench/` shows only `cobb`'s two dirty files.
3. **Figures.** `grep '1625\|\b169\b'` over the added lines of all four promoted files → 0 each (the
   entry-`evidence` counts stayed in the graph, as the record says). Every number `cobb` wrote
   re-derives: 44× `Depends(get_context)`; line numbers `890/992/299`, `mcp.py:41`,
   `storefront.py:685/:1494`, `modelconfig.py:605-606` (the phrase spans `:606-607` — within a
   line), `:730-790`, `embedding.py:216/:230/:127-138`, `ingestion.py:100-123`, `repository.py:193`;
   word counts exact — `review-techniques.md` 19,056→19,399, `model-bench/AGENTS.md` 1,950→2,035,
   `SERVER.md` 8,829→8,955, `BACKLOG.md` 896→1,020. The one untraceable quotation is Minor 3.
4. **Re-instance vs new method.** Baseline `review-techniques.md` neighbours read: L275 (a
   structural probe's negative is about shape), L652 (rebinding a class constant via a pytest
   plugin — *how* to mutate without touching source), L868 (SHRINK/WIDEN on a membership-guard
   pin), L906 (kill count is a draw), L965 (shared fixture couples two checks); the 09-16 U1 list
   (`git show 48e8a84:claude/analyst/kaizen/history.md`, 2026-09-16 entry) is EXPLAIN, docstring-nit
   transcription, fake timers, dev-instance repopulation, TanStack, React 18, TS excess-property,
   combined isolation test. None targets *what* to mutate when the math is already pinned — the
   glue's content, or a documented restriction's laxer variant. `grep -i 'glue\|directive\|laxer\|
   return a constant'` at baseline hits only unrelated text (L702-712 "glued fence"). Genuinely new;
   the Origin line's U168/U174 attributions both match their coordination rows verbatim.
5. **Altitude and doc-kind fit.** `model-bench/AGENTS.md`: present-tense invariant naming the regex,
   the forbidden shape, and the failure it prevents; same shape as its "Load-bearing invariants"
   peers (L96-110); 2,035 words < ~2,500. `BACKLOG.md`: an open item with a reproduction date as one
   dated clause, a named fix shape and a risk statement; matches the file's un-numbered bold-lead
   bullet convention. `SERVER.md` §1.3 read whole (L68-215): the two paragraphs read as one
   statement and nothing downstream contradicts them — except the intra-paragraph tension in
   Minor 2, which is the retained sentence's pre-existing looseness made visible by the merge.
6. **Bookkeeping.** History header 5 + 1 + 0 = 6; body 1 + 2 + 1 + 2 bullets, each of the six ids
   with `…` appears exactly once in the U2 entry; `git diff 48e8a84 -- claude/analyst/kaizen/
   history.md | grep -c '^-[^-]'` = 0 (U1 untouched, pure insertion). `grep '^## ' | sort | uniq -d`
   and the adjacent-`awk` scan: 0 on all seven files. `plan.md`: none of the six ids at baseline;
   `Last reviewed` bumped; one parking-lot pointer (the Info above). `cobb`'s history file list and
   word deltas match the diff.
7. **Graph (live, read-only).** `analyst` `PRODUCED` → exactly 1: `bb058e98-566b-43c6-b4fa-
   8e216accd573` (2026-09-18, U1's gate capture — out of scope). Prefix match on the six U2 ids → 0.
   `e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e` present, producers `[]`, `MENTIONS` → `['tico']`. This
   gate wrote no capture of its own, so `analyst` still reads 1.

### What's solid

- The discard bar was applied honestly: four model-bench instances confirmed fixed in git before
  being called fixed, and only the technique/invariant forms promoted — the instances live in the
  S7 fix chain already.
- The `validate_pack` item is a model open-item: the one remaining axis named, the crash
  reproduced (and re-reproduced here identically), the sibling's fix pattern and pinning test named,
  blast radius stated.
- The `review-techniques.md` section is a real new method at the right altitude, with the two
  mutants stated as one-edit-one-run procedures and a single reading rule.
- The SERVER.md merge is the right home and shape for U3's clause; naming all three
  once-constructed components closes a gap §1.8 only half-covered.
- The AGENTS.md paragraph anticipates U6 correctly — it forecloses the `rfind` advice the pending
  `tdd-engineer` entry carries.

### Open questions

None blocking. For `teco`: the three Minors are one-sentence edits in three different files; no
re-gate needed — a diff of those three sentences suffices. Sequence U3 per the Info above.

## U3 — `architect` (2026-09-18)

### Scope & verdict

Reviewed `cobb`'s uncommitted diff (`git diff 49ba441 -- <path>`, baseline `49ba441`, never `HEAD`)
for U3: the distillation of the 2 `architect`-produced entries `a1e6d9f4` (model-bench `report.py`
never rendered `LatencyBlock` in `compare_report`, S1–S6 — discarded) and `a1f3d9c2` (falkor-chat:
one process-constant actor behind every `get_context`-resolved call — promoted). Three files, each
read whole around the hunk: `falkor-chat/docs/SERVER.md` §1.3 (read whole, L68-215, plus §2.2),
`claude/architect/kaizen/history.md`, `claude/cobb/kaizen/history.md`. Every falkor-chat code claim
was checked at `git show 49ba441:<path>` only — `falkor-chat/server/**` was never read from the
working tree (a concurrent session's mid-edit). Not reviewed: `falkor-chat/AGENTS.md`,
`claude/graph-dba/falkordb-quirks.md`, the dirty `server/` files, the untracked `falkor-chat/` and
`claude/docs/plans/` files, `teco`'s coordination doc. The 2 source nodes are already cleared;
promotions were judged against the tree, git history, and the disposition record.

**Verdict: approve with suggestions** — no Blocker, no Major. One Minor (the promoted sentence's
middle clause is `agent-knowledge-base-strategy.md` §4.1's *design stance* restated as if it were a
code fact, and it sits two sentences above §1.3's own "when auth lands only `get_context` changes"
without the doc reconciling the two), two Infos (one of them pre-existing and out of scope), two
nits. Every code clause of the promoted sentence holds at `49ba441` — including the strongest one:
`ctx.actor` is the only value any production caller hands `create_document*` (exactly one
production call site, `services.py:1190`). The discard re-derives end-to-end (`9ef89d7`,
`_render_speed` at `report.py:913`/`:1098`, `HISTORY.md` S7 items 1–2, 6 tests green, and
`LatencyBlock` counted at 13 by reading). No figure from either entry's `evidence` reached
`SERVER.md`; the graph is exactly as reported; the bookkeeping mechanics all check. No Pass-2 re-gate
needed once the Minor is applied — it is one clause in one sentence.

**CPG: considered, not relevant — the diff is documentation-only but the promoted sentence asserts
five code facts; the brief flagged any CPG as unverified, so each was verified by direct source
reads pinned to `49ba441` (`config.py`, `mcp.py`, `services.py`, `repository.py`, `storefront.py`,
`api.py`, `app.py`) plus a repo-wide `git grep` caller enumeration and the executed model-bench
tests — evidence a possibly-stale graph could not supply.**

### Findings

#### Minor — the sentence's "needs its own additive per-call identity path *alongside* this seam rather than a change to it" is §4.1's design decision, not a code fact, and §1.3 now describes two futures for per-caller identity without saying how they relate

`falkor-chat/docs/SERVER.md` §1.3, working tree L94-99: *"…cannot read it off `ctx.actor` and
**needs its own additive per-call identity path alongside this seam rather than a change to it**,
the shape `Storefront.context_for` (below) already takes…"*. Compared clause-for-clause with
`claude/docs/plans/agent-knowledge-base-strategy.md` §4.1 L300-315 ("add an optional parameter …
**purely additive** … bolted a second, per-caller auth path … **alongside** the same
process-constant `get_context()` seam, **rather than reworking the seam itself**") and §1 L166-169:
the bolded clause is that plan's *chosen* design, with its rationale, not a statement about what the
code does. The disposition record (`claude/architect/kaizen/history.md`, the `a1f3d9c2` bullet) says
SERVER.md "received only the code fact, phrased so it neither anticipates that design" — the facts
around it do meet that bar; this clause does not. It also lands two sentences above §1.3's own
*"When auth lands (token → user + workspace claim…) **only `get_context` changes** — everything
below is untouched"* (L107-109): one sentence says per-caller identity arrives by changing the seam,
the next says it must arrive beside it. `cobb`'s history reconciles them ("auth is the seam's own
future; attribution rides beside it") but the doc itself does not — a reader gets both prescriptions
and no rule. **Fix (that clause only):** make it descriptive of today's code, e.g. "…cannot read it
off `ctx.actor`; the one per-caller precedent is `Storefront.context_for` (below), which builds its
own `CallContext` beside the seam and varies only `actor`" — and let §4.1 keep the "must be
additive" ruling, which is where a design decision belongs per root `AGENTS.md`'s doc-kind rule.

#### Info — `Repository.create_document` has no production caller at `49ba441`; the sentence names it as a peer of the production path

`git grep -n 'create_document(' 49ba441 -- falkor-chat`: the plain method (`repository.py:1036`) is
called from 24 sites, **all under `server/tests/`**; the only production `INGESTED_BY` write is
`services.py:1190` → `create_document_with_auto_supersede` (since document-ingestion2 Stage D, per
its own docstring at `services.py:1155-1163`). The sentence's "already resolve `INGESTED_BY` …
by the id they are handed" is true of both methods (both carry the identical `OPTIONAL MATCH …
coalesce(u, a)` clause, `:1061-1072` / `:1803-1820`), so nothing is false — but "the id they are
handed, today always `ctx.actor`" is vacuous for a method nothing in production hands anything to.
Optional one-word tweak: "`create_document_with_auto_supersede` (the production path) and the plain
`create_document` already resolve…". Tests hand it `u1`/`u2`/`bot1`/`ghost`, which is the
`User`-or-`Agent`-or-missing coverage the clause relies on — worth knowing, not worth promoting.

#### Info (pre-existing, out of U3's scope) — §2.2's tool table does not list the document tools the new sentence now names

`SERVER.md` §2.2 (L755-763) lists seven MCP tools; at `49ba441` `mcp.py` also exposes
`ingest_document` (`:287`), `ingest_documents` (`:323`) and further `@mcp.tool()`s below (`:363`
onward). `grep -n ingest_document falkor-chat/docs/SERVER.md` → the new sentence at L95 is the file's
**only** mention. The cross-reference the sentence makes to §2.2 is for `frm`, which §2.2 does state
(L765-766, "MCP ignores any client-supplied `frm`"), so the citation is correct; the gap is that a
reader following "such as `ingest_document`" to §2 finds no such tool. Not `cobb`'s diff and not a
distiller's job — route to the falkor-chat doc owner as a §2.2 refresh.

#### Nit — one line-number inconsistency in the disposition record

`claude/architect/kaizen/history.md`, `a1f3d9c2` bullet: "`:1804-1820` `create_document_with_auto_
supersede`" then, in the same sentence, "(`:1061-1072`, `:1803-1820`)". The clause opens at
`:1803` (`OPTIONAL MATCH (u:User …`); `:1804` is the `Agent` line. Make both `:1803-1820`.

#### Nit — the appended sentence is ~110 words joined by three semicolons

The paragraph (L86-99) is now the longest prose block in §1.3. Splitting at "the graph side needs
nothing new for it" into its own sentence costs nothing and reads better. Take or leave.

### Gate questions, with what was run

1. **The promoted sentence, clause by clause, at `49ba441`.** `config.py:264-273` `@dataclass(frozen=True)
   class CallContext(ws, actor)`; `:276-284` `get_context() → CallContext(ws=WS_ID, actor=USER_ID)`,
   docstring "MCP ignores any client-supplied `from`". `mcp.py:41` `_get_context = config.get_context`,
   swappable only via `configure(context_provider=…)` (`:147-169`; `app.py:348` `provider =
   context_provider or config.get_context`, `:404` passes it on); `send_message(body, re, mentions,
   frm)` at `:201-213` — `frm` is accepted and **never read** (docstring `:206` "reserved/ignored in
   M1"; body uses `ctx = _get_context()` only); `ingest_document` `:287`, `ctx = _get_context()`
   `:304`, → `Services.ingest_document` `services.py:1136` → `create_document_with_auto_supersede(…,
   ingested_by=ctx.actor)` `:1190-1194`; `ingest_documents` (`services.py:1205`) loops the same
   method; REST `api.py:165/:204` reach the same two service methods. `repository.py:1036`
   `create_document` and `:1741` `create_document_with_auto_supersede` both open with `OPTIONAL MATCH
   (u:User {userId: $ingestedBy}) OPTIONAL MATCH (a:Agent {agentId: $ingestedBy}) WITH … coalesce(u,
   a) AS ingestor` and `CREATE (d)-[:INGESTED_BY]->(ingestor)` (`:1061-1072`, `:1803-1820`);
   `sourceKind` derived from which label bound. `storefront.py:570` `self._ws = config.WS_ID if ws is
   None else ws`, `:678-685` `context_for(pid) → CallContext(ws=self._ws, actor=participant_id)`;
   `:1494` the agent's own ctx, same `ws`. **Caller enumeration** (`git grep` over all of
   `falkor-chat` at `49ba441`): production call sites of `create_document*` = **1**
   (`services.py:1190`, `ingested_by=ctx.actor`); every other `ingested_by=` site (28) is under
   `server/tests/`. §2.2's `frm` bullet confirmed at L765-766 — the cross-reference is correct.
2. **Paragraph coherence, read whole.** U2's antecedent ("every call that resolves through
   `get_context`, which is the legacy `api.py` router and `/mcp`") is precise, so the new clause
   stacks on a true subject. No contradiction with the storefront sentences: "their `actor` is per
   participant, their `ws` this same constant" (L92-93) and "no `/shop/api` request resolves through
   it … the `actor` half is per-request" (L175-179) agree with "the shape `Storefront.context_for`
   (below) already takes for `actor`". "`get_context` is the *only* place `ws` is fixed" (L101) is
   about `ws`; the new sentence is about `actor` — consistent. The one tension is the Minor: the
   normative "alongside … rather than a change to it" vs. L107-109's "only `get_context` changes".
3. **The discard.** `git show --stat 9ef89d7`: 2026-09-17 14:53:50 -0300, touches `report.py` (+68)
   and `test_report.py` (+165); `git show 9ef89d7 -- report.py | grep '^+.*_render_speed'` → the
   `def` and the `lines += _render_speed(runs, arm_names)` wiring; `git log -S _render_speed --
   report.py` → that one commit. At `49ba441`: `report.py:913` `def _render_speed`, `:1098` wired in
   `compare_report` (`:958`); `:917` docstring "prints exactly `LatencyBlock`'s own thirteen".
   `model-bench/docs/HISTORY.md:108` S7 entry, item 1 (L120-121) "prints `RunResult.latency` for the
   first time in this component's history", item 2 (L127-128) names `_render_speed` delivered. Ran
   from `model-bench/` with `.venv`: `pytest -q -k "speed or Speed or latency" tests/test_report.py`
   → **6 passed**, 125 deselected (`TestRenderSpeed` 5 cases at `:2971-3034`, helper `_latency_block`
   at `:2959`, regression `test_regression_existing_guard_judge_report_is_unchanged_and_speed_is_a_
   pure_addition` at `:3035`). **`LatencyBlock` counted by reading** `results.py:274-294` at
   `49ba441`: `latencyMsP50`, `latencyMsP95`, `latencyMsMax`, `latencyTimedCount`,
   `latencyItemCount`, `latencyWithheldForLoad`, `latencyWithheldForNoResponse`, `statsCoveredCount`,
   `callCount`, `ttftMsMedian`, `prefillMsPer1kMedian`, `tokensPerSecondMedian`, `unexplainedMsMax`
   — **13** field lines (`:282-294`); `callAttemptedCount` (`:296`) is a `@property`, not a field,
   and is the likely tenth-vs-thirteenth trap for any regex that anchors on `: ` rather than the
   dataclass body. Matches both history entries and the `:917` docstring.
4. **Figures.** `git diff 49ba441 -- SERVER.md | grep '^+' | grep -o '[0-9]\+'` → only "2" (twice,
   both `§2.2`). "13" appears in the two history logs only; "549", "913", "1098", "14:53" appear in
   the history logs only. Word count `git show 49ba441:… | wc -w` = 8,971 → working 9,076 — exact.
5. **Altitude / doc-kind fit.** Confirmed `agent-knowledge-base-strategy.md` §1 L150-188 ("Two
   costs …") and §4.1 L290-327 hold the finding *and* the design decision, as the record says. Of
   the promoted sentence's clauses, four are present-tense code facts (client self-description
   ignored; `ctx.actor` unreadable as per-caller identity; `context_for` varies only `actor`;
   `INGESTED_BY` resolves `User`/`Agent` by the handed id, today always `ctx.actor`) and one is
   §4.1's ruling (the Minor). The sentence names no uncommitted symbol (`produced_by` absent; `grep
   produced_by SERVER.md` → 0) and describes no working-tree behavior — the concurrent `mcp.py` work
   is neither described nor named, though the Minor's clause anticipates its *shape*.
6. **Bookkeeping.** Architect history header "1 promoted … 1 discarded … 0 kept open" = 2; body: one
   DISCARDED bullet (`a1e6d9f4`), one PROMOTED bullet (`a1f3d9c2`), each id exactly once in each
   history file's added lines and absent at baseline. `git diff 49ba441 -- <history> | grep -c
   '^-[^-]'` = 0 for both (pure insertion; U2 entries untouched). `grep '^## ' | sort | uniq -d` and
   the adjacent-`awk` scan → 0 on all three files. `git diff 49ba441 --stat -- claude/architect/
   kaizen/plan.md` → empty; neither id in `plan.md` at baseline. `cobb`'s history file list (3 files)
   and word delta match the diff; `git status --short` shows exactly those three plus the concurrent
   session's files and this review.
7. **Graph (live, read-only).** `MATCH (:Agent {agentId:'architect'})-[:PRODUCED]->(k)` → **0**.
   Prefix match on `a1e6d9f4`/`a1f3d9c2` → 0 nodes. `e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e` present,
   producers `[]`, `MENTIONS` → `['tico']`. This gate wrote no capture of its own.

### What's solid

- The strongest claim in the sentence — "only that id, today always `ctx.actor`, never varies" —
  is exactly right at `49ba441`, and was checked by enumerating every caller rather than by reading
  the one obvious path.
- The discard was traced to its fix in git, its test, and its `HISTORY.md` record before being
  called fixed, and the field count was enumerated rather than carried — the 09-16 U4 class is absent.
- Home and shape: the sentence lands on the exact antecedent U2 sharpened for it, cross-references
  §2.2 rather than restating it, and rejects the private-KB alternative for the right reason.
- The disposition record is complete and every line number in it re-derives (one off-by-one nit).

### Open questions

None blocking. For `teco`: the Minor is one clause in one sentence of `SERVER.md`; no re-gate needed —
a diff of that sentence suffices. Sequencing note from §U2 stands resolved by `cobb`'s wording
(nothing uncommitted is named), so U3 can commit ahead of the concurrent `mcp.py` work.
