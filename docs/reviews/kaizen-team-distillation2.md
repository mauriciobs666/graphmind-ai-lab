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
