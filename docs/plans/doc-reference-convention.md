# Cross-document reference convention — design & migration assessment

> **Status:** active · **Owner:** `architect` · **Tracks:** C-322 (repo-wide convention)
>
> **Version 1.4 · 2026-07-27 · author `architect` · design-only, nothing implemented.**
> **This is the final design pass. §12 is the execution contract — implementers read §12 and §9.6,
> not this changelog.**
>
> **v1.4 changelog — patch pass answering the `analyst` spot-check** (Part III of
> `docs/reviews/doc-reference-convention.md`, verdict *approve with suggestions*: **no blockers**,
> all six Part II findings verified closed, **3 majors**). **Verification text only — no design
> change, and no decision reopened (D1, D4, D6 and rename-nothing stand).** Three edits:
> **M20** — step 2's M9 coverage check asserted `**Status:**` in six prompts that the plan tells to
> carry a *pointer*, so a correct execution scored 1, not 7. **The indirection is kept and the check
> is fixed** (ruling and reasons in §11.3); the pointer becomes one canonical sentence, byte-identical
> in all six prompts, and the check greps for that sentence. **M21** — step 1's `git diff --stat`
> is corrected to **8 files** (`claude/README.md` is re-checked, not edited), and the *"zero `*.md`
> path-string edits"* invariant is now **defined** — a repath of an **existing** citation — with a
> command that implements the definition instead of one that cannot measure it. **M22** — the
> surviving *"completed plan documents move to `archive/`"* preamble at
> `falkor-chat/docs/BACKLOG.md`:5 is added to **step 4**, which already opens that file. Full
> dispositions in **§11.3**.
>
> **v1.3 changelog — stakeholder rulings + response to the `analyst` re-review** (Part II of
> `docs/reviews/doc-reference-convention.md`, verdict *needs changes*: 2 blockers · 4 majors ·
> 10 minors listed under a "9 minors" verdict line — all 16 answered, with the finding-ID →
> disposition map extended in **§11.2**). What changed:
>
> 1. **D1 is RULED: no clickable links.** References are plain backticked repo-root paths; the
>    composed form is *permitted, never required*, and the plan no longer *recommends* it anywhere.
>    Every construct that existed only to keep the mandate open is deleted — D1's option (b), the
>    "conditionally recommended in index documents" carve-out, the "S4 is a precondition" coupling,
>    and §6.1's "what would change the answer". **S4 is re-priced on its own merits** (§3.3, D2): it
>    detects Finding R2's 408 unresolvable references and archival rot; it was never needed for the
>    3 real broken links, which S2 deletes outright.
> 2. **D6 is RULED: adopt the naming convention**, in full, as recommended.
> 3. **D2 and D3 are taken by my stated defaults** and recorded as such — **D2 = (b)** report-only
>    script, no CI gate; **D3 = (a)** retire the 442 and replace it with nothing numeric. **D8
>    confirmed** unchanged (prompt-level enforcement only).
> 4. **B4 — the canonical header form is now stated verbatim, once** (§9.6). Labels are **bolded**;
>    `tico`'s two *values* remain byte-identical. N2c bolds `claude/tico/tico.md`:37's labels;
>    `tico.md`:71 and `claude/README.md`:8 stay genuinely untouched, proved by a **content**-matched
>    check (m15) rather than a line-number one.
> 5. **B5 — `teco` coordinates the `Status:` flip; each kind's owner performs it.** This matches the
>    existing `PreToolUse` guard topology exactly and needs no hook edit. **The same reasoning is
>    applied one step further than the reviewer took it: S2/N4 are re-owned from `teco` to an
>    implementer**, because `teco`'s guard allowlists `docs/plans/*` only and S2 writes
>    `docs/HISTORY.md`, `falkor-chat/docs/HISTORY.md` and `falkor-chat/docs/BACKLOG.md`.
> 6. **M6 / M7 / M8 — the three rule-5 / naming seams are closed with single answers** (§9.4, §9.5):
>    a sharpened branch selector that does not depend on a dropped token; header pointers declared
>    **metadata, not amendment**, with `Extends:` ⇄ `Extended by:` split from `Supersedes:` ⇄
>    `Superseded by:`; and one legal name for a milestone-scoped topic — the token goes **inside the
>    slug, never as a prefix**.
> 7. **M9 — the header gets a creation-time owner.** `cobb` adds one template line to the **six**
>    producing prompts that carry a docs-tree write path (`architect`, `analyst`, `qa-engineer`,
>    `teco`, `data-scientist`, `graph-dba`); `tico` already has one and N2c updates its form. This is
>    two prompts more than the reviewer priced, and §11.2 states why.
> 8. **New step S6 (m19)** — normalise the **16** module-anchored backticked `docs/…` refs in
>    `falkor-chat/AGENTS.md`, as its own step so **S1 keeps its zero-path-string-edit proof**.
> 9. **§12 is new**: the final numbered, ordered step list — owner, files, done-condition and a
>    self-verifying check per step — plus the per-document `Status:`/`Owner:`/`Tracks:` value table
>    (m16) that converts N3's 25 judgement calls into a reviewable list.
>
> **v1.2 changelog — response to the `analyst` review** (`docs/reviews/doc-reference-convention.md`,
> verdict *needs changes*: 3 blockers · 5 majors · 9 minors). Every finding is answered, with a
> finding-ID → disposition map in the new **§11**. The five substantive changes:
>
> 1. **B1 — the broken-link diagnosis was false and is corrected.** The 3 links point at documents
>    that **exist**; the defect is an extra `../`. All of them — in fact **4 of 4** repo-wide — are
>    the **composed** form this plan recommended. **The composed-form mandate is dropped** (D1
>    restated and shrunk).
> 2. **B2 — the grammar had no name for a second primary document.** §9.5 rule 5 is rewritten: the
>    ordinal moves from the role token to the **slug** (`executor2.md`), which *simplifies* the
>    grammar rather than extending it. `m3-executor-landing2.md` is re-derived as an **instance** of
>    that branch, not a failure of the default.
> 3. **B3 + M2 + M5 — the header is cut to three fields and the value set absorbs `tico`'s.**
>    `Updated:` is demoted to optional; `draft`/`delivered` are dropped; `Interviewing` and
>    `Ready for design` are absorbed **verbatim**, so `tico`'s stakeholder-gated transition is
>    untouched. The prompt-change scope is corrected from "two sentences" to **four prompts + root
>    `AGENTS.md`**.
> 4. **M1 — the naming rule is now a prohibition, not just a grammar**, and `qa-engineer.md`:28 is
>    rewritten rather than trimmed. One third of M1's suggestion is **rejected with evidence** (§11).
> 5. **M4 + D3 — the unreproducible numbers are dropped, not replaced.** `HISTORY` records only what
>    a committed one-liner regenerates. The reviewer's **4** broken links is confirmed by my own
>    re-measurement, and I found a further hazard the review did not: committing *this* document adds
>    **10** more "broken links", all illustrative (§1.3).
>
> **Reviewer-endorsed and deliberately unchanged:** D4 (documents stop moving), rename-nothing
> (§10.1), the §9.7 write-guard analysis, the S5 deferral, and the rejections of the bulk repath,
> flattening (B) and ID-based names (C). The measurement layer reproduced to the exact changed line
> and is not re-litigated.
>
> **v1.1 changelog.** Stakeholder ruled on §6: **D4 accepted** (documents stop moving; `Status:`
> marker in the existing header; existing `archive/` trees freeze as read-only history; no
> un-archiving). **D1 still open** — a remote *does* exist (`origin` →
> `github.com:mauriciobs666/graphmind-ai-lab`) but readership is unknown; D1 is restated below with
> an explicit both-consumers answer. **S5 (bulk repath) stays deferred.** New requirement, verbatim:
> *"we need to come up with a better naming convention — M-something is not good for longer."*
> Answered in **§9** (the convention) and **§10** (its migration). §§1–4 and 7 are unchanged and
> remain the evidence base; §5 and §6 are amended in place. One v1.0 arithmetic correction:
> §2.3 says *"33 files in `falkor-chat/docs/archive/`"* — the verified count is **34** (plus 2 in
> `docs/archive/`); the argument is unaffected.
>
> Trigger: three commits on 2026-07-26 (`c38f7dc`, `62714e8`, `9bbfbb5`) in which archiving **two**
> documents cost 22 path-string edits across 8 files. Stakeholder question, verbatim: *"the
> references to the docs/ and docs/archive are costing too much rework, how can we use a shorter
> relative path? which agents should we change? how to migrate the project?"*
> Scope note: this document assesses a **repo-wide documentation convention**; it is filed at the
> repo-root `docs/plans/` per the cross-component rule in `claude/architect/architect.md`.

---

## 0. Bottom line up front

**The premise is measurably wrong, and that is good news — the fix is much cheaper than a
migration.**

I reproduced the cost of both archival sweeps and classified every path string they touched.
Depth-sensitive relative paths (`../` → `../../`) accounted for **6 of 22 edits (27%)** in the cpg
sweep and **2 of ~157 edits (1.3%)** in the larger falkor-chat sweep — **8 of 179 across both,
4.5%**. The other 95.5% were **backticked, already-depth-independent path strings** that had to
change because the *target moved*, not because the citing file's depth changed. No path-spelling
convention, however short, avoids that class.

> **A shorter relative path buys ≤5% of the observed rework.** The cost driver is that the path
> **encodes lifecycle state** (`docs/plans/` vs. `docs/archive/plans/`), so closing a milestone
> mutates the middle segment of every inbound reference.

Two things eliminate the cost class rather than shaving it:

1. **Stop moving documents.** Mark a frozen document `Status: archived` in its own header and leave
   it where it is. Cost of a future "sweep": one line per document. Cost to adopt: **zero file
   moves, zero repaths** — existing `archive/` trees stay exactly as they are and become read-only
   history.
2. **Standardise the citation spelling on the repo-root-anchored backtick**, forward-only, because
   it fixes a *separate*, verified defect: **408 references in this repo are not resolvable by an
   agent reading from the repo root** (they are silently module-root-anchored). This is not about
   archiving; it is about agents being able to follow a citation at all.
   **A markdown link is permitted and never required** — **ruled, D1, 2026-07-27.**

**Recommendation: do #1 and #2. Do not migrate the 687 existing non-root-anchored references.** The
full conversion is a ~60-file mechanical-plus-judgement sweep buying ≤5% of future archival churn
on references that mostly point into frozen history. It costs more than it saves.

> **v1.3 (D1 ruled, 2026-07-27) — NO CLICKABLE LINKS.** The stakeholder's ruling closes this:
> **a cross-document reference is a plain backticked repo-root path.** The composed form
> ``[`falkor-chat/docs/QUERIES.md`](../QUERIES.md)`` is **permitted** — the existing 143 uses are not
> a defect and are not swept — but it is **never required and no longer recommended anywhere**,
> including in index documents. Nothing in this plan obliges anyone to write a path twice, and no
> step is contingent on that obligation. Option (b) (mandate + ship S4 with it) is **withdrawn**;
> **S4 is now justified solely by Finding R2** (§3.3, D2).
>
> **v1.2 correction (B1) — the one recommendation this review reversed.** v1.1 recommended
> *mandating* the composed form ``[`falkor-chat/docs/QUERIES.md`](../QUERIES.md)`` where a link adds
> navigational value, citing its 143 existing uses as proof it was "not an invention". That citation
> was self-defeating: **every broken relative link in this repo is that form** (4 of 4; 3 real
> defects plus 1 illustration — §1.1 R3). Writing a path twice creates two things that can disagree,
> and the only mechanism that detects the disagreement (S4) is optional and deferred. **v1.2 drops
> the mandate.** The composed form remains *permitted* — 143 uses are not a defect to sweep — but
> nothing requires it, so nothing requires anyone to write a path twice.

**Total recommended change (v1.3, final): root `AGENTS.md` + `falkor-chat/AGENTS.md` + 6 agent
prompts (~1 paragraph and ~8 sentences of prose), 25 one-line document headers, 3 deleted `../`
tokens, 16 module-prefix normalisations in one file. No file moved, nothing renamed, no path
repathed.** Plus one optional report-only script.

**The shipping set, all decisions now ruled:** **D4** (accepted) + **D6** (accepted) + the
**filename grammar** (§9.2–§9.5) + **S1–S3 + S6** + **N1–N4**, of which **N3** (the 25-document
`Status:` normalisation) is the single highest-value step — with D4 accepted, `Status:` is the
*only* lifecycle signal, and today it covers 9 of 26 active documents in 9 vocabularies while
looking complete.

> **Execution contract: §12.** Numbered, ordered, one owner per step, files listed, done-condition
> and a self-verifying check each. §9.6 states the canonical header block verbatim. An implementer
> needs those two sections and nothing else.

---

## 1. What I measured, and how

All numbers below are mine, produced by a read-only census I wrote for this assessment (scratch
script; not committed). Method, stated so it can be reproduced or disputed:

- Walked all **179 `*.md` files** outside `.git`, `.venv`, `node_modules`, `.cpg-artifacts`.
- Masked fenced code blocks. Extracted, per line: markdown links (`[text](target)`), and
  backticked tokens that look like a path (contain `/` **or** end in a known file extension).
- Resolved every reference **three ways**: relative to the citing document, relative to the repo
  root, relative to the citing file's top-level directory ("module root"). A reference is *dead*
  only if **none** resolve.
- Separately regex-scanned for **unbackticked, unlinked prose** mentions of a real `.md` filename.
- For both archival commits, diffed the pre-move against the post-move content of every renamed
  file to isolate the **outbound** edits, and classified every changed token by spelling
  (`../` / `./` / `docs/…` / bare / leading-slash) and container (markdown link / backtick / prose).

### 1.1 The reference census

| Spelling | Count | Share |
|---|---:|---:|
| Backticked path string (`` `docs/plans/x.md` ``) | 2,179 | 86.3% |
| Backticked path string *inside* a link label (``[`../x.md`](../x.md)``) | 143 | 5.7% |
| Relative markdown link, explicit (`./x.md`, `../x.md`) | 185 | 7.3% |
| Relative markdown link, bare (`x.md`) | 18 | 0.7% |
| Repo-root markdown link (`/docs/x.md`) | **0** | 0% |
| **Total references to a `.md` document** | **2,525** | |

**Finding R1 — the repo does not use markdown links to cite documents; it uses backticked
strings.** 92% (2,322) of document references are backticked path strings; only 203 are markdown
links. **A link checker validates 8% of this repo's references and is blind to the other 92%.** This
is the single most decisive fact in the assessment, and it reframes both the cost problem and the
tooling answer.

**Finding R2 — there are two silently competing anchoring conventions.** Of the 2,322 backticked
`.md` references:

| Resolves as | Count | Agent `Read` from repo root |
|---|---:|---|
| Repo-root-anchored only | 652 | ✅ works |
| **Module-root-anchored only** | **408** | ❌ **fails** |
| Both (root-level docs, ambiguous) | 333 | ✅ works |
| Citing-file-relative only | 279 | ❌ fails |
| Resolves nowhere | 650 | ❌ fails |

The 408 are concentrated in `falkor-chat/docs/HISTORY.md` (63), `falkor-chat/docs/BACKLOG.md` (58),
`falkor-chat/docs/archive/plans/m3-executor-coordination.md` (22), `falkor-chat/AGENTS.md` (13). A
citation reading `` `docs/QUERIES.md` `` inside `falkor-chat/docs/HISTORY.md` means
`falkor-chat/docs/QUERIES.md`; an agent handed that string verbatim and calling
`Read("docs/QUERIES.md")` gets the repo-root `docs/` tree instead — and there is no
`docs/QUERIES.md`, so it fails. **This is a live correctness defect independent of archiving.**

**Finding R3 — CORRECTED IN v1.2 (blocker B1). The v1.1 diagnosis was false, and correcting it
reverses this section's most decision-bearing conclusion.**

v1.1 stated that the broken links pointed at *"three never-created `workflow-def-structure-read` /
`k027-parse-robustness` docs"*. **All three targets exist.** Verified:

```
$ ls falkor-chat/docs/reviews/workflow-def-structure-read.md \
     falkor-chat/docs/plans/workflow-def-structure-read.md \
     falkor-chat/docs/reviews/k027-parse-robustness.md
(all three present)
```

The defect is an **extra `../`**. `falkor-chat/docs/BACKLOG.md`:785, 787, 895 each read
`` [`docs/reviews/…`](../reviews/…) `` where, from a file already inside `falkor-chat/docs/`, the
correct target is `reviews/…`. **Three deleted tokens, no judgement.** This is not a
never-built-deliverable question; **D5 collapses to a one-line fix folded into S2.**

| Class | Count (v1.2, re-measured) |
|---|---:|
| **Broken relative markdown links, tracked files** | **4** (v1.1 said 3 — see §1.3) |
| — of which real defects | **3** (`falkor-chat/docs/BACKLOG.md`:785, 787, 895) |
| — of which illustrative | **1** (`claude/architect/kaizen/inbox.md`:201 → `../relative.md`) |
| **Dead path-bearing backticked `.md` citations** | ~87 — **not reproducible without the uncommitted script; retired, see §1.3 and D3** |

> **The consequence that matters — this is direct evidence against the form v1.1 recommended.** All
> four broken links are the **composed** spelling (backticked repo-root label + relative target,
> the path written twice on one line), and in each the two halves disagree. The population:
>
> ```
> $ git grep -ohE '\[`[^`]+\.md`\]\([^)]+\)' -- '*.md' | wc -l
> 143
> ```
>
> **143 composed references; the form accounts for 100% of the repo's broken-link baseline, at a
> measured drift rate of 3/143 ≈ 2%.** §6.1 v1.1 cited those same 143 as evidence *for* the form
> (*"so it is not an invention"*) without noticing that R3's own number **is that form failing**.
> The 4th is `claude/architect/kaizen/inbox.md`:201 — my own v1.0 inbox note, illustrating the
> composed form with a deliberately fake target. Even the illustration of the form is a broken link.
>
> **Therefore: the composed-form mandate is dropped (D1, §6.1).** A convention that requires writing
> every path twice must ship with the checker that detects the two halves disagreeing; S4 is
> optional and gated behind D2, so under the recommended sequence the mitigation never lands. The
> honest options are *mandate the form **and** build S4*, or *don't mandate the form*. v1.2 takes
> the second, which is also the cheaper one — and the stakeholder is cost-sensitive.

**Finding R4 — the archival convention works inside a module and fails across modules.** Of the 87
dead citations, **15 point at a pre-move path whose archived twin exists** — i.e. missed archival
repaths. **All 15 are backticked. Zero are markdown links.** Location:

| File | Count | Character |
|---|---:|---|
| `claude/teco/kaizen/k001-run-brief.md` | 3 | point-in-time record |
| `claude/analyst/kaizen/plan.md` | 2 | **forward-looking** |
| `claude/teco/kaizen/history.md` | 2 | point-in-time record |
| `falkor-chat/docs/reviews/m3-archive-sweep.md` | 2 | deliberately quoting old paths |
| `claude/{analyst,architect}/kaizen/history.md` | 2 | point-in-time record |
| `claude/architect/kaizen/inbox.md` | 1 | point-in-time record |
| `claude/architect/kaizen/plan.md` | 1 | **forward-looking** |
| `claude/data-scientist/kaizen/inbox.md` | 1 | point-in-time record |
| `claude/qa-engineer/kaizen/history.md` | 1 | point-in-time record |

The analyst predicted this exactly, in `falkor-chat/docs/reviews/m3-archive-sweep.md` observation
O-2, and correctly ruled it out of that sweep's scope. Only **3** of the 15 are in forward-looking
documents and worth fixing; the rest are dated records where the pre-move path is arguably *correct
as written*.

**Finding R5 — prose mentions are a small, mostly-catchable class.**

| Class | Count | Automatically catchable? |
|---|---:|---|
| Path-bearing prose mention (`docs/requirements/x.md`, unbackticked, unlinked) | **24** (7 into a `docs/` tree) | ✅ yes — regex `\w+(/[\w.-]+)+\.md` outside fences |
| Bare-filename prose mention naming a real doc (`AGENTS.md`, `QUERIES.md`) | 367 | ❌ no — no directory, so the target is genuinely ambiguous |

The 367 are dominated by `AGENTS.md` (124), `QUERIES.md` (66), `SKILL.md` (25) — almost all generic
references to a *convention* ("per the module-doc convention in AGENTS.md"), not path citations.
They are not rot and should not be converted.

### 1.2 The cost of the two sweeps, decomposed

**Sweep A — `9bbfbb5`, 2026-07-26, cpg module: 2 documents archived, 22 path strings, 8 files.**

Outbound (inside the moved pair), verified by content-diffing old vs. new blobs — **9 strings**:

| Edit | Count | Would a repo-root spelling have avoided it? |
|---|---:|---|
| `../plans/…` → `../../plans/…` (3 in test-plan, 3 in report) | **6** | ✅ **yes** — pure depth churn |
| Self-citation `docs/test-plans/…` → `docs/archive/test-plans/…` | 2 | ❌ no — the file moved |
| `requirements/…` → `docs/requirements/…` normalisation | 1 | ❌ no — was a pre-existing anchoring bug |

Inbound — **13 strings** across `docs/BACKLOG.md` (1), `docs/HISTORY.md` (2),
`docs/plans/cpg-query-access.md` (4), `docs/plans/cpg-query-access-coordination.md` (2),
`docs/requirements/cpg-query-access.md` (1), `docs/reviews/cpg-mcp-containerization.md` (3). My
classification of the repathed tokens: **13 of 17 detected tokens (76%) were backticked
`docs/…`-anchored strings** — already depth-independent, changed solely because the target moved.
**No path convention avoids any inbound edit.**

> **Sweep A verdict: root-anchoring saves 6 of 22 edits (27%).**

Note also that each depth-sensitive citation is physically written **twice** —
``[`../../plans/x.md`](../../plans/x.md)`` — so every one of those 6 is two textual occurrences on
one line. That doubling is itself a small cost multiplier worth knowing about.

**Sweep B — `649b02a`, 2026-07-22, falkor-chat: 20 documents archived, 157 path strings, 26 files.**

I content-diffed all 20 rename pairs: **184 changed lines inside the moved documents**, of which
**2 lines involved a `../` relative path.** Every other outbound edit was a backticked
module-root-anchored `docs/…` string whose target had moved.

> **Sweep B verdict: root-anchoring saves 2 of ~157 edits (1.3%).**

**Combined: 8 of 179 edits (4.5%).** Sweep B is the better predictor of a real milestone-close
sweep — the more documents move together, the more the cost is dominated by target-moved edits and
the less by depth churn.

### 1.3 Reconciling the "442 unresolved backticked strings" baseline

Commit `9bbfbb5` records **442** unresolved backticked strings, identical before and after. I could
not reproduce that number and I believe it is a **methodology artifact, not a defect count**: the
script described in that message resolved *"every backticked repo-root `*.md` path"* — i.e. against
the repo root only. Under that single-strategy resolution, all **408 module-root-anchored**
references (Finding R2) count as unresolved, plus the genuinely dead ones — which lands in the right
neighbourhood.

**v1.2 (major M4) — the same criticism applies to my own numbers, and I accept it.** The reviewer's
objection is exact: D3 asks the stakeholder to retire the 442 on the grounds that *"a number nobody
can decompose cannot be driven down"*, while S2 writes **3 / 87 / 15** into `HISTORY.md` — three
numbers produced by a script I did not commit, i.e. **today exactly as unreproducible as the 442**.
Replacing an unauditable baseline with an unauditable baseline is not progress.

**Re-measured, and the reviewer is right about the count.** I re-ran an independent pass over
`git ls-files '*.md'` (fenced blocks masked, `<>*{}` placeholder targets skipped, targets resolved
relative to the citing file):

- **4** broken relative links in tracked files, not 3. The fourth is
  `claude/architect/kaizen/inbox.md`:201 → `../relative.md`. v1.1's **3** was defensible only under
  an "exclude illustrative paths" rule that **v1.1 never stated** — the 442's failure mode, at
  smaller scale.

**And a hazard neither version caught: this plan is itself a link-checker landmine.** The same pass
over the *untracked* tree flags **10** more broken links, **all of them inside this document** —
`../x.md`, `/docs/x.md`, `docs/plans/x.md`, `../QUERIES.md` — every one an illustration of a
spelling the plan is *about*. The moment this plan is committed, a naïve checker's baseline jumps
from 4 to 14 and every new entry is a false positive.

> **Consequence for any future checker (feeds D2/S4): documents that discuss link syntax must be
> excluded by an explicit, stated rule, or the metric measures prose about links rather than links.**
> A defensible rule: skip any target whose basename is a placeholder token (`x.md`, `relative.md`,
> `<slug>.md`) — but it must be *written down*, because it is precisely what makes "4" a choice
> rather than a fact.

**Recommendation (D3, restated): retire the 442 — and do not replace it with 87.** The 442 conflates
"written in the other anchoring convention" with "points at nothing". The 87 and the 2,525-reference
census are honest analysis numbers and belong **in this document, attributed to an uncommitted
scratch script**; they do **not** belong in `HISTORY.md`, where they would become permanent claims no
committed artifact can regenerate. `HISTORY` records only what a **one-line command reproduces**:

| Recorded in `HISTORY` | Regenerated by |
|---|---|
| broken relative links (4 → **1** after S2) | the census script, *if* D2 green-lights S4 — otherwise omit the number |
| composed-form references: **143** | `git grep -ohE '\[`[^`]+\.md`\]\([^)]+\)' -- '*.md' \| wc -l` |
| milestone-prefixed docs: **36** | `find . -path '*/docs/*' -name '*.md' \| grep -cE '/m[0-9]-'` |
| documents lacking a canonical `Status:`: **18 → 0** | the §10.3 N3 done-condition loop |

Everything else in the S2 entry is **qualitative**: two anchoring conventions coexist; archival rot
is confined to dated records; the composed form is the repo's only source of broken links.

### 1.4 Who actually consumes these references — verified, not assumed

The repo **is** pushed to a GitHub remote: `origin git@github.com:mauriciobs666/graphmind-ai-lab.git`,
`main` at `9bbfbb5`, **0 ahead / 0 behind**. So the GitHub renderer is a real *potential* consumer.
Whether the stakeholder actually reads these docs on github.com is a question only they can answer —
it decides between two sub-options below (see **D1**).

| Spelling | Agent (`Read`/`Grep`, cwd = repo root) | GitHub renderer | VS Code preview / ctrl-click |
|---|---|---|---|
| `` `docs/plans/x.md` `` (backtick, repo-root-anchored) | ✅ **verified**: `Read("docs/BACKLOG.md")` succeeds | n/a — not a link, renders as code | n/a |
| `` `docs/QUERIES.md` `` (backtick, module-root-anchored) | ❌ **verified**: no such path from root | n/a | n/a |
| `[t](/docs/x.md)` (leading slash) | ❌ **verified**: `Read("/docs/BACKLOG.md")` → *"File does not exist"* — a leading `/` is filesystem-absolute | ✅ **documented**: *"Links starting with `/` will be relative to the repository root"* | ✅ resolves to **workspace-folder** root (documented as-designed) — ⚠️ but VS Code's markdown *validation* flags root-relative links as missing in files using the `SKILL.md` language mode (open bug, microsoft/vscode#299488); preview and navigation still work |
| `[t](docs/x.md)` (bare, no `./`) | ✅ works if cwd = repo root | ❌ resolves relative to the **citing file's** directory | ❌ same |
| `[t](../plans/x.md)` (relative) | ⚠️ works, but the agent must compute the join itself | ✅ | ✅ |

**Verified vs. inferred, explicitly:**

- **Verified by execution:** `Read("/docs/BACKLOG.md")` fails; `Read("docs/BACKLOG.md")` succeeds.
  This is the load-bearing constraint.
- **Verified from official docs:** GitHub's leading-slash = repo root, and `./`/`../` = relative to
  the current file (GitHub Docs, *Basic writing and formatting syntax*). VS Code's leading-slash =
  workspace root, and the `SKILL.md`-language-mode validation wart (microsoft/vscode#299488, open).
- **Inferred, not verified:** that OpenCode's and Kiro's file-read tools behave like Claude Code's
  (leading slash = filesystem-absolute). This is near-certain — it is POSIX path semantics, not a
  harness choice — but I did not execute it in those harnesses.
- **Unknown, stakeholder-only:** whether anyone reads these documents in the GitHub UI.

> **Conclusion C1 — there is no single spelling that both an agent's file-read tool and a markdown
> renderer follow as a repo-root path.** `/docs/x.md` is renderer-correct and agent-broken;
> `docs/x.md` is agent-correct and renderer-broken. **Therefore a "shorter relative path" of the
> `/docs/…` form is not available to us.** Any convention must either serve one consumer, or
> separate the citation (for agents/grep) from the navigation link (for renderers).

---

## 2. Question 1 — "How can we use a shorter relative path?"

### 2.1 The options, with what each actually saves

Measured against the 179 real edits of sweeps A + B.

| # | Option | Eliminates | Share of the 179 | Cost to **adopt** | Cost to **live with** |
|---|---|---|---:|---|---|
| **A** | **Repo-root-anchored backtick citation** — `` `falkor-chat/docs/plans/x.md` ``, always from the repo root | outbound depth churn | 8 (4.5%) standalone — **0 on top of D** (see below) | ~0 if forward-only; ~690 edits / ~60 files if migrated | **very low** — one rule, and 652 refs already comply |
| A′ | Root-slash markdown link — `[x](/docs/plans/x.md)` | same as A | 8 (4.5%) | 203 link rewrites | **prohibitive** — **verified to break agent `Read`** (§1.4). Rejected. |
| **B** | Flatten the trees (fewer levels, so shorter `../`) | some depth churn | ≤8 (4.5%) | high — restructures every module's `docs/` | low, but loses the plans/reviews/requirements separation that agent prompts encode |
| **C** | Stable IDs + one lookup table — cite `C-320` / `K-022` / the slug, resolve via a table | inbound *and* outbound edits | ~171 (95%) — but replaces them with table maintenance | **very high** — 2,300+ references rewritten, and IDs must be minted for docs that have none | **high** — every reader takes two hops; an agent must read the table before it can read the doc. Not all docs have an ID today (`m3-archive-sweep.md`, `local-model-ram-budget-ml.md`, all four `cpg-analysis` references). |
| **D** | **Don't move documents** — `Status: archived` marker in the document, file stays put | **all of it** | **179 (100%)** | **~0** — no moves, no repaths; existing `archive/` trees frozen as-is | **~0** — one line when a milestone closes |
| **E** | Tooling — a resolution checker | 0 edits, but converts silent breakage into a visible failure | 0 | ~1 script (~80 lines) | low if report-only; medium if it gates |

> **v1.2 (minor m2) — A and D do not add up, and the table used to imply they did.** Under D nothing
> moves, so the 8 depth-churn edits A eliminates are **already inside** D's 179. **A's incremental
> archival saving on top of D is exactly 0.** A is recommended *solely* on Finding R2 — the 408
> references an agent cannot resolve — which is a live correctness defect and has nothing to do with
> archiving. The prose at the end of §2.2 always said this; the table now says it too, so a skim
> can't over-sell A.

### 2.2 Recommendation: **D + A (forward-only) + E (optional)**

**D is the answer to the stakeholder's actual problem.** It is the only option that removes the cost
class instead of shaving a slice, and it is the *cheapest to adopt* **and** the *cheapest to live
with* — an unusual combination that should make the decision easy.

Concretely: when a milestone closes, the document gains a marker in its existing header block, e.g.

```
> **Status: archived 2026-07-26 — M3 closed.** Frozen record; do not execute or amend.
```

and **nothing moves**. Zero inbound references change. Zero outbound references change.

**Why D is not a downgrade in the thing `archive/` was actually for.** Per
`falkor-chat/docs/HISTORY.md` (the 2026-07-05 consolidation entry) and root `AGENTS.md`, the purpose
is to stop a reader treating a frozen document as current. A directory name does that *only for a
reader who browses the directory*. A status line does it *for a reader who arrives via a link* —
which is how agents arrive, always. The evidence is in the trigger itself:

- Commit `9bbfbb5` deliberately chose **"no ARCHIVED banner stamped into the moved documents"**. So
  today, an agent handed `docs/archive/test-plans/cpg-query-access.md` and reading it sees **nothing
  in the document** saying it is frozen. The signal lives only in a path segment the agent may never
  have inspected.
- Review finding **m-26** is precisely that failure mode: the test plan records a wiring
  (`run.sh`) that no longer exists, and *"a QA engineer re-running this plan would set up an
  environment that no longer exists."* The chosen remedy was to **move** the file — but moving it
  does not stop a reader who follows a link from executing it. A status banner does.

So **D is strictly better at the convention's own stated goal**, at a fraction of the cost. That is
the trade-off that decides it.

**The honest cost of D — discoverability of the active set.** `ls docs/plans/` currently answers
"what is live?" in one command. Under D you need a negation grep:
`grep -L 'Status: archived' docs/plans/*.md`. That is a real loss. Mitigations, in order of
strength:

1. `docs/BACKLOG.md` (what is live) and `docs/HISTORY.md` (what is delivered) **already** index this,
   authoritatively and by design. The directory listing is a redundant, weaker index.
2. falkor-chat already prefixes slugs with the milestone (`m3-executor.md`), so frozen work already
   sorts together.
3. One line in each module's `AGENTS.md` giving the grep.

**A, forward-only, rides along** — not because of archiving (it buys 4.5% there) but because
Finding R2 is a live defect: **408 references an agent cannot follow.** The rule is one sentence, it
standardises on the *plurality* spelling (652 refs already comply) rather than inventing anything,
and it costs nothing to adopt because it applies only to new text.

**Spelling to adopt (recommended):**

> A citation of another document is a **backticked path from the repo root** —
> `` `falkor-chat/docs/QUERIES.md` ``, never `` `docs/QUERIES.md` `` from inside `falkor-chat/`, and
> never a bare filename when the repo has more than one file with that name. A markdown link is
> **optional and never required**; when you add one, its target must be **relative**
> (`./x.md`, `../plans/x.md`) — never `/docs/…`, which agents cannot resolve.

**v1.3 (D1 RULED): the composed form is permitted, never required, and not recommended.**
``[`falkor-chat/docs/QUERIES.md`](../QUERIES.md)`` serves every consumer — repo-root label for
agents and grep, relative target for renderers — and 143 references already use it. But it writes
the path **twice**, and this repo's only measured evidence about it is that the two halves drifted
in 3 of 143 cases, producing **100% of the repo's broken links** (§1.1 R3). The stakeholder's
ruling settles it:

- **Never mandated, and no longer recommended anywhere** — including index documents. v1.2's
  "conditionally recommended in `README`/`AGENTS`/`BACKLOG`/`HISTORY`" carve-out is **deleted**: it
  existed only to keep the mandate question open, and the question is closed.
- **Never swept.** The existing 143 are not a defect and are not migrated. A composed reference an
  author chooses to write must still carry a **repo-root label** and a **relative target**.
- **No step depends on it.** S4 is no longer coupled to this decision (§3.3, D2).

> **The rule an agent follows, in one sentence:** *write the reference as a backticked repo-root
> path. That is the whole convention. If you want a clickable target you may add one and it must be
> relative — but nothing asks you to.*

### 2.3 Options rejected, and why

- **A′ (`/docs/…` links)** — the most natural reading of "shorter relative path", and **verified
  broken** for the primary consumer. This is the single most important negative result here: had it
  been adopted on renderer-compatibility grounds, it would have quietly broken every agent
  handoff-by-path.
- **B (flatten)** — buys at most the same 4.5% as A, at restructuring cost, and the
  `plans/reviews/requirements/test-plans/test-reports` split is hard-coded into 7 agent prompts
  (§3.1). Fighting the grain for no measurable gain.
- **C (IDs)** — the theoretically correct answer, and the wrong one here. It targets the right 95%,
  but only by moving the cost from "repath N references" to "maintain a table + every reader takes
  two hops + mint IDs for the ~40% of docs that have none". D achieves the same 95%+ at zero cost.
  Keep C on the shelf for the day the repo has enough documents that the status-marker grep stops
  scaling.
- **Un-archiving the existing `archive/` trees** — **34** files in `falkor-chat/docs/archive/` + 2
  in `docs/archive/` (v1.2, minor m4: the body still said 33; re-verified **34**). Reversing them is
  another 157-edit sweep, i.e. paying the exact cost we are
  trying to abolish, to buy tidiness. **Explicitly rejected.** Existing `archive/` trees stay and
  are redesignated read-only history.

---

## 3. Question 2 — "Which agents should we change?"

### 3.1 Where the convention is encoded today — grepped, not assumed

**The normative source** is root `AGENTS.md` lines 159–166, the *"Module documentation convention"*
bullet. The operative sentence is:

> *"…`archive/<same-subdir>/` for frozen ones — a doc moves to `archive/` when its milestone closes,
> with inbound links fixed in the same change."*

That clause is the entire cost driver, in one place. **This is the primary edit.**

**Agent prompts mentioning the docs tree** (`grep -cE 'docs/(plans|reviews|requirements|test-plans|test-reports)|archive/'`,
excluding `kaizen/`):

| Agent definition | docs-tree hits | mentions `archive/` |
|---|---:|---:|
| `claude/teco/teco.md` | 8 | 0 |
| `claude/analyst/analyst.md` | 5 | 0 |
| `claude/architect/architect.md` | 4 | 0 |
| **`claude/qa-engineer/qa-engineer.md`** | 3 | **1** |
| `claude/tico/tico.md` | 3 | 0 |
| `claude/data-scientist/data-scientist.md` | 3 | 0 |
| `claude/{coder,tdd-engineer,frontend-engineer,graph-dba,joern}/…` | 1 each | 0 |
| `claude/{cobb,devops}/…` | 0 | 0 |

> **Finding A1 — `qa-engineer` is the *only* agent whose prompt encodes the `archive/` move rule.**
> `claude/qa-engineer/qa-engineer.md:28`: *"Completed-milestone docs live in
> `docs/archive/<same-subdir>/` — write new plans to the active dirs, never into `archive/`."*
> Every other agent references only the *active* write paths, which **option D does not change**.

> **Finding A2 — no agent prompt says how to spell a cross-document reference.** I grepped all 15
> agent definitions plus every `AGENTS.md` and `SKILL.md` for `relative (link|path)`,
> `markdown link`, `repo-root`, `leading slash`: **zero hits stating a citation convention.** All
> the prompts say is where an agent *writes its own* file. **The reference convention is entirely
> tacit** — which is exactly why two anchoring conventions coexist (R2) and why 15 references rotted
> across module boundaries (R4).

**Other encodings** (verified):

| Location | What it says | Affected by D? |
|---|---|---|
| root `AGENTS.md`:159–166 | the normative convention, incl. the move rule | ✅ **must change** |
| `claude/qa-engineer/qa-engineer.md`:28 | never write into `archive/` | ✅ **must change** |
| `falkor-chat/AGENTS.md`:112 | *"`docs/archive/` — a doc moves here when its milestone closes"* | ✅ **must change** |
| `falkor-chat/AGENTS.md`:115–116 | two key-doc rows citing `docs/archive/plans/…` | ❌ no — those files stay archived |
| root `CLAUDE.md` | `@AGENTS.md` import stub only | ❌ no |
| `claude/AGENTS.md` | **zero** docs-tree / `archive` mentions (verified) | ❌ no |
| `claude/README.md` | catalog rows cite each agent's *write* path, not the archive rule | ❌ no |
| `skills/*/SKILL.md` | exactly **one** incidental hit — `skills/joern-cpg/references/cpg-model.md:66`, a `docs/plans/<slug>-graph.md` write path | ❌ no |
| `salesperson/AGENTS.md`, `opencode/agents/severino/AGENTS.md` | no `docs/` tree (components haven't adopted it) | ❌ no |

> **Finding A3 — the shared-`skills/` three-harness constraint is satisfied trivially.** The
> convention lives in `AGENTS.md` files and one agent prompt, not in any `SKILL.md`. Nothing that
> must port across Claude Code / OpenCode / Kiro changes.

### 3.2 The recommended edit list, with owners

Ordered by necessity. **`cobb` owns agent-definition and skill edits** — this section scopes that
work, it does not do it.

| # | File | Change | Owner |
|---|---|---|---|
| **1** | root **`AGENTS.md`** (the *"Module documentation convention"* bullet) | Replace the move rule with the status-marker rule; redesignate existing `archive/` trees as read-only history; add the one-sentence **citation-spelling** rule (§2.2). ~1 paragraph. | `cobb` (repo-standard prose), coordinated by `teco` |
| **2** | **`claude/qa-engineer/qa-engineer.md`**:28 | Replace *"Completed-milestone docs live in `docs/archive/<same-subdir>/` — … never into `archive/`"* with the status-marker rule. **1 sentence.** Also update `claude/qa-engineer/kaizen/{plan,history}.md` and re-check its `claude/README.md` row, per `claude/AGENTS.md`'s same-change rule. | **`cobb`** |
| **3** | **`falkor-chat/AGENTS.md`**:112 | Reword the `docs/archive/` key-doc row: frozen history from the old convention, not a destination. **1 sentence.** | `teco` or `coder` (component doc) |
| **4** | `docs/HISTORY.md` **and** `falkor-chat/docs/HISTORY.md` | One dated entry each recording the convention change. **v1.2 (M4/D3):** carry the **qualitative** findings plus only the counts a committed one-liner regenerates (§1.3 table) — **not** the 3/87/15 triple, which no committed artifact reproduces. Retires the 442 without enshrining a successor nobody can audit. | `teco` |
| **5** | **`claude/teco/teco.md`** — *"Documentation curation"* bullet 1 (line 65) | **v1.2: promoted from *optional* to REQUIRED (M2). v1.3 (B5): the wording changes — `teco` *coordinates* the flip; each kind's owner *performs* it.** Add to the documentation-impact scan: when a milestone closes, `teco` lists every document whose work the close freezes and makes the `Status: archived` flip a **done-condition of the closing unit, routed to that document's owner** (§9.6's "who flips it" column). `teco` does not perform the flips itself — its `PreToolUse` guard allowlists `docs/plans/*` only, so a flip on a review, requirements doc, test plan or test report would escalate to an interactive human approval *per file*. Plus its kaizen + `claude/README.md` row re-check. | **`cobb`** |
| **6** | **`claude/tico/tico.md`**:37 | **v1.2: NEW, B3's fix. v1.3 (B4): also bolds the field labels.** The header template becomes the canonical block of §9.6 — `**Status:**` / `**Last updated:**` bolded, gaining `**Owner:**`/`**Tracks:**`. **The two `Status:` *values* (`Interviewing`, `Ready for design`) are byte-identical** — so `tico.md`:71's stakeholder-gated transition and `claude/README.md`:8's user-facing promise are **verified unchanged, by a content match** (m15). Plus its kaizen + README row re-check. | **`cobb`** |
| **7** | **`claude/{architect,analyst,qa-engineer,teco,data-scientist,graph-dba}/*.md`** — the write-path line in each | **v1.3: NEW (major M9).** Each gains **one template line** telling the agent to open the document it writes with the §9.6 header block. Today `claude/tico/tico.md`:37 is the **only** header contract in all 13 prompts (verified), so without this N3's normalisation decays from document 27 onward. Four of the six files are already being edited (items 2, 5, 6 + `analyst`'s `-impl` row); `architect`, `data-scientist` and `graph-dba` are the marginal additions. Plus 2 extra kaizen pairs + 2 extra README re-checks. | **`cobb`** |
| 8 | `claude/{coder,tdd-engineer,frontend-engineer,devops,cobb,joern}/*.md` | **No change** — no docs-tree write path, no document-filename contract. The filename **prohibition** is still stated once only (root `AGENTS.md`) — see the M1 note below; M9 adds a *header* line to the six that write documents, which is a different rule with different evidence. | — |
| 9 | `claude/scripts/audit-team.sh` | *Optional.* See §3.3. **D2 ruled by default = (b): build the standalone report-only script, do not wire a CI gate.** | `devops` or `cobb` |

> **v1.2 — the prompt-change scope is corrected (blocker B3).** v1.1 §9.7 claimed *"six of the seven
> fit exactly; two prompt sentences change."* That is true of every agent's **write path** and false
> of the **header contract**, because §9.6 introduces a `Status:` field into a namespace `tico`
> already owns and gates. The real scope:
>
> | # | Prompt | What changes | Gated? |
> |---|---|---|---|
> | 1 | root `AGENTS.md` | the convention bullet: status-marker rule, citation rule, filename grammar **+ prohibition**, role set, collision rules, 3-field header | — |
> | 2 | `claude/qa-engineer/qa-engineer.md`:28 | archive sentence **and** the milestone clause — rewritten, not trimmed (M1) | no |
> | 3 | `claude/analyst/analyst.md`:51 | `-impl` role documented | no |
> | 4 | `claude/tico/tico.md`:37 | header template gains `Owner:`/`Tracks:`; **values unchanged** | **:71 untouched** |
> | 5 | `claude/teco/teco.md`:65 | named owner + trigger for the `Status:` flip (M2) | no |
>
> **Four agent prompts plus root `AGENTS.md`** — with 4 `kaizen/{plan,history}.md` pairs and 4
> `claude/README.md` row re-checks riding along, per `claude/AGENTS.md`'s same-change rule. Not two
> sentences. This is the single largest cost increase in v1.2 and it is stated up front.
>
> | 6 | `claude/architect/architect.md`, `claude/data-scientist/data-scientist.md`, `claude/graph-dba/graph-dba.md` | **v1.3 (M9)**: one header-template line each | no |
>
> **v1.3 correction to the scope, again: SIX agent prompts plus root `AGENTS.md`.** The two added
> rows are M9's — nothing today tells a producing agent to *write* the header. The cost delta over
> v1.2 is 3 one-line template additions (`architect`, `data-scientist`, `graph-dba`) plus 2 kaizen
> pairs and 2 README re-checks; the other three M9 lines ride on files already being edited.

**Cost of items 1–7: root `AGENTS.md` + 6 prompts + 2 `HISTORY` entries — roughly one paragraph and
eight sentences of prose.** No file moves, no path repaths; the path-string edits anywhere in the
plan are S2's **three deleted `../` tokens** and S6's **16 module-prefix normalisations in one
file**, each isolated in its own step so every other step's diff is provably path-free.

**Why the citation rule and the filename rule are priced separately (major M1).** v1.1 treated them
as one decision and refused to duplicate either into agent prompts. That argument is **sound for the
citation rule** — no prompt states it today, so root `AGENTS.md` is genuinely the only place it lives
— and **weaker for the filename rule**, because 7 prompts already carry a filename template. I
grepped for which of them could *contradict* an `AGENTS.md` prohibition:

```
$ grep -niE "follow \*?that|detect the convention|discover them" claude/*/[a-z]*.md
```

**Exactly one document-filename override exists: `qa-engineer.md`:27–28.** The other hits are about
*graph* label naming (`graph-dba.md`:46), *code* style (`coder.md`:13), *build* conventions
(`devops.md`:44) and *test-framework* naming (`tdd-engineer.md`:33) — none is a document-filename
contract. **Therefore item 7 stands: the prohibition in root `AGENTS.md` is contradicted nowhere
else, so duplicating it into 6 more prompts buys nothing.** M1's fix lands entirely on
`qa-engineer.md` (rewritten, §9.7) and on making the `AGENTS.md` rule a **prohibition** rather than a
permissive grammar. This is a partial, evidence-backed rejection of M1's third suggestion — recorded
as such in §11.

### 3.3 Can `audit-team.sh` enforce it?

> **v1.3 — S4 re-priced on its own merits (D1 ruled).** In v1.2, S4 carried a second job: it was the
> stated precondition for ever *mandating* the composed citation form. **D1's ruling deletes that
> job**, so S4 must now justify itself alone. It does, on one ground and one only: **Finding R2's
> 408 module-anchored references that an agent reading from the repo root cannot resolve**, plus
> archival-rot delta detection on a future sweep. It is *not* justified by the 4 broken links —
> **S2 deletes the 3 real ones outright, and the 4th is a deliberate illustration.** So S4 is
> genuinely optional, its value is *delta detection over time*, and **D2's default answer (b)
> — build it report-only, do not gate — is taken** (§6). A link-only checker would still be the
> wrong tool: it validates 8% of this repo's references (R1).

**Partly, and only in the spelling that matters.** A **check 8 — doc-reference resolution** is
feasible in ~80 lines:

- Enumerate `git ls-files '*.md'`; mask fenced code; extract relative markdown links and backticked
  path-bearing tokens; assert each resolves (link → relative to the citing file; backtick → relative
  to the repo root). Skip tokens containing `<`, `>`, `*`, `{`, `}` (template placeholders — 118 of
  my 205 raw hits are these).
- **The high-value half is the backtick check**, because that is 92% of the references and 87 of the
  90 defects (R1, R3). A conventional markdown link-checker would find 3 problems and miss 87.

Four caveats, all load-bearing:

0. **v1.2 — it must exclude documents that discuss link syntax, by a written rule.** Measured
   (§1.3): committing *this* plan adds **10** broken links to the census, every one an illustration
   (`../x.md`, `/docs/x.md`, `../QUERIES.md`). Without a stated placeholder-basename exclusion the
   checker's first act is to flag the document defining the convention. This caveat is new in v1.2
   and is the strongest single argument for keeping S4 **report-only**.
1. **It cannot start green.** Baseline is 4 broken links (3 real) + ~87 dead citations. It must run
   **report-only** (print counts, exit 0) or carry a recorded allowlist. Otherwise it is red on day
   one — and **`audit-team.sh` is already red** (C-309a: two pre-existing check-7 home-path/username
   leaks). A second permanent red source makes the gate worthless.
2. **`git ls-files` is blind to untracked files** — the same limitation the brief notes for check 7.
   For a convention about *committed* documents this is acceptable and should be stated in the
   script header, not silently inherited.
3. **Scope mismatch.** `audit-team.sh` is *"the deterministic half of the team-coherence
   certification"*, scoped to the `claude/` agent collection. A repo-wide doc-link census is a
   different concern. Cleaner: a standalone **`claude/scripts/check-doc-links.sh`** that
   `audit-team.sh` invokes as check 8 — runnable alone, and honest about what it owns. Note there is
   **no repo-root `scripts/` directory** today, so `claude/scripts/` is the only existing home.
4. **CI is available but path-filtered.** `.github/workflows/falkor-chat.yml` exists and is filtered
   to `falkor-chat/**` — so today a docs change elsewhere runs nothing. A tiny second workflow
   (checkout + run the script, no service container, ~20s) is the **only** mechanism that actually
   *prevents* rot; an on-demand script only detects it when someone remembers to run it. That is a
   real choice with a real cost (see **D2**).

---

## 4. Question 3 — "How to migrate the project?"

### 4.1 Is it mechanical or judgement? — it depends entirely on which option you pick

| Migration | Mechanical? | Files touched | Verifiable? |
|---|---|---:|---|
| **D (recommended)** | **No migration at all.** Only rule text changes. | **4** | Trivially — the diff must contain **zero** path-string edits |
| A forward-only (recommended) | No migration. New text only. | 0 | n/a |
| A full conversion (**not** recommended) | ~70% mechanical, 30% judgement | ~60 | Yes, by the §3.3 script |
| C (IDs) | Judgement throughout | ~100+ | Weakly — a table can be stale and still resolve |

**Why full-A is only 70% mechanical.** The rewrite "prefix module-root-anchored refs with their
module" is scriptable for the 408 clear cases. It is *not* scriptable for:

- the **279 citing-file-relative** refs — is `` `../QUERIES.md` `` meant as a path or as prose?
- the **333 ambiguous** refs where both anchorings resolve (root-level docs) — no signal.
- the **650 unresolved-anywhere** refs, which are mostly bare filenames and `<placeholder>` forms
  that must be *left alone*.
- every reference inside a **dated record** (`HISTORY.md`, `kaizen/history.md`, `kaizen/inbox.md`,
  review findings) where the as-written path is the historically correct one — exactly the judgement
  the analyst made in `m3-archive-sweep.md` O-2 and that `9bbfbb5` made for m-26's dated resolution
  note. **This is the class that makes bulk repathing expensive and unreviewable**, and it is large:
  `falkor-chat/docs/HISTORY.md` alone holds 174 `.md` references.

### 4.2 Incremental or atomic?

**Incremental, and that is a property of the recommendation, not a compromise.**

- **D is atomic but trivially so** — the rule flips in one commit and no document changes meaning.
  There is no half-migrated state, because there is no migration.
- **A is inherently incremental**: new text uses the repo-root spelling; existing references are
  left. A mixed repo is not *broken* — the 408 module-root refs resolve for a human reading in
  context and for anyone whose cwd is the module. The convention only asserts what **new** citations
  must be.
- **The one thing that must NOT be incremental** is D itself: root `AGENTS.md` and
  `qa-engineer.md` must flip in the **same commit**, or `qa-engineer` will keep writing "never into
  `archive/`" against a rule that no longer has an `archive/` destination.

### 4.3 Sequenced plan — **rationale; the executable list is §12**

> **v1.3:** S0–S6 below define *what each step is and why*. **§12 is the ordered execution contract**
> — it merges these with the N-steps, assigns one owner each, lists every file, and gives a
> self-verifying check per step. Where the two differ on an **owner**, §12 wins: it carries v1.3's
> B5 re-routing (S2/N4 moved from `teco` to an implementer).

**S0 — decision gate: CLOSED (v1.3).** D4 accepted, **D1 and D6 ruled**, D2 and D3 taken by their
stated defaults, D5/D7/D8 closed (§6). **Nothing waits on anyone; execution starts at §12 step 1.**

**S1 — flip the rule.** Root `AGENTS.md` convention bullet + `claude/qa-engineer/qa-engineer.md`:28
+ `falkor-chat/AGENTS.md`:112, **one commit**. Add the citation-spelling sentence to root
`AGENTS.md` in the same commit. Include `claude/qa-engineer/kaizen/{plan,history}.md`.
*Owner: `cobb` (agent + standards prose), `teco` coordinating the component doc.*
**Done when:** the three files state the status-marker rule consistently; `git diff` shows **no
path-string edits**; `claude/scripts/audit-team.sh` shows no new failures beyond the known C-309a
baseline (2 FAILs).

**S2 — record it, and fix the three links.** *Owner: **an implementer (`coder`)**, coordinated by
`teco`.* **v1.3 (B5, extended):** v1.2 assigned this to `teco`, but S2 writes `docs/HISTORY.md`,
`falkor-chat/docs/HISTORY.md`, `falkor-chat/docs/BACKLOG.md` and `docs/BACKLOG.md` — none of which
is in `teco`'s `docs/plans/*` allowlist, so every one escalates to an interactive human approval.
The reviewer applied this reasoning to the recurring flip (B5) and to N3; it applies here too, and
this is the third place v1.2 missed it. `coder` has **no** `PreToolUse` doc guard at all. Three
things, one commit:

1. **Dated entries** in `docs/HISTORY.md` and `falkor-chat/docs/HISTORY.md`. **v1.2 (M4/D3): carry
   only the reproducible numbers and the qualitative findings** from §1.3's table — **not** the
   3/87/15 triple, which no committed artifact regenerates. State the exclusion rules
   (placeholders, illustrative paths) beside any count, since they are what make a number a choice
   rather than a fact.
2. **v1.2 (B1/D5) — delete three `../` tokens** in `falkor-chat/docs/BACKLOG.md`:785, 787, 895.
   Folded here, not deliberated: all three targets exist, the fix has no judgement content, and it
   takes the repo's real broken-link count from 3 to 0 (1 remaining is the illustration in
   `claude/architect/kaizen/inbox.md`:201, deliberately left).
3. **Correct the false claim**: `docs/HISTORY.md`'s 2026-07-26 entry says *"`docs/test-plans/` and
   `docs/test-reports/` remain as empty active directories"* — **verified false**; git tracks no
   empty directories and neither path exists on disk.

**Done when:** both HISTORY files carry the entry; the false claim is gone; and
`ls falkor-chat/docs/{reviews,plans}/workflow-def-structure-read.md` resolves from each of the three
fixed links. **This is the only step in the plan that edits a path string** — 3 tokens, all
deletions.

**S3 — the 3 forward-looking rot fixes.** Repath the archival-rot references in
`claude/analyst/kaizen/plan.md` (2) and `claude/architect/kaizen/plan.md` (1) to their
`falkor-chat/docs/archive/…` targets. **Leave the 12 in `history.md`/`inbox.md`/`k001-run-brief.md`
alone** — dated records, where the pre-move path is correct as written (analyst O-2's reasoning,
which I concur with). *Owner: `cobb` (kaizen files are agent-scoped).* **Done when:** 3 lines
changed; dead-citation count drops 87 → 84.

**S6 — normalise `falkor-chat/AGENTS.md`'s own citations (v1.3, minor m19).** The file an agent
reads *before* writing anything in that component carries **16** backticked `` `docs/…` `` refs that
are module-root-anchored — the exact spelling the new rule forbids. Prefix each with
`falkor-chat/`. This is live guidance, not a dated record, so analyst O-2's "correct as written"
protection does not apply to it. **Deliberately its own step, not folded into S1**, so S1's
"zero path-string edits" proof stays intact and *this* step's diff is exactly 16 prefix insertions.
*Owner: an implementer (`coder`).*
**Done when:** `grep -ohE '`docs/[^`]+`' falkor-chat/AGENTS.md | wc -l` → **0**, and every
`` `falkor-chat/docs/…` `` token in the file resolves with `Read` from the repo root.

**S4 — the checker (optional; D2 ruled by default = (b), so build it report-only, no CI gate).**
`claude/scripts/check-doc-links.sh`, report-only, wired as `audit-team.sh` check 8. Header must
state the `git ls-files` blindness and the recorded baseline. **v1.3: its justification is Finding
R2 alone** (§3.3) — it is no longer coupled to D1. *Owner: `devops` or `cobb`.* **Done when:** the
script reproduces §1's numbers in §1.3's table, states its placeholder-exclusion rule, records its
baseline **in its own header/report and not in `HISTORY.md`** (m11), and exits 0.

**S5 — bulk repath to full root-anchoring. NOT RECOMMENDED. Do not schedule.** File as a backlog
item with this document's cost analysis attached, so the decision is recorded rather than
re-litigated. Buys ≤4.5% of future archival churn plus agent-resolvability of references that mostly
point into frozen history, for a ~60-file diff nobody will review carefully. *Owner: unassigned.*

### 4.4 Verifying zero regressions

**The recommendation is self-verifying, which is the strongest argument for it.** S1–S2 change no
path, so link resolution *cannot* regress. The verification is a one-line assertion on the diff:

> `git diff` for S1/S2 must contain no changed line in which a `*.md` path string differs — only
> prose.

For S3 (the only path-changing step), and as a repo-wide baseline going forward, the numbers to hold
are:

**v1.2 (M4): the table below is this document's analysis record, not `HISTORY`'s baseline.** Rows
marked *script* require the uncommitted census and are **not** written into `HISTORY` unless D2
green-lights S4 and the script is committed with it (§1.3).

| Metric | Baseline (working tree at `583e132`) | After S2 + S3 | Reproducible by |
|---|---:|---:|---|
| Broken relative markdown links, tracked | **4** (3 real + 1 illustration) | **1** (the illustration) | *script* |
| — same, once this plan is committed | **14** (10 new, all illustrative) | 11 | *script* |
| Composed-form references | 143 | 143 | **one-line `git grep`** |
| Dead path-bearing backticked `.md` citations | ~87 | ~84 | *script* |
| Archival-rot references (twin exists in `archive/`) | 15 | 12 (dated records, deliberately kept) | *script* |
| Active feature docs lacking a canonical `Status:` **(v1.3, m13: re-counted)** | **25 of 26** — 17 have no `Status:` line at all, 8 have a non-canonical one, 1 (`docs/plans/doc-reference-convention.md`) already conforms | **0 of 26** (after N3) | **the §12 N3 loop** |
| Module-anchored `` `docs/…` `` refs in `falkor-chat/AGENTS.md` **(v1.3, m19)** | **16** | **0** (after S6) | ``grep -c '`docs/' falkor-chat/AGENTS.md`` |
| `audit-team.sh` | FAIL, 2 (C-309a: username + home path) | unchanged | `bash claude/scripts/audit-team.sh` |

**Retire the 442** (§1.3) **and do not enshrine 3/87/15 in its place.** The one metric that is both
decomposable *and* cheaply reproducible is the last row — which is N3's, and is the reason N3 is the
highest-value step.

**Can prose mentions be caught automatically?** Partly, and I quantified it: **yes for the 24
path-bearing prose mentions** (regex `[\w.-]+(/[\w.-]+)+\.md` outside code fences — that is how I
found them). **No for the 367 bare-filename mentions** — with no directory there is no way to know
which `AGENTS.md` is meant, and they are overwhelmingly legitimate prose about a convention, not
citations. Making the class checkable requires the convention to mandate a **directory component in
every citation**, which is part of the §2.2 spelling rule.

### 4.5 Rollback

Uniquely clean, because nothing moves:

- **S1–S3** are single-commit prose edits → `git revert` restores the previous convention exactly.
  The old `archive/` trees were never touched, so reverting the *rule* does not orphan any file.
- **S4** is additive and report-only → deleting the script is a complete rollback.
- **There is no data motion anywhere in this plan.** The one destructive-ish thing available — moving
  documents — is precisely what we are choosing to stop doing.

### 4.6 Is it worth doing at all? — the honest answer

**Yes for D, yes for A-forward-only, defer E, and no for the bulk repath.**

- **Doing nothing** and paying at each sweep: sweep A cost 22 edits / 8 files for 2 documents; sweep
  B cost 157 edits / 26 files for 20 documents. Rough rate: **~8 path edits per archived document.**
  Two of the four modules haven't adopted the docs tree yet, so future sweeps get *more* frequent,
  not less. Both sweeps also needed a bespoke verification script written on the spot, plus (in
  sweep A) a review finding that *existed purely as a pathing artifact* — m-26, which then needed a
  dated resolution note because its reasoning argued from the pre-move state. That is a real,
  compounding tax on top of the edits.
- **Doing D** costs one paragraph and two sentences, once, and drops the rate from ~8 edits per
  archived document to **1 line per archived document.** It pays back on the very next milestone
  close.
- **Doing the bulk repath** costs a ~60-file judgement-heavy sweep to buy 4.5% of a cost that D
  already reduced to ~zero. **The migration costs more than it saves. Do only S1–S3.**

---

## 5. Documentation impact

Rows marked **[naming]** were added in v1.1 and are specified in §9–§10. `S*` = archiving/citation
steps (§4.3); `N*` = naming steps (§10.3).

> **v1.2 changes to this table (B3, M1, M2, M5, B1):** **two new rows** — `claude/tico/tico.md` and
> `claude/teco/teco.md` are now **required** prompt edits, not "no change" / "optional". The
> `qa-engineer` row becomes a **rewrite**, not a 4-word trim. The 17-file header row becomes **26
> documents** — **v1.3 re-derived: 17 additions + 8 normalisations = 25 touched, 24 tracked; the
> 26th already conforms** (m13) — with a **3-field** header, not 4.
> `falkor-chat/docs/BACKLOG.md` gains the **3-link fix** (folded into S2). The row asserting *"no
> change — their write-path contracts match"* for `{architect,tico,data-scientist,graph-dba,teco}` is
> **narrowed to `{architect,data-scientist,graph-dba}`**.

| Document | Change | Trigger step |
|---|---|---|
| **`AGENTS.md`** (root) | *"Module documentation convention"* bullet (lines 159–166): status-marker rule replaces the move rule; existing `archive/` trees redesignated read-only history; **new** one-sentence citation-spelling rule (§2.2) | S1 |
| **`AGENTS.md`** (root), *same bullet* | **[naming]** filename grammar `<topic-slug>[-<role>].md`; **v1.2 (M1): stated as a PROHIBITION** — *"a new document's basename never begins with `m<digit>`, `k<digit>`, or a date"*, because a permissive grammar forbids nothing; the closed role set (§9.4); the collision rules (§9.5, incl. the **slug-ordinal successor** branch, B2); the **3-field header block + its 5-value `Status:` set** (§9.6, v1.2); the one sentence saying an existing `m<n>-` prefix is part of a name, not a lifecycle claim (§10.1) | **N1** — *same bullet, same commit as S1* |
| **`claude/analyst/analyst.md`** | **[naming]** add `-impl` to the review-document convention (lines 51/60). Used **4×** in the repo, documented **nowhere** — the gap already produced a divergent-slug defect (§9.5). + kaizen + `claude/README.md` row re-check | **N2** · **`cobb`** |
| **`claude/qa-engineer/qa-engineer.md`**:28 | **[naming] v1.2 (M1) — REWRITE, not a 4-word trim.** Drop `/milestone` **and** subordinate *"If a component uses a different convention, follow **that**"* — otherwise the agent re-derives the prefix from a corpus that is 59% milestone-prefixed and, in `archive/test-plans/`, **5 of 5**. Same line already edited by S1 ⇒ **one commit**. + kaizen + README row re-check | **N2a** · **`cobb`** |
| **26 active-tree feature docs** (**per-file value table in §12 step 3**) | **[naming] v1.3 (M2, m13, m16):** the **3-field** header (§9.6) — **17 additions + 8 normalisations = 25 documents touched, 24 tracked**; the 26th (`docs/plans/doc-reference-convention.md`) already conforms. **Content-only, zero path strings.** Without the normalisation the census reads complete but returns 8 non-lifecycle answers | **N3** · `coder` |
| **`claude/tico/tico.md`**:37 (+ kaizen, README row) | **[naming] v1.2 (B3) — NEW REQUIRED EDIT.** Header template gains `Owner:`/`Tracks:`. `Status:` **values absorbed verbatim**, so `:71`'s stakeholder-gated transition and `claude/README.md`:8 are **unchanged — verify byte-identical** | **N2c** · **`cobb`** |
| **`claude/teco/teco.md`**:65 (+ kaizen) | **[naming] v1.2 (M2) — PROMOTED from optional to required.** The documentation-curation scan names `teco` as **owner** of the `Status: archived` flip at milestone close. Under D4 this clause *is* the archival sweep | **N2d** · **`cobb`** |
| **`falkor-chat/docs/BACKLOG.md`**:785, 787, 895 | **v1.2 (B1/D5):** delete three extra `../` tokens. Together with S6 these are the only path-string edits in the plan | **S2** · implementer (`coder`) **— v1.3 (B5): re-owned from `teco`, whose guard allowlists `docs/plans/*` only** |
| `docs/HISTORY.md`, `falkor-chat/docs/HISTORY.md` | **[naming]** the S2 entries also record: naming convention adopted forward-only; **renames explicitly declined**, with the 39-occurrence / 15-file measurement (§10.2) | **N4** (fold into S2) · implementer (`coder`), `teco` coordinating |
| `falkor-chat/docs/BACKLOG.md` | **[naming]** file the `k031-structure-read-impl.md` → `workflow-def-structure-read-impl.md` re-slug as an **opportunistic nit** (4 occurrences, 3 files — verified), not scheduled work | **N4** · implementer (`coder`) |
| `claude/{architect,data-scientist,graph-dba}/*.md` | **[naming] no *filename* change** — write-path contracts already `<slug>[-role].md` (§9.7), **and verified to contain no convention-override clause** that could contradict the `AGENTS.md` prohibition (§3.2 item 8). **v1.3 narrows this row again:** they now take M9's one-line **header** template addition (row above). Filename rule: still stated once, in root `AGENTS.md` | N2e |
| `claude/{coder,tdd-engineer,frontend-engineer,devops,cobb,joern}/*.md` | **no change** — no docs-tree write path, no document-filename contract | — |
| `claude/*/hooks/guard-*doc-writes.sh`, `claude/scripts/guard-doc-writes.sh` | **[naming] no change** — verified directory-globbed only; they neither break nor can enforce a filename rule (§9.7) | — |
| **`claude/qa-engineer/qa-engineer.md`** | line 28: replace the never-write-into-`archive/` sentence | S1 · **`cobb`** |
| `claude/qa-engineer/kaizen/{plan,history}.md` | log the prompt edit (`claude/AGENTS.md` same-change rule) | S1 · **`cobb`** |
| `claude/README.md` | re-check the `qa-engineer` row for a restated archive rule (currently only cites write paths — likely **no edit**) | S1 · **`cobb`** |
| **`falkor-chat/AGENTS.md`** | line 112: reword the `docs/archive/` key-doc row. Lines 115–116 unchanged. | S1 |
| **`docs/HISTORY.md`** | dated entry + baseline table; **correct the false "empty active directories" claim** in the 2026-07-26 entry | S2 |
| **`falkor-chat/docs/HISTORY.md`** | dated entry noting the convention change applies forward-only | S2 |
| `claude/analyst/kaizen/plan.md`, `claude/architect/kaizen/plan.md` | 3 archival-rot repaths | S3 · **`cobb`** |
| ~~`claude/teco/teco.md` (+ kaizen) — *optional*~~ | **v1.3 (m10): DELETED.** This row contradicted the `teco` row above, which makes the same edit **required** (N2d). Two rows for one edit with opposite dispositions, in the table an implementer works from. | — |
| **`claude/{architect,data-scientist,graph-dba}/*.md`** (+ 2 kaizen pairs, 2 README re-checks) | **v1.3 (M9) — NEW REQUIRED EDIT.** One template line each: the document this agent writes opens with the §9.6 header block. Without it the header decays from document 27 (`tico.md`:37 is today the only header contract in all 13 prompts — verified) | **N2e** · **`cobb`** |
| **`falkor-chat/AGENTS.md`** (16 backticked `` `docs/…` `` refs) | **v1.3 (m19) — NEW.** Prefix each with `falkor-chat/`. Live guidance an agent reads before writing, so the "dated record, correct as written" protection does not apply. **Its own step, so S1's diff stays path-free** | **S6** · implementer (`coder`) |
| `claude/scripts/audit-team.sh` + new `claude/scripts/check-doc-links.sh` | *optional* — check 8, report-only. **D2 ruled by default = (b)** | S4 |
| `docs/BACKLOG.md` | file S5 (bulk repath) as **deliberately deferred**, with this document's cost analysis cited | S2 |
| `skills/*/SKILL.md` | **no change** — one incidental hit, unaffected. The three-harness portability constraint is satisfied trivially. | — |
| `claude/AGENTS.md`, root `CLAUDE.md` | **no change** — verified zero docs-tree/archive mentions | — |

**Backlog interactions** (noted only, not expanded per the brief):

- **C-309** — `audit-team.sh` is already FAIL on 2 pre-existing check-7 leaks, and check 7's
  `git grep` is untracked-blind. A new check 8 **must** be report-only or it compounds a gate that is
  already ignored. C-309 should land before S4, or S4 should stay standalone.
- **C-310** — OpenCode/Kiro MCP wiring. No interaction: nothing recommended here touches MCP or
  `SKILL.md`.
- **C-321** — containerized scratch-graph naming. No interaction.
- *Out of scope, noticed in passing:* root `AGENTS.md`'s **Structure** section does not mention the
  tracked `kiro/DESIGN.md`. Unrelated drift; flagging only so it isn't mistaken for something this
  plan introduces.

---

## 6. Decisions — all ruled or taken

> **v1.3: nothing here is open.** D4 and D1 and D6 are the stakeholder's rulings; D2 and D3 are
> taken by the defaults this document stated in v1.2 and are recorded as *taken*, not *ruled*; D5,
> D7 and D8 were already closed. **No step below waits on anyone.**

| # | Decision | Options | Outcome |
|---|---|---|---|
| **D1** ✅ **RULED 2026-07-27 — (a)** | **Should the composed form ever be *mandated*?** | **(a)** Link permitted, never required. ~~(b) Mandate it where a link adds navigational value, with S4 as a precondition~~ | **(a) — NO CLICKABLE LINKS.** Ruling, in the stakeholder's terms: *a reference is a plain backticked repo-root path; the composed form is permitted, never required.* **Consequences folded into v1.3:** the "recommended in index documents" carve-out is **deleted** (§2.2); option (b) and its S4 coupling are **withdrawn**; §6.1's "what would change the answer" is **deleted**; **S4 is re-priced on Finding R2 alone** (§3.3) and survives on those merits, since the 3 real broken links are deleted by S2 regardless and were never S4's justification. In no case is `[x](/docs/…)` acceptable (**verified** to break agent `Read`). |
| **D2** ✅ **TAKEN BY DEFAULT 2026-07-27 — (b)** | **Build the checker (S4), and does it gate?** | **(a)** None. **(b)** Report-only script, run on demand. **(c)** Report-only + a CI workflow. | **(b), taken by the stated default** (v1.2 recommended *"(b) now, (c) later"*; no ruling arrived, so the default is taken and recorded here and in S2's HISTORY entry). Build `claude/scripts/check-doc-links.sh` report-only; **do not add a CI workflow and do not gate**, because `audit-team.sh` is already red (C-309a) and a second permanent red makes the gate worthless. S4 stays **optional and last** in §12 — it is a nice-to-have, not a dependency of anything. |
| **D3** ✅ **TAKEN BY DEFAULT 2026-07-27 — (a)** | **Retire the "442 unresolved backticked strings" baseline — and what replaces it?** | **(a)** Retire it, replace with nothing numeric. **(b)** Retire it and commit the census script with S2. **(c)** Keep the 442. | **(a), taken by the stated default.** `HISTORY` records the qualitative findings plus only the counts a committed one-liner regenerates (§1.3 table) — **not** 3/87/15. **(b) is not taken even though D2 green-lights S4**: S4 is optional and last, so making S2's HISTORY content depend on it would re-create exactly the sequencing defect M4 raised. If S4 ships, its numbers live **in the script's own header/report** (m11), never in `HISTORY`. |
| **D4** ✅ **ACCEPTED 2026-07-27** | **Adopt D — stop moving documents to `archive/`?** | **(a)** Adopt D forward-only; existing `archive/` trees frozen as read-only history. ~~(b)~~ ~~(c)~~ | **(a)** — **accepted as recommended.** Ruling: documents stop moving; the `Status:` marker goes in the document's **existing header**; existing `archive/` trees freeze as read-only history; **no un-archiving.** This makes §9–§10 load-bearing: with nothing moving, the filename is the only lifecycle signal in a directory listing, and **D7 is what replaces it.** |
| ~~**D5**~~ ✅ **CLOSED in v1.2 — not a decision (B1)** | v1.1 asked whether to fix 3 links pointing at *"never-created documents"*, offering **(b) leave them, they document planned-and-never-built deliverables**. **The premise was false: all three targets exist** and the defect is an extra `../`. Option (b) had no factual basis and option (a) required no judgement. | — | **Folded into S2 as three deleted tokens.** No stakeholder input needed. |
| **D6** ✅ **RULED 2026-07-27 — (a), ADOPT IN FULL** | **Adopt the naming convention in §9?** — `<component>/docs/<kind>/<topic-slug>[-<role>].md`, no milestone/ID/date prefix (**stated as a prohibition**, M1), closed role set, shared slug across kinds, slug-ordinal successors (§9.5 rule 5), the milestone-scoped-topic exception (§9.4, M8). **v1.3 price, final:** 1 `AGENTS.md` paragraph + **6 prompts** + 6 kaizen pairs. | **(a)** Adopt in full (N1+N2). ~~(b) only the "no milestone prefix" half~~ ~~(c) do nothing~~ | **(a) — accepted as recommended.** The whole of §9 is now normative and lands in root `AGENTS.md` via N1. **This also closes m18:** v1.2 flagged that if D6 were declined, N3 would ship 25 headers speaking a vocabulary written down nowhere normative — with D6 ruled (a), N1 carries **both** halves (the §9.6 header block *and* the filename grammar / role set / collision rules) and the question is moot. For the record, the split m18 asked for still holds if the convention is ever partly reverted: **the §9.6 header block is D4-consequent and ships regardless; only the filename grammar is D6-gated.** |
| ~~**D7**~~ ✅ **FOLDED into D4 in v1.2 (minor m9)** | v1.1 posed the `Status:` backfill as a stakeholder decision with a **(c) Skip** option, while the plan itself argued skipping *"is not defensible"*. An option the document argues against is ceremony, and it dilutes the two real forks (D1, D6). | — | **N3 is a consequence of the already-accepted D4, not a separate decision.** D4 made `Status:` the sole lifecycle signal; a signal covering 8 of 26 documents while *looking* complete is worse than the directory convention it replaced. **Scope corrected (M2): normalise all 26, don't backfill 17.** If exactly one thing ships, ship N3. |
| ~~**D8**~~ ✅ **CONFIRMED in v1.3 — architect decision, unchanged** | v1.1 escalated "script check or prompt-level?" to the stakeholder while §10.5 had already decided it on four verified grounds. That is a technical call, not a product one. | — | **Decision, confirmed and unchanged: prompt-level enforcement only (N5).** The four verified reasons stand (§10.5): scope mismatch; `audit-team.sh` already FAILs (C-309a); the check cannot start green (6 active + 30 archived `m<n>-` files); check 7's `git grep` is untracked-blind at exactly the moment an agent writes a new document. **v1.3 addition:** D2's default (b) *does* green-light S4, so the ~12-line naming+header census inside `check-doc-links.sh` is now in scope as **§12 step 8's optional half** — still report-only, still not a gate, and still not the enforcement mechanism. **M9's prompt-template line is the real enforcement**, which is the same conclusion by a stronger route. The reviewer endorsed this handling twice; it is not reopened. |

### 6.1 D1 — the both-consumers answer, stated explicitly

The brief asks for a spelling robust under **both** an agent resolving from the repo root and
GitHub's renderer. **Conclusion C1 (§1.4) stands and is the honest answer: there is no single bare
spelling that both follow as a repo-root path.** `/docs/x.md` is renderer-correct and
**verified agent-broken**; `docs/x.md` is agent-correct and renderer-broken (GitHub resolves it
relative to the *citing file's* directory). Two refinements narrow — but do not close — the gap:

1. **Documents that live at the repo root are a free special case.** For `AGENTS.md`, `CLAUDE.md`,
   `README.md`, the bare relative link `[x](docs/plans/x.md)` is **dual-valid**: relative-to-citing-file
   *is* the repo root, which *is* the agent's cwd. Zero duplication. Conveniently, this covers the
   normative documents where the convention itself lives — so root `AGENTS.md` can cite in a
   spelling that works everywhere.
   *Confidence: reasoned from the two verified rules (GitHub — no leading `/`, no `./` ⇒ relative to
   the citing file's directory; agent — cwd = repo root). I did **not** execute a GitHub render.*
2. **Everywhere else, the only both-consumers form is the composed one** —
   ``[`falkor-chat/docs/QUERIES.md`](../QUERIES.md)``. **v1.2 (B1): this is where the argument
   turned.** v1.1 cited the form's 143 existing uses as evidence that mandating it *"is not an
   invention"*. The review's decisive observation: **§1.1 R3's broken-link count is that same form
   failing.** All 4 broken relative links in the repo are composed references whose two halves
   disagree — 3 real defects in `falkor-chat/docs/BACKLOG.md` plus, fittingly, the *illustration* of
   the form in `claude/architect/kaizen/inbox.md`:201. Measured drift rate: **3/143 ≈ 2%**, and
   **100% of the repo's broken-link baseline.**

   Citing a population as proof of safety, when that population *is* the defect population, is the
   error v1.1 made. The population proves the form is **familiar**, not that it is **reliable**.

**The trade-off, now settled by the ruling (v1.3).** The backticked repo-root citation is mandated
always (the agent contract, non-negotiable — Finding R2's 408 unresolvable references). The relative
link is **permitted and never required, and is not recommended**. A citation with no link is not
clickable on GitHub; the reader copies the path. **That is the accepted price.**

v1.2's *"what would change the answer"* paragraph — the (b)-plus-S4 escape hatch — is **deleted**.
It existed only to keep the mandate question open for a stakeholder who might browse on github.com;
the stakeholder has ruled and does not. Nothing in §§9–12 is contingent on it. **The one durable
fact worth keeping from that analysis:** if the composed form is ever revisited, it must ship with a
label/target drift detector, because the form's only measured behaviour in this repo is that it
drifts (3/143 ≈ 2%, 100% of the broken-link baseline). Recorded so it isn't rediscovered — not
scheduled.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Under D, the active set becomes harder to see** — no directory listing answers "what is live?" | **Medium** — the one real regression | ~~milestone-prefixed slugs already sort frozen work together~~ **[v1.1: this mitigation is withdrawn — §9 removes the prefix. See §9.3.]** Remaining, and stronger: `BACKLOG.md`/`HISTORY.md` already index this authoritatively; **N3 makes `grep -m1 -H 'Status:' <dir>/*.md` a *complete* listing** (8→26 of 26 active docs, v1.2); add that one-liner to each module's `AGENTS.md` |
| **A mixed convention is worse than either** — half `archive/`, half status-marker, for years | Medium | Redesignate existing `archive/` explicitly as *history of the previous convention*, in root `AGENTS.md`, so the mix is **documented state** rather than drift. Never move anything into `archive/` again. |
| **Nobody follows the citation rule**, because it lives only in root `AGENTS.md` | Medium | It standardises the **plurality** spelling (652 refs already comply), so the default behaviour is already mostly right. S4 makes violations visible. Accepted: a convention with 5 copies drifts worse than one with 1. |
| **Status markers rot** — a document is frozen but nobody adds the line | **Medium** *(raised in v1.2 — M2)* | v1.1 rated this Low **on the strength of a teco clause it had marked "optional"** — the mitigation was not actually being bought. **v1.2 made it required (N2d); v1.3 (B5) makes it *performable*: `teco` coordinates, each kind's owner performs.** Residual risk stands: under D4 the status line *is* the archival sweep, and it is enforced by a prompt clause, not a script (D8). |
| **The archival flip is a guarded write for 3 of the 5 document kinds** — `teco`'s `PreToolUse` allowlist is `docs/plans/*` only, so a flip on a review, requirements doc, test plan or test report escalates to an **interactive human approval per file** | **High if unfixed — mitigated to Low** *(v1.3 — B5)* | **Route by the existing guard topology, no hook edit** (§9.6's "who flips it" column): `plans/` → `architect` (or `teco` for `-coordination`) · `reviews/` → `analyst` · `requirements/` → `tico` · `plans|reviews/*-ml` → `data-scientist` · `test-plans/`, `test-reports/` → `qa-engineer` (**no doc guard at all**). Verified against all 9 `claude/*/hooks/` scripts. The rejected alternative — widening `teco`'s allowlist — is a permanent loosening of a deliberately narrow guardrail and must be argued on its own merits, not smuggled in as a side effect of a docs convention. **The same defect bit S2/N4** (`HISTORY.md`/`BACKLOG.md` writes), re-owned to an implementer in v1.3. |
| **New documents omit the header** — N3 normalises 26 once, then document 27 starts the drift back | **Medium if unfixed — mitigated to Low** *(v1.3 — M9)* | Verified: `claude/tico/tico.md`:37 is the **only** header contract in all 13 agent prompts, and the three producing prompts that carry full document skeletons (`analyst`:53–61, `qa-engineer`:29, `architect`:40) mention no header field. **N2e adds one template line to all six producing prompts.** This is the same argument M1 made for the filename rule — a prompt-level template beats a general `AGENTS.md` rule — applied to the field N3 exists to establish. Owner: `cobb`. |
| **`tico`'s stakeholder-gated `Status:` transition is broken by the new vocabulary** | **High if mishandled — mitigated to Low** *(v1.2 — B3)* | `tico.md`:71 gates *"Ready for design"* on explicit stakeholder confirmation and `claude/README.md`:8 promises it to the user. **§9.6 absorbs both `tico` values verbatim rather than renaming them**, so neither line changes. **N2c's done-condition is that `tico.md`:71 and `README.md`:8 are byte-identical to HEAD** — the gate's survival is *verified*, not assumed. |
| **The composed citation form drifts** — label and target disagree | **Low, and now measured** *(v1.2 — B1; closed by the D1 ruling)* | Measured at **3/143 ≈ 2%**, producing 100% of the repo's broken links. **Mitigated by the ruling**: the form is never required and no longer recommended, so the population cannot grow by obligation; the existing 143 stay untouched. |
| **The checker's baseline (87) never goes down** and becomes noise | Low | Report-only by design; the value is *delta detection* on a sweep, not reaching zero. **v1.3 (m11): record the baseline in the script's own header/report — NOT in `HISTORY.md`.** The v1.2 wording here still said "in HISTORY", contradicting D3(a), which retires unreproducible numbers rather than minting new ones. |
| **`/docs/…` gets adopted later** by someone optimising for the renderer, silently breaking agent handoffs | Low but nasty | Root `AGENTS.md` must state the prohibition **with the reason** (leading `/` is filesystem-absolute to a file-read tool), not just the rule. |
| **Kiro never sees the convention** — it reads steering files, not `AGENTS.md`, and this repo has **no** `.kiro/steering/` (verified: no such directory) | Low | Unchanged from today — Kiro does not see the *current* convention either. If Kiro adoption becomes real, the convention needs a steering file; note it, do not build it now. |
| **My census disagrees with the committed 442** — if my methodology is wrong, some conclusions shift | Low | Every number above states its method (§1) and the load-bearing claim (agent `Read` behaviour) is **verified by execution**, not inference. The two sweep decompositions are content-diffs of real git blobs and are independently reproducible. |

---

## 8. Ready to implement

> Consolidated summary of §§1–11. The naming work is specified in **§9** (the convention) and
> **§10** (its rationale and migration).
>
> ## → **The executable step list is §12. Go there.**
>
> §12 is the contract: numbered, ordered, one owner per step, every file listed, a done-condition
> and a self-verifying check each. The canonical header block is stated verbatim in **§9.6**. An
> implementer needs §12 + §9.6 and nothing else in this document.

**Plan:** `docs/plans/doc-reference-convention.md` (this document, **v1.3 — final design pass**).
**Review answered:** `docs/reviews/doc-reference-convention.md` — Part I map in **§11.1**, Part II
map in **§11.2**.

**Decision state — v1.3: nothing is open.**

| # | State | Gates |
|---|---|---|
| **D4** | ✅ **ACCEPTED 2026-07-27** — documents stop moving | S1; makes N3 load-bearing |
| **D1** | ✅ **RULED 2026-07-27 — (a)**, no clickable links; composed form permitted, never required, not recommended | nothing — the mandate and its S4 coupling are withdrawn |
| **D6** | ✅ **RULED 2026-07-27 — (a)**, adopt the naming convention in full | N1, N2 (both ship) |
| **D2** | ✅ **TAKEN BY DEFAULT — (b)** report-only script, no CI gate | S4 (optional, last) |
| **D3** | ✅ **TAKEN BY DEFAULT — (a)** retire the 442, nothing numeric replaces it | S2's HISTORY content |
| ~~D5~~ | ✅ **CLOSED** — false premise (B1); folded into S2 as 3 deleted tokens | — |
| ~~D7~~ | ✅ **FOLDED into D4** — N3 is a consequence, not a decision (m9) | — |
| ~~D8~~ | ✅ **CONFIRMED** — prompt-level enforcement only; the S4 census is report-only and is not the enforcement | N5 |

**Global verification, holding across every step:** every diff except **S2's three deletions** and
**S6's 16 prefix insertions** must contain **zero** `*.md` path-string edits — nothing can regress
because **nothing moves and nothing is renamed**. `bash claude/scripts/audit-team.sh` stays at the
C-309a baseline (**FAIL, 2**: username leak + home-path leak); no new FAILs. And the one
done-condition that proves the riskiest change was safe:
`git diff HEAD -- claude/tico/tico.md | grep -E '^[-+].*Ready for design'` returns **nothing**, and
`git diff HEAD -- claude/README.md` is **empty**.

---

## 9. Document naming convention

New in v1.1. Answers the stakeholder's *"M-something is not good for longer."* This section is
grounded in a full census of the 67 documents under `*/docs/`, not in the abstract.

### 9.1 The inventory — three schemes already coexist

`find . -path '*/docs/*' -name '*.md'` → **67 files**; 6 are module fixtures
(`BACKLOG`/`HISTORY`/`DESIGN`/`QUERIES`), leaving **61 feature documents**: **25 active**, **36
archived** (34 in `falkor-chat/docs/archive/`, 2 in `docs/archive/`).

| Scheme | Count | Share of 61 | Active / archived | Examples |
|---|---:|---:|---|---|
| **Milestone-prefixed** `m<n>-…` | **36** | **59%** | **6 / 30** | `docs/plans/m2-cpg-analysis-skill.md`, `falkor-chat/docs/archive/plans/m3-executor.md`, `…/m3-process-flow-coordination.md`, `…/m1-chat-mcp.md` |
| **Backlog-ID-prefixed** `k<nnn>-…` | **4** | **7%** | 2 / 2 | `falkor-chat/docs/archive/test-plans/k007-m2-groundwork.md` (**ID *and* milestone**), `falkor-chat/docs/reviews/k027-parse-robustness.md`, `…/k031-structure-read-impl.md` |
| **Bare slug** | **21** | **34%** | 17 / 4 | `docs/plans/cpg-query-access.md`, `docs/requirements/joern-cpg-pipeline.md`, `falkor-chat/docs/plans/demo-environment-bringup.md` |

Active-tree breakdown: **12 `plans/`, 8 `reviews/`, 5 `requirements/`, 0 `test-plans/`, 0
`test-reports/`** (the last two confirm §4.3-S2's finding that `docs/HISTORY.md`'s "empty active
directories" claim is false — those directories do not exist).

> **v1.3 (minor m14) — one inventory number, stated once.** The counts in this table are **as of
> `583e132` plus this plan** (25 active). Every later section — §10.3, §10.5 and §12 — counts **26
> active feature documents**, the difference being this plan's **review**, which arrived in
> `docs/reviews/` after §9.1 was written. **26 is the operative number** and decomposes as
> **12 `plans/` + 9 `reviews/` + 5 `requirements/`**, of which **24 are tracked** and 2 (this plan
> and its review) are untracked. Re-verified today.

**Finding N1 — the three schemes are not converging; the newest documents use all three.** In the
active trees alone: `m2-cpg-analysis-skill.md` (milestone), `k031-structure-read-impl.md` (ID),
`cpg-mcp-containerization.md` (bare) — all written within the same fortnight. Left alone this goes
to four schemes, because **`claude/qa-engineer/qa-engineer.md`:28 actively licenses** the milestone
form (*"named for the feature/milestone under test"*) while every other write-path contract says
`<slug>`/`<kebab>`. Verified: `grep -rniE 'milestone' claude/*/[a-z]*.md` filtered to naming
context returns **exactly that one line** across all 13 agents.

**Finding N2 — the ID-prefix scheme is already lossy.** All 4 files spell the ID `k007`/`k027`/`k031`,
which does **not** match the repo's canonical `K-007`/`K-027`/`K-031`. **Verified:
`git grep 'K-007'` does not find `k007-m2-groundwork.md`.** The prefix fails at the one thing an ID
is for — greppability.

**Finding N3 — ad-hoc suffixes, classified.** Enumerated across all 61:

| Suffix | Count | What it actually means | Verdict (§9.4) |
|---|---:|---|---|
| `-coordination` | 6 | teco's orchestration ledger, co-located in `plans/` | **role — keep** (in teco's prompt) |
| `-report` | 6 | qa test report in `test-reports/` | **role — keep** (in qa's + teco's prompts) |
| `-ml` | 4 | data-scientist method note in `plans/` | **role — keep** (in ds's prompt) |
| `-impl` | 4 | analyst review of an **implementation** vs. of a plan | **role — keep, but it is *undocumented*** (see N4) |
| `-landing2` | 1 | a *phase* of K-022, one-off | **retire** — §9.5 rule 5 |
| `-queries` | 1 | graph-dba deliverable, predates the `-graph` contract | **retire** — superseded by `-graph` |
| `-skill`, `-sweep` | 2, 2 | part of the **topic**, not a role (`cpg-analysis-skill`, `archive-sweep`) | **reclassify, no action** |
| `-graph`, `-rca` | **0**, **0** | in `graph-dba`'s and `analyst`'s prompts; **never yet used in this repo** | **role — keep** (contract exists, unexercised) |

**Finding N4 — CORRECTED IN v1.2 (major M3). The missing `-impl` rule has produced a defect, but a
narrower one than v1.1 claimed.**

v1.1 asserted *"`git grep workflow-def-structure-read` does not surface the impl review."* **That is
false**, and reproducibly so:

```
$ git grep -c workflow-def-structure-read -- 'falkor-chat/docs/reviews/k031-structure-read-impl.md'
falkor-chat/docs/reviews/k031-structure-read-impl.md:3
$ git grep -l workflow-def-structure-read -- '*.md' | wc -l
7
```

The impl review cites the plan and both its gates in its own header. **The family is fully
discoverable by content grep; only the *basename* diverges.** The real cost is therefore narrower
and should be stated as such:

> **K-031's implementation review is invisible to `ls falkor-chat/docs/reviews/` and to any
> filename glob for the topic, though visible to content grep.** A reader who knows the topic finds
> it; a reader who lists the directory does not, and neither does any tooling keyed on the filename.

**The stronger evidence, which v1.1 missed.** `claude/analyst/analyst.md`:51 **already** mandates
*"kebab-case slug matching the artifact under review"* — that is §9.5 rule 2 in embryo, already
written into a prompt, **and already breached** by `k031-structure-read-impl.md`. This is better
evidence than the grep claim on two counts: it is true, and it is evidence about **enforcement**
rather than about the rule's absence. The rule exists; an agent departed from it anyway, because the
prompt states the slug rule but not the `-impl` role, leaving the agent to invent both halves.

**What this does to D6.** v1.1 called option (b) — *"adopt only the no-milestone-prefix half"* — *"a
false economy"* on N4's strength. With N4 corrected, that verdict needs re-deriving, and it does
survive, but for a different reason: not *"families become undiscoverable"* (they don't — content
grep works) but *"`analyst` has a slug rule it already breached, and the missing `-impl` role is
why."* Documenting `-impl` is what makes an existing prompt rule followable. See D6.

### 9.2 The convention

```
<component>/docs/<kind>/<topic-slug>[-<role>].md
```

- **`<kind>`** ∈ `plans` · `reviews` · `requirements` · `test-plans` · `test-reports`. **The
  directory carries the document's kind** — the filename never repeats it.
- **`<topic-slug>`** — kebab-case, lowercase, a noun phrase naming the **feature or topic**. Stable
  for the life of the topic. **Deliberately shared across kinds** (§9.5 rule 2).
- **`[-<role>]`** — optional, from the **closed set** in §9.4. Distinguishes co-located documents of
  the same kind on the same topic, usually by producing agent.
- **No milestone prefix. No backlog-ID prefix. No date prefix.** (§9.3)

Examples, using the repo's own topics:

| Today | Under the convention |
|---|---|
| `docs/plans/m2-cpg-analysis-skill.md` | `docs/plans/cpg-analysis-skill.md` |
| `falkor-chat/docs/archive/plans/m3-executor-ml.md` | `falkor-chat/docs/plans/executor-ml.md` |
| `falkor-chat/docs/reviews/k031-structure-read-impl.md` | `falkor-chat/docs/reviews/workflow-def-structure-read-impl.md` |
| `falkor-chat/docs/archive/test-plans/k007-m2-groundwork.md` | `falkor-chat/docs/test-plans/groundwork.md` |
| `docs/plans/cpg-query-access.md` | *unchanged — already conforms* |

> **These are illustrations of the grammar, not a work order. §10 recommends renaming nothing.**

### 9.3 What identifies a document — slug, ID, or date?

**The topic slug. Not the backlog ID, not the milestone, not the date.** Four reasons, each
measured:

1. **A backlog ID is not 1:1 with a document, so it cannot identify one.** K-022 alone owns
   **6** documents (`m3-executor.md`, `-ml`, `-coordination`, `-landing2`, `-impl`,
   `-landing2-impl`, plus `m3-capability-probe-ml.md` and `m3-guard-calibration.md`). An ID prefix
   would still need slug + role appended — it is **pure prefix cost**, buying nothing the rest of
   the name doesn't already do.
2. **Both IDs and milestones are already recoverable from the body.** Verified across the 61 feature
   documents: **55 name a `[CK]-\d{3}` ID** in their own text; **54 name an `M<n>` milestone**.
   `docs/BACKLOG.md` carries 27 milestone tokens and `falkor-chat/docs/BACKLOG.md` 78. Nothing is
   lost by removing them from the filename.
3. **The slug is what ties a *family* together across kinds** — see §9.5 rule 2. An ID prefix
   fragments the family whenever the ID changes mid-feature, which is exactly what happened to
   K-031 (N4).
4. **Greppability.** A slug is a word a human remembers and greps. The repo's actual ID prefixes are
   `k007`-style and **don't even match the canonical ID spelling** (N2).

**Which sort order serves a reader here? Alphabetical-by-topic.** Under D4 nothing moves, so a
directory listing is a mixed, permanent list — and the adjacency you want in it is the **family**:
`cpg-query-access-coordination.md` next to `cpg-query-access.md`. A milestone or ID prefix instead
sorts by *when unrelated topics happened to be filed*, which is useful only while the milestone is
current — precisely when you least need help finding the document. Chronology is not lost: it lives
in `git log` and in `HISTORY.md`. (**v1.2, M5:** v1.1 also claimed the header's `Updated:` field —
that field is withdrawn, §9.6; the two remaining sources were always the stronger half of the
argument.) **Chronology is a query, not
a filename.**

> **A v1.0 claim this withdraws, stated plainly.** §2.2 mitigation #2 and the first §7 risk row
> argued that D's discoverability loss was softened because *"falkor-chat already prefixes slugs
> with the milestone, so frozen work already sorts together."* **§9 removes that prefix, so that
> mitigation is gone.** The replacement is strictly stronger, and it is **N3**: with the header
> backfilled, `grep -m1 -H 'Status:' docs/plans/*.md` is a *complete* listing that **names** each
> document's state, where a prefix only ever let you **guess** it from a milestone number you had to
> remember. This is the interaction between D4 and the naming rule, and it is why **N3 is the single
> highest-value step in the plan rather than a tidy-up**. (v1.2: this was D7, now folded into the
> accepted D4 as a consequence — §6, minor m9. v1.2 also **extends N3 from a 17-document backfill to
> a 26-document normalisation**, per M2: presence alone is not a lifecycle.)

### 9.4 Does the milestone belong in the name? No — and where it goes instead

The stakeholder's instinct is right, and D4 sharpens it:

- **It ages into noise.** `m3-` was informative in July 2026 and is meaningless at M7 unless the
  reader remembers the calendar. `executor` is informative forever.
- **Under D4 it becomes a *competing, unmaintained* lifecycle signal.** D4 makes `Status:` the
  authoritative signal. A milestone prefix is a second signal that can never update itself — it can
  only drift out of agreement. Two lifecycle signals is strictly worse than one.
- **It is already stored in three maintained places.** Recoverable from **`Tracks:` in the header**
  (per-document, §9.6), **`BACKLOG.md`** (authoritative, per-item), and **`HISTORY.md`**
  (authoritative, per-delivery) — plus the body text in 54 of 61 documents.

> **Where the milestone must be recoverable from: the header's `Tracks:` field first, then
> `BACKLOG.md`/`HISTORY.md`. Never as a filename *prefix*.**

**The one exception — a topic that genuinely *is* a milestone (v1.3, major M8).** Three rules used
to meet here and give three different answers: the prohibition (*"never begins with `m<digit>`"*),
§9.4's *"never the filename"*, and §9.5 rule 3 (*"a slug is never reused for a different topic"*).
Name the M4 follow-ups coordination document under those three and you get, respectively:
`m4-followups-coordination.md` **prohibited**; `followups-coordination.md` **legal once and then
blocked for M5**; `followups-m4-coordination.md` **legal by the prohibition but forbidden by
§9.4**. An agent hitting three mutually exclusive rules improvises — which is Finding N1's
"three schemes become four", reintroduced by the rule meant to stop it. **This is not theoretical:
the repo has produced the class twice already** (`m3-followups-coordination.md`,
`m3-archive-sweep.md`) and the stakeholder's M4 follow-ups are the next instance. The single answer:

> **When a document's topic genuinely *is* a milestone or a recurring per-milestone activity, the
> milestone token goes *inside* the slug — never as a prefix.** `followups-m4-coordination.md`,
> `archive-sweep-m4.md`. **The prohibition is on the *prefix***, which is what makes a directory
> listing sort and read by topic; it is not a prohibition on the token. `Tracks:` still carries the
> milestone; the slug carries it too *only* in this case, because without it the topic has no name.

Three consequences, stated so nothing is left to interpret:

1. **§9.4's *"never the filename"* is restated as *"never as a prefix, and never as a lifecycle
   claim."*** A mid-slug milestone token is part of a **name**, exactly like the `m<n>-` prefixes
   §10.1 tells readers not to fix.
2. **Rule 3 is satisfied, not bent.** "M4 follow-ups" and "M5 follow-ups" *are* different topics, and
   `followups-m4` / `followups-m5` are different slugs. The rule needs no amendment.
3. **The test for "genuinely is a milestone"**, so it isn't used as a loophole: *the topic has no
   name without the milestone token* — i.e. removing the token leaves a slug that would collide with
   the same activity in another milestone. §10.2 already applied exactly this test to
   `m3-followups-coordination.md` (*"renaming it requires **inventing** a topic name"*) and got the
   right answer for the rename question; this is that test pointed forward.

**The closed role set.** Everything not on this list is part of the topic slug, not a role:

| Role | Directory | Meaning | Producer | In a prompt today? |
|---|---|---|---|---|
| *(none)* | any | the primary document of that kind | the kind's owner | ✅ |
| `-coordination` | `plans/` | teco orchestration ledger | `teco` | ✅ `teco.md`:56 |
| `-ml` | `plans/`, `reviews/` | method note / methodology review | `data-scientist` | ✅ `data-scientist.md`:71–72 |
| `-graph` | `plans/` | graph data-model design note | `graph-dba` | ✅ `graph-dba.md`:51 (0 files yet) |
| `-rca` | `reviews/` | root-cause analysis | `analyst` | ✅ `analyst.md`:60 (0 files yet) |
| `-impl` | `reviews/` | review of an **implementation** (default = review of a **plan**) | `analyst` | ❌ **used 4×, documented nowhere** |
| `-report` | `test-reports/` | test report | `qa-engineer` | ✅ `qa-engineer.md`:41, `teco.md`:49 |

Two judgement calls, stated:

- **`-report` is redundant with `test-reports/` — keep it anyway.** It is in two prompts and six
  files, and because **92% of this repo's citations are backticked strings a human reads as text**
  (Finding R1), a self-describing basename has real value: `m2-graphrag-report.md` is unambiguous in
  a grep hit where `m2-graphrag.md` is not. Dropping it costs 2 prompt edits for zero gain.
- **`-impl` is *not* redundant — keep it and document it.** `reviews/` legitimately holds two
  different artifacts on one slug: the plan review and the implementation review. The directory
  cannot tell them apart, and the absence of the rule already broke a family (N4). This is the one
  role that needs a prompt edit (§9.7).

### 9.5 Collision and uniqueness rules

1. **Primary key: `(component, kind, topic-slug, role)`.** Unique by construction within a directory.
2. **The same slug across several kinds is *required*, not merely tolerated — it is the family.**
   `requirements/x.md` → `plans/x.md` → `plans/x-coordination.md` → `reviews/x.md` →
   `test-plans/x.md` → `test-reports/x-report.md`. **A downstream document inventing a new slug is a
   defect** (N4). Nine such families exist today; the healthiest is `cpg-query-access`, spanning
   `plans` + `requirements` + `reviews` + `test-plans`.
3. **A topic slug is never reused for a different topic**, in any component or kind.
4. **Cross-directory basename collision is safe *because every citation carries a directory*** —
   the §2.2 spelling rule. Verified: all **16** citations of `m2-cpg-analysis-skill.md` (which
   exists in both `plans/` and `reviews/`) carry a directory prefix; **zero** are bare.
   > **Rules 2 and 4 are a matched pair: the shared slug is only a feature because the citation rule
   > makes it unambiguous. Adopt both or neither.** Adopting the shared slug *without* the citation
   > rule would convert a feature into exactly the ambiguity the brief worries about.
5. **Two documents of the same kind + topic, at different times.** **REWRITTEN IN v1.2 (blocker
   B2); the branch selector and the back-pointers CORRECTED IN v1.3 (majors M6, M7).** v1.1 put the
   ordinal on the *role* token, which left a **primary** document — role `(none)`, §9.4 — with no
   legal name at all.

   > **v1.3 (M6) — the selector no longer keys on a token that was dropped.** v1.2 chose the branch
   > by the earlier document's `Status:` (*"while `active` … once `archived`/`superseded`"*). But
   > M2 removed `delivered`, so §9.6 note 2's lifecycle is *"written `active`, flipped once to
   > `archived`"* — meaning **an approved, gated, partly-executed plan whose milestone is still open
   > is `active`**. Read literally, v1.2's selector routed exactly the `m3-executor.md` situation to
   > *revise in place* — the disposition this very section spends 18 lines proving destroys
   > information. The whole weight fell on *"or otherwise must stay intact"*, an undefined,
   > subjective test doing the work a token used to do. **The fix is not to re-add `delivered`** (it
   > would cost a touch per document to re-encode what `BACKLOG.md`/`HISTORY.md` already say); it is
   > to make the selector test an **event**, not a token:

   **The selector — one question, answered by the record, not by a field:**

   > **Has the earlier document been approved, gated, or executed against?**

   - **No → revise it in place.** Bump the optional `Version:` field and add a dated revision note;
     for reviews, append a dated `## Pass N` section. Three documents already do this:
     `falkor-chat/docs/plans/workflow-def-structure-read.md` (*"Version: v2 — 2026-07-24"*),
     `…/m3-process-flow.md` (*"patch v2.1"*), and this document (v1.3). **Cost: zero new filenames,
     zero new inbound references.** This is the common case and stays the default.

   - **Yes → it must stay intact. Write a successor, ordinal on the SLUG** — *even if its milestone
     is still open and its `Status:` is still `active`*:

     ```
     <topic-slug><n>[-<role>].md      →  executor2.md, executor2-coordination.md, executor2-impl.md
     ```

     Because the ordinal is a **slug suffix**, `git grep executor` and `ls plans/executor*` both
     still return the whole family.

   **The back-pointers (v1.3, major M7 — two defects fixed).** v1.2 said the successor carries
   `Supersedes:` **or** `Extends:` while *"the earlier document gains `Superseded by:`"*. That (1)
   instructed an amendment to a document whose `archived` status means *"do not execute or amend"*,
   and (2) applied a *superseded* label to the `Extends:` case, where the plan itself says the
   earlier document **remains authoritative** — `m3-executor.md` was not superseded by `-landing2`,
   and marking it so would misstate the very history the separate document existed to protect. The
   corrected rule:

   | Relationship | Successor's header | Earlier document gains | Earlier document's `Status:` |
   |---|---|---|---|
   | The successor **replaces** the earlier one | `Supersedes: <path>` | `Superseded by: <path>` | flips to **`superseded`** |
   | The successor **adds to** an earlier one that stays authoritative | `Extends: <path>` | `Extended by: <path>` | **unchanged** |

   > **Adding or updating a header pointer is *metadata, not an amendment*. It is the one edit
   > permitted on an `archived` document** — the whole point of the back-pointer is that a reader
   > arriving at the frozen document by link learns a successor exists. Stated here and in §9.6's
   > `archived` row so an implementer never has to weigh the two rules against each other.

   > **This is a simplification, not an addition.** The v1.1 escape hatch (ordinal on the role
   > token: `x-impl2.md`, `x-report2.md`) is **withdrawn** — one ordinal rule replaces two, the role
   > set stays clean, and `x2-impl.md` reads correctly as "the impl review of the second plan on
   > topic x". A genuine second review of the *same* artifact is a `## Pass 2` section, which the
   > first branch already covers.

   **Re-derivation of `m3-executor-landing2.md` (B2 — v1.1 misdiagnosed this).** v1.1 asserted it
   *"would have been a `Version: 2` section of `executor.md`"*. Its own header
   (`falkor-chat/docs/archive/plans/m3-executor-landing2.md`:3–7) says otherwise:

   ```
   > **Status:** proposed (architect design-patch, 2026-07-12). Planning-only — no code/DDL changed.
   > **Extends:** `docs/archive/plans/m3-executor.md` (approved plan, §6 trigger / §7 safety / Phases 4–5)
   ```

   It is a **design-patch extending an already-approved, already-partly-executed plan**, deliberately
   kept separate so the artifact a review gate had signed off — and an implementer had built from —
   stayed intact. Folding it into `executor.md` would have **rewritten a signed-off document**, i.e.
   destroyed information, in the one case the repo actually produced. **The separate document was
   the right call; only the filename was wrong.** Under v1.2 it is
   `falkor-chat/docs/plans/executor2.md`, `Status: active`, `Extends: falkor-chat/docs/plans/executor.md`
   — an *instance* of the second branch, and the case that motivates it rather than a counter-example
   to it. What v1.1 correctly disliked — `-landing2`, a project *phase* smuggled in as a role — is
   still prohibited by the closed role set (§9.4).

### 9.6 The header block — v1.2 (blockers B3; majors M2, M5) · **canonical form fixed in v1.3 (B4)**

> **This section is normative and is the one place the header block is stated. §12 points here;
> root `AGENTS.md` copies from here; implementers copy, they do not paraphrase.**

v1.1 specified **four** required fields with a **five-value** `Status:` set of its own invention.
Three findings landed on it and all three are accepted:

- **B3** — the set **collided with a vocabulary `tico` already owns and gates.** `claude/tico/tico.md`:37
  mandates `> Status: Interviewing | Ready for design · Last updated: YYYY-MM-DD`, and `:71` flips to
  *"Ready for design"* **only on the stakeholder's explicit confirmation** — restated as a
  user-facing promise in `claude/README.md`:8. All 5 `requirements/` documents speak it. v1.1's set
  contained neither value.
- **M2** — the field was **unowned** (its only enforcement point was an *optional* clause) and the
  8 existing `Status:` lines were **left un-normalized**, so the post-backfill grep would have been
  complete in presence and incoherent in content.
- **M5** — **`Updated:` duplicates `git log -1 --format=%ad -- <file>`**, has no maintainer, and
  nothing checks it. By §9.6's own standard (*"every addition erodes the reason to adopt it"*) it
  does not earn a required slot.

#### The canonical header block — copy this, do not interpret it

**v1.3 (blocker B4): the *form* is now specified, not only the vocabulary.** v1.2 absorbed `tico`'s
`Status:` **values** but not `tico`'s **syntax**: `claude/tico/tico.md`:37 and all three requirements
documents written from it spell it `> Status: Interviewing` — **unbolded** — while the anchored
done-condition m1 asked for demands `^> \*\*Status:\*\*`. N3 therefore could not pass its own gate on
the three files it was told must not change. **The canonical form is BOLD LABELS.** Verbatim, one
line, **immediately under the H1** (a blank line between them is permitted; nothing else may precede
it):

```markdown
# <Document title>

> **Status:** <token> · **Owner:** `<agent>` · **Tracks:** <id(s)> (<M<n>>)
```

Two conformant real examples, one of each dialect:

```markdown
# CPG query access — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** C-101 (M2)
```

```markdown
# Summary Nodes — Feature Requirements

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-07-12
```

**The exact rules, so two implementers produce byte-identical output:**

1. **Labels are bolded** (`**Status:**`, `**Owner:**`, `**Tracks:**`) — and so is `tico`'s optional
   `**Last updated:**`. **`tico`'s two `Status` *values* are byte-identical to today**
   (`Interviewing`, `Ready for design`); only the label's asterisks change. **Bolding a label is a
   form change, not a value change** — it does not touch `tico.md`:71's gated transition or
   `claude/README.md`:8's user-facing promise, and §12 step 2 proves that by content match.
2. **Field separator is ` · `** (space, U+00B7, space). Field order is `Status:`, `Owner:`,
   `Tracks:`, then any optional fields.
3. **The canonical token is the first thing after `Status:`.** Free text is preserved *after* the
   token, never before it — `> **Status:** archived · delivered ✅ AC-1…AC-4 met (M3, 2026-07-25)`
   is conformant. This is what makes N3 non-destructive.
4. **One window, everywhere (v1.3, minor m12).** The canonical line is written **immediately under
   the H1**, and every check in this plan uses **`head -6`**. Where a document already carries a
   `Status:` line *lower down* its header block — `docs/plans/cpg-query-access.md` has one at
   **line 11** — the line is **folded into the canonical one** (its text becomes the trailing clause
   per rule 3) and **removed from its old position**. It is never left in two places. v1.2's three
   disagreeing windows (`head -8`, "first 12 lines", "immediately under the H1") are replaced by
   this one rule.
5. **The regex that defines conformance**, used identically by N3's gate and N5's census:

   ```bash
   grep -qE '^> \*\*Status:\*\* (Interviewing|Ready for design|active|superseded|archived)\b'
   ```

| Field | Values | Why it earns its keep |
|---|---|---|
| **`Status:`** | the closed set below; **the canonical token is the first thing after `Status:`** | **D4's lifecycle signal** — the job the milestone prefix was doing by accident. Makes `grep -m1 -H 'Status:'` a *complete* listing that **names** the state instead of implying it. |
| **`Owner:`** | the producing agent/role, backticked | Who an amendment routes to — **and, under B5, who performs the `archived` flip.** Already present as `Author:`/`Reviewer:` in 24 documents. |
| **`Tracks:`** | backlog ID(s) + milestone, e.g. `K-022 (M3)`; `—` if none | **The only genuinely new field.** Carries exactly what leaves the filename (§9.3, §9.4). |

**The closed `Status:` set — five values; `tico`'s two are absorbed verbatim (B3), and the "who
flips it" column is the B5 routing table (v1.3).**

| Token | Applies to | Means | **Who flips it, and when** |
|---|---|---|---|
| `Interviewing` | `requirements/` | interview open | `tico`, at creation — **existing contract, unchanged** |
| `Ready for design` | `requirements/` | handed off to design | `tico`, **only on explicit stakeholder confirmation** — **existing gate, unchanged** |
| `active` | all other kinds | live; **amendable in place until it has been approved, gated, or executed against** — after that it stays intact and a successor is written (§9.5 rule 5), *even while it is still `active`* | the producing agent, at creation |
| `superseded` | any | a **replacement** exists; `Superseded by:` required (the `Extends:` case does **not** use this token — §9.5 rule 5) | whoever writes the successor |
| `archived` | any | frozen record; **do not execute or amend — except a header pointer, which is metadata** (§9.5 rule 5) | **the document's own owner, at milestone close, on `teco`'s coordination** (B5) |

**Who performs the `archived` flip, by kind (v1.3, blocker B5) — this is the existing
`PreToolUse` guard topology, verified across all 9 `claude/*/hooks/` scripts; no hook is edited:**

| Kind | Performer | Its guard allowlist | Silent write? |
|---|---|---|---|
| `plans/<slug>.md` | `architect` | `docs/plans/*\|*/docs/plans/*` | ✅ |
| `plans/<slug>-coordination.md` | `teco` | `docs/plans/*\|*/docs/plans/*` | ✅ |
| `plans/<slug>-ml.md`, `reviews/<slug>-ml.md` | `data-scientist` | `docs/plans/*\|…\|docs/reviews/*\|…` | ✅ |
| `plans/<slug>-graph.md` | `graph-dba` | no doc guard (`guard-destructive-ops.sh` only) | ✅ |
| `reviews/*` | `analyst` | `docs/reviews/*\|*/docs/reviews/*` | ✅ |
| `requirements/*` | `tico` | `docs/requirements/*\|*/docs/requirements/*` | ✅ |
| `test-plans/*`, `test-reports/*` | `qa-engineer` | no doc guard | ✅ |

> **`teco` coordinates; it does not perform.** Its allowlist is `docs/plans/*` only, so a flip it
> performed on a review, requirements doc, test plan or test report would raise an **interactive
> human approval prompt per file** — `falkor-chat/docs/reviews/` alone holds 4 active documents
> today. Routing to each kind's owner costs nothing, matches `teco`'s own charter
> (*"coordinates — doesn't do"*), and is the reason no hook needs widening.

Three design choices, stated so they aren't re-litigated:

1. **`tico`'s values are absorbed, not renamed.** They are legal members of the set spelled exactly
   as `tico` spells them. The alternative — renaming them to `draft`/`ready` for tidiness — would
   have rewritten `tico.md`:71's gate sentence and `README.md`:8's user-facing promise to buy a
   lowercase token. **The gated transition is product behaviour and is not the architect's to
   change.** Cost of absorbing: a machine census matches `Ready for design|Interviewing|active|superseded|archived`
   instead of one lowercase pattern. That is the whole price.
2. **`draft` and `delivered` are dropped (M2).** They cost a touch each and duplicate
   `BACKLOG.md`/`HISTORY.md` — the *same* argument §9.3 uses to strip the milestone from the
   filename, applied consistently. **Touches per document under v1.2: a plan/review is written
   `active` and flipped once to `archived`; a requirements doc keeps its two existing `tico`
   transitions and gains one flip to `archived`.** Exactly one net-new touch per document, which is
   D4's replacement for a `git mv` plus ~8 inbound repaths.
3. **Free text is preserved after the token, never before it.**
   `Status: archived · delivered ✅ AC-1…AC-4 met and accepted (M3, 2026-07-25)` is conformant.
   This is what makes N3's normalisation **non-destructive**: nothing in the 8 existing rich status
   lines is thrown away, it just stops being the first token.

**Optional fields, only when there is something to say:** `Version:` (when a revision must be
citable), `Supersedes:` / `Superseded by:` / `Extends:` / **`Extended by:`** (real pointers — §9.5
rule 5, v1.3), `Last updated:` (**`tico` keeps it; it is not required of anyone else**), `Reviews:`
(a review's artifact, when the slug doesn't make it obvious). All take the same bolded-label form.

**Explicitly *not* fields:** `Updated:` (M5 — `git log` answers it authoritatively and for free, and
an unmaintained freshness claim is worse than none), a separate milestone field (folded into
`Tracks:`), `Type:`, `Scope:`, `Under test:`, `Baseline:`, `Gate:` — body content. The census found
**59 distinct bolded header labels** across the 67 documents' first 10 lines; **the point of this
block is to replace that vocabulary with three.**

**One consequence to record: §9.3's chronology claim is withdrawn.** v1.1 argued the milestone could
leave the filename partly because *"chronology … now lives in the header's `Updated:` field"*. With
`Updated:` demoted, chronology lives in `git log` and `HISTORY.md` — which was always the stronger
half of that argument, and which §9.3 also states. Nothing else in §9.3 depends on it.

Two notes: a document under `docs/archive/` is `Status: archived` **by construction** and needs no
backfill (§10.3). And a `Status:` value is the *document's* state, not the milestone's.

### 9.7 Interaction with the agent handoff contracts

A convention that silently contradicts these breaks every agent. **v1.2 corrects this table: the
*write paths* fit almost everywhere, but once §9.6's header is part of the convention the change
lands on SIX prompts, not two** (blocker B3 + major M9, v1.3), **and `qa-engineer`'s edit must be a
rewrite, not a 4-word trim** (major M1).

| Contract as written today | Write path fits? | Prompt change (**v1.3, final**) |
|---|---|---|
| `architect` → `<component>/docs/plans/<slug>.md` (`architect.md`:40) | ✅ identical | ⚠️ **M9 — +1 template line**: the plan opens with the §9.6 header block. + kaizen + README row re-check |
| `data-scientist` → `<slug>-ml.md` in `plans/` and `reviews/` (`data-scientist.md`:71–72) | ✅ identical | ⚠️ **M9 — +1 template line.** + kaizen + README row re-check |
| `graph-dba` → `<component>/docs/plans/<slug>-graph.md` (`graph-dba.md`:51) | ✅ identical | ⚠️ **M9 — +1 template line.** + kaizen + README row re-check |
| `tico` → `<component>/docs/requirements/<slug>.md` (`tico.md`:33) | ✅ identical | ⚠️ **B3 + B4 — the header template at `:37` gains `**Owner:**`/`**Tracks:**` AND bolds its labels.** Its `Status:` **values stay byte-identical** (§9.6), so **`:71`'s gated transition and `README.md`:8 are verified unchanged, by content match** (m15). + kaizen + README row re-check |
| `teco` → `<component>/docs/plans/<slug>-coordination.md` (`teco.md`:56) | ✅ identical | ⚠️ **M2 + B5 — `:65`'s documentation-curation bullet makes the `Status: archived` flip a done-condition of the closing unit, *routed to each document's owner*; `teco` coordinates, does not perform.** Required, not optional. **+ M9's template line** for its own coordination docs. + kaizen |
| `analyst` → `docs/reviews/<slug>.md` + `<slug>-rca.md` (`analyst.md`:51, 60) | ⚠️ **incomplete** — `-impl` used 4×, specified nowhere; `:51`'s slug rule is **already breached** (§9.1 N4) | **+1 sentence** naming `-impl` as the implementation-review role. **+ M9's template line** in the review skeleton (`:53–61`). + kaizen + README row re-check |
| `qa-engineer` → `docs/test-plans/<kebab>.md` + `<kebab>-report.md` (`qa-engineer.md`:28, 41) | ⚠️ paths fit; **:28 licenses the milestone prefix AND overrides the repo rule — and so does `:54`** (m17) | **M1 — rewrite `:28`, don't trim it, and subordinate `:54`'s "doc conventions" override too.** See below. **+ M9's template line** in the `:29` test-plan structure and the `:41` report structure. + kaizen + README row re-check |

> **v1.3 (minor m17) — a second override clause the §3.2 grep missed.**
> `claude/qa-engineer/qa-engineer.md`:54 reads *"**Match the project.** Discover and follow each
> component's framework, runner, file layout, naming, and **doc conventions** … Read the component's
> `AGENTS.md` first."* It does not match the pattern §3.2 grepped for
> (`follow that|detect the convention|discover them`) but does the same work as `:28`. It does
> **not** disturb M1's rejection — it is inside the file already being rewritten, so it is zero
> marginal cost — but **N2a must cover it, or the rewritten `:28` is contradicted 26 lines below.**
> Note the reviewer independently re-grepped with a wider net and confirmed that the *other*
> override clauses (`frontend-engineer`:18, `coder`:13, `tdd-engineer`:33, `graph-dba`:46,
> `devops`:70) are about UI code, code style, test frameworks, graph labels and infra — **none is a
> document-filename contract.** M1's rejection stands.

**M1 — why `qa-engineer.md`:28 needs a rewrite.** v1.1 priced this at *"≈4 words — drop
`/milestone`"*. The line in full is:

> *"**Detect the convention first.** Look at how the component already stores docs/plans … named for
> the feature/milestone under test. Completed-milestone docs live in `docs/archive/<same-subdir>/` …
> **If a component uses a different convention, follow *that*.**"*

Deleting `/milestone` leaves *"Detect the convention first"* and *"follow **that**"* intact — and the
corpus a `qa-engineer` detects **from** is `falkor-chat/docs/archive/test-plans/`, whose contents are
`k007-m2-groundwork.md`, `m1-chat-mcp.md`, `m1-hardening-regression.md`, `m2-graphrag.md`,
`m3-workflow-engine.md`: **5 of 5 carry a milestone token** (verified). An agent obeying its own
prompt would re-derive the prefix, correctly. The repo-wide corpus is 59% milestone-prefixed
(36 of 61) and, under rename-nothing, stays that way permanently. **Forward-only naming does not
survive a "follow that" override pointed at a corpus that disagrees with it.** The rewrite:

- **Keep** "detect the convention" for genuinely component-specific matters (directory layout,
  whether a `test-plans/` dir exists).
- **Subordinate the override:** *"…follow that — **except the filename grammar, which is repo-wide
  (root `AGENTS.md`) and not component-negotiable**."*
- **Delete** the `archive/` sentence (S1 already does this) and the `/milestone` clause.

**And root `AGENTS.md` must state a prohibition, not only a grammar (M1).** Every write-path template
says `<slug>`/`<kebab>`, and **none of them forbids anything** — `m4-executor` is a perfectly good
kebab-case slug. The rule therefore carries an explicit negative clause:

> *"A new document's basename never begins with `m<digit>`, `k<digit>`, or a date."*

Per §3.2 item 7 this is **not** duplicated into the other six prompts: I grepped for
convention-override clauses and **`qa-engineer.md`:27–28 is the only document-filename override in
the collection** — the rule is contradicted nowhere else, so one statement suffices.

**Verified non-interaction — the `PreToolUse` write guards.** All five doc-scoped agents
(`architect`, `analyst`, `data-scientist`, `teco`, `tico`) wrap `claude/scripts/guard-doc-writes.sh`
with **directory-only** globs (e.g. architect: `'docs/plans/*|*/docs/plans/*|architect/kaizen/inbox.md|…'`).
A filename convention therefore **cannot break them**, and should not be enforced by them:

- **v1.2 (minor m5) — the "cannot express it" claim is withdrawn, and so is the reviewer's fix.**
  v1.1 said a shell `case` glob *"cannot express 'must not start with `m<digit>`'"*. With
  `shopt -s extglob` it can — but **not** with the pattern the review suggested. Tested:
  `docs/plans/!([m][0-9])*` **matches** `docs/plans/m3-x.md` (the trailing `*` lets `!()` consume
  just `m`), so it would silently fail to flag anything. The form that actually works is
  `docs/plans/!([mk][0-9]*)` — verified to flag `m3-x.md` and `k031-x.md` while correctly passing
  `executor.md` and `machine-learning.md`. Recording both, because a claim of the form "X cannot be
  done" invites exactly one counter-example and then loses the argument.
- **The sound reasons stand, unchanged:** the escalation is an **interactive human prompt**, which is
  the wrong altitude for a naming nit; and **`qa-engineer` and `coder` — who write test-plans,
  test-reports and N3's headers — have no such hook at all**, so the guard could never cover the
  cases that matter. Feeds **D8**.

**Cross-harness (Claude Code / OpenCode / Kiro).** The convention lives in root `AGENTS.md` plus
**six** Claude agent prompts (v1.3, B3 + M9). **`skills/` is untouched** — §3.1 verified exactly one incidental docs-tree
mention repo-wide (`skills/joern-cpg/references/cpg-model.md`:66, a `docs/plans/<slug>-graph.md`
write path), and that spelling is *unchanged* by this convention. OpenCode reads `AGENTS.md`, so it
inherits the rule. **Kiro: unverified and unchanged from today** — it reads steering files and this
repo has no `.kiro/steering/` (§7), so Kiro sees neither the current convention nor the new one.
Flagged, not solved.

---

## 10. Migration plan for the naming

### 10.1 Recommendation: **rename nothing. Forward-only.**

**Measured, not asserted.** Renaming only the **6 active** milestone-prefixed documents would touch
**39 path-string occurrences across 15 files**:

```
git grep -l  -E '(m2-cpg-analysis-skill|m2-cpg-analysis-coordination|m2-cpg-analysis|m3-followups-coordination|m3-archive-sweep)\.md' -- '*.md'   # → 15 files
git grep -ohE '(m2-cpg-analysis-skill|m2-cpg-analysis-coordination|m2-cpg-analysis|m3-followups-coordination|m3-archive-sweep)\.md' -- '*.md' | wc -l   # → 39
```

Compare the sweep that triggered this entire assessment: **`9bbfbb5` archived 2 documents for 22
edits across 8 files.**

> **Renaming 6 documents costs ≈1.8× the whole sweep the stakeholder called too expensive — and
> buys cosmetics. A rename *is* a target move: it is the same 95.5% cost class §1 measured.
> Renaming to escape rename costs is self-defeating.**

Worse, **4 of those 15 citing files are dated records** (`docs/HISTORY.md` and three
`kaizen/inbox.md`), where the old name is *correct as written* — the same unreviewable judgement
class that sank the bulk repath (§4.1). And the **30 archived** milestone-prefixed documents are
off-limits by construction: under D4 they are frozen read-only history, and renaming them would
rewrite the record `9bbfbb5`/`649b02a` created.

**How the repo lives with mixed naming — and how a reader tells old from new.**

Honestly: **from the filename, they can't — and they don't need to.** Two sentences in root
`AGENTS.md` make the mix *documented state* rather than drift:

1. **An `m<n>-` or `k<nnn>-` prefix on an existing filename is part of that document's name, not a
   claim about its lifecycle.** `m3-executor.md` is simply what that document is called. Nobody
   should read meaning into it, and nobody should "fix" it.
2. **Lifecycle is read from the header, never from the filename** (§9.6). This is why **N3 is not
   optional**.

Old-vs-new is then answerable *mechanically*, and by the right signal — **one one-liner, not two**:

```bash
grep -m1 -H 'Status:' docs/plans/*.md        # the complete lifecycle listing
```

**v1.2 (minor m6): the `grep -L 'Status:'` one-liner is dropped.** v1.1 offered it as the mechanical
"old vs new" answer, but **N3 empties it for every active document in the same change** — post-N3 it
distinguishes nothing. Only the census one-liner earns a place in each module's `AGENTS.md`. (It is
also substring-matched; see the N3 done-condition, which is anchored.)

**v1.2 (minor m7): the 39/15 rename pricing below is a `git grep` figure and therefore
untracked-blind — read it as a lower bound.** This plan and its review are untracked today and cite
`m2-cpg-analysis-skill.md`, `m3-archive-sweep.md` and `m3-followups-coordination.md` several times;
once committed, the true rename cost is higher. This strengthens the rename-nothing conclusion, so
no action follows from it — but the same blindness applies to the S4 checker, where §3.3 caveat 2
already calls it out.

### 10.2 Are *any* renames worth it? — priced individually

| Candidate | Occurrences | Citing files | Payoff | Verdict |
|---|---:|---:|---|---|
| `falkor-chat/docs/reviews/m3-archive-sweep.md` → `archive-sweep.md` | **1** | **1** (`claude/architect/kaizen/inbox.md`) | mild | **No.** One edit is cheap, but it is a precedent that invites the other five, and the single citing file is a dated record. |
| `docs/plans/m2-cpg-analysis-skill.md` + its `reviews/` twin + `-coordination` + `docs/reviews/m2-cpg-analysis.md` | 13 + 3 + 4 + 4 = **24** | ~11 | family → `cpg-analysis-skill` | **No.** Entangled: `reviews/m2-cpg-analysis.md` and `reviews/m2-cpg-analysis-skill.md` are two reviews with *different* slugs for one topic (an N4-class defect), so renaming correctly means **re-slugging** — judgement, not `sed`. |
| `falkor-chat/docs/plans/m3-followups-coordination.md` | **7** (+4 bare mentions) | 3 | none | **No — and it is the counter-example that settles the question.** Its slug *is* the milestone: "M3 follow-ups" has no meaning without "M3". Renaming it requires **inventing** a topic name. That is design work, not migration. |
| Any of the 30 archived `m<n>-` documents | — | — | none | **No.** Frozen history under D4. |

> **Net: zero renames. No step below contains a `git mv`.**

One exception, recorded so it isn't rediscovered: **if
`falkor-chat/docs/reviews/k031-structure-read-impl.md` is ever touched for other reasons**, re-slug
it to `workflow-def-structure-read-impl.md` to repair the N4 family break — **4 occurrences across
3 files** (`falkor-chat/docs/{BACKLOG,HISTORY}.md`, `…/plans/m3-followups-coordination.md`,
verified). **File it as a backlog nit; do not schedule it.**

### 10.3 Sequenced steps → **see §12**

> **v1.3: the step definitions moved.** N0–N4 used to live here and S0–S5 in §4.3, which meant an
> implementer had to interleave two lists and reconcile their owners. **§12 is now the single,
> authoritative, ordered step list** — every S- and N-step, one owner each, files listed,
> done-condition and self-verifying check. §4.3 and this section remain as the *rationale* behind
> those steps; §12 is what gets executed.

**N0 — decision gate: CLOSED.** D6 is **ruled (a), adopt in full**; D4 was already accepted; D7 is
folded into D4 and D8 confirmed (§6). *No gate remains.*

**N5 — enforcement: prompt-level only** (D8, confirmed). **Do *not* add a check to `audit-team.sh`.**
Four verified reasons:

1. **Scope mismatch.** `audit-team.sh` declares itself *"the deterministic half of the team-coherence
   certification"* over the `claude/` agent collection (header, lines 1–6). A repo-wide docs-naming
   census is a different concern — the same argument as §3.3 item 3.
2. **It cannot start green.** 6 active + 30 archived `m<n>-` files, 4 `k<nnn>-` files, and **25 of
   26** documents without a *canonical* `Status:` line until N3 lands (v1.3, m13).
3. **`audit-team.sh` is already FAIL** (C-309a, 2 check-7 leaks). A second permanent red trains
   everyone to ignore the gate — the same reasoning that made S4 report-only.
4. **Check 7 is `git grep`-based and untracked-blind** (stated in the brief and confirmed by the
   script header). A naming check built the same way is blind at precisely the moment that matters:
   the file is untracked when the agent writes it. The write-time guards can't cover the gap either
   (§9.7, verified).

**What *is* worth doing — D2 is ruled (b), so S4 ships and this rides along:** add ~12 lines to
`claude/scripts/check-doc-links.sh` — a report-only naming + header census over the **active trees
only** (`--exclude-dir=archive`), flagging any feature document whose basename matches `^[mk][0-9]`
or that fails §9.6's conformance regex **within `head -6`** (v1.3, m12 — one window everywhere; the
"first 12 lines" wording is retired), with the **post-N3 state as the recorded baseline**
(expected: 6 legacy `m<n>-` + 2 legacy `k<nnn>-` names, 0 missing headers). The baseline is recorded
**in the script's own header**, not in `HISTORY.md` (m11). *Owner: `devops`/`cobb`.* **This is a
report, not the enforcement — M9's prompt-template line is the enforcement.**

### 10.4 Rollback

**There is no state to unwind.** Nothing is renamed, nothing is moved.

- **N1, N2, N4** are prose in tracked files → `git revert` restores the previous convention exactly.
- **N3** is 25 documents' worth of header lines (17 additions + 8 in-place normalisations) →
  `git revert`. **No citation anywhere points at a header field**, so removing them breaks nothing;
  the only loss is the lifecycle grep. The 8 normalisations are non-destructive by construction
  (§9.6 rule 3 preserves the existing text after the token), so a revert loses nothing either.
- **S6** is 16 prefix insertions in one file → `git revert`.
- The convention's forward-only nature means a revert leaves **no half-migrated state** — documents
  written under it are simply documents with sensible names.

### 10.5 "Do nothing" — the honest comparison

| | Do nothing | **N1–N4 (recommended)** | N1–N4 + renames (**rejected**) |
|---|---|---|---|
| Cost now | 0 | **v1.3 (final):** 1 `AGENTS.md` paragraph + **6 prompt edits** (+6 kaizen pairs, +6 README re-checks) + **25 header lines** + 16 prefix normalisations in `falkor-chat/AGENTS.md` + 3 deleted `../` tokens | + **≥39 path-string edits / 15 files** (a lower bound — m7), 4 of them dated records needing judgement — **for 6 documents** |
| Cost per new document | 0, but a coin-flip among 3 schemes | 0 — the grammar is what agents already emit, **minus a prefix** | 0 |
| Fixes *"M-something is not good for longer"* | ✗ — `qa-engineer.md`:28 actively licenses it (N1) | ✓ forward | ✓ forward + cosmetically backward |
| Fixes divergent-slug families (the K-031 defect, N4) | ✗ — recurs on every impl review; **`analyst.md`:51's existing slug rule stays unfollowable** (M3) | ✓ (rule 2 + `-impl` documented) | ✓ |
| Gives D4 a *working* lifecycle signal | ✗ — `Status:` census is **9 of 26**, and those 9 answer in 9 different vocabularies (only this plan's is canonical) | ✓ **26 of 26, one closed set** (M2; v1.3 per-file values in §12 step 3) | ✓ |
| Names a legal successor for a frozen document (B2) | ✗ — improvised (`-landing2`) | ✓ slug ordinal, §9.5 rule 5 | ✓ |
| Risk introduced | 3 schemes → 4, as each new agent guesses | mixed filenames, **documented as state** | rewrites frozen history; 4 dated records become wrong-as-written |

**Verdict, split honestly:**

- **Do-nothing is genuinely defensible for the naming *rule* alone.** Nobody is blocked by
  `m3-executor.md`; the cost of the status quo is slow (one more scheme, one more broken family per
  impl review), and the fix is cheap but not urgent. If the stakeholder wants to spend nothing here,
  the loss is real but small.
- **Do-nothing is *not* defensible for N3.** D4 is already accepted, which means `Status:` is now
  the lifecycle signal — and today it covers **9 of 26** active documents, in **9 different
  vocabularies** (only `docs/plans/doc-reference-convention.md` is canonical, and only because v1.2
  wrote it that way), while *looking* complete.
  A partial signal that presents as total is worse than the directory convention it replaced.

> **If exactly one thing on this plan gets done, do N3** — §12 step 3. It is 25 one-line edits, zero
> path strings, self-verifying, and it is what makes the already-accepted D4 actually work. **But
> pair it with N2e (§12 step 2), or it decays from document 27** (M9).

---

---

## 11. Review response — finding-by-finding

### 11.1 Part I (review of v1.1) — answered in v1.2, all 17 verified closed by the reviewer

Answering `docs/reviews/doc-reference-convention.md` Part I (verdict *needs changes*: 3 blockers · 5
majors · 9 minors). **Fixed: 14. Fixed with a variation: 2. Rejected with reasons: 1 (partial).**
**Part II's §8 audit independently re-verified all 17 and confirmed the arithmetic
(14 + 2 + 1 = 17 = 3 + 5 + 9); all three v1.1 blockers are closed at the level raised, and the three
corrections aimed back at the reviewer — including the extglob pattern — were upheld.** The rows
below are unchanged from v1.2 and are the audit trail; v1.3's work is in §11.2.

| ID | Finding | Disposition | Where |
|---|---|---|---|
| **B1** | Broken-link diagnosis false; composed form produced 100% of the baseline | ✅ **Fixed.** R3 rewritten; §1.3, §4.4, §6.1 corrected; **D1's recommendation reversed** to "permitted, never required"; D5 collapsed into S2 | §1.1 R3, §1.3, §2.2, §4.3 S2, §4.4, §6 D1/D5, §6.1 |
| **B2** | No legal name for a second primary document; `landing2` example misdiagnosed | ✅ **Fixed, with a variation.** Added the branch — but the ordinal goes on the **slug**, and v1.1's ordinal-on-the-role hatch is **withdrawn**, so the grammar gets *smaller*. `landing2` re-derived as an instance of the branch | §9.5 rule 5 |
| **B3** | `Status:` collides with `tico`'s gated vocabulary; "two prompt sentences" wrong | ✅ **Fixed** via the reviewer's preferred option: absorb `tico`'s values — **verbatim, not renamed**, so `:71` and `README.md`:8 never change. Scope corrected to **4 prompts + root `AGENTS.md`** | §9.6, §9.7, §3.2, §5, §10.3 N2 |
| **M1** | Forward-only naming won't hold; `qa-engineer.md`:28's "follow that" survives | ⚠️ **Fixed (2 of 3 suggestions); third rejected with evidence.** Adopted: rewrite `:28` with the override subordinated; state the rule as a **prohibition**. **Rejected:** duplicating it into the 5 other filename templates — I grepped for override clauses and **`qa-engineer.md`:27–28 is the only document-filename override in the collection** (the other hits are graph-label, code-style, build and test-framework conventions), so the `AGENTS.md` rule is contradicted nowhere else and 6 more prompt edits buy nothing | §9.7, §3.2 item 7 |
| **M2** | `Status:` unowned, un-normalized, over-modelled | ✅ **Fixed, all three.** N3 extended to **normalise all 26**; **`teco` named as owner** and §3.2 item 5 promoted to required; set cut — `draft`/`delivered` dropped (5 values, of which 2 are `tico`'s absorbed pair) | §9.6, §10.3 N3, §3.2 item 5, §5, §7 |
| **M3** | Finding N4's grep claim is false | ✅ **Fixed.** N4 restated (`ls`/glob-invisible, content-grep-visible); the **`analyst.md`:51-already-breached** observation adopted as the stronger evidence; **D6's (b)-is-a-false-economy claim re-derived** on the corrected premise | §9.1 N4, §6 D6 |
| **M4** | Census unreproducible, yet its numbers go into HISTORY; 4 broken links not 3 | ✅ **Fixed, with a variation.** Re-measured independently: **the reviewer's 4 is correct.** Took the *second* of the two offered fixes — **drop the numeric baseline** rather than commit the script, since S4 is optional and the stakeholder is cost-sensitive. **Plus a finding the review didn't have:** committing this plan adds **10** more illustrative "broken links", so any checker needs a stated placeholder-exclusion rule | §1.3, §4.4, §4.3 S2, §3.3 caveat 0, §6 D3 |
| **M5** | `Updated:` duplicates `git log`, unowned, unchecked | ✅ **Fixed.** Demoted to optional; required block cut to **3 fields**. §9.3's dependent "chronology lives in `Updated:`" claim explicitly withdrawn | §9.6 |
| **m1** | N3's self-check is a substring grep that false-positives on this plan | ✅ **Fixed.** Anchored to `^> \*\*Status:\*\*` with a value whitelist, checks all 3 fields; the false-positive document added to N3's list; `git diff --stat` corrected to the tracked count | §10.3 N3 |
| **m2** | §2.1 double-counts option A's saving | ✅ **Fixed.** Table now prices A's incremental saving over D as **0**; A justified on R2 alone | §2.1 |
| **m3** | §8's decision state inconsistent with §6 | ✅ **Fixed.** §8 now tables **all** decisions with their state and what each gates | §8 |
| **m4** | §2.3 still says 33 archive files | ✅ **Fixed.** Re-verified **34**; body corrected | §2.3 |
| **m5** | The `case`-glob "cannot express" claim is overstated | ✅ **Fixed, with a correction to the reviewer.** Claim withdrawn — **and the suggested pattern `!([m][0-9])*` doesn't work either**: tested, it matches `m3-x.md`. The working form is `!([mk][0-9]*)`. Both recorded; the two sound reasons stand | §9.7 |
| **m6** | `grep -L 'Status:'` stops discriminating once N3 lands | ✅ **Fixed.** Dropped; only the census one-liner goes into `AGENTS.md` | §10.1 |
| **m7** | Rename pricing is `git grep`-based, untracked-blind | ✅ **Fixed.** 39/15 labelled a **lower bound**, with the note that this strengthens the conclusion | §10.1 |
| **m8** | N3 carries ~17 judgement calls but is priced as mechanical | ✅ **Fixed.** Explicit derivation rules for `Status:`/`Owner:`/`Tracks:`, incl. `—` when nothing is tracked | §10.3 N3 |
| **m9** | D7 and D8 aren't stakeholder decisions | ✅ **Fixed.** **D7 folded into D4** as a consequence; **D8 recorded as an architect decision**, flagged for objection. D1 and D6 left as the two genuine forks | §6, §8 |

**Part I's open questions, answered.** (1) *D1 is stakeholder-only* — agreed, and it is now
**ruled**: no clickable links. (2) *B3's fork is a product question* — **avoided rather than
answered**: absorbing `tico`'s values verbatim means nobody has to rule on the gated semantics,
because they do not change. If the stakeholder later *wants* one lowercase vocabulary, that is a
separate, optional tidy-up with a real product cost, and I recommend against it. (3) *B2 needs a
naming call* — taken: **ordinal on the slug** (`executor2.md`), with v1.1's competing role-ordinal
hatch removed so only one rule exists.

---

### 11.2 Part II (re-review of v1.2) — answered in v1.3

Answering `docs/reviews/doc-reference-convention.md` **Part II** (verdict *needs changes*: 2
blockers · 4 majors · **10 minors listed under a "9 minors" verdict line** — m10 through m19; all 10
are disposed of below, and the discrepancy is noted, not silently absorbed).
**Fixed: 14. Fixed with a variation: 1. Fixed beyond what was asked: 1. Rejected with reasons: 0.**

| ID | Finding | Disposition | Where |
|---|---|---|---|
| **B4** | N3 can't pass its own done-condition on the three `tico` documents it's told not to change — `tico` writes `> Status:` **unbolded**, the anchored gate demands `^> \*\*Status:\*\*` | ✅ **Fixed, canonical form stated.** **Bold labels are canonical.** §9.6 now specifies the block **verbatim** with five explicit rules (bolding, separator, token-first, one window, the conformance regex). N2c bolds `tico.md`:37's labels; the three requirements documents are normalised **in form only**, values byte-identical. The distinction is stated in the plan's own words: **bolding a label is a form change, not a value change** — so `tico.md`:71 and `claude/README.md`:8 stay genuinely untouched, and §12 step 2 proves it by **content** match (m15) rather than by line number | §9.6, §9.7, §12 steps 2–3 |
| **B5** | `teco` is named owner of the recurring flip but its guard allowlists `docs/plans/*` only ⇒ every flip on a review/requirements/test document escalates to an interactive human approval | ✅ **Fixed as recommended, and extended.** **`teco` coordinates; each kind's owner performs** — the preferred option, matching the existing guard topology with **no hook edit**. §9.6 carries the full routing table, verified against all **9** `claude/*/hooks/` scripts. §7 gains the risk row. The rejected alternative (widen `teco`'s allowlist) is recorded with the reviewer's own reason: it permanently loosens a deliberately narrow guardrail and must be argued on its merits, not smuggled in. **Extended beyond the finding:** the same defect afflicts **S2/N4**, which v1.2 assigned to `teco` while it writes `docs/HISTORY.md`, `falkor-chat/docs/HISTORY.md` and two `BACKLOG.md`s — **re-owned to an implementer** | §9.6, §3.2 item 5, §4.3 S2, §5, §7, §12 steps 2 & 4 |
| **M6** | Dropping `delivered` removed the token rule 5's selector keys on; it now routes its own `landing2` example to "revise in place" | ✅ **Fixed as recommended — no token re-added.** The selector is now an **event test**, not a field read: *"Has the earlier document been approved, gated, or executed against?"* — Yes ⇒ successor, *even while `Status:` is still `active`*. `active`'s meaning column is amended to match (*"amendable in place until it has been approved, gated, or executed against"*) | §9.5 rule 5, §9.6 |
| **M7** | Rule 5 requires amending an `archived` document the plan says must not be amended, and applies `Superseded by:` to the `Extends:` case | ✅ **Fixed, both halves.** (1) *"Adding or updating a header pointer is **metadata, not an amendment** — it is the one edit permitted on an `archived` document"*, stated in both §9.5 and §9.6's `archived` row. (2) The pointers are **split into two pairs** in a table: `Supersedes:` ⇄ `Superseded by:` (earlier flips to `superseded`) and `Extends:` ⇄ **`Extended by:`** (earlier's `Status:` **unchanged** — it stays authoritative). `Extended by:` is a new optional field | §9.5 rule 5, §9.6 |
| **M8** | No legal name for a **new** milestone-scoped topic; the prohibition, §9.4 and rule 3 give three different answers | ✅ **Fixed as recommended, with the loophole closed.** The token goes **inside the slug, never as a prefix** — `followups-m4-coordination.md`. §9.4's *"never the filename"* is restated as *"never as a prefix, and never as a lifecycle claim"*; rule 3 needs no amendment because "M4 follow-ups" and "M5 follow-ups" **are** different topics. **Added beyond the suggestion: an explicit test for when the exception applies** — *the topic has no name without the token* — so it can't be used as a general escape from the prohibition. This is §10.2's own reasoning about `m3-followups-coordination.md`, pointed forward | §9.4 |
| **M9** | No producing agent's prompt tells anyone to **write** the header; `tico.md`:37 is the only header contract in all 13 prompts, so N3 decays from document 27 | ✅ **Fixed with the cheapest option, applied wider than suggested.** `cobb` adds **one template line to six prompts** — `architect`, `analyst`, `qa-engineer`, `teco`, `data-scientist`, `graph-dba` (N2e). The reviewer priced three (the skeleton-carrying prompts already open in N2); I take six. **Why the extra three:** `data-scientist` writes `plans/<slug>-ml.md` (4 files today) and `graph-dba` writes `plans/<slug>-graph.md`, and `teco` writes `plans/<slug>-coordination.md` (6 files) — covering half the producers guarantees decay in exactly those classes, which is the defect M9 names. Marginal cost over the reviewer's version: 3 one-line template additions + 2 kaizen pairs + 2 README re-checks. §7 gains the risk row too, so the residual is stated rather than assumed away | §3.2 item 7, §5, §7, §9.7, §12 step 2 |
| **m10** | §5 still carries a stale *"optional"* `teco` row contradicting the required one | ✅ **Fixed.** Row deleted, with a struck-through marker so the deletion is visible rather than silent | §5 |
| **m11** | §7's checker row says *"record the baseline in HISTORY"*, contradicting D3(a) | ✅ **Fixed.** Restated: the baseline lives **in the script's own header/report**. The same correction applied at N5 | §7, §10.3 N5 |
| **m12** | Three disagreeing `Status:` windows (`head -8` / "first 12 lines" / "immediately under the H1"), and `cpg-query-access.md`:11 sits outside all of them | ✅ **Fixed.** **One window, `head -6`, everywhere** — §9.6 rule 4, N3's gate, N5's census. And the missing instruction is now stated: an existing `Status:` elsewhere in the header block is **folded into the canonical line and removed from its old position** — never left in two places | §9.6, §12 step 3, §10.3 N5 |
| **m13** | *"The 17 to add"* lists **18** paths; this plan already conforms; the real workload is 25 | ✅ **Fixed by re-derivation, not by patching the prose.** Re-counted from disk: **26 active = 1 already conformant (`docs/plans/doc-reference-convention.md`) + 8 to normalise + 17 to add = 25 documents touched, 24 tracked.** §12's per-file table lists all 25 explicitly, so the arithmetic is no longer load-bearing — the list is | §4.4, §12 step 3 |
| **m14** | §9.1 (25 active, 8 reviews) and §10.3/§10.5 (26, 9) disagree inside one document | ✅ **Fixed by note + re-verification.** §9.1's table is labelled *as of `583e132` plus this plan*; **26 is the operative number** everywhere after it, decomposing as 12 plans + 9 reviews + 5 requirements, 24 tracked. Re-verified today | §9.1, §10.3, §12 |
| **m15** | N2c's done-condition is line-number-addressed and breaks the moment `:37` gains a line | ✅ **Fixed as recommended.** The check is now content-based: `git diff HEAD -- claude/tico/tico.md \| grep -E '^[-+].*Ready for design'` returns nothing, and `git diff HEAD -- claude/README.md` is empty. Promoted to §8's global verification because it is the proof that the riskiest change was safe | §8, §12 step 2 |
| **m16** | The `Status:` derivation rule marks live work *"do not execute or amend"* — `cpg-query-access.md` says *"→ in implementation"* yet its milestone is closed | ✅ **Fixed, both halves.** The rule is re-derived from **the document's own work** (*"its backlog item is closed in `BACKLOG.md`"*), **defaulting to `active` when in doubt** — not from the milestone. **And the reviewer's ten-minute suggestion is taken in full: §12 step 3 writes the expected `Status:`/`Owner:`/`Tracks:` triple beside each of the 25 paths**, converting 25 judgement calls into a reviewable list. Each value was derived against `docs/BACKLOG.md` / `falkor-chat/docs/BACKLOG.md` today | §12 step 3 |
| **m17** | `qa-engineer.md`:54 is a second override clause the §3.2 grep missed | ✅ **Fixed.** N2a's scope now covers `:54` as well as `:28`; §9.7 quotes it and states why it does **not** disturb M1's rejection (it is inside the file already being rewritten, so zero marginal cost) | §9.7, §12 step 1 |
| **m18** | The D6 fork and N3's independence aren't reconciled — if D6 = no, N3 ships a vocabulary written nowhere normative | ✅ **Dissolved by the ruling, and the rule recorded anyway.** D6 is **ruled (a)**, so N1 carries both halves and the fork is gone. The split the reviewer asked for is recorded for any future partial revert: **the §9.6 header block is D4-consequent and ships regardless; only the filename grammar / role set / collision rules are D6-gated** | §6 D6 |
| **m19** | The citation rule inherits M1's problem: it's asserted in one place against a corpus where the opposite spelling dominates the files agents read first (15 in `falkor-chat/AGENTS.md`, 59 in its `HISTORY.md`, 64 in its `BACKLOG.md`) | ✅ **Fixed with a variation — taken, but as its own step.** The reviewer suggested folding the `falkor-chat/AGENTS.md` normalisation into S1, which already edits its line 112. **I take the change but not the folding: it becomes S6**, because folding 16 path-string edits into S1 would destroy S1's *"the diff contains zero `*.md` path-string edits"* proof — the plan's single strongest verification. As its own step it has its own exact check (16 → 0). Re-counted on disk: **16**, not 15. The `HISTORY.md`/`BACKLOG.md` populations stay untouched — they are dated records (analyst O-2), and that is the distinction the reviewer drew | §4.3 S6, §4.4, §5, §12 step 6 |

**Part II's open questions, answered.** (1) *B5's routing is partly a policy question* — **resolved
without needing the policy call**: each kind's owner performs the flip, so `teco`'s guard is never
widened and the stakeholder is never asked to approve a loosening. (2) *D1 is yours and only yours* —
**ruled: no clickable links.** (3) *D6 is the one place to cut cost* — **ruled: adopt in full**;
the cut was available and was declined. (4) *M9 costs three sentences in three prompts* — **taken,
and widened to six**, with the cost delta stated in §3.2 and §10.5 rather than buried.

---

### 11.3 Part III (targeted spot-check of v1.3) — answered in v1.4

Answering `docs/reviews/doc-reference-convention.md` **Part III** (verdict *approve with
suggestions*: **0 blockers** · 3 majors · 0 minors; the six Part II findings B4, B5 and M6–M9 were
re-verified against the repo and confirmed closed, and §12 step 1 was confirmed executable as
written). All three majors are in **verification text, not design** — no decision is reopened.
**Fixed as recommended: 2. Fixed with a variation (the reviewer's own alternative reading taken): 1.
Rejected: 0.**

| ID | Finding | Disposition | Where |
|---|---|---|---|
| **M20** | Step 2's M9 coverage check `grep -lE '\*\*Status:\*\*' … # → 7` cannot pass: the plan tells six of those seven prompts to carry a **pointer** (*"Open the document with the header block from root `AGENTS.md`."*), which contains no `**Status:**`. A correct execution scores **1**, not 7 | ⚠️ **Fixed with a variation — the check was wrong, not the design.** The reviewer's suggestion was to inline the header template in all six prompts so the existing grep passes; I **keep the indirection and fix the check**. Two reasons, both verifiable: (1) root `AGENTS.md` is imported by the root `CLAUDE.md` (`@AGENTS.md`) and is therefore **already in every Claude Code agent's context, subagents included** — the "second hop" M20 prices is a lookup that has already happened, so the inline copy buys nothing an agent doesn't already have; (2) §9.6's governing rule is *"the one place the header block is stated … implementers copy, they do not paraphrase"* — inlining makes **eight** copies (§9.6, root `AGENTS.md`, six prompts) of a line whose fields are still settling, and the drift it invites is the failure mode B4 already cost this plan one round. What the reviewer is right about is the **instruction/gate contradiction**, and that is removed: the pointer is promoted to **one canonical M9 sentence, byte-identical wherever it lands** (stated once, in step 1), step 2's four rows are marked as *placement, not wording*, and the check becomes ``grep -lF 'the header block from root `AGENTS.md`' … # → 6`` plus `grep -c '\*\*Status:\*\*' claude/tico/tico.md # → 1`. Both measured at **0** today, so both are clean signals. Step 1 gains its own two-file version of the same grep | §12 steps 1 & 2 |
| **M21** | Step 1's done-condition is wrong twice: `git diff --stat # 9 files` (the 9th path, `claude/README.md`, is *"expected: no edit"*), and *"zero `*.md` path-string edits"* is violated by step 1 itself, which must write the filename grammar, its examples and the census one-liner into root `AGENTS.md` | ✅ **Fixed as recommended, both halves, and the invariant is defined rather than weakened.** The count is **8 files changed, 9 paths** — stated in the files-touched heading *and* in the check. The invariant now carries the reviewer's definition, promoted to §12's global-invariant block so every step inherits it: **a path-string edit is a change to an *existing* citation's target path**; newly authored grammar, examples and prohibitions are new text, and a line deleted whole is not a repath. Because no `grep -c` can decide that on its own, the plan now says so out loud and splits the checks by kind: steps with no new `.md`-bearing text keep an exact count; steps 1 and 4 get `git diff -U0 … \| grep -E '^[-+].*\.md'` **plus a stated inspection** — *no `-`/`+` pair is the same citation with a different path*. Steps 4 and 6 keep their exact numbers. The invariant §11.2/m19 called *"the plan's single strongest verification"* is therefore stronger, not looser: it now says what it means | §12 global invariants, §12 step 1 |
| **M22** | `falkor-chat/docs/BACKLOG.md`:5's **preamble** still says *"completed plan documents move to `archive/`"* — D4's abolished rule, in a living document, reachable by no step's file list (pre-existing; not introduced by v1.3) | ✅ **Fixed as recommended.** Added to **step 4**, which already opens that file, as item 4(c) with the replacement sentence written out. Two things verified before relying on the reviewer's note rather than after: (1) `:5` itself carries **no** `.md` string, and the sentence starts with *"completed"* at the end of `:4`, so `:4`'s ``[`HISTORY.md`](./HISTORY.md)`` citation need not move — **but the replacement text cites root `AGENTS.md`**, so step 4's `grep -c '^[-+].*\.md'` would have gone from 6 to 7. The check is therefore **split by side** — `-` → **3** (exact, the three `../` citations) and `+` → **4** (the same three plus the one `AGENTS.md` pointer, robust to however the new sentence wraps) — and a `grep -cE 'move[sd]? (to\|into) .{0,3}archive' falkor-chat/docs/BACKLOG.md # → 0` is added, measured at **1** today. (2) The reviewer's repo-wide sweep is reproduced in the step, so the implementer can see this is the last such sentence outside dated records | §12 step 4 |

**Part III's open questions:** none were raised, and none is opened here. D1, D4, D6 and
rename-nothing are untouched by v1.4, and **step 1 may be executed as written** now that its two
check lines are correct.

---

## 12. Execution — the final ordered step list

> **This section is the implementation contract.** Every decision is ruled (§6); nothing waits on
> anyone. Steps are **ordered**; each is **one commit** unless stated. **Owner convention:** `cobb`
> owns every agent-prompt, kaizen and `claude/README.md` edit; **an implementer** (`coder` — the one
> agent in the collection with **no** `PreToolUse` doc guard) owns every document edit; `teco`
> coordinates and verifies but performs no write outside `docs/plans/*`.
>
> **The canonical header block is stated once and verbatim in §9.6 — copy it from there, do not
> paraphrase it.** Its shape, for orientation only:
> `> **Status:** <token> · **Owner:** \`<agent>\` · **Tracks:** <id(s)> (<M<n>>)`.
> Value set: `Interviewing` · `Ready for design` · `active` · `superseded` · `archived`.
>
> **Ordering provenance.** The reviewer cleared **S1 + N1 + N2a + N2b as one commit, then S2 + N4,
> then S3**, with **N3 / N2c / N2d blocked only by B4 and B5**. v1.3 resolves B4 and B5, so the
> previously blocked work slots in as steps 2 and 3, and **S2+N4 moves after N3** so its `HISTORY`
> entry can record the final 26-of-26 figure rather than a projection. Steps 1, 4 and 5 are the
> reviewer-cleared sequence, unchanged in content.

**Global invariants, asserted after every step:**

```bash
bash claude/scripts/audit-team.sh          # FAIL, 2 — the C-309a baseline. No new FAILs, ever.
git diff HEAD -- claude/tico/tico.md | grep -E '^[-+].*Ready for design'   # → nothing
git diff HEAD -- claude/README.md          # → empty, unless a catalog row genuinely changed
```

**The path-string invariant, defined (v1.4, M21).** A **`*.md` path-string edit** is a change to an
**existing** citation's target path — a repath. Text that merely *contains* a `.md` string but is
**newly authored** — the filename grammar, its examples, a prohibition, the census one-liner — is
**not** one; nor is deleting a line whole. Under that definition, **every step's diff must contain
zero path-string edits except step 4** (exactly 3 `../` repaths) **and step 6** (exactly 16 prefix
insertions).

No single `grep -c` implements that definition, because it turns on whether a `-`/`+` pair is *the
same citation with a different path*. So: where a step's diff contains no new `.md`-bearing text,
the count is the check; where it does (steps 1 and 4), the check is `git diff -U0 -- <files> | grep
-vE '^(--- |\+\+\+ )' | grep -E '^[-+].*\.md'` — the filter drops the `--- a/…` / `+++ b/…` file
headers, which match `.md` and are not edits — **plus the stated inspection**: *no `-`/`+` pair
differs only in a path*. Steps 4 and 6 keep their exact numeric checks.

---

### Step 1 — S1 + N1 + N2a + N2b · *the convention lands* · **one commit**

**Owners:** `cobb` (root `AGENTS.md`, both prompts, both kaizen pairs, the `claude/README.md`
re-check) · **an implementer (`coder`)** for `falkor-chat/AGENTS.md` · `teco` coordinates and lands
them together. **The two `AGENTS.md` files and `qa-engineer.md` must flip in the same commit** —
otherwise `qa-engineer` keeps writing *"never into `archive/`"* against a rule with no `archive/`
destination (§4.2).

**Files touched — 9 paths, of which 8 are edited** (v1.4, M21: `claude/README.md` is a re-check with
an expected empty diff, so a correct execution reports **8 files changed**)**:**

| File | Edit |
|---|---|
| `AGENTS.md` (root), *"Module documentation convention"* bullet, lines 159–166 | Rewrite. Must state **all eight** items below |
| `falkor-chat/AGENTS.md`:112 | Reword the `docs/archive/` key-doc row: **frozen history from the previous convention, not a destination.** Lines 115–116 unchanged |
| `claude/qa-engineer/qa-engineer.md`:28 | **Rewrite** (M1): delete the `archive/` sentence **and** the `/milestone` clause; subordinate *"follow **that**"* → *"…follow that — **except the filename grammar, which is repo-wide (root `AGENTS.md`) and not component-negotiable**."* |
| `claude/qa-engineer/qa-engineer.md`:54 | **(m17)** Subordinate *"…and **doc conventions**…"* the same way, or the rewritten `:28` is contradicted 26 lines below |
| `claude/qa-engineer/qa-engineer.md`:29, :41 | **(M9)** +1 line in each of the test-plan and test-report structures: **the canonical M9 sentence** below |
| `claude/analyst/analyst.md`:51 | **+1 sentence** naming `-impl` as the implementation-review role (§9.4) |
| `claude/analyst/analyst.md`:53–61 | **(M9)** +1 line in the review skeleton: same header instruction |
| `claude/{qa-engineer,analyst}/kaizen/{plan,history}.md` (4 files) | Log the prompt edit, per `claude/AGENTS.md`'s same-change rule |
| `claude/README.md` | Re-check the `qa-engineer` and `analyst` rows — **expected: no edit** (they cite write paths, not the archive rule) |

**The canonical M9 sentence (v1.4, M20) — one string, byte-identical wherever it is added** (here,
and in step 2's four prompts). Copy it; do not paraphrase it, or step 2's coverage check will not
find it:

> Open the document with the header block from root `AGENTS.md`.

**Why a pointer and not an inlined template:** root `AGENTS.md` is imported by the root `CLAUDE.md`
(`@AGENTS.md`) and is therefore already in every Claude Code agent's context, subagents included —
the "second hop" costs nothing to resolve — and §9.6's *"one place the header block is stated"* rule
survives with two copies (§9.6, root `AGENTS.md`) instead of eight. Reasoning in full: §11.3, M20.

**The eight items root `AGENTS.md` must state** (each traced to its section):

1. **Status-marker rule** replacing the move rule — a doc that freezes gets `Status: archived` in its
   own header and **does not move** (§2.2, D4).
2. **Existing `archive/` trees are read-only history of the previous convention.** Nothing is ever
   moved into them again; nothing is un-archived (§2.3).
3. **Citation spelling:** a reference to another document is a **backticked path from the repo root**.
   A markdown link is **permitted and never required**; if written, its target must be **relative** —
   **never `/docs/…`, which agents cannot resolve** (state the reason, not just the rule — §7). (§2.2, D1)
4. **Filename grammar** `<component>/docs/<kind>/<topic-slug>[-<role>].md` (§9.2).
5. **The prohibition, stated as a prohibition:** *"a new document's basename never begins with
   `m<digit>`, `k<digit>`, or a date"* — plus **M8's exception**: when a topic genuinely *is* a
   milestone, the token goes **inside the slug, never as a prefix** (`followups-m4-coordination.md`),
   the test being *the topic has no name without it* (§9.4).
6. **The closed role set** — `(none)` · `-coordination` · `-ml` · `-graph` · `-rca` · `-impl` ·
   `-report`. Everything else is part of the topic slug (§9.4).
7. **Collision rules 1–5** (§9.5), including the **rules-2-and-4 pairing**, the **slug-ordinal
   successor** (`executor2.md`), M6's **event-based selector**, and M7's **two pointer pairs**
   (`Supersedes:` ⇄ `Superseded by:` / `Extends:` ⇄ `Extended by:`) with *header pointers are
   metadata, not amendment*.
8. **The 3-field header block, verbatim from §9.6**, its 5-value `Status:` set, the **B5 routing
   table** (who flips `archived`, by kind), the lifecycle one-liner
   `grep -m1 -H 'Status:' docs/plans/*.md`, and §10.1's sentence that **an existing `m<n>-` prefix is
   part of a name, not a lifecycle claim — nobody should read meaning into it, and nobody should
   "fix" it.**

**Done when:** all eight items are present in the one bullet; `qa-engineer.md` contains no filename
clause licensing a milestone and does contain the "not component-negotiable" clause; `analyst.md`
documents `-impl`; the canonical M9 sentence appears verbatim in `qa-engineer.md` (twice) and
`analyst.md` (once); and **8 files changed**, with **zero path-string edits** as §12's global
invariant defines the term.

**Self-verifying checks:**

```bash
git diff --stat                                   # → 8 files changed (9th path, claude/README.md,
                                                  #   is re-checked and must show NO diff)
# The path-string invariant (defined above): a repath of an EXISTING citation. Newly authored
# grammar, examples and prohibitions contain '.md' and are exempt, so this one is inspected, not
# counted:
git diff -U0 -- AGENTS.md falkor-chat/AGENTS.md \
        claude/qa-engineer/qa-engineer.md claude/analyst/analyst.md \
  | grep -vE '^(--- |\+\+\+ )' | grep -E '^[-+].*\.md'     # ('--- a/…'/'+++ b/…' headers filtered:
                                                           #  they match '\.md' but are not edits)
   # inspect: every '+' hit is NEW normative text (the grammar, its two examples, the census
   # one-liner, the M9 sentence); every '-' hit is text removed outright (qa-engineer.md:28's
   # archive sentence, root AGENTS.md's old bullet). NO -/+ pair may be the SAME citation
   # carrying a different path — that, and only that, is the invariant.
grep -cF 'the header block from root `AGENTS.md`' claude/qa-engineer/qa-engineer.md   # → 2  (:29, :41)
grep -cF 'the header block from root `AGENTS.md`' claude/analyst/analyst.md           # → 1
grep -n 'milestone' claude/qa-engineer/qa-engineer.md      # no filename/naming clause remains
grep -c 'not component-negotiable' claude/qa-engineer/qa-engineer.md   # ≥ 1
grep -c -- '-impl' claude/analyst/analyst.md      # ≥ 1  (verified 0 today — a clean signal)
grep -c 'archive/' claude/qa-engineer/qa-engineer.md       # 0
bash claude/scripts/audit-team.sh                 # FAIL, 2 (unchanged)
```

---

### Step 2 — N2c + N2d + N2e · *the prompts that produce and maintain the header* · **one commit**

**Owner:** `cobb` (all of it). This step contains the two fixes that blocked v1.2 — **B4** (the
canonical form) and **B5** (who performs the flip).

**Files touched (16):**

| File | Edit | Finding |
|---|---|---|
| `claude/tico/tico.md`:37 | The template becomes the §9.6 block: **bold `**Status:**` and `**Last updated:**`, add `**Owner:**`/`**Tracks:**`.** The two value strings `Interviewing` and `Ready for design` are **byte-identical** | **B4** |
| `claude/teco/teco.md`:65 | The documentation-curation scan: at milestone close, `teco` **lists** every document the close freezes and makes the `Status: archived` flip **a done-condition of the closing unit, routed to that document's owner** (§9.6 routing table). `teco` **coordinates; it does not perform** — its allowlist is `docs/plans/*` only. Required, not optional | **B5**, M2 |
| `claude/teco/teco.md`:56 | +1 line: coordination docs open with the header block | M9 |
| `claude/architect/architect.md`:40 | +1 line: plans open with the header block | M9 |
| `claude/data-scientist/data-scientist.md`:71–72 | +1 line: method notes and methodology reviews open with the header block | M9 |
| `claude/graph-dba/graph-dba.md`:51 | +1 line: design notes open with the header block | M9 |
| `claude/{tico,teco,architect,data-scientist,graph-dba}/kaizen/{plan,history}.md` (10 files) | Log each prompt edit | — |
| `claude/README.md` | Re-check the 5 rows — **expected: no edit** | — |

**The four M9 rows above add the canonical M9 sentence from step 1 — verbatim, byte-identical**
(*"Open the document with the header block from root `AGENTS.md`."*). The rows say **where** the
line goes; they are not the wording. `tico`'s row is the exception: it edits `tico`'s own template,
which carries `**Status:**` literally. (v1.4, M20.)

**Explicitly NOT touched, and this is the step's proof obligation:** `claude/tico/tico.md`:71 (the
stakeholder-gated *"Ready for design"* transition) and `claude/README.md`:8 (its user-facing
promise). **Bolding a label at `:37` is a form change, not a value change.** No hook file is edited
anywhere in this plan.

**Done when:** `tico.md`:37 matches §9.6's block; `teco.md`:65 names both the trigger (milestone
close) and the routing (each document's owner performs); all six producing prompts carry the
canonical M9 sentence verbatim (two of them landed in step 1).

**Self-verifying checks:**

```bash
# The B4 proof — content-matched, not line-numbered (m15):
git diff HEAD -- claude/tico/tico.md | grep -E '^[-+].*Ready for design'   # → NOTHING
git diff HEAD -- claude/README.md                                          # → EMPTY
# The B5 proof — no hook was touched:
git diff --stat -- 'claude/*/hooks/*' 'claude/scripts/guard-doc-writes.sh' # → EMPTY
# M9 coverage (v1.4, M20) — the six producing prompts carry the canonical M9 SENTENCE, not a
# '**Status:**' template: the header block stays stated in one normative place. Measured today: 0.
grep -lF 'the header block from root `AGENTS.md`' \
     claude/{architect,analyst,qa-engineer,teco,data-scientist,graph-dba}/*.md | wc -l    # → 6
# tico is the seventh producer and the one file that carries the block literally (B4). Today: 0.
grep -c '\*\*Status:\*\*' claude/tico/tico.md                              # → 1
bash claude/scripts/audit-team.sh                                          # FAIL, 2 (unchanged)
```

---

### Step 3 — N3 · *`Status:` normalisation across the 26 active feature documents* · **one commit**

**Owner:** an implementer (**`coder`**) — the write guards make a bulk cross-directory pass awkward
for the doc-scoped agents (`architect` can only write `plans/`, `analyst` only `reviews/`,
`tico` only `requirements/`), and `coder` has no doc guard at all. **This is the one-time backfill
only; the recurring flip thereafter follows §9.6's B5 routing table.**

**Content-only. Zero path strings. Zero renames. No archived document is touched** — under
`docs/archive/` a document is `archived` by construction.

**The arithmetic, re-derived from disk (m13, m14):** **26** active feature documents =
**1** already conformant (`docs/plans/doc-reference-convention.md` — this plan, no work) +
**8 to normalise** + **17 to add** = **25 documents touched, 24 of them tracked.**

**How to write each header** — §9.6, rules 1–5. Two rules do all the work:

- **Fold, don't duplicate (m12):** the canonical line goes **immediately under the H1**. Where a
  `Status:` line already exists lower in the header block — `docs/plans/cpg-query-access.md` has one
  at **line 11** — its text becomes the trailing clause and **the old line is deleted**. Never two.
- **Preserve, don't discard:** existing status prose is kept verbatim **after** the canonical fields
  (`· <existing text>`). **One stated exception, below.**

**The 25 documents, with the exact value to write (m16 — this list replaces 25 judgement calls).**
`Status:` is derived from **the document's own work** (its backlog item closed in `BACKLOG.md`),
**defaulting to `active` when in doubt** — *not* from whether its milestone is closed.

| # | Path | `Status:` | `Owner:` | `Tracks:` | Note |
|---:|---|---|---|---|---|
| **8 to normalise** ||||||
| 1 | `docs/plans/cpg-query-access.md` | `archived` | `architect` | `C-301…C-307 (M3)` | fold line 11 up; preserve its text |
| 2 | `docs/requirements/cpg-query-access.md` | `archived` | `tico` | `C-301…C-307 (M3)` | preserve *"Delivered ✅ — AC-1…AC-4 met…"*; keep `**Last updated:** 2026-07-25` |
| 3 | `docs/requirements/joern-cpg-pipeline.md` | `active` | `tico` | `C-201…C-208 · C-301…C-307 (M1–M3)` | **living** component requirements — C-308/C-310/C-312 still open ⇒ default `active` |
| 4 | `falkor-chat/docs/plans/graphrag-eval-ml.md` | `active` | `data-scientist` | `— (M2.5 quality track)` | M2.5 deferred, never delivered |
| 5 | `falkor-chat/docs/plans/workflow-def-structure-read.md` | `archived` | `architect` | `K-031 (M3 follow-ups)` | **the one preserve-verbatim exception**: the existing *"revised, awaiting re-gate"* is **stale** — the re-gate happened (`falkor-chat/docs/reviews/k031-structure-read-impl.md`, verdict *approve with suggestions*). Write the trailing clause as `v2 2026-07-24; re-gated and delivered` |
| 6 | `falkor-chat/docs/requirements/agent-import.md` | `Ready for design` | `tico` | `—` | **FORM ONLY** — bold the labels, add `Owner:`/`Tracks:`; value + `Last updated: 2026-07-22` byte-identical |
| 7 | `falkor-chat/docs/requirements/summary-nodes.md` | `Interviewing` | `tico` | `—` | **FORM ONLY** — `Last updated: 2026-07-12` |
| 8 | `falkor-chat/docs/requirements/workflow-dependence-overlay.md` | `Interviewing` | `tico` | `K-032 (M3 follow-ups)` | **FORM ONLY** — `Last updated: 2026-07-23` |
| **17 to add** ||||||
| 9 | `docs/plans/cpg-mcp-containerization.md` | `archived` | `devops` | `C-320 (M3)` | C-320 ✅ 2026-07-26 |
| 10 | `docs/plans/cpg-query-access-coordination.md` | `archived` | `teco` | `C-301…C-307 (M3)` | |
| 11 | `docs/plans/m2-cpg-analysis-coordination.md` | `archived` | `teco` | `C-201…C-208 (M2)` | |
| 12 | `docs/plans/m2-cpg-analysis-skill.md` | `archived` | `architect` | `C-201…C-208 (M2)` | |
| 13 | `docs/reviews/cpg-mcp-containerization.md` | `archived` | `analyst` | `C-320 (M3)` | |
| 14 | `docs/reviews/cpg-query-access.md` | `archived` | `analyst` | `C-301…C-307 (M3)` | |
| 15 | `docs/reviews/doc-reference-convention.md` | `active` | `analyst` | `C-322` | this plan's own review — **untracked** |
| 16 | `docs/reviews/m2-cpg-analysis-skill.md` | `archived` | `analyst` | `C-201…C-208 (M2)` | |
| 17 | `docs/reviews/m2-cpg-analysis.md` | `archived` | `analyst` | `C-201…C-208 (M2)` | |
| 18 | `falkor-chat/docs/plans/demo-environment-bringup.md` | `active` | `devops` | `—` | a **reusable runbook**, explicitly re-runnable — never archive it |
| 19 | `falkor-chat/docs/plans/local-model-ram-budget-ml.md` | `active` | `data-scientist` | `K-022 (M3)` | live advisory guidance; K-022 closing doesn't retire the RAM budget |
| 20 | `falkor-chat/docs/plans/m3-followups-coordination.md` | `active` | `teco` | `K-027 · K-031 (M3 follow-ups)` | K-031 ✅ but **K-027 still open** ⇒ the coordination run is live |
| 21 | `falkor-chat/docs/plans/wsl2-memory-diagnostic.md` | `archived` | `devops` | `—` | a dated point-in-time diagnostic; the run is over |
| 22 | `falkor-chat/docs/reviews/k027-parse-robustness.md` | `active` | `analyst` | `K-027 (M3 follow-ups)` | K-027 still open |
| 23 | `falkor-chat/docs/reviews/k031-structure-read-impl.md` | `archived` | `analyst` | `K-031 (M3 follow-ups)` | |
| 24 | `falkor-chat/docs/reviews/m3-archive-sweep.md` | `archived` | `analyst` | `K-025 (M3)` | |
| 25 | `falkor-chat/docs/reviews/workflow-def-structure-read.md` | `archived` | `analyst` | `K-031 (M3 follow-ups)` | |
| **not touched** ||||||
| — | `docs/plans/doc-reference-convention.md` | `active` | `architect` | `C-322` | **already conformant** — v1.3 wrote it |

**Done when:** the loop below prints nothing, and a spot check of documents 1, 2 and 5 shows the
pre-existing free text preserved *after* the canonical fields (§9.6 rule 3's non-destructive claim).

**Self-verifying check** — one window, `head -6`, matching §9.6 rule 4 and N5's census:

```bash
for f in docs/{plans,reviews,requirements}/*.md \
         falkor-chat/docs/{plans,reviews,requirements}/*.md; do
  h=$(head -6 "$f")
  grep -qE '^> \*\*Status:\*\* (Interviewing|Ready for design|active|superseded|archived)\b' <<<"$h" \
    && grep -q '\*\*Owner:\*\*' <<<"$h" && grep -q '\*\*Tracks:\*\*' <<<"$h" \
    || echo "NONCONFORMING $f"
done                                            # → prints nothing
git diff --stat                                 # → 24 tracked files, 0 path-string edits
grep -rn 'Status:' docs/plans/cpg-query-access.md | wc -l   # → 1 (the fold worked; no duplicate)
grep -m1 -H 'Status:' docs/plans/*.md           # → the complete lifecycle listing, the D4 payoff
```

---

### Step 4 — S2 + N4 · *record it, and fix the three links* · **one commit**

**Owner:** an implementer (**`coder`**), `teco` coordinating. **Re-owned from `teco` in v1.3 (B5,
extended):** this step writes `docs/HISTORY.md`, `falkor-chat/docs/HISTORY.md`,
`falkor-chat/docs/BACKLOG.md` and `docs/BACKLOG.md`, **none** of which is in `teco`'s
`docs/plans/*` allowlist — as assigned in v1.2 it would have raised four interactive approval
prompts.

**Files touched (4):**

1. **`docs/HISTORY.md`** — dated entry. **Carry only reproducible numbers** (D3(a)): the §1.3 table's
   one-line-regenerable counts (composed-form references **143**; milestone-prefixed docs **36**;
   active docs lacking a canonical `Status:` **25 → 0**) plus the **qualitative** findings (two
   anchoring conventions coexist; archival rot is confined to dated records; the composed form is the
   repo's only source of broken links). **Do NOT write 3 / 87 / 15** — no committed artifact
   regenerates them. State the exclusion rules (placeholder basenames, illustrative paths) beside any
   count, since they are what makes a number a choice rather than a fact. Record that **D1 was ruled
   (no clickable links)**, **D6 ruled (adopt)**, and **D2/D3 taken by default**.
2. **`docs/HISTORY.md`, the 2026-07-26 entry** — **correct the false claim** that *"`docs/test-plans/`
   and `docs/test-reports/` remain as empty active directories"*. **Verified false:** git tracks no
   empty directories and neither path exists.
3. **`falkor-chat/docs/HISTORY.md`** — dated entry: the convention change applies **forward-only**;
   the naming convention was adopted forward-only and **renames were explicitly declined**, citing
   §10.1's measurement (**39 occurrences / 15 files** to rename 6 documents, vs. **22 edits / 8
   files** for the entire triggering sweep) so the decision is recorded rather than re-litigated (N4).
4. **`falkor-chat/docs/BACKLOG.md`** — (a) **delete three extra `../` tokens** at **:785, :787, :895**
   (all three targets exist; the fix has no judgement content); (b) file the `k031-structure-read-impl.md`
   → `workflow-def-structure-read-impl.md` re-slug as an **opportunistic nit** (4 occurrences, 3
   files), **not scheduled work**; (c) **(v1.4, M22)** reword the **preamble blockquote at `:5`**,
   which still states the rule D4 abolishes — *"completed plan documents move to
   [`archive/`](./archive/)"*. This is a **living** document every agent working in the component
   reads, not a dated record, so O-2's "correct as written" protection does not apply, and after
   step 1 it would contradict root `AGENTS.md` in the same repo. Write:
   *"completed plan documents stay in place and are marked `Status: archived` (root `AGENTS.md`);
   `archive/` holds frozen documents from the previous convention."* **Line `:4` is not touched** —
   the sentence begins with the word *"completed"* at the end of `:4`, and `:4`'s
   ``[`HISTORY.md`](./HISTORY.md)`` citation stays exactly as it is. This is the only such surviving
   sentence outside dated records: a repo-wide `git grep -iE 'moves? (to|here|into) .?archive'`
   returns root `AGENTS.md`:163 and `falkor-chat/AGENTS.md`:112 (both step 1), `docs/HISTORY.md`:9
   and `falkor-chat/docs/HISTORY.md`:331, :697 (dated records, left alone), and this one.
5. **`docs/BACKLOG.md`** — two entries. (a) **`C-322` — documentation reference & naming convention**,
   recorded as **delivered** by steps 1–3 and citing this plan (`docs/plans/doc-reference-convention.md`)
   and its review. (b) **`C-323` — bulk repath to full root-anchoring (S5), deliberately deferred**,
   with this document's cost analysis cited: it buys ≤4.5% of future archival churn for a ~60-file
   judgement-heavy sweep. **Do not schedule C-323.** (Highest existing ID today is `C-321` — verified.)

**Self-verifying checks:**

```bash
# The path-string count, split by side (v1.4, M22): the ':5' reword deletes a line carrying no
# '.md' and adds one citing root `AGENTS.md`, so the two sides are no longer symmetric. The
# '--- a/…' / '+++ b/…' file headers also match '\.md' and are filtered out — they are not edits.
D() { git diff -U0 -- falkor-chat/docs/BACKLOG.md | grep -vE '^(--- |\+\+\+ )'; }
D | grep -cE '^-.*\.md'      # → 3  — the three '../' citations, and nothing else
D | grep -cE '^\+.*\.md'     # → 4  — the same 3, corrected, + the ':5' reword's 'root `AGENTS.md`'
                             #        pointer (1 occurrence, however the new sentence wraps)
grep -cE 'move[sd]? (to|into) .{0,3}archive' falkor-chat/docs/BACKLOG.md   # → 0  (was 1: the ':5' preamble)
ls falkor-chat/docs/reviews/workflow-def-structure-read.md \
   falkor-chat/docs/plans/workflow-def-structure-read.md \
   falkor-chat/docs/reviews/k027-parse-robustness.md              # all three resolve from the fixed links
grep -c 'empty active directories' docs/HISTORY.md                # → 0
grep -cE 'C-32[23]' docs/BACKLOG.md                               # ≥ 2 (C-322 delivered, C-323 deferred)
grep -cE '\b(87|15)\b' <(git diff -- docs/HISTORY.md falkor-chat/docs/HISTORY.md)  # inspect: no 3/87/15 baseline
```

---

### Step 5 — S3 · *the 3 forward-looking archival-rot repaths* · **one commit**

**Owner:** **`cobb`** — `kaizen/` files are agent-scoped.

**Files touched (2):** `claude/analyst/kaizen/plan.md` (**2** references) ·
`claude/architect/kaizen/plan.md` (**1** reference). Repath each to its
`falkor-chat/docs/archive/…` target.

**Explicitly left alone: the other 12** archival-rot references, in
`claude/teco/kaizen/{k001-run-brief,history}.md`, `claude/{analyst,architect,data-scientist,qa-engineer}/kaizen/{history,inbox}.md`
and `falkor-chat/docs/reviews/m3-archive-sweep.md` — **dated records, where the pre-move path is
correct as written** (analyst O-2's reasoning, concurred with).

**Done when:** 3 lines changed, in exactly 2 files; the 12 dated-record references are untouched.

```bash
git diff --stat            # 2 files, 3 changed lines
```

---

### Step 6 — S6 · *normalise `falkor-chat/AGENTS.md`'s own citations* · **one commit** *(v1.3, m19)*

**Owner:** an implementer (**`coder`**).

**File touched (1):** `falkor-chat/AGENTS.md` — prefix each of its **16** backticked
`` `docs/…` `` references with `falkor-chat/`. This is the file an agent reads **before** writing
anything in that component, so it is live guidance, not a dated record; O-2's "correct as written"
protection does not apply. **Deliberately its own step** so step 1 keeps its *zero path-string
edits* proof intact.

**Not in scope:** the 59 module-anchored refs in `falkor-chat/docs/HISTORY.md` and 64 in
`falkor-chat/docs/BACKLOG.md` — dated records and a per-item ledger. Leaving them is the same
judgement S3 makes, and S5 remains deferred.

```bash
grep -c '`docs/' falkor-chat/AGENTS.md     # → 0
git diff --stat                            # 1 file, ~16 changed lines, all prefix insertions
```

---

### Step 7 — S4 · *the checker* · **optional, last, and gates nothing** *(D2 = (b))*

**Owner:** `devops` or `cobb`. New file `claude/scripts/check-doc-links.sh`, invoked by
`audit-team.sh` as **check 8**, **report-only (prints counts, exits 0)**. No CI workflow — `C-309a`
leaves `audit-team.sh` already red, and a second permanent red makes the gate worthless.

**Must carry, in the script header:** (a) the **placeholder-exclusion rule**, written down —
*skip any target whose basename is a placeholder token (`x.md`, `relative.md`, `<slug>.md`)* —
without which the checker's first act is to flag **this very document** 10–13 times; (b) the
`git ls-files` untracked-blindness, stated rather than silently inherited; (c) **its baseline, in
the header — not in `HISTORY.md`** (m11). Plus the ~12-line naming + header census over the active
trees only (`--exclude-dir=archive`), using **`head -6`** and §9.6's regex.

**Justified by Finding R2 alone** (§3.3) — the 408 references an agent cannot resolve, plus
archival-rot delta detection. It is **not** justified by the 4 broken links, which step 4 removes.

---

### Not scheduled

- **S5 — the bulk repath of the 687 non-root-anchored references.** Filed as deferred in step 4
  (`C-322`). Costs more than it saves.
- **Renames: none.** Zero `git mv` anywhere in this plan (§10.2).
- **Hook edits: none.** No file under `claude/*/hooks/` or `claude/scripts/guard-doc-writes.sh` is
  touched (§9.6's B5 routing exists precisely to avoid this).

---

### If only part of this ships

**Steps 1 → 2 → 3 are the value.** Step 1 makes the rule normative, step 2 makes it self-sustaining
(M9), step 3 makes D4's lifecycle signal real. **Step 3 alone is the single highest-value step, but
pair it with step 2 or it decays from document 27.** Steps 4–6 are hygiene with real but smaller
payoffs; step 7 is genuinely optional.

