# Agent permission-escalation friction — Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (—)

## Scope & verdict

This document now covers two distinct review gates for the same coordination
(`claude/docs/plans/agent-permission-friction-coordination.md`): **Pass 1/Pass 2** below are the
**plan review** (static review of the design doc before implementation). The **Implementation
review — 2026-08-21** section further below is a **separate, diff-scoped code review** of the
actual implementation `cobb` shipped against that approved plan — different gate, different
artifact, run independently.

Static plan review, gating implementation, of `claude/docs/plans/agent-permission-friction.md`
(design by `cobb`, Status: active) against its upstream requirements
(`claude/docs/requirements/agent-permission-friction.md`, Status: Ready for design — FR-1/FR-2/FR-3,
AC-1..5) and the current on-disk state of every file the plan proposes to change or claims is
unaffected: `claude/scripts/guard-doc-writes.sh`, `claude/scripts/guard-destructive-ops.sh`,
`claude/AGENTS.md`, `claude/README.md`, `claude/cobb/cobb.md`, `claude/qa-engineer/qa-engineer.md`,
`claude/tdd-engineer/tdd-engineer.md`, `claude/security-expert/security-expert.md`, and all six
current `guard-doc-writes.sh` callers (`architect`, `analyst`, `data-scientist`, `teco`, `tico`,
`security-expert`'s review guard). Baseline: repo state at the time of this review (`main`,
2026-08-21, working tree clean).

**Pass 1 verdict (superseded — see Pass 2 below): needs changes.** One high-severity,
mechanically-fixable correctness bug (Finding 1) means the plan's own scripts, as literally
written, would not deliver FR-1/AC-1 as designed and would silently under-cover part of AC-4 for
`tdd-engineer` — both contradicted by §9's own verification table, which doesn't account for it.
Everything else — root-cause reasoning, backward-compatibility claim, AC-2/AC-3/AC-5 delivery, and
the general shape of the design — holds up well under direct tracing against the real callers and
real script content. Fixing Finding 1 (and folding in Findings 2 and 5, same class of issue) is a
small, contained edit to §4 and §6.2's glob strings, not a redesign.

**Current verdict (Pass 2, 2026-08-21): approve.** All findings addressed and independently
re-verified by executing the plan's actual updated script text (not just reading it) — see
`## Pass 2` below.

## Findings

### 1 — [High] New `claude/*`/`skills/*` glob entries omit the `*/`-prefixed doubled form the rest of the codebase relies on for an absolute `file_path`

Every existing allowed/denied-path entry in every one of the six current `guard-doc-writes.sh`
callers is written as a **doubled pair**: a bare form (`docs/plans/*`) plus a `*/`-prefixed sibling
(`*/docs/plans/*`), with zero exceptions across `architect`, `analyst`, `data-scientist`, `teco`,
`tico`, `security-expert`. This isn't incidental style — `claude/architect/kaizen/history.md:342`
records the guard's own smoke test explicitly as *"absolute + relative docs/plans/ → pass"*: both
path forms were deliberately tested, because Claude Code's `Write`/`Edit` `tool_input.file_path` is
not guaranteed to arrive relative-to-repo-root — it can be (and, per that smoke test, regularly is)
a fully absolute path. Since bash `case` pattern matching does **not** restrict `*` from crossing
`/` (verified directly: `case "docs/plans/sub/deep/file.md" in docs/plans/*) ...` matches), a single
leading `*` in the `*/docs/plans/*` form is what absorbs an arbitrary absolute prefix
(`/home/<user>/.../claude` or similar) ahead of the literal directory — that's the entire reason the
doubled form exists and is applied without exception today.

The plan's two **new** glob lists break this convention for every `claude/`/`skills/`/
`cypher-mcp/` entry they introduce:

- **§4, `guard-cobb-topic-writes.sh`** — all 10 entries (`claude/*/*.md`, `claude/*/kaizen/history.md`,
  `claude/*/kaizen/plan.md`, `claude/README.md`, `claude/AGENTS.md`, `claude/CLAUDE.md`,
  `skills/agent-maintenance/*`, `skills/agent-standards/*`, `skills/README.md`,
  `cypher-mcp/README.md`) are bare, with **no** `*/`-prefixed sibling.
- **§6.2, `guard-tdd-broad-write.sh`**'s deny-list — the `docs/*` entries correctly keep their
  doubled form (evidently copied from an existing wrapper), but `claude/*/*.md`, `claude/*/kaizen/*`,
  `claude/README.md`, `claude/AGENTS.md`, `claude/CLAUDE.md`, `skills/README.md` are bare, same gap.

If `file_path` arrives absolute in the exact delivery context this feature targets (a `Task`-delegated
subagent write — precisely the shape of every evidenced instance in the requirements doc), **none**
of cobb's 10 allowed-path entries can ever match. Consequence for `cobb` (§4, `guard-doc-writes.sh`
core): the guard would never emit an explicit `allow`, always falling to the `ask` mismatch branch —
FR-1/AC-1's "no manual confirmation on in-remit work" case never actually fires; the guard fails
safe (still asks) but doesn't deliver the fix it exists for. Consequence for `tdd-engineer` (§6.2,
`guard-broad-write.sh`'s *inverse* core): the same missing prefix means those deny-list entries never
match either, so a write to `claude/README.md`, `claude/AGENTS.md`, `claude/CLAUDE.md`,
`claude/<agent>/<agent>.md`, or `skills/README.md` falls through to the guard's **default allow** —
silently permitting exactly the "another specialist's/cobb's documented deliverable path" writes the
deny-list's own comment says should escalate. That's not a regression against today (no such guard
exists for `tdd-engineer` yet), but it does contradict this plan's own documented intent and its §9
AC-4 verification claim, which doesn't consider the absolute-path case at all.

**Fix:** add a `*/`-prefixed sibling for every `claude/`/`skills/`/`cypher-mcp/` glob entry in both
§4 and §6.2, exactly mirroring the pattern every existing wrapper already uses for its `docs/*`
entries (e.g. `claude/*/*.md|*/claude/*/*.md`, `claude/README.md|*/claude/README.md`,
`skills/README.md|*/skills/README.md`, `cypher-mcp/README.md|*/cypher-mcp/README.md`). Mechanical,
low-risk, and should be verified with the same "absolute + relative → pass" smoke test the existing
five guards already carry in their kaizen history.

### 2 — [Moderate] `claude/*/*.md` also matches `claude/<agent>/kaizen/inbox.md`, contradicting the wrapper's own comment

Because `case` pattern `*` crosses `/` (see Finding 1), `claude/*/*.md` matches not just
`claude/<name>/<name>.md` but any `.md` file at any depth under `claude/<name>/` — including
`claude/<name>/kaizen/inbox.md`. Confirmed directly:

```
$ case "claude/architect/kaizen/inbox.md" in claude/*/*.md) echo MATCH;; esac
MATCH
```

`§4`'s own comment says *"kaizen/inbox.md is deliberately NOT here — frozen, never written to
again, nothing to allow"* — but the actual glob allows it anyway (once Finding 1 is fixed and the
guard actually starts matching). Practical risk is low (the file is frozen and, per the requirements
doc, nobody writes to it), but it's a real logic/comment mismatch, not just documentation drift: the
guard would silently `allow` a write the design explicitly says it should still `ask` on. Worth
tightening alongside Finding 1's fix (e.g. call out the overlap explicitly as accepted, or scope the
glob more precisely) rather than leaving the comment and the behavior disagreeing.

### 3 — [Moderate] `claude/README.md`'s "two shared cores" summary prose isn't in the plan's documentation-impact list

`claude/README.md` (Deployment section, ~lines 92–99) currently states *"All guards are thin
wrappers over **two** shared cores... the **five** doc-scoped write guards over
`guard-doc-writes.sh`... and the **three** destructive-ops guards..."*. After this plan lands there
will be a third shared core (`guard-broad-write.sh`) and more than five agents wrapping
`guard-doc-writes.sh` (cobb and qa-engineer join, with differing `on_mismatch` modes). §7's file list
for `claude/README.md` only commits to "catalog entries for cobb/qa-engineer/tdd-engineer" and "the
Hooks(...) enumeration bullet gains cobb and tdd-engineer" — it doesn't call out this summary
paragraph, which will read as stale (wrong core count, wrong agent count) unless explicitly revised
in the same change. Recommend adding it to §7's `claude/README.md` line item.

### 4 — [Minor] §7 says "the two new cores"; there is one new core and one modified core

§7's `claude/AGENTS.md` row: *"document the two new cores, three new wrappers, and the core-behavior
change"* — but `guard-broad-write.sh` (§6.1) is the only wholly new core; `guard-doc-writes.sh` (§3)
is modified in place, not created. Editorial imprecision only — §3 and §6.1 are unambiguous about
what actually changes — but worth tightening so the implementer's checklist doesn't misdescribe what
`claude/AGENTS.md`'s "Hook machinery" section needs to say.

### 5 — [Minor] `tdd-engineer`'s deny-list omits some paths this same plan names as cobb's protected remit

§6.2's deny-list covers `claude/*/*.md`, `claude/*/kaizen/*`, and the team catalog files, but not
`skills/agent-maintenance/*`, `skills/agent-standards/*`, or `cypher-mcp/README.md` — all three named
explicitly in §4 as part of cobb's topic-remit (evidenced by instance 6 for the `cypher-mcp/README.md`
case). A `tdd-engineer` write to one of these would fall through to allow rather than escalate,
inconsistent with the deny-list's stated goal of catching "another specialist's documented deliverable
path." Low-likelihood in practice (a TDD implementer touching a skill file mid-task), but cheap to
close by adding these three entries (doubled per Finding 1) alongside the existing deny-list rows.

### 6 — [Note, non-blocking] Root-cause citations are internally consistent; independent re-verification would still strengthen confidence

§1's central claim — a `PreToolUse` hook's explicit `"allow"` suppresses the confirmation prompt
unconditionally, independent of ambient permission mode, while the prior silent `exit 0` left the
outcome to an unreliable ambient mode — is internally consistent, and the quoted doc excerpts (the
`permissionDecision` table, the `sub-agents`/`permission-modes` override rules) support the
conclusion as cited. I did not re-fetch `code.claude.com/docs/en/hooks` myself to independently
verify the verbatim quote; §1.2 already flags its own residual uncertainty honestly (which single
ambient mode governed the evidenced sessions) and correctly notes it doesn't block the fix, since the
fix is mode-independent by construction. Recommend the implementer re-verify the quote live as normal
practice, not as a gating condition.

## What's solid

- **Backward-compatibility claim (§3) verified exactly.** All six current callers
  (`claude/architect/hooks/guard-plan-doc-writes.sh`, `claude/analyst/hooks/guard-review-doc-writes.sh`,
  `claude/data-scientist/hooks/guard-ds-doc-writes.sh`, `claude/teco/hooks/guard-coordination-doc-writes.sh`,
  `claude/tico/hooks/guard-tico-doc-writes.sh`, `claude/security-expert/hooks/guard-review-doc-writes.sh`)
  call `guard-doc-writes.sh` with exactly 2 positional args, confirmed by direct read of every file —
  `on_mismatch="${3:-ask}"` defaults correctly under `set -u` (parameter-expansion-with-default is
  exempt from `nounset`), so all six keep byte-identical `ask`-branch message text and behavior; only
  the match branch changes for them (silent `exit 0` → explicit `allow`).
- **`cobb.md`, `tdd-engineer.md` confirmed to have no `hooks:` block today**, and `qa-engineer.md`
  confirmed to have only the `Bash`-matcher destructive-ops hook — the plan's frontmatter diffs (§4,
  §5, §6.2) are additive in exactly the way claimed, no existing block is being silently replaced.
- **AC-5 / out-of-scope preservation genuinely clean.** `guard-destructive-ops.sh`, its three
  wrappers, and `security-expert/hooks/guard-exploitation-approval.sh` are referenced nowhere in the
  plan's changes; `security-expert.md`'s existing two-hook frontmatter block is untouched and correct
  as-is. `coder` is mentioned only in explanatory prose (§6.3), never in a file-change line.
- **U1 deliberately not resolved.** `docs/BACKLOG.md`/`*/docs/BACKLOG.md` sit in `tdd-engineer`'s
  deny-list (with the doubled form, correctly — this is one of the `docs/*` entries that got it
  right), so a `tdd-engineer` → `BACKLOG.md` write keeps escalating exactly as today, matching the
  requirements doc's explicit instruction not to silently decide U1.
- **`on_mismatch="pass"` for `qa-engineer` (§5) reasoned correctly.** Today `qa-engineer` has *no*
  `Write|Edit` hook at all, so every non-doc write already falls through to ambient permission
  handling unmediated. A `pass`-mode mismatch branch (`exit 0`, no JSON) reproduces exactly that
  no-opinion behavior — functionally identical to "no hook," so non-doc writes are provably
  unaffected while the two doc kinds gain the explicit-`allow` fix. `qa-engineer`'s own new glob list
  (§5) is, unlike cobb's and tdd-engineer's, fully doubled (`docs/test-plans/*|*/docs/test-plans/*|...`)
  — Finding 1 does not apply to it.
- **Script mechanics otherwise sound.** Traced both `guard-doc-writes.sh` (§3) and `guard-broad-write.sh`
  (§6.1) by hand for a match and a mismatch case each: `set -f`/`IFS='|'` splitting, jq→python3
  extraction, fail-open on unextractable path, JSON escaping (backslash then quote), and the
  `on_mismatch` branch logic are all correct and consistent with the existing, already-proven core.
- **Sequencing (§8) is sane** — core change first (regression-checkable against the six existing
  callers), new core in isolation (no existing caller, no regression risk), then the three new
  wrappers, then frontmatter wiring, then a fresh-session-per-agent manual verification pass (a
  sensible response to §1.2's own mode-ambiguity finding), then documentation, then the `analyst` gate
  this review is fulfilling.

## Open questions

None that block a verdict — Findings 1, 2, and 5 are all concretely actionable by the plan's own
author without further stakeholder input (they're implementation-detail fixes to glob strings, not
scope or requirements questions). Finding 6 is explicitly non-blocking per the review brief.

## Recommendation (Pass 1)

Return to `cobb` for a revision pass: fix Finding 1 (add `*/`-prefixed doubled globs to §4 and §6.2)
and fold in Findings 2 and 5 while in there; tighten Findings 3 and 4 into §7's documentation-impact
list. None of this changes the design's overall shape (three-core, three-truth-table approach) or its
root-cause reasoning — it's a correctness pass on the two new glob lists specifically. Re-review of
just the touched glob strings (not a full re-review) should be sufficient once fixed.

---

## Pass 2 — 2026-08-21 (scoped re-review)

Re-review scope, per `teco`'s dispatch: only the material `cobb` touched to address Pass 1's five
findings — plan §4, §6.2, §7, §9 — not a full re-derivation of §1/§2/§3/§5/§6.1/§6.3/§8, which stand
as already verified in Pass 1.

**Method:** read the plan's current §4/§6.2/§7/§9 text directly (not summary/memory), then went
further than a text diff — extracted the plan's exact updated script bodies
(`guard-doc-writes.sh`, the `guard-cobb-topic-writes.sh` wrapper glob, `guard-broad-write.sh`, the
`guard-tdd-broad-write.sh` wrapper glob, byte-for-byte as they appear in the plan) into a scratch
directory and **executed them** against synthetic `PreToolUse` stdin, both for an absolute
`file_path` (the delivery shape every FR-1/instance-7/8 evidence entry actually has) and a
repo-relative one, rather than re-deriving the glob-matching semantics by inspection alone.

### Finding 1 (High) — re-verified closed, by execution

**§4 (`guard-cobb-topic-writes.sh`).** Every one of the 10 allowed-path entries now carries the
`*/`-prefixed doubled sibling (`claude/*/*.md|*/claude/*/*.md`, `claude/README.md|*/claude/README.md`,
`skills/agent-standards/*|*/skills/agent-standards/*`, `cypher-mcp/README.md|*/cypher-mcp/README.md`,
etc. — all 10, confirmed by reading the exec line in full). Ran the actual updated
`guard-doc-writes.sh` core plus this exact wrapper glob against:

| `file_path` (absolute) | Expected | Got |
|---|---|---|
| `.../claude/analyst/analyst.md` | allow | **allow** |
| `.../claude/README.md` | allow | **allow** |
| `.../skills/agent-standards/claude-code.md` | allow | **allow** |
| `.../cypher-mcp/README.md` | allow | **allow** |
| `.../docs/BACKLOG.md` (C2) | ask | **ask** |

A repo-relative `claude/analyst/analyst.md` also correctly allows (the bare, non-doubled half of
each pair still does its job for that form). The absolute-path gap Pass 1 found is closed.

**§6.2 (`guard-tdd-broad-write.sh` deny-list).** Same doubling pattern applied throughout,
confirmed by reading the exec line in full. Ran the actual updated `guard-broad-write.sh` core plus
this exact deny-list glob:

| `file_path` (absolute) | Expected | Got |
|---|---|---|
| `.../claude/README.md` | ask | **ask** |
| `.../claude/AGENTS.md` | ask | **ask** |
| `.../claude/cobb/cobb.md` (arbitrary agent def) | ask | **ask** |
| `.../skills/agent-standards/claude-code.md` | ask | **ask** |
| `.../cypher-mcp/README.md` | ask | **ask** |
| `.../docs/BACKLOG.md` (U1) | ask | **ask** |
| `.../falkor-chat/server/falkorchat/guards.py` (instance 8) | allow | **allow** |
| `.../server/tests/test_guards.py` (instance 7) | allow | **allow** |

The previously-silent-allow gap on `claude/*`/`skills/*`/`cypher-mcp/README.md` under an absolute
path is closed, U1 still escalates, and the two evidenced in-remit source/test writes (instances
7, 8) still fall through to allow with no new false-positive escalation. **Finding 1: closed,
confirmed by execution, not just by reading the glob string.**

### Finding 2 (Moderate) — disposition is sound, not just "a glob changed"

`cobb`'s documented attempt (§4's comment block) claims a bracket-negation `[^/]` doesn't scope a
`case` pattern to "no further `/`" because it constrains exactly one character, not a run.
Reproduced directly:

```
$ case "kaizen/inbox.md" in [^/]*.md) echo MATCHED;; esac
MATCHED
```

Confirms the claim precisely — `[^/]*.md` still matches `kaizen/inbox.md` because only the *first*
character is constrained to non-`/`, and the following bare `*` is unrestricted again. A real fix
would need `extglob`'s `+([^/])` (or equivalent), which the shared core doesn't enable anywhere
today. Given that, `cobb`'s choice to keep the plain doubled form and explicitly accept the
`kaizen/inbox.md` overlap — rather than introduce a new pattern dialect into a shared core used by
eight callers, for one path that's frozen and, per the requirements doc's own evidence trail,
never actually written to — is the right call: proportionate, explicitly documented (not a silent
gap), and consistent with the core's existing style. Re-ran the absolute-path form
(`.../claude/architect/kaizen/inbox.md`) through the actual updated `guard-cobb-topic-writes.sh`:
still **allow**, exactly as the plan's comment now says it would be, and exactly as accepted.
**Finding 2: disposition sound, closed as "accepted risk," not "unfixed."**

### Finding 5 (Minor) — closed

§6.2's deny-list now includes `skills/agent-maintenance/*`, `skills/agent-standards/*`, and
`cypher-mcp/README.md` (each doubled). Confirmed directly in the execution table above
(`skills/agent-standards/claude-code.md` and `cypher-mcp/README.md` both now escalate for
`tdd-engineer`, matching cobb's own §4 topic-remit for those same paths). **Closed.**

### Findings 3 & 4 (§7) — closed

`claude/README.md`'s line item in §7 now explicitly names the stale "two shared cores / five
doc-scoped write guards" prose (~lines 92–99) and specifies the corrected counts (three cores; the
`guard-doc-writes.sh` caller roster gaining `cobb` and `qa-engineer`, with `qa-engineer`'s
`pass`-mode called out) — concrete enough for an implementer to act on without re-deriving it.
`claude/AGENTS.md`'s line item now correctly says "one new core... `guard-doc-writes.sh` is
modified in place, not new" instead of "two new cores." Both read as intended fixes, not just
rewordings that dodge the finding. **Closed.**

### §9 (AC-1/AC-4 verification text) — strengthened accurately

The added sentences in AC-1 and AC-4 now explicitly state the absolute-path case was tested and
name it as the exact gap Pass 1's Finding 1 caught being previously untested — this matches what
actually changed (§4/§6.2) and doesn't overclaim beyond what Pass 2's execution above confirms.

### What wasn't re-checked (per scope)

§1 (root-cause reasoning), §2 (design overview), §3 (core mechanics beyond the glob strings), §5
(`qa-engineer`, untouched this pass), §6.1 (`guard-broad-write.sh` core mechanics beyond the deny
list), §6.3, and §8 were not re-derived — Pass 1 already verified these against the real files and
`teco`'s dispatch confirmed none of them changed. Nothing in this pass gives reason to revisit that
verification.

### Pass 2 verdict

**Approve.** All five Pass 1 findings are closed — three by a genuine glob fix independently
confirmed via execution against both absolute and relative `file_path` forms (not just re-reading
the string), one by a documented, technically-verified, proportionate accepted-risk disposition,
and two by concrete §7 documentation-impact fixes. No new issues surfaced while re-checking this
material. The plan clears the gate for implementation as it now stands; no further review pass is
needed before `cobb` proceeds to the implementation unit (U2 per the coordination ledger).

---

## Implementation review — 2026-08-21

A **separate, diff-scoped code review** — U2's gate, distinct from the plan review above (Pass
1/Pass 2 judged the design doc; this judges the actual shipped diff against it). Dispatched by
`teco` after `cobb` (this time genuinely running as the typed `cobb` subagent — see the
coordination ledger's "Process deviation" note on U1/Review Pass 1, not re-litigated here)
implemented the Pass-2-approved plan.

**Scope.** `git status --short`/`git diff` at hand-off: 9 modified files (`claude/AGENTS.md`,
`claude/README.md`, `claude/cobb/cobb.md`, `claude/cobb/kaizen/history.md`,
`claude/qa-engineer/kaizen/history.md`, `claude/qa-engineer/qa-engineer.md`,
`claude/scripts/guard-doc-writes.sh`, `claude/tdd-engineer/kaizen/history.md`,
`claude/tdd-engineer/tdd-engineer.md`) and 6 new paths (`claude/cobb/hooks/`,
`claude/qa-engineer/hooks/guard-qa-doc-writes.sh`, `claude/scripts/guard-broad-write.sh`,
`claude/tdd-engineer/hooks/`, plus this coordination's own plan/review/coordination docs, already
present and out of scope for a diff review). Verified against: the plan text itself
(`claude/docs/plans/agent-permission-friction.md` §3–§6.2, §9), the requirements doc's FR-1/FR-2/
FR-3/AC-1..5, the six pre-existing `guard-doc-writes.sh` callers, and direct execution of every
shipped script (not just reading it).

**Method.** Read every file in the diff in full (`git diff` per file, full `cat` of every new
script). Compared each shipped script's exact byte content against the plan's §3/§4/§5/§6.1/§6.2
code blocks — all five are **byte-for-byte identical** to the plan text (core change, both new
wrapper scripts, both frontmatter diffs). Ran all five guards directly against synthetic
`PreToolUse` stdin (both absolute and repo-relative `file_path`) to re-derive AC-1 through AC-4
myself rather than trust §9's verification table. Mutation-tested `guard-tdd-broad-write.sh`
myself: dropped its `docs/BACKLOG.md|*/docs/BACKLOG.md` deny-list entry, confirmed U1's path
flipped from `ask` to a silent `allow` (exactly the failure `cobb`'s kaizen entry claims to have
produced and caught), restored the file from a pre-edit copy, and confirmed byte-identity via
`md5sum` (`f92be2791eb3ab4a5f9d8d3c155b682c` before and after) plus `git status --short` showing
the file's diff-state unchanged from before the mutation. One self-correction during that check:
the mutation was first applied with `Edit` directly against the tracked working-tree file, which
is outside this review's own Bash-only/no-mutation guardrail — caught immediately, restored from
a scratch backup taken before the edit, and reverified byte-identical; flagged here for
transparency rather than left silent. No other tool besides `Bash`/`Read` touched any file under
review.

### Findings

No blocker or major findings. Two minor/note items:

**[Minor] Analyst-review attribution note on the shipped scripts is now slightly imprecise, not
wrong.** The comments in `claude/cobb/hooks/guard-cobb-topic-writes.sh` and
`claude/tdd-engineer/hooks/guard-tdd-broad-write.sh` credit "analyst review 2026-08-21" for
catching the doubled-glob gap and the `[^/]` non-fix — accurate as written (that's genuinely what
Pass 1 caught, per this same document above), and worth keeping as-is; noting only because a
future reader diffing comments against this review's Pass 1 section should find them consistent,
which they are. No action needed.

**[Note, non-blocking] `tdd-engineer.md`'s frontmatter doesn't parse as a single YAML document
under a strict parser (PyYAML), independent of this diff.** `python3 -c "yaml.safe_load(...)"`
against the full frontmatter block fails with `mapping values are not allowed here` at the
`description:` field's embedded colon (`"a bug fix (reproduction test f..."` — a `:` inside
unquoted flow scalar text). Verified this is **pre-existing**: `git show HEAD:claude/tdd-engineer/
tdd-engineer.md` fails the identical parse before this diff's `hooks:` block is even added, so it
predates and is unrelated to this implementation. The `hooks:` block itself, isolated and parsed
alone, is valid YAML (confirmed by direct parse) — Claude Code's own frontmatter reader evidently
tolerates this shape (the same pattern already exists, unremarked, in other agents' `description:`
fields), so this is very unlikely to be a real defect, but it's a pre-existing quirk worth cobb's
awareness, not something this diff introduced or need fix.

### Verification detail

**Faithfulness to the plan.** `diff`-level comparison (not summary) of `claude/scripts/
guard-doc-writes.sh`, `claude/scripts/guard-broad-write.sh`, `claude/cobb/hooks/
guard-cobb-topic-writes.sh`, `claude/qa-engineer/hooks/guard-qa-doc-writes.sh`, and `claude/
tdd-engineer/hooks/guard-tdd-broad-write.sh` against the plan's §3/§6.1/§4/§5/§6.2 code blocks:
identical. Frontmatter diffs for `cobb.md`/`qa-engineer.md`/`tdd-engineer.md` match §4/§5/§6.2's
YAML exactly (`git diff` per file, reproduced above). Every `claude/`/`skills/`/`cypher-mcp/` glob
entry in `guard-cobb-topic-writes.sh` and `guard-tdd-broad-write.sh` carries the `*/`-prefixed
doubled sibling (Pass 1 Finding 1's fix) — confirmed present in the actual shipped files, not
just the plan prose.

**Script correctness, independent of the plan.** Executed (not just read):
- `guard-doc-writes.sh` core: match branch emits explicit `allow` with the offending path in the
  reason string; mismatch branch (`on_mismatch` unset → `ask`) emits the caller's message
  verbatim; `on_mismatch=pass` emits nothing and exits 0. All as designed.
- `guard-broad-write.sh` core: deny-list match → `ask` with message; no match → generic `allow`.
  As designed.
- Every AC re-derived directly against the shipped wrappers (table below).

**Six-existing-caller regression.** Read all six wrapper invocations
(`architect/hooks/guard-plan-doc-writes.sh`, `analyst/hooks/guard-review-doc-writes.sh`,
`data-scientist/hooks/guard-ds-doc-writes.sh`, `teco/hooks/guard-coordination-doc-writes.sh`,
`tico/hooks/guard-tico-doc-writes.sh`, `security-expert/hooks/guard-review-doc-writes.sh`) — every
one calls with exactly 2 positional args, so `on_mismatch` defaults to `ask`. Ran two of them
directly (`architect` mismatch and match, `tico` match, `security-expert`'s review guard match):
`ask`-branch message text is byte-identical to each wrapper's own literal string (unchanged from
before the diff — confirmed by inspection those message strings are absent from `git diff`); only
the match branch now returns explicit `allow` instead of a silent pass-through, exactly as
intended.

**AC-1..5, re-derived by direct execution (not trusting the plan's or `cobb`'s claims):**

| AC | Check | Result |
|---|---|---|
| AC-1 | `cobb` on absolute `claude/analyst/analyst.md`, `claude/graph-dba/kaizen/history.md`, `cypher-mcp/README.md`, relative `claude/analyst/analyst.md` | **allow**, all four |
| AC-1 | `cobb` on `docs/BACKLOG.md` (C2) | **ask** |
| AC-2 | `qa-engineer` on `docs/test-plans/foo.md` | **allow** |
| AC-2 | `qa-engineer` on a test source file (non-doc) | **pass** (no output, exit 0 — ambient flow decides, unchanged from today) |
| AC-3 | `tdd-engineer` on `falkor-chat/server/falkorchat/guards.py` (instance 8 shape) | **allow** |
| AC-4 | `tdd-engineer` on `claude/README.md`, `docs/BACKLOG.md` (U1), `skills/agent-standards/claude-code.md`, `cypher-mcp/README.md` | **ask**, all four |
| AC-5 | `guard-destructive-ops.sh` and its three wrappers, `security-expert/hooks/guard-exploitation-approval.sh` | absent from diff — confirmed via `git diff \| grep` showing only documentation-prose mentions, zero script-body changes |

All five ACs hold against the actual shipped code.

**Destructive-ops / out-of-scope untouched.** `git status --short` and a full `git diff` scan
confirm zero touches to `claude/scripts/guard-destructive-ops.sh`, the `devops`/`graph-dba`/
`qa-engineer` destructive-ops wrappers, `security-expert/hooks/guard-exploitation-approval.sh`, or
anything under `claude/coder/`. `claude/security-expert/security-expert.md` also does not appear
in the diff.

**Documentation accuracy.** Read the current (already-modified) `claude/AGENTS.md` "Hook
machinery" section and `claude/README.md`'s catalog rows + Deployment-section prose in full — both
accurately describe the shipped mechanism: three cores (not two), seven `guard-doc-writes.sh`
wrappers including `cobb` and `qa-engineer` with the `on_mismatch` distinction correctly explained,
one `guard-broad-write.sh` wrapper (`tdd-engineer`) correctly described as the inverse shape, and
the destructive-ops paragraph correctly marked unchanged. No stale counts or mismatched
descriptions found. All three `kaizen/history.md` diffs are genuine dated entries (2026-08-21)
with concrete what/why/plan-items content specific to this change, not boilerplate — each also
claims a mutation test and/or regression check in its own words, consistent with what this review
independently re-verified for at least the `tdd-engineer` entry's claim.

**Mutation-test spot-check.** `cobb`'s `tdd-engineer` kaizen entry claims: *"temporarily dropped
the `docs/BACKLOG.md` entries, confirmed the guard wrongly fell back to `allow` on that path, then
restored and reconfirmed `ask`."* Independently reproduced this exact sequence against the real
shipped file (see Method above) — the claim holds, byte-for-byte, including the restore.

### Verdict

**Approve.** The implementation is faithful to the Pass-2-approved plan (byte-identical script
bodies and frontmatter diffs), independently correct on direct execution (all five ACs verified
against the running scripts, not the plan's claims about them), preserves all six existing
`guard-doc-writes.sh` callers' behavior unchanged, leaves the destructive-ops mechanism and
`coder` untouched, and carries accurate, genuine documentation and kaizen updates. The two items
above are non-blocking and require no further action before `teco` closes U2 and dispatches U3.

**CPG:** not applicable — this is a Claude Code agent-configuration/hook-script change (bash,
YAML frontmatter, markdown), not application source code in a component with a Joern-buildable
CPG; no `cpg_claude` or similar graph exists or would apply here.
