# `permission-default-mode.md` — Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (U10, `write-guard-classifier-gap-coordination.md`)

**Scope:** static review of `claude/docs/plans/permission-default-mode.md` (design, authored by
`cobb`) — the investigation into whether switching Claude Code's `defaultMode` away from `auto`
can close the delegated-write classifier gap left open after `write-guard-classifier-gap.md`'s
`permissions.allow`-rule approach was empirically refuted (`write-guard-classifier-gap-coordination.md`
§U7). Read in full, against: the coordination ledger (`write-guard-classifier-gap-coordination.md`,
U1–U10), the earlier refuted design (`write-guard-classifier-gap.md`), the durable KB entry this
document folds into (`skills/agent-standards/claude-code.md`, already committed at `773328c`
alongside the plan), the live repo state (`~/.claude/settings.json`, `.claude/settings.json`,
`.claude/settings.local.json`, all 13 agents' frontmatter), and current official docs, fetched
directly (`code.claude.com/docs/en/sub-agents.md`, `.../permission-modes.md`, `.../permissions.md`)
rather than taken on the document's word. This is a fresh, independent pass, not a continuation of
Pass 1/Pass 2 on the earlier, different document.

**Verdict: approve with suggestions.** The document's central factual work — the subagent
parent-mode-inheritance mechanism, the "every agent's `acceptEdits` frontmatter is dead
configuration" finding, and the recommendation to stay on `auto` — is accurately grounded and the
recommendation is sound. Two Major findings below identify places where the document's own
"docs-verified" standard slips: an explicit "gap in the docs" claim that the docs actually resolve,
and a safety claim about hooks that ignores this exact repo's own already-documented counter-evidence
sitting in the very file this document edits. Neither overturns the recommendation, but both should
be corrected before this reasoning is relied on again (it already has been, once, by inclusion in
`skills/agent-standards/claude-code.md`).

**CPG:** considered, not relevant — `cpg_claude` isn't loaded (confirmed via `mcp__cypher__query`
against `cpg_claude`, which reports only `kaizen_team`/`cpg_salesperson`/`cpg_falkorchat`/etc.
loaded), and even if it were, this review turns on permission-mode semantics and doc-verification,
not call-graph structure in the guard scripts the document references.

---

## Findings

### Major — §1.2's "Gap in the docs" claim is false; the docs resolve it, and the resolution should have been checked against the option space in §3

§1.2 states: *"Gap in the docs: neither passage states what happens when the parent is in
`default`, `plan`, or `dontAsk` — only `bypassPermissions`/`acceptEdits` (takes precedence) and
`auto` (forces auto) are covered."*

This is checkable, and wrong. `code.claude.com/docs/en/sub-agents`, "Permission modes" subsection,
the paragraph immediately preceding the one the document quotes (fetched verbatim,
`sub-agents.md:494`):

> "Set `permissionMode` to choose the permission mode a subagent runs in... If you leave it unset,
> the subagent inherits the main conversation's mode, which starts as auto mode on Pro, Max, and
> Team plans unless your settings or your organization change it. **Setting it overrides that mode,
> except in the cases described below.**"

The "cases described below" are exactly the two the document quotes (`bypassPermissions`/
`acceptEdits`-parent takes precedence; `auto`-parent forces `auto`). By the document's own logic,
that's an exhaustive if/else: for every *other* parent mode — `default` (Manual), `plan`, `dontAsk`
— the general rule applies and a subagent's own frontmatter `permissionMode` governs independently
of the parent's. This is not an edge case the docs are silent on; it's the stated default behavior,
with the two exceptions being the only carve-outs.

This matters because it directly touches §3's "No finer scope exists" claim, which was written
without ever considering (or ruling out) this branch: a coordinator's own session running in
`default` (Manual, via project/global `defaultMode: "default"` or `--permission-mode default`)
would leave every dispatched subagent's *already-declared* `permissionMode: acceptEdits`
frontmatter live at dispatch time — exactly the per-subagent-independent behavior §3 says doesn't
exist ("You cannot keep `auto`'s classifier coverage for a coordinator's own actions while getting
`acceptEdits`-style silent delegation for what it dispatches").

Having worked this branch through myself: it does **not** actually beat the `acceptEdits`-parent
candidate §3 already analyzes. Both a `default`-parent and an `acceptEdits`-parent land the
dispatched subagent in the same place (its own frontmatter's `acceptEdits`, either by the
general-rule branch or by the takes-precedence branch), but `default` is strictly worse for the
*parent's own* actions — Manual mode auto-approves nothing (every edit/Bash/network call prompts),
where `acceptEdits` at least silently clears file edits and the small filesystem allowlist. So the
`default`-parent option is dominated, and the document's ultimate "no viable narrower scope, stay on
`auto`" conclusion survives — but the document doesn't show this reasoning; it asserts a docs gap
that isn't there and never surfaces the option to dismiss it. A reader relying on this document's
"docs-verified" standard is told something false about what the docs cover.

**Suggested fix:** replace the "Gap in the docs" callout with the actual quote (`sub-agents.md`,
"Setting it overrides that mode, except in the cases described below") and add one paragraph in §3
explicitly naming the `default`-parent option and showing why it's dominated by the `acceptEdits`-
parent candidate already on the table, rather than omitting it from the option space entirely.

### Major — §1.3's hook-safety claim doesn't grapple with this repo's own already-documented counter-evidence, in the exact file it's being folded into

§1.3 concludes: *"switching a delegating parent to `acceptEdits` would **not** weaken the per-agent
escalation guarantee the guard hooks provide for a genuinely out-of-remit path: the hook's `"ask"`
still fires; only the in-remit `"allow"` path changes."* This rests entirely on the docs quote
"Extend permissions with hooks" being mode-independent (verified — the quote is accurate, checked
against `permissions.md:465`), plus the inference that nothing in it gates on permission mode.

But `skills/agent-standards/claude-code.md` — the exact file this document's cross-references
section says its §1.2/§1.3 findings are "being folded in as a durable addition in the same change"
(and which, per `git show --stat 773328c`, already *was* folded in, in the same commit as this
plan) — already carries a dated, live-reproduced, filed-upstream finding directly contradicting the
premise that a `PreToolUse` hook's `"ask"` decision reliably fires:

> "Hooks — `PreToolUse` 'ask' enforcement gap observed 2026-08-21, filed upstream (Claude Code
> 2.1.238, Auto Mode, not doc-sourced — contradicts current docs...) — four isolated live tests
> found a `PreToolUse` hook..., confirmed correctly wired and confirmed to compute `ask` in
> isolation, does not pause execution for the real matching command from either a Task-dispatched
> subagent or the main session itself... Treat `PreToolUse` 'ask' enforcement as unverified under
> Auto Mode until re-confirmed — matcher-agnostic and context-agnostic, not a narrow
> subagent-dispatch-only gap."

The four tests that produced this finding were all run under `auto` mode, so it's not proven that
the same failure reproduces under `acceptEdits` — the working hypothesis in that same file is that
*auto mode's classifier layer* is what silently overrides the `ask` decision, which would mean
removing the classifier (by switching to `acceptEdits`) could plausibly fix it as a side effect.
But that's exactly the point: it's a hypothesis, untested under `acceptEdits`, sitting in the same
document this design edits — and §1.3 doesn't cite it, weigh it, or flag it as an open question
affecting the safety of the §5 pilot. Presenting "the hook's `ask` still fires" as an established,
docs-supported conclusion, one paragraph away from folding new text into a file that already
documents this exact claim as empirically unconfirmed in this environment, is an internal
inconsistency the document should resolve rather than silently drop.

This doesn't change the recommendation (§5 doesn't propose adopting `acceptEdits` as a standing
default, and if anything an unresolved `ask`-reliability risk strengthens the case for staying on
`auto`) — but it does mean the §5 "if the mechanism is worth empirically confirming anyway" pilot
is missing a caveat it should carry: the pilot observes whether the *allow* path gets silently
approved, but says nothing about verifying whether the *ask* path (the actual safety net for a
genuinely out-of-remit write) still fires under `acceptEdits` either — which, per the KB's own
open question, is not something to assume.

**Suggested fix:** add a sentence to §1.3 acknowledging the `PreToolUse` "ask" enforcement gap
already on record in `skills/agent-standards/claude-code.md` (2026-08-21 entry) as unresolved under
`acceptEdits` specifically, and extend §5's pilot description to also observe whether an
out-of-remit write's hook `"ask"` actually prompts during the same `acceptEdits`-parent session,
not only whether an in-remit write's `"allow"` is silently approved.

### Minor — §4's "which agents carry the Agent tool" list undercounts by roster inspection

§4 names `architect`/`analyst`/`data-scientist`/`security-expert`/`tico` (plus `teco`) as "all of
which also carry the Agent tool per the roster," implying these are the delegation-capable set.
Checked directly against every agent's frontmatter (`claude/*/*.md`): those six do declare an
explicit `tools:` list that includes `Agent`, but the other seven (`coder`, `tdd-engineer`,
`qa-engineer`, `graph-dba`, `devops`, `frontend-engineer`, `cobb`) declare **no** `tools:` field at
all — and per the subagent tool-inheritance rule ("Agents that omit `tools:` inherit everything"),
they inherit the `Agent` tool too. So the Bash-friction blast radius §4 describes ("every subagent
[a delegating session] dispatches") should be understood to potentially apply through any of the 13
agents dispatching further, not only the six named — a broader exposure than stated, which if
anything makes §4's cost case *stronger*, not weaker. Doesn't change the conclusion; the roster
citation should just be accurate.

**Suggested fix:** either say "every one of the 13 agents can, in principle, dispatch via `Agent`
(the six named declare it explicitly; the other seven inherit it by omitting `tools:`)," or narrow
the claim to "agents whose documented role routes through delegation" if that's what was meant.

---

## What's solid

- **§1.1's mode table and §1.2's core inheritance quotes are verbatim-accurate.** Checked both
  quoted passages directly against a fresh `curl` of `code.claude.com/docs/en/sub-agents.md` and
  `.../permission-modes.md` (not the document's own paraphrase) — byte-for-byte matches, including
  the `acceptEdits` Bash allowlist (`mkdir, touch, rm, rmdir, mv, cp, sed`), the "How auto mode
  handles subagents" three-step description, the "which mode a session starts in" three-step
  decision order, the classifier's default-blocked-actions list (force push, `git reset --hard`,
  `curl | bash`, secret exfiltration, IaC destroy), and "Asking Claude in chat to change the
  permission mode doesn't work." This is a document that does the verification work it claims to,
  in every place I independently re-checked except the two Major findings above.
- **§2's "dead configuration" finding is independently confirmed.** `grep -l 'permissionMode:
  acceptEdits' claude/*/*.md` returns all 13 agents, exactly as claimed; `~/.claude/settings.json`
  does carry an explicit `"defaultMode": "auto"` pin at user scope; the project's `.claude/settings.json`
  and `.claude/settings.local.json` carry no `permissions.defaultMode` key at all — all matching the
  document's own direct reads.
- **§4's cost-benefit reasoning is sound on its own terms.** The Bash-allowlist gap between `auto`
  and `acceptEdits` (silent classifier clearance vs. a small fixed filesystem list, with everything
  else — test suites, `git commit`, `docker`/`redis-cli` — reverting to a plain prompt) is correctly
  characterized, and the §1.2 inheritance rule's consequence (a parent's own mode switch propagates
  to everything it dispatches) is the right lens for judging blast radius.
- **§5/§6 are honest about what they are** — a recommendation not to adopt, an optional low-commitment
  empirical pilot clearly marked as such, and a rollback story that's actually trivial (a JSON key).
  No overclaiming past what §1–4 established, aside from the two gaps above.
- **The AC-3 cross-reference is accurate** — checked `agent-permission-friction2.md`'s AC-3 directly;
  the document's characterization of it ("a pre-existing condition shared with all five
  phase-1-fixed agents, not a regression... outside this document's scope") matches verbatim.

## Open questions

- Given `skills/agent-standards/claude-code.md` already carries this document's §1.2/§1.3 findings
  as a durable addition (committed at `773328c`, same commit as the plan), should the two
  corrections above land as an amendment to *both* documents, or just this one with a pointer? The
  KB entry doesn't repeat the false "docs gap" claim, but it also doesn't cross-reference the
  `PreToolUse` "ask" enforcement gap the way §1.3 should — that omission exists in both places.
- Is the §5 optional per-launch pilot actually going to run, and if so, should its scope be widened
  (per the second Major finding) to also test an out-of-remit write's `"ask"` path, not just the
  in-remit `"allow"` path? That's a stakeholder call, not one this review can make.
