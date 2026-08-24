# Write-guard auto-mode classifier gap — Design review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

**Reviewed:** `claude/docs/plans/write-guard-classifier-gap.md` (Status: active, Owner: `cobb`).
**Baseline:** the root-cause investigation it builds on (commit `6193083`,
`skills/agent-standards/claude-code.md`), `claude/docs/requirements/agent-permission-friction.md`
AC-4, `claude/docs/requirements/agent-permission-friction2.md`, and the live hook wrapper scripts
under `claude/*/hooks/*.sh` (read directly, not taken from the plan's citations). This is a design
review, not a re-check of the already-committed RCA — the root cause is taken as given.

**CPG:** considered, not relevant — no `cpg_claude` graph is loaded (`mcp__cypher__query` against
this instance lists `kaizen_team`, `cpg_salesperson`, `cpg_falkorchat`, and three `ws:`/`reference`
graphs; nothing for `claude/`). The reviewed artifact is a settings/hook-mechanics design touching
a handful of short, directly-readable bash wrappers (`claude/*/hooks/*.sh`, `claude/scripts/guard-*.sh`)
and a `.claude/settings.json` rule-syntax proposal — reading and grepping those files directly was
more precise than a call-graph tool would offer at this scale, even if a graph existed.

**Verdict: needs changes.** The docs-grounding and the general reasoning are sound and mostly
well-evidenced (re-verified independently below), but §5's candidate list contains one glob group
that should not go through the review gate as currently framed — it trades away more than the
plan's own §4 analysis accounts for — and two of the table's per-candidate glob descriptions are
inaccurate in a way that misrepresents how much overlap/redundancy exists among the "distinct"
decisions §8 asks the gate to sign off on individually.

---

## Findings

### Blocker — cobb's candidate glob union (§5, last row) includes prompt-governing files that §4's tradeoff analysis doesn't differentiate from the doc-kind rows

§5 recommends "one `Edit(...)` rule pair per union member, same list" for `guard-cobb-topic-writes.sh`'s
allowlist. I read the live wrapper
(`claude/cobb/hooks/guard-cobb-topic-writes.sh`) — its union is not homogeneous the way the plan's
one-line gloss ("agent defs, kaizen, catalogs, cobb's own skills, `cypher-mcp/README.md`") suggests:

- `claude/*/*.md` / `*/claude/*/*.md` — **every agent's own definition file**, i.e. its system
  prompt, tool grants, `hooks:` wiring, and `permissionMode`.
- `skills/agent-maintenance/*`, `skills/agent-standards/*` — skill packages whose bodies get
  injected into sessions that reference them (per-agent behavioral content, same trust class as
  an agent definition).
- `claude/README.md`, `claude/AGENTS.md`, `claude/CLAUDE.md` — `AGENTS.md`/`CLAUDE.md` auto-load
  into **every** subagent's context via the always-loaded memory hierarchy (confirmed in
  `skills/agent-standards/claude-code.md`, "What loads into a subagent": "The full `CLAUDE.md`/
  memory hierarchy still auto-loads... a `CLAUDE.md` of just `@AGENTS.md` reaches the subagent").
- vs. the kaizen-history/plan files and `cypher-mcp/README.md`, which are lower-stakes raw-capture
  and cross-component pointer docs.

The first three bullets are **prompt-governing artifacts**, not process documentation. I checked
whether Claude Code's own "protected path" carve-out (which forces classifier review even past a
matching allow rule — verified verbatim against `code.claude.com/docs/en/permissions`: "writes to
protected paths route to the classifier even when an allow rule matches") would catch these and
close this gap on its own. It doesn't: the doc names `.claude` (dot-prefixed) as a protected path
example ("`bypassPermissions` mode skips permission prompts, including for writes to protected
paths such as `.git` and `.claude`"), but the repo's actual agent definitions are edited at
`claude/<name>/<name>.md` — a **plain, non-dot-prefixed** top-level directory in this repo,
confirmed by `ls -la ~/.claude/agents/`: every entry there is a **symlink out to
`/home/mauricio/prg/graphmind-ai-lab/claude/<name>/`**, so the `tool_input.file_path` an agent
actually edits is the `claude/...` project path, not anything under a literal `.claude/` prefix.
Nothing about this design's candidate rules would route through the protected-path carve-out.

Consequence: adding `Edit(claude/**/*.md)` (or the doubled equivalent) as a project-wide allow rule
would let **any** agent, or a prompt-injected/compromised session, silently rewrite its own or
another agent's system prompt, hook wiring, or tool grants — or the `agent-maintenance`/
`agent-standards` skill bodies that define the team's own maintenance protocol — with **zero**
classifier involvement and zero human visibility. This directly reopens the exact hazard
`skills/agent-standards/claude-code.md` already documents as caught and correctly stopped: the
2026-08-20 `cobb` incident where "a delegate proposed adding a persistent bypass rule to its own
settings, citing the coordinator's authorization rather than the user's" was flagged
("Auto-Mode Bypass/Self-Modification") and not acted on. A standing `Edit(claude/**/*.md)` allow
rule is a standing, always-on version of exactly that route — the classifier that caught the
one-off attempt would never see a future one, because the write would resolve at rule-step-1,
before the classifier is ever invoked.

§4's framing treats every candidate glob as the same shape of trade ("per-agent → per-path"
narrowing of AC-4's escalation guarantee). That's the right frame for the doc-kind rows (see next
section), but it understates this one: for `claude/*/*.md` and the two skill-package globs, the
trade isn't "any agent can silently write a review/plan doc instead of only the one that owns it"
— it's "any agent can silently rewrite the prompts and hooks that define every agent's own
constraints, including its own." That is a categorically different risk, and §8's "is this an
acceptable trade for each candidate glob" question needs an explicit **no** for these three glob
members specifically, not a blanket answer alongside the rest of cobb's row.

**Suggested fix:** split cobb's row in §5 into two: (a) `claude/*/kaizen/history.md`,
`claude/*/kaizen/plan.md`, `claude/README.md`/`AGENTS.md`/`CLAUDE.md`, `cypher-mcp/README.md` —
keep as candidates, same reasoning as the doc-kind rows below (still worth an explicit call-out
that `AGENTS.md`/`CLAUDE.md` land in every session's context, so a bad silent edit has unusually
wide blast radius even though it isn't a privilege-escalation vector); (b) `claude/*/*.md`,
`skills/agent-maintenance/*`, `skills/agent-standards/*` — **exclude**, same conclusion and same
reasoning §6 already applies to `tdd-engineer`: leave these hook-only, accept the friction, because
the exposure this specific glob group would create is worse than the friction it removes.

### Major — §5's per-candidate risk isn't differentiated for "gate/verdict" doc kinds vs. plain process docs (task 1 of the brief)

Per the brief's request to judge each candidate glob rather than the block as a whole:

| Candidate | My read |
|---|---|
| `docs/reviews/**` (analyst, security-expert) | **Real, non-trivial exposure — flag to the gate explicitly, don't wave through.** A review document is the artifact that records an independent verdict on someone else's work (`AGENTS.md`'s "Owner altitude" convention: `analyst`/`security-expert` verify, `teco` integrates only after). If any agent can silently write there, the agent whose work is under review can, in principle, edit its own review's verdict with no escalation — the one thing this doc kind exists to prevent. Acceptable only if the team is comfortable treating "review integrity" as backstopped by git history/diffing rather than by the write-time guarantee — that's a real call, not a rubber-stamp. |
| `docs/test-reports/**` (qa-engineer) | **Same category as reviews** — a test report is QA's independent verification result. Same self-approval risk, same recommendation: flag explicitly, don't lump in with `docs/test-plans/**`. |
| `docs/test-plans/**` (qa-engineer) | **Fine as the pilot.** A test *plan* is a forward-looking declaration of what will be tested, not a pass/fail verdict — lower stakes, and it's already the plan's own choice of pilot (§5 sequencing) for good, independent reasons (narrow, low ambient traffic). |
| `docs/plans/**` (architect, data-scientist, teco) | **Moderate, acceptable.** A silently-edited plan could mask a mid-implementation deviation from what was approved, but plans get a companion review (`reviews/<slug>.md`) in the normal family flow, which is a real backstop the review/test-report kinds don't have to the same degree. |
| `docs/requirements/**` (tico) | **Moderate-to-major, worth a callout.** Requirements documents carry the acceptance criteria a deliverable is judged against — silently loosening AC text is the same "move the goalposts unnoticed" risk as a review self-edit, just one step upstream. Worth the same explicit flag as reviews/test-reports rather than bundling with `docs/manuals/**`. |
| `docs/manuals/**` (tico) | **Fine.** End-user-facing prose, no gate/verdict role. |

None of this rises to the cobb-row blocker — these are documentation-integrity risks backstopped by
version control, not privilege escalation — but the brief specifically asked for a differentiated
answer, and §4/§8 as written invite the gate to treat "is this an acceptable trade" as one judgment
per row rather than naming which rows carry the sharper trade. Recommend the design call out
`docs/reviews/**`, `docs/test-reports/**`, and `docs/requirements/**` as the three needing an
explicit "yes, we accept losing the write-time self-approval guard here" from the stakeholder,
separately from the rest.

### Major — two of §5's "verify exact glob" rows describe a narrower scope than the live hook actually has, which changes the exposure picture the gate is asked to sign off on

I read the live wrappers rather than trusting the table (which itself flags this as unverified —
§5's own caveat). Two of the four rows marked "verify exact glob before implementing" are not just
stale citations, they're **descriptively wrong about the shape**, in a way that matters for §8:

- **`teco`'s row** says `docs/plans/*-coordination.md` **shape** (suffix-restricted). The live
  wrapper (`claude/teco/hooks/guard-coordination-doc-writes.sh`) has no suffix restriction at all
  — its glob is `docs/plans/*|*/docs/plans/*`, byte-identical to `architect`'s row.
- **`data-scientist`'s row** says `docs/plans/*|docs/reviews/*` **shape** ("ML-scoped"), implying
  a `-ml.md`-suffix restriction. The live wrapper
  (`claude/data-scientist/hooks/guard-ds-doc-writes.sh`) has no such restriction either — its glob
  is `docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*`, i.e. the union of `architect`'s
  and `analyst`'s globs, verbatim.

This isn't academic: if implemented "mirroring the live glob" as §5 instructs, the `teco` and
`data-scientist` rows would produce **rules that are exact duplicates** of the `architect` and
`analyst`/`architect`-combined rows respectively — not five/six independently-scoped exposure
decisions as the eight-row table visually presents, but effectively three distinct glob surfaces
(`docs/plans/**`, `docs/reviews/**`, `docs/requirements/**`+`docs/manuals/**`) each claimed by
multiple agents. That's actually a point *in favor* of the design (less incremental surface than
the table suggests, since `docs/plans/**` gets exposed once regardless of how many agents' hooks
already write there) — but the table's current wording would mislead a reader who doesn't
independently check the scripts (as I did) into believing `teco`'s and `data-scientist`'s rows are
narrower, agent-specific carve-outs. Fix the table's prose to say what the globs actually are
before this goes to implementation — the design's own "before implementation, re-`grep`" note
(§5, closing paragraph) is the right instinct but undersells the gap as a staleness risk rather
than a description-accuracy one.

### Minor — §6's rejection of the deny-list translation doesn't show the "ask-rules-only, no blanket allow" alternative it's implicitly ruling out

Per the brief's task 2: I worked through whether an `ask`-only translation (mirroring
`guard-broad-write.sh`'s deny-list as settings.json `ask` rules, with **no** companion blanket
`allow`) would help without cobb's flagged repo-wide-exposure cost. It wouldn't, and I think cobb's
conclusion is correct — but the design doesn't show this reasoning, so the gate has to re-derive it
rather than check it:

`tdd-engineer`'s guard (`claude/tdd-engineer/hooks/guard-tdd-broad-write.sh`) allows "everything
except a fixed, enumerable list of other specialists' doc-kind paths." The allow side is
*unbounded* — any source/test file in any current or future component directory. A rule-based
translation needs something to match at decision-step 1 to skip the classifier; `ask`-only rules
for the deny-list paths do nothing for the actual friction, because they only ever fire on paths
that were *already* correctly escalating — the overwhelming majority of `tdd-engineer`'s in-remit
writes (the actual friction source) would still fall through to steps 2/3, i.e. still riding the
same unreliable classifier path they're on today. The only way to get in-remit writes onto the
rule-resolves-first path is an allow rule that covers them, and since the allow side is unbounded
by design (new component directories are supposed to be auto-covered without a guard-list edit),
enumerating it as discrete `Edit(<dir>/**)` rules is operationally close to `Edit(**)` anyway (most
of the repo, modulo the same short exclusion list) while adding an ongoing maintenance burden the
current deny-list shape was specifically built to avoid. So: no, there isn't a narrower rule-based
option here — "leave it hook-only, accept the friction" is the least-bad choice, but say so
explicitly in §6 rather than presenting only the blanket-`Edit(**)` translation as the sole
alternative considered.

### Note — §2.1's "subagent same-rules" reading is plausible but not iron-clad, and the design already treats it as unconfirmed (no action needed beyond what §7 already does)

I re-fetched both docs pages cited and confirmed §2.1's and §2.2's quotes verbatim (see "What's
solid" below) — but I want to flag one interpretive gap the design doesn't call out, since the
brief asked me to check whether the reasoning holds. "How auto mode handles subagents" step 2
reads: "each of its actions goes through the classifier with the same rules as the parent
session." That sentence supports the plan's reading (rule-step-1 resolution applies to a subagent
action the same way it does to the parent's own) — but it's also consistent with a **narrower**
reading: every subagent action reaches the classifier regardless, and "the same rules as the
parent session" describes what *context* the classifier consults, not that rules can short-circuit
the classifier for a subagent action the way they do for a top-level one. Docs prose alone can't
disambiguate these two readings, and the design already treats its whole premise as unconfirmed
(§7) rather than asserting it — so this doesn't change my verdict on §7's own gating language. I'm
flagging it only so that whoever runs the eventual live test knows there are two live hypotheses
to distinguish, not one: "rules bypass the classifier for subagent writes" vs. "rules bypass it for
top-level writes but subagent writes always reach the classifier regardless of rules" (which would
mean this design's fix doesn't close the gap for exactly the Task/Agent-delegated case that
motivated it, even though it might still help a same-agent-interactively-run write).

## Answering the brief's task 3 — should review wait for the empirical test to land?

No — reviewing the design's structural soundness now was worth doing, and I'd recommend against
gating the review itself on the live test. The two substantive defects above (the cobb blocker and
the two mis-described globs) are static, verifiable-without-the-test problems; catching them now
means the eventual test run and any settings.json edit start from a corrected candidate list rather
than needing a second review pass after the fact. What *should* stay gated on the test — and the
document is already explicit about this in §7 ("Nothing in §5 or §6 should be implemented until
that test lands") — is implementation. My recommendation to `teco`: treat this as "design approved
for its reasoning framework, with the corrections above folded in before implementation," not
"ready to build" — the empirical gate in §7 stands regardless of this review's own findings, and
this review doesn't remove it.

## What's solid

- Both direct docs quotes in §1/§2 are byte-verbatim, re-verified independently against a fresh
  fetch of `code.claude.com/docs/en/permission-modes` and `.../permissions` (not from cache): the
  three-step "How the classifier evaluates actions" order, "How auto mode handles subagents" step
  2, the `Edit(path)`/`Read(path)`-only rule-matching restriction (verbatim, including its
  "Requires Claude Code v2.1.210 or later" qualifier — this repo's sessions run v2.1.240/241 per
  the RCA, so no version gap), and "Hook decisions don't bypass permission rules... a matching ask
  rule still prompts even when the hook returned `"allow"`" are all accurate quotes of the current
  page content.
- §6's core math (rule evaluation order is deny → ask → allow, first match wins — verbatim-checked
  against the same page) correctly supports its claim that a targeted `ask` rule would still win
  over a blanket `allow`, even though (per the Minor finding above) that doesn't actually rescue
  the deny-list translation.
- The `analyst`, `architect`, `tico`, `qa-engineer`, and `security-expert` rows in §5's table are
  accurate against the live wrapper scripts — I diffed each one by hand.
- §4's general framing (a settings.json rule has no agent-scoping mechanism, unlike a hook) is
  correct and is the right lens for the doc-kind rows; it just needed to be applied unevenly rather
  than uniformly across §5's full candidate set (the Blocker and first Major finding above).
- The pilot choice (`qa-engineer`'s `docs/test-plans/**`) is a genuinely good low-risk starting
  point independent of my test-plans/test-reports split above — it's narrow, low ambient traffic,
  and (per my table) one of the lower-stakes members either way.
- §7's honest account of its own blocked non-invasive test attempt, and its explicit "nothing
  should be implemented until that test lands," is exactly the right posture for a design resting
  on an unconfirmed premise — no notes.

## Open questions

- Does the stakeholder want `docs/reviews/**`/`docs/test-reports/**`/`docs/requirements/**` treated
  as a single "accept the self-approval trade" decision, or does each need its own sign-off? The
  design as written doesn't surface this as a question at all (Major finding above); I'd suggest
  `cobb` add it explicitly to a revised §8 rather than the gate inferring it.
- Should the kaizen/catalog half of cobb's row (history/plan/README/AGENTS/CLAUDE/`cypher-mcp/README.md`)
  proceed as a candidate on its own once the agent-definition/skill-package half is dropped, or
  should the whole row wait for the same live test as everything else regardless? I don't see a
  reason it needs to wait longer than the rest, but it's the stakeholder's call given `AGENTS.md`'s
  wide blast radius (every session).
