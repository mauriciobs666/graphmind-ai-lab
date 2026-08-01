# Minimal Kiro Demo Agent for falkor-chat — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (—)

## Scope & verdict

Static pre-implementation review of `kiro/docs/plans/kiro-demo-agent.md` (Owner: `architect`)
against `docs/requirements/kiro-demo-agent.md` (Status: Ready for design), the related
`docs/requirements/kiro-vision-followups.md` item 4, and root `AGENTS.md`'s documentation
conventions. Scope covered: FR-1…FR-6 / AC-1…AC-4 coverage, plausibility and internal consistency
of the plan's `kiro-cli` v2.14.1 behavioral claims (independently re-verified live — see below),
the zero-`falkor-chat/`-files hard constraint, the §3.5 relocation recommendation against the
actual filename-grammar/family rules, doc-impact completeness for `kiro/`'s first structured
build-out, the header block, and general plan quality (sequencing, step concreteness, §7 risk
completeness). Not in scope: anything under `falkor-chat/` itself (out of scope for this feature
by its own requirements), and live/interactive verification of the demo flow (explicitly deferred
to coordination unit 5, `qa-engineer`, live — this review is static only).

**Verdict: approve with suggestions.** No blockers. The plan is well-grounded — its central,
load-bearing `kiro-cli` claims were independently re-verified against a live `kiro-cli 2.14.1`
install (see "Independent verification" below) and all matched. One major finding: a rationale
bullet in §3.1 asserts a "generated default" that is directly contradicted by the plan's own §2.3
evidence and by my own reproduction — worth fixing before or during implementation since it
undermines the plan's core "empirically verified, not asserted" claim, though it doesn't change
the (correct) config value chosen.

## Independent verification performed

`kiro-cli` **is** installed on this machine (`kiro-cli 2.14.1`, matches the plan's stated version).
I re-ran the plan's most load-bearing empirical claims from a fresh scratch directory, independent
of the plan's own investigation:

- `kiro-cli agent create <name> -d .kiro/agents` (EDITOR=true) — generated template matches §2.3's
  captured JSON **exactly**, including `"tools": ["*"]`, `"resources": []`, `"includeMcpJson": true`.
- `kiro-cli mcp add --name falkor-chat --url http://localhost:8000/mcp --agent <name> --force` —
  wrote `"mcpServers": {"falkor-chat": {"url": "http://localhost:8000/mcp"}}`, no `"type"` key.
  Confirmed against `falkor-chat/docs/DESIGN.md` §15.3, which does use `{"type": "streamable-http",
  "url": "..."}` — the plan's claimed divergence between the two schemas is real.
- Exact-CWD local-agent discovery: `kiro-cli agent list` from the directory containing
  `.kiro/agents/<name>.json` lists it as `Workspace`; from a `subdir/` one level down it does not
  appear and there is no fallback to the parent — matches §2.3 exactly.
- `kiro-cli agent validate --path <file>` against a config pointing at an unreachable
  `localhost:8000` (nothing listening) returned exit `0` in ~1.4s, no hang, no output — confirms
  the "pure static/schema check" claim.
- `kiro-cli chat --help` confirms `--require-mcp-startup` exists with exactly the documented
  semantics ("Require all enabled MCP servers to start successfully; exit with code 3 if any
  fail") — used in the plan's §4 step 4 run block and §5's AC-1 recipe.
- `tools` vs `allowedTools` and the remote-MCP-server schema were cross-checked against
  `kiro.dev/docs/cli/custom-agents/configuration-reference/` and
  `kiro.dev/docs/cli/mcp/configuration/` directly (WebFetch) — both confirm the plan's claims
  verbatim: `tools` is the reachability allowlist, `allowedTools` is a strict subset that cannot
  expand reachability, and remote MCP entries use `url` (+ optional `headers`/`oauth`/etc.), never
  `type`.

All of this corroborates the plan's central grounding claim (built on real `kiro-cli` behavior,
not `kiro/DESIGN.md`'s stale sketch) — with one exception below.

## Findings

### Major

**§3.1's `resources: []` rationale is factually wrong, contradicted by the plan's own evidence.**
§3.1 states: *"`resources: []` (not the generated default, which pulls in `file://AGENTS.md`,
`file://README.md`, `skill://...`)"*. But the plan's own §2.3 captured default-template JSON, 60
lines earlier, shows `"resources": []` **is** the generated default — not pre-populated with any
file/skill references. I independently reproduced this: a fresh `kiro-cli agent create` in a
scratch directory produced `"resources": []` verbatim, same as §2.3's own snippet. So the actual
generated default and the plan's chosen value are identical — `resources: []` isn't a deviation at
all, it's what `agent create` already writes.

This is the exact failure mode the review brief asked me to check for ("unverified assertion
dressed as fact") — and it's directly falsifiable by the plan's own text, not a subtle inference.
It doesn't change the final config (`[]` is still the right choice either way, so nothing needs
re-implementing), but it undermines the plan's central credibility claim — that every `kiro-cli`
fact in this document was empirically checked, not asserted — the one thing the document
repeatedly stakes its authority on (§2.3's own framing: "NOT taken from `kiro/DESIGN.md`'s
illustrative JSON sketch"). Note there is a real, adjacent mechanism the plan doesn't mention:
`kiro.dev`'s docs describe a separate global setting, `chat.disableInheritingDefaultResources`,
that governs whether a custom agent inherits default resources (steering files, skills,
`AGENTS.md`) **at runtime**, independent of its own `resources` array — worth a one-line footnote
if the coder wants to be thorough, though it likely doesn't affect this demo since `kiro/` has no
`AGENTS.md`/steering files to inherit.

**Suggested fix:** rewrite the bullet to state `resources: []` is already the generated default
value and is being kept as-is because the agent has no need for repo file context — not framed as
an override of a populated default.

### Minor

**The README's suggested demo phrasing isn't pinned to the risk-avoiding form the plan itself
already identified.** §2.3/§7 flag a real, only-partially-verified risk: typing a literal `@`
character in the interactive TUI could trigger kiro-cli's own file/prompt-completion dropdown
before the presenter finishes typing `@assistant`. §5's own AC-1 test recipe sidesteps this
cleanly by phrasing the example instruction in plain English — *"post 'hello from the kiro demo'
and mention assistant"* — never asking the presenter to type a literal `@`. But §4 step 4 (the
`kiro/README.md` content spec) only asks for "one line each on what to type for AC-1 ... and AC-2,"
leaving the exact wording to the coder's invention. Nothing in step 4 tells the coder to use the
same non-`@`-prefixed phrasing §5 already uses — so there's a real chance the README ends up
suggesting the more demo-natural but riskier `"@assistant, ..."` phrasing, reintroducing the exact
risk the plan already found a way around. **Suggested fix:** step 4 should explicitly say the
suggested demo input mirrors §5's phrasing (no literal `@` character) rather than leaving it open.

**FR-4/AC-1's "@mention-ing @assistant" wording is ambiguous, and the plan's design silently picks
one reading.** The requirement text could mean either "the presenter's literal keystrokes contain
`@assistant`" or "the resulting falkor-chat message ends up mentioning `assistant` via falkor-chat's
own mention mechanism" (i.e., the MCP tool's `mentions` argument, set structurally by the agent
regardless of what the human typed). I checked `falkor-chat/server/falkorchat/services.py:618-644`
(`post_message`) — `mentions` is a distinct, structured argument from `body`, never parsed out of
the message text — so the plan's system prompt design (call `send_message` with `mentions:
["assistant"]` whenever the user's intent implies it, independent of literal `@`-typing) is a
defensible, well-grounded implementation of the second reading, and it's the one that also avoids
the `@`-completion risk above. This isn't a plan defect — the design is sound and internally
consistent — but the plan never states the interpretation explicitly as a decision, so a reader
comparing FR-4's text to §3.1's system prompt could reasonably wonder if literal `@`-typing was
required and dropped by accident. **Suggested fix:** one sentence in §3.1 or §7 noting this is the
adopted reading, so it reads as a decision rather than an implicit byproduct.

### Nit

`kiro/DESIGN.md`'s header (`**Status**: Draft | **Last updated**: 2026-06-20`) predates root
`AGENTS.md`'s current header-block convention (bolded `Status:`/`Owner:`/`Tracks:` line
immediately under the H1) and the plan doesn't touch it — reasonable since it's explicitly out of
scope and untouched by this feature, but worth a passing mention to `tico`/whoever next revisits
`kiro/DESIGN.md` for the vision-followups work, since this plan is otherwise bringing `kiro/` up
to the repo's current documentation conventions.

## What's solid

- **Grounding is genuinely strong, not just claimed.** Every empirical `kiro-cli` claim I
  independently re-tested (default template, remote-MCP schema, exact-CWD discovery,
  `agent validate`'s no-network behavior, `--require-mcp-startup`'s exact semantics, `tools` vs
  `allowedTools` reachability) matched exactly, cross-checked against both live reproduction and
  the official `kiro.dev` docs. This is a plan that did the actual legwork instead of trusting
  `kiro/DESIGN.md`'s stale sketch, and it shows.
- **FR-1…FR-6 and AC-1…AC-4 are all genuinely satisfied by the design**, not just asserted in a
  checklist — traced each one against §3.1's config and §5's test recipe; no gaps found beyond the
  minor interpretive note above.
- **The §3.5 relocation recommendation is correctly grounded.** Root `docs/BACKLOG.md` and
  `docs/HISTORY.md` do explicitly scope themselves to "CPG code-graph component" (verified by
  reading both headers); `docs/requirements/` does currently hold exactly the two CPG files plus
  the two Kiro files (verified via `ls`); the family co-location rule genuinely does require
  `kiro-demo-agent.md`'s requirements/plans/coordination members to sit together. The §3.5
  exhaustive-grep claim (3 numbered fix sites) matches my own independent re-run of
  `grep -rn "docs/requirements/kiro-demo-agent\|docs/requirements/kiro-vision-followups"
  --include="*.md" .` line-for-line — no missed reference, no non-Markdown reference anywhere in
  the repo either.
- **The hard constraint holds.** No step in §4 touches any `falkor-chat/` path; `git status
  --porcelain -- falkor-chat/` is currently clean (the K-034 work noted in-flight in the
  coordination doc has since been committed), and every file this plan's step list creates or
  moves sits under `kiro/` or root `AGENTS.md`/`docs/requirements/`.
- **Doc-impact completeness is well-reasoned, not just checked off.** The deferrals (no
  `reviews/`/`test-plans/`/`test-reports/` scaffolding since git doesn't track empty dirs, no
  `BACKLOG.md` since there's only one forward-looking item) are each given an actual reason, not
  asserted by fiat, and the plan correctly notes the precedent (`cpg/` also has no
  component-level `AGENTS.md`, confirmed by directory listing) for keeping `kiro/`'s doc footprint
  minimal at this size.
- **Header block is compliant** — `Status:`/`Owner:`/`Tracks:` line immediately under the H1,
  correct bolding and separator, matches root `AGENTS.md`'s required format exactly.

## Open questions

- Should `resources: []`'s rationale bullet (Major finding) be corrected in the plan itself before
  unit 3 starts, or is a note to the implementer sufficient given the config value is unaffected?
  Per root `AGENTS.md`'s document lifecycle rules, this plan is still `active` and un-executed, so
  revising it in place (rather than waiting for a successor doc) is the lower-friction path if
  `architect` wants to take it.
- Confirm with `qa-engineer` (unit 5) whether the `@`-completion risk should get an explicit
  interactive pass even if the shipped `kiro/README.md` avoids literal `@`-typing in its own
  suggested wording — a live presenter could still improvise `"@assistant..."` regardless of what
  the README suggests, so the edge case in §5 remains worth keeping even after the README fix
  suggested above.
