# Review — Typed Decision Models Plan Successor 6

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1 (2026-11-01)

**Scope.** I checked `docs/plans/typed-decision-models6.md` v1.0 (4,402 words, uncommitted) against these sources:
- plan4 v1.2 and plan5 v1.4;
- the coordination ledger's 2026-11-01 entries: the background ruling, the Pass 8 reconciliation, the stakeholder decisions, "Probe attempt 1" and "Probe complete";
- `docs/reviews/typed-decision-models4-impl.md` Pass 8;
- the probe evidence, read-only: `scratchpad/dm-probe/results-after-{kill,restart}.txt`, `hook-log.jsonl`, `panes/`, the attempt-1 and attempt-2 transcripts and `.meta.json` files under `~/.claude/projects/-tmp-…-dm-probe{,-attempt1}/`;
- `claude --help` for 2.1.316. I started no session.

I wrote nothing outside this file.

**Verdict: needs changes.** There is no blocker, but there are three majors, all routed to `architect`:
- **M1:** unattended operation depends on what a usage-limit stop looks like under `-p`, and that is neither probed nor routed.
- **M2:** the stop switch cannot run while a run is active.
- **M3:** U6.2 leaves three headless premises without a pass/fail route, and does not run under the runner's environment.

Most of the design is sound. The fixes are to probe gates, one audit-row fallback and the lock rule.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the plan is design-level, and the harness evidence is transcripts, not code.

### Findings

**M1 — major — the routine usage-limit path under `-p` is unprobed, and the table turns any unexpected shape into a halt in every spent window.** *Owner: `architect`. Inferred from the table and the help text; there is no live evidence under `-p`.*
- §3.2 relies on "a fire within a spent window … ends as a usage stop". That is the normal case, roughly every window. Its shape, though, comes only from plan4's interactive `<synthetic>` row.
- **Driver at the limit.** If `-p` exits on the limit without writing a transcript, the runner sees "no transcript" and classes the batch never spawned. The second fire then halts the campaign.
- **Subagent at the limit.** If a foreground subagent's driver `tool_result` arrives with a status other than `completed`, the "any other `status`" row matches. That row is `spawnShapeUnknown`, a halt, and halts precede usage stops.
- **User settings.** The user's `~/.claude/settings.json` has an `autoContinueAtUsageLimit` key (false today). If it is set, a session may wait out the limit instead of ending. `batchTimeout` then makes it `incomplete`, and the second occurrence halts.
- No H-case covers any of these.

**Fix:**
- State the usage-stop classification for `-p` explicitly, including the runner's own evidence. The `--output-format json` result in `store/runs/<uuid>.json` is currently "diagnostic only".
- Name where the shape is first recorded (U5.7's smoke or the first spent pilot window), and make that the red check: a usage stop that halts the campaign means the row is wrong.
- Have the runner refuse to start if user settings set `autoContinueAtUsageLimit: true`.

**M2 — major — `pause` and `pause --now` refuse exactly when they are needed.** *Owner: `architect` (§3.2 "Single-flight", "Stop switch"). Verified by reading.*
- §3.2 says `pause`, `resume`, `resolve`, `install` and `finalize` "take the same lock, without blocking. They refuse while a run holds it".
- Yet `pause` is meant to let "a run finish the batch in flight and exit", and `pause --now` signals the running runner. Both need a run to be holding the lock, so both always refuse.
- That leaves disabling the timer as the only stop, and disabling it does not stop a run in flight.
- U6.3's "`pause --now` stops the runner" test and U6.4's "after `pause`" check cannot both pass under the lock rule as written.

**Fix:** let `pause` and `pause --now` bypass the lock and leave the ledger alone. One option: write a sentinel file, `store/pause`, that `next` reads. The runner, which holds the lock, then appends `paused` to the ledger. That keeps one ledger writer.
- `pause --now` should signal the process group recorded in the ledger's `issued` event, after checking `/proc/<pid>/cmdline`. A PID read from the lock file may have been reused, and the original runner may already be dead (see m1).
- Add a U6.3 test in which `pause` succeeds while a fake child holds the lock.

**M3 — major — U6.2 does not give every headless premise a pass/fail route, and it does not probe under the runner's environment.** *Owner: `architect` (§4 U6.2). Verified by reading §3.1–§3.5 against H1–H5 and gates (a)–(d).*

The rule I applied is plan4 review Pass 1's proposed line (not yet in root `REVIEW.md`; see the proposals below): each harness premise cites a probe, and a worst-case smoke goes red if the premise is false. Measured against it:

| Premise (§) | U6.2 case | Route on failure |
|---|---|---|
| foreground under `-p` (3.1) | H1 | (a) → `architect` |
| settings hooks fire in subagents (3.3) | H1 | (b) → `architect` |
| exactly one guard entry per tool use (3.3) | none | none |
| `--session-id` fixes the transcript path (3.2, 3.4) | H5 | **none** |
| kill shapes; subagent outliving its parent (3.2, 3.4) | H3 | (c) "recorded" only |
| `--permission-prompts none` + `acceptEdits` allow the `Agent` call and the subagent's `req/` Read and `ans/` Write (3.2) | none | none |
| the env allowlist is enough to authenticate (3.2) | none | none |
| `--setting-sources project` (3.2) | H4 | (d) ✓ |
| usage-limit shape (3.2) | none | M1 |

Two of these fail silently rather than closed:
- **Permission auto-deny.** If `--permission-prompts none` auto-denies the subagent's `Write`, no hook denies anything. The run voids as `answerMissing`, a counted void, so items drop as `retryCapExceeded` with no halt.
- **Subagent outliving its parent.** If H3 shows the subagent outlives a parent SIGKILL, the runner's timeout kill no longer bounds a run.

The probe environment is also unstated. If devops launches `claude -p` from inside its own Claude Code session, the child inherits that session's `CLAUDE*` variables. H1 would then not be evidence for the runner.

**Fix:**
- Run U6.2 through the runner's exact env allowlist (`env -i` plus the list), from a shell that is not a Claude Code session.
- Give H5 and H3 gate outcomes:
  - H5 → `architect` if the path differs;
  - H3 subagent survival → `architect`, because the timeout model changes.
- Add to H1's gate:
  - no permission-denied `tool_result` anywhere;
  - exactly one guard `hook_success` per tool use;
  - one planted subagent `Write` outside `ans/`, which records the auto-deny shape.
- Add an audit row so that shape is an environmental halt, not a counted void.

**m1 — minor — an orphan can hold the lock indefinitely, with no timeout enforced.** *Owner: `architect` (§3.2, §6). Inferred from reading.*
- `pass_fds` correctly turns a race into a block. But once the runner is dead, nothing applies `batchTimeout` to the orphan, and §6 accepts "stalls until a human kills it" in a mode that has no human by design.
- Any grandchild that inherits the descriptor extends the hold too.

**Fix:** when `run` finds the lock held, it reads the last `issued` event (start time and process group). Past `batchTimeout` plus a grace period, it SIGKILLs that group and exits; the next fire then reconciles. Test this with U6.3's surviving fake child.

**m2 — minor — the env allowlist does not isolate user-level settings.** *Owner: `architect` (§3.2). Verified by reading the user settings' keys and the values of `autoContinueAtUsageLimit` and `modelSettings` only.*
- Without `--setting-sources project` (conditional on H4), `~/.claude/settings.json` still applies `env`, `hooks`, `permissions`, `modelSettings` and `autoContinueAtUsageLimit`.
- This user's file carries `modelSettings` (`claude-sonnet-5` effortLevel `high`) and `autoContinueAtUsageLimit`.
- §3.2's fallback ("P8-F4's `guard_command` filter") covers hooks only.

**Fix:** if H4 rules the flag out, `install` records which user-settings keys are present and their hash. `run` then refuses on `env`, `hooks` or `autoContinueAtUsageLimit: true`, or on a changed hash.

**m3 — minor — the safeguards risk in the subagent depends on `teco` watching, when it could be a row.** *Owner: `architect` (§3.4, §6). Verified: attempt-1 `c5ce5291…` line 46 is an Opus assistant entry with `stop_reason: "refusal"`, and line 49 is a `user` entry carrying the "safety classifier" text.*

**Fix:** add a row that halts as `safeguardStop` on that user text in either transcript, even without a `refusal` entry. That turns §6's manual watch into a check that can go red. Seed a U5.5 fixture from the line-49 text.

**m4 — minor — citation seams.** *Owner: `architect`. Verified by reading.*
- **(a)** The driver template "joins plan4 §3.7's `promptSha256` set", but no key is named. U5.10's "matches §3.7's keys exactly" row is not amended. Name the key, for example `spawnTemplate.driver`, and add it to U5.10 or U5.7.
- **(b)** The U5.5 row says "P8-F3: as Pass 8 states", but P8-F3's planted case is a driver `Bash` `tool_result`, and the driver no longer has `Bash`. Restate the plant as any second `tool_result` with a different id.
- **(c)** F5 cites the "diff of the two results files", but that diff shows only the `Stop`/`SubagentStop` counts (7 → 8). The table lists S4's notification as "none" both times, because `read_results.py` matches notifications by tool-use-id. The `stopped` notification is only in the driver transcript `fb3a881a…`. U6.1's traceability done-condition needs that excerpt.
- **(d)** Plan4 R-18 ("hooks fire only in a trusted folder") is stale under `-p`, which skips the trust dialog. Amend it, or point it at gate (b).

**n1 — nit — growth-stops.** Plan6 is a new document (v1.0), so its words are new scope.
- §2's F1–F7 table becomes a second home for the probe facts once U6.1 lands. At the next revision, cut each to one line citing its report row.
- The U5.5 test row (about 300 words) is the longest cell. It is acceptable as a build instruction.
- I found no finding→resolution table and no plan4 or plan5 restatement.

### Check answers

1. **Headless premises:** M3 (table) and M1. Gates (a) and (b) route failures back to `architect` and the tmux fallback. They do not fall back silently.
2. **§3.4 against the observed shapes.** Each was verified against the transcripts and meta files:
   - S3 (foreground, `completed`, `finished: true`) is the only completion row.
   - S4 (`finished` absent, last entry `user:tool_result`) is incomplete, because there is no matched result.
   - The `stopped` notification with no `<tool-use-id>` is ignored.
   - The foreign `SubagentStop` is ignored: every transcript `hook_success` I counted (82) has `hookEvent: "PreToolUse"`.
   - P8-F1 closes by elimination (no background path). F2 closes through the terminal rule, F4 through `guard_command`, and F5/F6 as carried; F3 needs m4(b).
   - Precedence is consistent:
     - `safeguardStop` and the model-pin row are both halts;
     - the incomplete row excludes `refusal`;
     - the driver usage stop is tie-broken over plan4's no-`tool_use` row.
   - One gap remains: the subagent usage stop (M1).
3. **Relaunch safety.**
   - **No double-issue:** `issued` is fsync'd before launch, with a fresh UUID each time.
   - **No double-count:** the lock and ledger-first idempotence prevent it.
   - **Lock-holding orphans:** m1.
   - **The stop switch:** M2.
4. **CLI flags.** All exist in 2.1.316: `--session-id`, `--permission-prompts none` (auto-deny), `--fallback-model`, `--setting-sources user,project,local`, `--strict-mcp-config` and `--output-format json`. The help text also says `-p` "silently ignores" settings files that fail validation. A malformed `exchange/.claude/settings.json` therefore removes the guard. The audit's `hookMissing` row fails closed on that, so there is no finding.
5. **Units.** The order is coherent: U6.2 gates everything that depends on `-p`, U5.5 takes H3's shapes, and U6.3 follows U5.6 on the shared `driver.py`. The owners fit: `devops` can write `docs/test-reports/` (its guard is for destructive operations only), and `tico` owns the decision log. Cross-plan seams: m4.
6. **Growth-stops:** n1.

### What's solid

- Moving the loop out of the LLM into a stdlib runner, with one Sonnet driver session per batch and a runner-chosen `--session-id`, removes the in-session drift and modal-dialog risks (F7, verified).
- The background path halts and is never supported. This is simpler than Pass 8's dual-shape audit, and the probe backs it (S1/S2/S3b background, S3 foreground).
- Wiring the hook once in settings, with the at-least-one, all-allow rule, matches F4 (verified: 9 `hook_success` entries per 3 tool uses).
- The `refusal` row precedes every counted row, which is the right default for safeguards.

### Open questions

- None is a stakeholder decision beyond those already in §6. M1's fix may surface one: whether a halt in each spent window is acceptable until the `-p` usage shape is recorded.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*.
- **Addition (source: `docs/reviews/typed-decision-models4.md` Pass 1 F2/F4/F5, proposed there and not yet applied; and M3 here):** A plan step that relies on agent-harness behaviour (frontmatter fields, hook firing, transcript fields, CLI flags, tool size limits) cites a live probe or transcript for each premise. The probe runs under the production invocation's exact flags and environment. Each premise has a named pass/fail route, and a worst-case-sized smoke that goes red if the premise is false.
- **Addition (source: M2):** A control command that shares a single-flight lock with the long-running process it controls (stop, pause, signal) is checked for reachability while that process holds the lock.

## Pass 2 (2026-11-01)

**Scope.** Plan6 v1.1 (5,887 words) against Pass 1, re-checked against:
- `docs/test-reports/typed-decision-models-harness-probe-report.md` §1 and §7;
- the attempt-1 and attempt-2 transcripts;
- `model-bench/scripts/import_decision_pack.py` and its tests at `362dc56b`;
- plan5 §3.5.

I also did one web search on usage-window resets. I only read files; I wrote nothing outside this file.

**Verdict: needs changes.** There are two new majors:
- **P2-M1** is produced by the m2 fix.
- **P2-M2** is a seam in the new §3.7.

The rest is minor. Every Pass 1 finding is fixed, or fixed in part. See the stopping condition at the end of this pass.

CPG: considered, not relevant — no `cpg_decisionmodels`/`cpg_modelbench` graph is loaded; the importer was read directly.

### Pass 1 dispositions

- **M1: fixed in part.**
  - The `launchFailed` time bound, the `autoContinueAtUsageLimit` refusal, usage-stop precedence over `spawnShapeUnknown` and the first-spent-window red check are all present.
  - "A usage window resets within 5 hours" is cited nowhere in the plan or the ledger. The only repo trace is the ledger's "Session hit the 5-hour usage limit" (line 161). Secondary web sources agree: a 5-hour session reset plus a separate weekly limit, which the plan already halts on. Cite one source.
  - **Gap:** see P2-m1.
- **M2: fixed.**
  - `store/pause` is the only pause state.
  - `pause` and `resume` take no lock and write nothing to the ledger.
  - The U5.6 test checks that the ledger is byte-equal across a pause, and U6.3 checks that `pause` works while the lock is held.
- **M3: fixed.**
  - U6.2 now launches with `env -i` plus the allowlist.
  - H5 and H6 are added, gates (a)–(f) are complete, and a failed gate goes back to `architect`.
- **m1: fixed.** The held-lock rules match the runner by PID, start time and cmdline. The residue is P2-m2.
- **m2: fixed, but the fix creates P2-M1.**
- **m3: fixed. I verified it.**
  - In `c5ce5291…`, line 46 has `stop_reason: "refusal"` with message id `msg_011CgcKif6g5nak2GR7J2dyB`.
  - Line 49 is a `user` entry with `isMeta: True` and `refusedMessageId` set to that same id.
  - Report §7 lists both key sets. No attempt-2 transcript contains `refusedMessageId`, so the marker is not background noise.
- **m4: fixed.**
  - (a) Fixed: the key is `spawnTemplate.driver`, and U5.10 covers it.
  - (b) Fixed: the planted `tool_result` is restated.
  - (c) Fixed: report §7, lines 100–101, has the `stopped` tag set with `has tool-use-id=False` for `fb3a881a`.
  - (d) Fixed: §6 now names R-18.
- **n1: fixed.** §2 now has one line per fact and cites report §1.

### Checked against each other

These were verified by reading the precedence list against the rows and the §5 fixtures:
- **`rate_limit` vs `backgroundSpawn`:** a halt sits in tier 1, above the usage stop in tier 2, and a fixture pins it.
- **`permissionDenied` vs `answerMissing`:** a halt beats a run-attributable row. The row is limited to tool uses the guard allowed, so it does not collide with `hookDenied`.
- **`safeguardStop` vs everything:** it is in tier 1, matches either marker, and has a mutant.
- **Single writer:** `pause --now` makes the runner, which holds the lock, append `interrupted`. The orphan paths write nothing. One writer holds.

Three interactions do not hold together: P2-M1, P2-m1 and P2-m2.

### New findings

**P2-M1 — major — the user-settings hash refusal will halt the campaign whenever the user changes their own settings.** *Owner: `architect` (§3.2 "User settings"). Verified by reading the file's keys. Produced by the m2 fix.*
- `run` refuses any launch when the sha256 of `~/.claude/settings.json` has changed since `install`.
- That file is the user's everyday config. It holds `model`, `theme`, `editorMode` and `verbose`, which `/config` and `/model` rewrite during ordinary interactive work in this repo.
- So any such change stops the campaign. The refusal writes nothing to the ledger, and no `status` line is specified, so the stop is silent. Q-1's unattended mode then stalls without anyone knowing.

**Fix:**
- Refuse on named keys only (`env`, `hooks`, `autoContinueAtUsageLimit: true`), plus a hash of the subset that can reach a child: `permissions` and `modelSettings`. Drop the whole-file hash.
- Have `status` print the refusal reason.
- Add a U6.3 test: a `theme` or `model` edit does not refuse, and a `permissions` edit does.

**P2-M2 — major — §3.7's recursive key-set check cannot pass on correct output, and it leaves out plan5's keys.** *Owner: `architect` (§3.7, U5.10/U6.6 rows). Verified by reading plan5 §3.5 and §3.7.*
- The vector must be "a complete §3.7 object: every key". U5.10 then requires `generation.json` key sets to be equal, recursively.
- **Plan5's keys are missing.** Plan5 §3.5 adds `dropped.formMismatch: {<form>: n}` and `dropped.bindingMismatch`, but plan6 declares plan5 out of scope, so the vector omits them.
- **Some keys depend on the data.** `formMismatch`'s keys depend on which forms dropped, and `runs[]` is a list of varying length. A correct `finalize` can therefore fail an equality on key sets.

**Fix:** cite plan5 §3.5's keys in §3.7. Name the data-keyed maps and lists, and compare only their value types. Alternatively, fix `formMismatch` as all six forms with zeros included. Then add one U5.10 fixture where a form dropped and one where none did.

**P2-m1 — minor — M1's partial fix covers one of three possible `-p` usage-limit shapes.** *Owner: `architect` (§3.4). Inferred from the rows.*

A spent-window `-p` driver can leave one of three shapes:
- no transcript: the time-bounded `launchFailed` row;
- a transcript with a `<synthetic>` `rate_limit`: a usage stop;
- a transcript with the prompt only, no `<synthetic>` and no `Agent` call: the plan4 never-spawned row, whose second occurrence halts.

The third shape halts the campaign in every spent window. The red check catches it, so it fails closed, but each window then costs a `resolve`.

**Fix:** apply `launchFailed`'s time bound to a never-spawned driver transcript when the child exited non-zero. This keeps one rule for "the launch produced nothing".

**P2-m2 — minor — the runner's new paths are inconsistent about counting, and one window has no record.** *Owner: `architect` (§3.2). Inferred from reading.*
- **Counting.** `pause --now` on an orphan, and the m1 orphan kill, write no `interrupted` event, so the cap counts them as harness faults. Yet §3.2 says "a human stop is not a harness fault". Rule on it: either let the next `run` append `interrupted` when it finds a `store/pause` file and an incomplete batch, or state that orphan stops count.
- **Unrecorded window.** `runner.json` is written after launch (step 3). A runner that dies between steps 2 and 3 leaves the lock held by a child with no record of it, and the held-lock rules do not cover that. Write the runner fields before launch and add `childPgid` after, or let the held-lock path scan `/proc` for the `issued` event's `--session-id`.

**P2-m3 — minor — one §3.7 clause has no check.** *Owner: `architect`. Verified by reading.*
- "Each list holds `itemId` strings, in `items.jsonl` order" is checked by nobody: the importer does not check it, and U5.10's check compares key sets and types only.
- The importer's own test builder breaks it: `tests/test_import_decision_pack.py:123-131` reorders the items after collecting `eval_ids`.

Everything else in §3.7 matches the importer:
- `_eval_ids` requires the `eval` list to hold non-empty unique strings (`:204-210`);
- train and calibration are not read;
- `labelVotes` is checked only for non-negative non-bool ints summing to 3 (`:232-237`). The key-set refusal is owned by U6.6.

**Fix:** drop the ordering clause, or add it to U5.10's check.

**P2-n1 — nit — growth-stops.** The words went from 4,402 to 5,887 (+1,485).
- **New scope:** §3.7 with U6.6 (P24-m2), the stakeholder's §3.5 rule with `--skip-batch`, and U6.2's H5 and H6.
- **Rework that added mechanism:**
  - in §3.2: `runner.json`, the held-lock rules and the pause file;
  - in §3.4: the precedence list, `launchFailed` and the two "why" paragraphs.

  These are design content, not restatement, but they are where P2-M1, P2-m1 and P2-m2 came from.
- **Residue:**
  - §3.6's "U6.1 has delivered" and "U6.1 needs no addition for m4(c)" are a status line and a finding disposition. The rules allow neither in a plan.
  - §3.6 also re-lists what the report holds.

  Cut §3.6 to the report path and one sentence.

### Stopping condition

**Most of what is left was produced by the fixes.** P2-M1, P2-m1 and P2-m2 all come from the runner mechanics added in v1.1. P2-M2 and P2-m3 are design seams in new scope.

The next revision, v1.2, passes as **approve with suggestions** by a `teco` line check, with no full Pass 3, if **all** of the following hold:
1. P2-M1: the refusal is limited to named keys, and the theme/model-edit and permissions-edit tests are present.
2. P2-M2: §3.7 cites plan5 §3.5's keys and names the maps and lists that are compared by type only.
3. P2-m1 to P2-m3: each is ruled in a single sentence.
4. §3.2 and §3.4 gain no new mechanism beyond those rulings.

Any further defect in the runner's internals (signals, `/proc`, lock handoff) then belongs to U6.3's impl gate, where the fake-`claude` tests can show it.

**Falsifier:** v1.2 is not accepted by line check if it adds a mechanism in §3.2 or §3.4 that the four items above do not call for, or if a line check finds a new major in those sections. Either case shows the runner design is still moving. The response should then be to remove features, not add a v1.3. The candidates are `pause --now`'s orphan path and the held-lock orphan kill: the timer-off switch and §6's "a human kills it" already cover both.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: P2-M1, and Pass 1 M2 together with the P2-m2 residue.
- **Addition:** In a plan for unattended operation, every refusal or halt the plan adds is checked against what an ordinary user action on the same machine changes (config edits, logins, other sessions). Each refusal must also show up in a status command, not only in a run's exit line.

## Pass 3 (2026-11-01)

**Scope.** Plan6 v1.3 (7,025 words, uncommitted) against v1.2 at `276d0869` (`git diff`). I checked it against:
- report §9 (all subsections);
- the U6.2 headless transcripts, the kit's `hook-log.jsonl`, and the attempt-1 transcripts;
- the local transcript corpus under `~/.claude/projects/`, for `permissionDecision` shapes and parallel tool-use ordering;
- `code.claude.com/docs/en/hooks` and `/permission-modes`, read 2026-11-01;
- the installed 2.1.316 bundle, by `grep -a`.

I ran no `claude` session. My scripts were read-only and live in the scratchpad (`p3scan.py`, `p3pd.py`, `p3err.py`, `p3par.py`).

**Verdict: needs changes.** There are two majors:
- **P3-M1:** the guard's new fail-closed `deny` turns a broken guard into counted voids, which drain silently.
- **P3-M2:** U6.7's gates can pass without testing anything.

Both come from v1.3's own fixes, as do three of the four minors. The design premises hold: `onFailure: "block"` and "deny/ask rules after `allow`" are both confirmed. See the stopping condition.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the evidence is transcripts, docs and the CLI bundle.

### Earlier findings

- **P2-M1, P2-M2, P2-m1–m3, P2-n1:** fixed at v1.2 (`teco` line check, ledger 2026-11-01). §3.7 is unchanged in v1.3, with a 0-word delta. §3.2's named-key refusals stand, and v1.3 adds a correct reason to the hash clause.
- **Folded in since Pass 2:** U6.2's spend approval was used, and the row is now delivered. U6.7 needs a new approval, which is an open item, not a finding.

### The six checks

1. **`onFailure: "block"`: holds.**
   - **Docs** (verified): `"onFailure": "block"` exists on `command` hooks from v2.1.295. A failure is one of: cannot start, exits other than 0 or 2, times out, or prints invalid or schema-failing JSON. With `block`, any of these "does what exit code 2 does": on `PreToolUse` it blocks the call. It is ignored for async hooks and on `Stop`/`SubagentStop`.
   - **Bundle** (verified, 2.1.316): the command-hook schema has `onFailure: U(["continue","block"])`, with the same description ("the failure counts as exit code 2"). The handler sets `blocked: true` with "timed out"/"failed; blocking because onFailure is \"block\"".
   - **Deny/ask after `allow`:** the hooks page says "Deny and ask rules are still evaluated regardless of what the hook returns" (verified, docs only; not probed).
   - **Unstated exception:** `allow` also does not skip "actions no mode auto-approves" or network-path reads. Neither applies to `Read`/`Write`/`Agent` here, except an explicit ask rule, which §3.2's hash already covers. No finding.
2. **In-flight rule: holds for completed runs; fails on one shape of killed run.**
   - **Completed runs** (verified on H1, H2, H4 and H6): every driver `Agent` and all 13 subagent tool uses have a guard entry and a `tool_result`, so the in-flight set is empty. That is also true by construction, because the first conjunct needs the `Agent` result to be absent.
   - **H3a** (verified): the in-flight set is exactly the driver's `Agent`. All 11 subagent tool uses are paired.
   - **The driver's `hook_success` is buffered** until the `tool_result` is written. H1 shows it after two `tools-used` lines, stamped 33.7 s, written with the result at 47.3 s. So an in-flight call never has a guard entry, and the stated trade-off is real.
   - **The defect:** "the last `tool_use`" misses parallel calls. See P3-m1.
   - **The trade-off:** I judge it acceptable, and it can be narrowed (P3-n1).
3. **`permissionDenied` / `guardBreach`: consistent, total, one gap.**
   - All 28 recorded values are `{"decision":"accept","source":"hook","reasonType":"hook"}`: 24 subagent `tool_result`s and the driver's 4 `Agent` results. H3a's driver has none. (Verified; §2's "the driver's included" is right.)
   - In the corpus, tool errors after a hook `allow` still carry `accept`/`hook` (41 `is_error` cases). So the re-keyed row does not misfire on tool failures.
   - The rows partition the guard decision: allow + overridden → `permissionDenied`; allow + out of bounds → `guardBreach`; no entry → `hookMissing`; deny → `hookDenied` (tier 5); allow + in bounds + foreign part → plan4's narrowed row.
   - The precedence is still a total order: tiers, then table order. `audit.py` already breaks ties that way.
   - **Gaps:** the absent-`permissionDecision` clause needs a `tool_result` to exist (P3-m1). Pre-hook tool errors fall to `hookMissing` (P3-m2). A hooked hand-back is not exempt from `permissionDenied`, and its `permissionDecision` has never been observed (P3-n1).
4. **U6.7: routes are present, discrimination is not guaranteed.**
   - Each gate failure returns to `architect` before U5.5.
   - "Inside the kit, outside `ans/`" is the right axis: `acceptEdits` allows a new file in the working directory. But a path under `<kit>/.claude/` (a protected path) or an existing file such as `hook.py` (refused by `Write`'s read-first check, before any hook runs) would also leave the file absent or unchanged without the hook doing anything.
   - Gate (h) does not require D2's `Write` ever to have been attempted. See P3-M2.
5. **Same-revision interactions.** Checked against each other:
   - fail-closed `deny` × `hookDenied`: P3-M1;
   - `onFailure` × `hookMissing`: a blocked failure leaves no guard entry, so it halts. That is consistent;
   - in-flight × `permissionDenied`: P3-m1;
   - `guardBreach` × U5.7's bound tests: P3-m3.
   - Dropping `--setting-sources` × the `hooks` refusal: plugin hooks (`enabledPlugins`, not refused) still load. They cannot override a guard `deny` (precedence deny > ask > allow), and a plugin deny or ask halts as `permissionDenied`. That is loud, so this is a nit (P3-n1).
6. **Growth-stops.** The plan grew by 1,099 words. By section: §2 +185, §3.3 +212, §3.4 +404, §5 +246, §4 +25 (U6.2 −195, U6.7 +208), §7 +47.
   - **New scope:** U6.7; the guard as the only bound (canonical paths, fail-closed); the in-flight rule; the two new rows; and their §5 rows. All of these answer a failed gate.
   - **Rework growth:** about 100 words in §2 and about 120 elsewhere. See P3-n2.
   - **§2 is no longer citations only.** The H-lines restate §9's facts in prose.

### New findings

**P3-M1 — major — fail-closed `deny` makes a broken guard drain the campaign silently.** *Owner: `architect` (§3.3 "fail closed", §3.4 `hookDenied`). Traced through plan4 §3.3's cap rule. Inferred, because there is no live run.*
- v1.3 makes a missing or unknown `agent_type`, a missing `tool_input`, or any exception print a normal `deny` with exit 0.
- §3.4 classes every denying guard entry as `hookDenied`, which is run-attributable and counted. After 3 counted attempts, a batch's items drop as `retryCapExceeded` (plan4 §3.3), with no halt.
- So a guard defect, or a CLI release that renames `agent_type`, voids every batch as the model's fault and empties the campaign into `dropped` while the timer keeps firing.
- In v1.2, the same crash exited non-zero, left no guard entry and halted as `hookMissing`. The new path is quieter than the one it replaced.
- In the same revision, the failures the guard does not catch (`onFailure: "block"`) still halt. One root cause gets two classes.

**Fix:** the guard's fail-closed denies carry a structured marker, for example `permissionDecisionReason` starting `guardError:`. The transcript records the guard's stdout verbatim (verified in every U6.2 guard entry). §3.4 then classes a marked deny as **environmental: halt, `guardError`**.
- U5.7: each fail-closed path prints the marker.
- U5.5: a marked deny → `HALT guardError`, and an unmarked out-of-bounds deny → counted `hookDenied`. A mutant that ignores the marker reddens.

**P3-M2 — major — U6.7's gates (g) and (h) can pass without the hook having done anything.** *Owner: `architect` (§4 U6.7). Protected paths verified from the docs. That the read-first error comes before the hook is inferred from attempt-1's hooked validation case (P3-m2). The skipped call is inferred.*
- Gate (h) is "the file does not exist". That also holds if:
  - the subagent never attempts D2, for example because it stops after D1's deny;
  - the path is protected, such as `<kit>/.claude/…`, where `acceptEdits` prompts and `--permission-prompts none` denies;
  - the path is an existing file. `Write` then fails its read-first check before any hook runs; the corpus has a `Write` "modified since read" error with no `PreToolUse` attachment.
- Gate (g) does require D1's `hook_success`, but it has no control showing that `acceptEdits` would have allowed D1's path.

**Fix:**
- Name the paths: new files at the kit root, for example `<kit>/d1.txt` and `<kit>/d2.txt`, not under `.claude/`, deleted before the run.
- Tell the subagent to attempt every listed `Write` whatever the earlier results.
- Add **D3**, a third path for which the hook exits 0 with no output. Its file must exist, with `permissionDecision.source` other than `hook`. D3 is the positive control: if D3's file is absent, (g) and (h) prove nothing. It also records the "permission flow decided" shape that §3.4's guard-entry bullet asserts. This is the variant report §9.7(3) asks for, in the same session at no extra spend.
- Gates (g) and (h) also require D1's and D2's `tool_use` in the transcript and a hook-log line for each, with D2's exit code 1.

**P3-m1 — minor — "the last `tool_use`" misses parallel calls, and a killed run can still halt.** *Owner: `architect` (§3.4 in-flight bullet, `permissionDenied` row). Verified that parallel calls are persisted this way in the corpus. Inferred that `-p` persists them the same way.*
- In the corpus's subagent transcripts, 10 hooked parallel pairs were persisted as `TU0 TU1 HS0 TR0 HS1 TR1`, and 193 hookless pairs as `TU0 TU1 TR0 TR1`.
- A kill after `TU1` leaves `TU0` with no guard entry and no result. `TU0` is not "the last `tool_use`", so it falls to `hookMissing` and the batch halts. That is gate (c)'s defect again, for a labeller that reads two pages at once.
- The `permissionDenied` row's "`permissionDecision` is absent" also reads a missing `tool_result` as an absent field.

**Fix:**
- In flight = **every** `tool_use` with no `tool_result`, in a run whose `Agent` result is absent. Completed runs stay unaffected, because the first conjunct is unchanged.
- `permissionDenied` requires a `tool_result` entry.
- U5.5: a killed shape `TU0 TU1` with no attachments → `REISSUE incomplete`.

**P3-m2 — minor — a pre-hook tool error is read as a skipped guard and halts.** *Owner: `architect` (§3.4 guard-entry bullet, `hookMissing`). Verified on 16 of 16 corpus instances.*
- When a tool input fails validation (`InputValidationError`, `Write`/`Edit` precondition errors), the result is a `tool_result` with `is_error`, **no** `permissionDecision` and no `PreToolUse` attachment.
- Example: attempt-1 `c5ce5291` line 46, an `Agent` call in a session hooked on `*`, whose other `Agent` call has its entry.
- §3.4's "the guard did not decide, so the permission flow did" is false here: neither ran, and the call did nothing. A `Write` truncated at the output limit (no `content`) would therefore halt the campaign as `hookMissing`.
- A really skipped guard is a different shape: the permission flow then records `permissionDecision` with `source: "config"` (12,134 corpus instances).

**Fix:** a tool use with no guard entry whose `tool_result` has `is_error` and no `permissionDecision` is run-attributable, because the call never ran. Seed the U5.5 fixture from line 46's keys.

**P3-m3 — minor — `guardBreach` can share the guard's defects, and the bounds tests miss the sibling-prefix axis.** *Owner: `architect` (§3.4 `guardBreach`, §5 U5.5/U5.7). Inferred from reading. The sibling-prefix case is verified from H6's path.*
- "As the audit recomputes them" does not forbid importing the guard's own bounds function. With shared code, the canonicalization defect the plan fears would be missed by both. That voids §3.4's "Any run that completes shows such a call as `guardBreach`".
- U5.5's only breach fixture is outside `exchange/` entirely.
- H6's own planted path, `…/dm-probe-headless-outside/…`, string-prefixes the kit folder `…/dm-probe-headless`. So `ans-x/…` against `ans/` is a real axis. U5.7 lists `..`, a symlink, `req/` and relative paths, but not this one.

**Fix:**
- State that the audit's bounds code does not import `guard_hook.py`.
- Give U5.5 and U5.7 the same **axis table**, not a list of shapes: relative/absolute × `..` × symlinked file/directory × sibling prefix (`ans-x/`) × the bound directory itself × empty/non-string. Each axis needs a planted case in both suites, with its expected value written in the test.

**P3-n1 — nit — four small seams.**
- **Trade-off can be narrowed.** The guard logs every call to `hook-log.jsonl`, including calls that were never persisted (H3a's `f010`). A session-scoped "any out-of-bounds allow" scan of that log is not a one-to-one join, and it would see a breach that a kill cut short.
- **Hand-back.** Exempt it from `permissionDenied`, as from `guardBreach`, or have U5.7's smoke record its `permissionDecision`.
- **Plugins.** §3.2's "no user-level hook exists at any launch" overstates: `enabledPlugins` hooks are not refused. The consequences are loud (check 5).
- **Hook timeout.** The default command-hook timeout is 600 s (docs). Set an explicit `timeout` in the wiring so that a hung guard blocks within seconds.

**P3-n2 — nit — growth-stops residue.**
- §2's H-lines restate report §9 in prose. H3a alone carries five facts. Cut each H-line to one line citing its §9.x subsection, as n1 did for F1–F7.
- "Cited here as H1–H6" lists no H4.
- "F6 does not occur under `-p`" overstates §9.3, which says it "does not prove it".
- The U6.2 row's "(a), (b) and (e) pass … (f) fails …" is a gate-disposition list. The same is true of "U6.2 saw none" (§3.3), "U6.2 showed exactly one entry" (§3.4) and "U6.2 appended its §9" (§3.6). Leave the row as "Delivered (report §9)".
- §7's bullets restate §3.3/§3.4 (+47 words).

### Stopping condition

Every remaining finding except P3-m2 comes from v1.3's own fixes, and P3-m2 came to light through v1.3's guard-entry definition. **v1.4 passes as approve with suggestions by a `teco` line check, with no Pass 4,** if all of the following hold:
1. **P3-M1:** fail-closed denies carry a marker that §3.4 classes as a halt, with the U5.5 marked/unmarked fixture pair.
2. **P3-M2:** U6.7 names new kit-root paths and has D3 as a positive control, and (g)/(h) require D1/D2's `tool_use` and hook-log lines.
3. **P3-m1–m3:** each is ruled in at most two sentences plus its fixture row.
4. **No other new mechanism** is added in §3.3, §3.4 or U6.7.

**Falsifier:** if a line check finds a new major in §3.3 or §3.4, the guard design is still moving. The response is then to remove features, not to write v1.5. The candidate is the in-script fail-closed path: let the guard exit non-zero on internal error, so that `onFailure: "block"` gives the single, halting class. That depends on gate (h).

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*.
- **Addition (source: P3-M1):** A fail-closed path is traced to its consumer. The guard's own failure must reach a class that halts and is distinguishable from a policy decision. A fail-closed default that the consumer counts as the actor's fault turns a broken guard into silent attrition.
- **Addition (source: P3-m1, P3-m2, and gate (c) in report §9.1):** A row that keys on a missing record (no hook entry, no result, no field) is checked against every producer of that absence: a kill, batched or parallel writes, and validation before the recorder runs. Checking only the producer that motivated the row is not enough.
