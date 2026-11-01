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
