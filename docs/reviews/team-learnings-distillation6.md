# Team learnings distillation 6 — gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063 · **Extends:** `docs/reviews/team-learnings-distillation5.md`

## Scope & verdict

Reviewed: `cobb`'s uncommitted working-tree diff (11 modified files), the untracked sweep report `docs/reviews/team-learnings-distillation6-sweep.md` and the ledger `docs/plans/team-learnings-distillation6-coordination.md`. `resume_live_spec.sh` ignored. Baseline: HEAD `271eb087`. Root `REVIEW.md` applied; no component `REVIEW.md` is touched. Result gated: 18 raw entries, 14 promoted or merged, 4 rejected, 0 kept open, no raw deletion.

**Verdict: approve with suggestions.** No blocker. The 4 rejections hold: each cited file and line carries the fact. All 14 promoted claims reproduce, on a corpus that grew by one session. Two statements need a one-line edit before the commit (F1, F2); the rest are minor.

CPG: considered, not relevant — the diff is skill prose, one README paragraph, a JSON manifest and kaizen rows; the only code touched by a claim (`_kill_group`) I read directly.

## Findings

### F1 — major — the K-063 close text keeps a phantom owed action (verified)

`claude/cobb/kaizen/plan.md` K-063, open item (3): "Stale KB claim `040a2630` (superseded by `68509210`) awaits the user's delete." It does not: `MATCH (d:Document) WHERE d.documentId STARTS WITH '040a2630' RETURN count(d)` on `ws:agent-team` → 0, and `git show 1338e2ac` is titled "...superseded claim 040a2630 deleted by user". The item was carried over from the pre-sweep row. The sweep's Findings 1-2 are right; only this item is stale, and a close that lists a nonexistent delete sends the user to a dead end.
Fix: drop item (3) (renumber), leaving the raw-delete, `u1`, description, `2885564b` and `GRAPH.MEMORY` items. The other K-063 claims checked out: `_dispositioned_raw_distillation{3,4,5,6}` cover all 134 `Fact:` documents in the graph (Cypher `NOT IN` over the 138 manifest ids, 0 rows, so `d88545b5` is in the excluded set); `480ef825`, `a1d388f4`, `a93905a6` are cited by `model-bench` code, fixtures and plans; the 18 new ids appear in no repo document outside kaizen files, this report and the coordination row (`git grep`).

### F2 — major — "Every assistant `tool_use` is also replayed" is stronger than the counts show (verified, scan re-run)

`skills/agent-standards/claude-code.md` bullet "Transcript shape a reader must not assume", point (1): "Every assistant `tool_use` is also replayed with the same `id` in `api-request-blob` entries (4,919 distinct ids ...; 4,927 `tool_use` blocks in the blobs)". The two numbers compare distinct ids with non-distinct blocks, so they do not show containment. My per-id scan of the 29 main files (4,922 assistant ids, 4,930 blob blocks): 12 blob blocks are duplicate ids, 0 blob ids are missing from the assistant entries, and **4 assistant ids are absent from every blob**, all in one 2026-10-31 file (`bc52754a...`). "Every" is false for 4 of 4,922. The advice (count from `assistant` entries, or dedupe by id) is unaffected.
Fix: "almost every one (4,918 of 4,922 ids; 12 duplicate blocks)", or give the per-id containment check instead of the two totals.

### F3 — minor — the `710a0b5e` rejection argues against a rule the entry did not propose (verified, static)

Row 16 says "A rule 'never name a test path in an author's brief' would contradict that ruling". The entry's suggestion reads "never name a test path in a document-author's brief **without an owner and a size limit**" plus "split it into a separate implementer unit". The ruling it cites (`docs/plans/connected-code-graph-coordination.md` line 330: "fine for now, test carefully") accepts that outcome; it does not forbid a bounded-brief rule. Line 330, `docs/HISTORY.md` line 119, `4c5ee802` and 1,851 lines all check out, and `claude/architect/architect.md` line 54 states the role limit but did not stop the write (the brief overrode it), so "already stated" is thin. Rejecting as a one-off the user accepted is defensible; the stated reason is not.
Fix: re-word the row to "user accepted the outcome; one incident, no rule promoted", or promote the bounded-check-script line to `claude/teco/coordination-techniques.md` (`grep -n -i 'scratchpad\|check script'` there finds nothing now). Owner's call; no re-run is needed either way.

### F4 — minor — heading of the hook bullet asserts what its body disclaims (static)

Bullet "Hook decisions as the transcript records them, and what a hook 'allow' skips under `-p`". The body says the counterfactual (same write without the hook) was not run, so what an allow "skips" is not shown; a reader scanning headings takes it as established. I confirmed the observed half: `973830b1` has `accept/hook/hook` on the out-of-bounds `Write` (probe report §9.4, H6).
Fix: "...and what the transcript shows for a hook 'allow' under `-p`". Same bullet: "which never reached the permission step" for the 107 `<tool_use_error>` results is an inference (no run shows the step); say "carry no decision".

### F5 — minor — cites in row 10 point at the weaker line (static)

`docs/test-reports/standard-local-model-report.md` line 27 only shows `unload --all`; the lines that carry "`unload --all` removes the embedder" and "a CLI `lms load` without `--ttl` sets none" are 39 and 175. `claude/data-scientist/lm-studio-model-notes.md` 251-253 and `docs/test-plans/standard-local-model.md` 86 are right. The rejection holds; fix the cite.

### F6 — nit — small wording points

- Point (3) of the transcript bullet: "tool_result in 456 and end_turn in 4" leaves 6 of 466 unstated (my scan: 5 `user` entries that are not a `tool_result`, 1 `tool_use`). Add "the other 6 are neither".
- `python -I` section: "Drop `-I` (use `-P` or `-s` for isolation)": `-s` only drops the user site and neither flag ignores `PYTHON*` variables. Say "to avoid the bytecode write".
- Pre-existing, not this diff: `claude-code.md` ~lines 535-540 still say there is "no caller-facing toggle ... only" `DISABLE_BACKGROUND_TASKS` and "it isn't [in the env-vars reference]". Fetched today, `env-vars.md` has the `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` row and `sub-agents.md` names `CLAUDE_CODE_FORK_SUBAGENT=0`. Follow-up for the next skill pass.

## Checks run (what I re-executed)

| Claim | Result |
|---|---|
| `python -I` vs `PYTHONPYCACHEPREFIX` (scratch pkg, 3.12.3) | `-I`: prefix never created, `pkg/__pycache__` in tree; `-P`, `-s`, no flag: prefix created, tree clean; `sys.dont_write_bytecode` under `-I`: clean. Matches |
| pytest fd 0 (pytest 9.1.1, pipe on stdin) | default capture: parent and child `/dev/null`; `-s` and `--capture=sys`: `pipe:[...]`. Matches. The suggested recipe (subprocess with `stdin=PIPE`, child reads `/proc/self/fd/0`) run: the `DEVNULL`-dropping mutant fails, the real form passes |
| group-kill (`sh` leader, TERM-ignoring grandchild, flock `pass_fds`) | form A (wait on leader): group alive, lock held; form B (poll `killpg(pgid,0)`): group gone, lock free. Zombie leader reports alive before `wait()`, gone after. `runner.py` 376-388 holds the B form |
| Mutation copy (`model-bench`) | named tests: `2 failed` without `results/`, `2 passed` with it. `pytest` launcher from a `tests/` file resolved into the live tree; with `PYTHONPATH=<copy>` and under `python -m pytest` into the copy. (A test at the copy root resolved into the copy: the claim needs the file under `tests/`, as the repo has it.) |
| Offline `tsc` (`ccg-toolchain-truth:a82faf5219b3`, `--network none`) | `tsc` 6.0.3 at the cited path; `command -v tsc` empty, `node` at `/opt/truth/node/bin/node`; host has `javac 21.0.12.1`, no `node`/`tsc`. Beyond the claim: `tsc --noEmit --strict` exits 2 on a planted type error and 0 on a clean file |
| Transcript counts (467 subagent files, 44 main, 29 with tool_use) | 465 `Agent` results all `async_launched`, 380 without `run_in_background`; 463 `SubagentHandback` files, 457 end in `tool_result`, 4 in `end_turn`; `finished: true` 431, 36 absent; 280 files with a coordinator leg, 274 after a hand-back; 2 `parentAgentId`, both `spawnDepth` 2; median ratio 40.8 with a leg, 19.2 without (19.1 in the sweep; one new no-leg file, mine); `a24f...`: 86 messages, 14,153,036 / 258,383. The +1 or +3 drifts against the sweep are this session's own files |
| Hook decisions | 6,709 `accept/hook/hook` results, 0 without an earlier PreToolUse `hook_success`; 110 `is_error` results with no decision, 108 `<tool_use_error>`; 0 `reject/hook` in the repo's files. Probe `ada23a78` (2.1.316): two `reject/hook/hook:PreToolUse` (a deny, an `onFailure` block; content text differs), `permission-rule`, no attachment; one `accept/config/mode` without attachment. Matches |
| Settings page (fetched today, 449,263 bytes) | `disableAllHooks` "Any file"; `autoContinueAtUsageLimit` default `true`, "User or managed", project file can only turn it off; `switchModelsOnFlag` unset switches, `false` ends a `-p` request as an error. Docs: `FORK_SUBAGENT=0` and `DISABLE_BACKGROUND_TASKS=1` both named as wait levers (so "the docs name" is true) |
| Rejections' live state | `RESULTSET_SIZE` 10000; 9 `telemetry{*` keys against 9 graphs; `GRAPH.MEMORY USAGE` integer `_mb` only; no `ccg_*` graph loaded; `sspaths_pathcount_probe.json` tracked, 300,000 → 19,666 and `pathCount 0` → 3,521; `lms.exe ps`: embedder TTL cell empty, qwen shows a TTL (`8h / 10h` now, `10h / 10h` then) |
| Manifest | parses; `_dispositioned_raw_distillation6`: 18 ids = 14 promoted + 4 rejected, disjoint; matches the raw ids of the four rejections and 14 promotions |
| Leaks and size | `git diff -U0` adds 0 lines matching the user name, `/home/`, `/mnt/c/Users` or the 3-character hostname; the sweep and ledger have none. `awk` bar prints only the two pre-existing `falkor-chat/AGENTS.md` lines (87, 88); no always-loaded file is touched. `audit-team.sh`: FAIL on the same two repo-wide checks (146 files / 743 lines; 2 / 2) and 3 NOTEs, as the sweep reports. Longest added line over 700 chars is the K-063 plan row (1,223), not always-loaded |
| Cites after edits | `python-web-quirks/SKILL.md` line 999 is the `-q -q` section; `typed-decision-models6.md` 38-46 and 111-128 carry the settings keys; probe report F2/F3, §4.2/§4.3 carry the spawn-mode facts; no `claude/*/*.md` definition names `SubagentHandback` at HEAD |

## What's solid

- The cleared half: `605f76ea`, `15de3552` and `d0a55c70` are doc'd with matching numbers (`falkordb-quirks.md` 979-1000, 1044, 1256-1275, 1300; the recorded probe JSON; `lm-studio-model-notes.md` 251-253). `710a0b5e`'s facts all check out.
- Narrowings kept the original invocation form: the mutation-copy row ran `python -c`, `python -m pytest` and the launcher; `-I` ran four flag forms. The unrun parts (hook-allow counterfactual, `sys.pycache_prefix`, `tsc` effect on a running CLI, mutant M31) are stated as unrun in the promoted text.
- The two hook-related bullets (new transcript bullet; the pre-existing "allow is not a full override" paragraph at ~435) do not contradict: the old one is about settings rules and the classifier, the new one about what the transcript records; cobb's follow-up 5 names the overlap.
- No leaks, no always-loaded growth, the `description:` breach (6,690 chars) not widened and recorded as a follow-up.

## Open questions

- F3: reject as a one-off the user accepted, or promote the bounded-check-script line to `coordination-techniques.md`? (`cobb`/`teco`.)
- F1 touches `cobb`'s own kaizen row; `teco` applies it or routes it back.

### Proposed REVIEW.md lines

None. F2 is the same class as distillation 5 F1/F2 (a skill claim wider than its evidence) and is covered by the evidence bar.
