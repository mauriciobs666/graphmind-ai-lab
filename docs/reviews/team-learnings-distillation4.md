# Team learnings distillation 4 — review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063 · **Extends:** `docs/reviews/team-learnings-distillation3.md`

## Scope & verdict

Reviewed the uncommitted working-tree diff of `cobb`'s second raw-learnings sweep (14 modified files plus the untracked sweep report and coordination ledger) against `docs/reviews/team-learnings-distillation4-sweep.md` and `docs/plans/team-learnings-distillation4-coordination.md`. Checked: 7 promoted / 33 rejected of 40; the 3 ingested KB claims; the manifest; size and leak rules; root `REVIEW.md` applied. Nothing in `ws:agent-team` was deleted and no git state was mutated (scratch repos live in the session scratchpad).

**Verdict: approve with suggestions.** No blocker. One major: a promoted KB claim states a measurement that does not reproduce. The rejection set held up: all 33 dispositions are right, though two citations point at the wrong lines or files.

CPG: considered, not relevant — documentation/KB sweep; the only code reads were test-collection and a one-line `normalize_name` call.

## Findings

### Major

**M1. KB claim `040a2630` (and its `.md` home) states an exit code that does not reproduce, and over-generalizes `git add -A <dir>`.** Text: "`git add -A <dir>` skips it without output ... only an explicit `git add <path>` refuses ('paths are ignored', rc 0 under `--dry-run`)". Verified in a scratch repo with `lib/` ignored (git 2.43.0, the same git the claim was measured on):
- `git add -A --dry-run comp` (parent of `comp/lib`) → silent, rc 0. True.
- `git add -A --dry-run comp/lib` (naming the ignored dir) → "The following paths are ignored", rc **1**.
- `git add --dry-run comp/lib/x.py` (explicit) → refuses, rc **1**, not 0. A real `git add` is also rc 1.

So the "rc 0 under --dry-run" figure is wrong, and "`-A <dir>` skips silently" holds only when `<dir>` is an ancestor of the ignored dir. The practical advice (rename the dir, run `check-ignore -v`) is sound. The claim is labelled "Verified 2026-11-06", so the false figure is a verified-claim defect, and it sits in a retrievable KB document. Fix: edit the `.md` section in `claude/devops/ops-quirks.md` to "a plain `git add <path>` of an ignored file exits 1 with the 'paths are ignored' hint; `git add -A <ancestor-dir>` skips it silently, rc 0". Then re-ingest the claim and replace `040a2630` in the manifest's `_cobb_distillation4_2026_11_06`. The delete of the old document is the user's call, not mine. Verified (run).

### Minor

**m1. Row 22 (`a14fafa1`) cites files that do not carry the fact.** `cpg/eval/headless/README.md` contains no `--bare`, `--safe-mode` or `--strict-mcp-config` (line 136 only says OAuth works under `--restricted`), and `S52-NOTES.md` has one `--restricted --settings` hook note (line 93). The fact is documented, but in `docs/plans/connected-code-graph-p0.md` lines 1061-1084 (`--safe-mode` rejected, `--bare` unusable with OAuth) and `docs/plans/connected-code-graph-p0-2.md` lines 1085/1104. The rejection stands; correct the citation in the sweep report row and the matching `history.md` line. Verified (read, `git grep -E -e '--bare|--safe-mode|strict-mcp-config'`).

**m2. Row 15 (`f92a6f31`) cites wrong line numbers.** `docs/plans/salesperson-query-graph-data.md` lines 742 and 761 are a command block and a `git archive` pathspec note. The "no in-tree ministral baseline, 39 records, no `toolCompletion`" statement is at lines 93-95 of the plan; `falkor-chat/docs/HISTORY.md` line 102 holds the ministral numbers. The rejection is right. Verified (read).

**m3. Row 17 id typo.** The sweep report prints `b7176409`; the real id is `b71764b0...` (manifest and `history.md` have it right). Under my first recompute the typo showed as a phantom pending id. Fix the report cell.

**m4. Row 10 (`510c9f16`): "misattributed" is correct but incomplete.** `language_salience_probe.py` has no p95 (it records `turn_ms` only, line 433). But `salesperson/scripts/single_model_analysis.py` (lines 117-119, 178) also computes the replied-only p95; the row names only `cart_outcome_analysis.py`. Add it, since the K-078 S7 single-model run is likelier to have used it. Verified (grep).

**m5. Row 13 wording.** "NBSP is stripped": `normalize_name('Foo Bar')` returns `'foo bar'`, so NBSP becomes a space. `normalize_name('Foo​Bar')` does return `'foo​bar'`, so the open-defect finding stands. No `falkor-chat/docs/BACKLOG.md` entry exists (grep: reviews only), as the report says. Verified (run).

**m6. The `redis-cli` KB claim `ec14609e` partly duplicates existing docs.** "`redis-cli` exits 0 on an error reply" is already at `skills/joern-cpg/SKILL.md` line 112 and `cpg/infra/ccg-falkordb.sh` lines 160/262. The genuinely new content is the wrong-`REDISCLI_AUTH` → `PONG` rc 0 case, which I reproduced. Acceptable; consider trimming the title to that.

### Nits

- The sweep's "33 rejected" includes row 2 (`9f9c7e1e`), which also yields the `pkill` bullet; the manifest lists it only under `rejected`. Harmless, but "7 promoted" understates by one bullet.
- `32f13a00` and `0f3111eb` were both rejected as "doc'd" in single-incident RCAs (`salesperson-fabricated-prices-rca.md` lines 35/43, `salesperson-cart-product-id-rca.md` DEF-1). The citations are accurate (I read them). Both raw entries said "Suggested home: knowledge base"; the RCA is the only retrievable home. Defensible under the standing "no one-off incident facts" rule; flagged as a judgment call only.

## Checks run

**(a) Rejections — 22 of 33 checked (every one marked "Not re-run" for which the cited line was cheap to read).** Quoted results:
- `4b9c6680`: `run_nlq_golden_set_eval.py` lines 98-103 build `_RESULTS_PATH` from `NLQ_EVAL_RESULTS_PATH`, docstring lines 35-38 say unset overwrites the tracked file; `git log -S'NLQ_EVAL_RESULTS_PATH'` → `ad01aee7`. Fixed, confirmed.
- `510c9f16`: misattribution confirmed (m4).
- `39c9b4cf` / `b3257162`: the correcting document opens "This corrects document 39c9b4cf..."; the sandbox design basis is `docs/plans/connected-code-graph-egress-spec.md` line 529 ("Never present: `/mnt/*`, `/run/WSL`, `/init`, any `docker.sock`") and line 535 (`WSLInterop-late`); `cpg/sandbox/README.md` line 170 (N1/N2). Confirmed.
- `0f3111eb`: `salesperson-cart-product-id-rca.md` DEF-1 ("89 failure groups, each exactly 3 calls") and `docs/plans/standard-local-model.md` line 98 (`kv_unified`). Confirmed; non-promotion is right.
- `32f13a00`: RCA lines 35 and 43 carry both halves. Confirmed.
- Also read and confirmed: `443b6cf8` (`DESIGN.md` 453), `eddc90b4` (`forbidden-flags.txt` lines 10/18), `9b26f2eb` (`falkor-chat/AGENTS.md` 115 names qwen JIT), `7bf10aeb` (`ensure_demo_models.sh` header, `lm-studio-model-notes.md`), `56fac9ed` (`claude-code.md` 255-262), `e02bb248` (report line 298), `31115c98` (`toolchain/README.md` 119-124), `9b6886df` (`Dockerfile` 336-346), `cef174b2`, `78040b73` (`Dockerfile` 8-14), `a0d7766f`, `eec5f55e` (spec line 598-606), `72e164b3` (README 151-158), `26255ae0`, `2ff81eb9`, `7e7fa532`, `b0060e9a` (spec H8 line 1011), `dfbcdf08`, `2cc6debd`, `9a5e835d` (`python-web-quirks` line 860). Not checked beyond the sweep's own cite: `f7c51ef6`, `ebcf3cf5` (offline-run.sh line 240 comment is as quoted).
- Two rejections have wrong or mismatched citations (m1, m2). Neither changes a disposition. No rejection cites a doc that fails to say the fact.

**(b) Promoted claims — 6 re-run.** (pytest 9.1.1, scratch project, `falkor-chat/server/.venv`)
- Absolute `--deselect`: from the rootdir, `--deselect <abs>/tests/test_x.py::test_b_live` → `2 passed`; relative id → `1 passed, 1 deselected`; `-k "not _live"` from `/` → `1 passed, 1 deselected`; absolute id from `/` → `2 passed`. Matches the new `python-web-quirks` section, including the correction.
- `grep -qv`: `printf '' | grep -qv x` shim 0 / `/usr/bin/grep` 1; `printf 'a\nb\n' | grep -qv a` shim 1 / GNU 0; `printf 'a\n' | grep -qv x` 0 / 0. `type grep` → "grep is a function". ugrep 7.8.4, GNU grep 3.11. Matches exactly.
- `.ino`: `c2cpg.sh` on one `a.ino` → rc 0, `cpg.bin` 16,341 B, the figure claimed; same source as `a.cpp` → rc 0, 38,637 B (different source from the claim's 24,532 B; the pattern holds, and the claim states its own size). Method count not inspected, as stated.
- `SERVER.md` §1.7: `tests/test_tools.py --collect-only -q` → `129 tests collected`; with `-k "not _live"` → `126/129 ... (3 deselected)`. The only `test_tools.py` tests taking `repo`/`conn`/`fresh_vector_index` are the three `*_live` ones (lines 1981, 2041, 2061). The bound "collection from `server/`" is stated.
- Pathspec (`92bdc7df`): from `falkor-chat/server`, commit `6286de44`: root-relative `git diff --quiet` rc 0, `':/falkor-chat/server'` rc 1, `../server` rc 1, `git archive -- falkor-chat/server` rc 128; `git status --short` prints only a stderr warning. All as written.
- `redis-cli` 7.0.15: `FOO` → `ERR unknown command 'FOO'`, rc 0; `REDISCLI_AUTH=bogus ... PING` → `AUTH failed` on stderr, `PONG`, rc 0. The `NOAUTH` half is declared not re-run.
- `lib/`: `git check-ignore -v` prints `.gitignore:17:lib/` for both paths; `falkor-chat/scripts/shlib/` exists. The `git add` details failed (M1).

**(c) KB claims.** `92bdc7df`, `ec14609e`, `040a2630`: `status` = `ready`; chunk counts 2, 2, 1 and every chunk has a non-null embedding (cypher on `ws:agent-team`). `get_document` text compared by eye against the three `.md` section bodies: identical. `ingestedById` is `cobb` on all three. No raw document was touched.

**(d) Manifest.** JSON parses. `_dispositioned_raw_distillation4` holds 40 ids (`dispositioned_not_deleted` 40, `promoted` 7, `rejected` 33, `deleted` 0; promoted and rejected partition the 40). `_cobb_distillation4_2026_11_06` holds 3 ids matching (c). All 40 ids exist in the graph. Recompute: 134 `Fact:` documents; minus the 36 present ids of `_dispositioned_raw_distillation3` (4 deleted, documented) minus the 40 → **58 pending**, matching K-063. The 40 are exactly the oldest 40 of the 98 pending at sweep start (set equality). The `_note` in the new key honestly says the byte compare was not run.

**(e) Size discipline.** No `AGENTS.md`/`CLAUDE.md` touched. Additions are new sections or bullets; no existing entry was left wrong, and the only replaced prose is the K-063 row (rewritten in place, not appended). One added line over 700 chars: a `claude/cobb/kaizen/history.md` entry (743 chars), a lookup-read file, not an always-loaded one. Manifest +100 lines is two new keys and no change to existing ones; sane. The sweep report is a new, active document (growth-stops not triggered).

**(f) Leaks.** Over `git diff -U0` added lines: 0 matches for the user name, `/home/`, `/mnt/c/Users/<name>`, or the host name. This review contains none.

**(g) Contradictions.** Promoted text does not contradict a rejection: `--deselect` is not relied on in the rejected rows; the KV, qwen-JIT and `unshare` premises are not repeated. Against existing docs: the `-k` guidance sits beside, and does not conflict with, `claude/analyst/review-techniques.md` lines 235-243 (`-k` vs `-m` counts differ per project). The new `grep -qv` bullet is consistent with the existing shim bullets in `claude-code.md`. The pathspec section's "`':/...'` works from any cwd" is true for `git diff`/`git archive` here; the plan note at `salesperson-query-graph-data.md` line 761 (`git archive` rejects `:/`) did not reproduce from `falkor-chat/server` (rc 0) — not a defect in this diff.

**(h) Root `REVIEW.md`.** Evidence bar met above. The only check the sweep relies on that can silently pass is the `grep -qv` test advice ("test with `/usr/bin/grep`"), which I ran with a planted positive control (empty input) that flips shim 0 vs GNU 1. The sweep report's own "no latency figure carried" lesson holds: none appears in promoted text. No spec-update or growth-stops rule applies to this diff.

## What's solid

- The rejection set is accurate on facts: 33 of 33 dispositions are right; the high rate reflects that the oldest entries come from CPG/sandbox work already well documented.
- Promoted claims were re-run by `cobb` and honestly narrowed: `--deselect` correction (the "outside rootdir" premise did not reproduce) and the `grep -qv` single-line case marked as not reproducing. Both agree with my runs.
- KB hygiene: all 3 claims ready and embedded, text matches the `.md` homes, manifest accounts for every id, remainder of 58 correct.
- No raw deletes, no leaks, no always-loaded file touched.

## Open questions

1. `040a2630` needs re-ingestion after the M1 fix, which implies deleting the old document. That delete is the user's decision (the harness has blocked deletes before). Alternatively `cobb` can leave the KB claim and fix only the `.md`, but then the two diverge; I recommend re-ingest.
2. Should `normalize_name` keeping U+200B (sweep Findings 1) get a `falkor-chat/docs/BACKLOG.md` item? Owner decision, outside this gate.
3. Root `.gitignore` line 17 `lib/` anchoring (sweep Finding 2) is a repo-owner call; unchanged by this review.
