# Identifier-leak cleanup — review of U1 (check 7) and U2 (redactions)

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

**Reviewed:** the uncommitted U1 and U2 diffs against commit `dfa5d537`, per
`claude/docs/plans/identifier-leak-cleanup-coordination.md`.

- **U1 (`cobb`):** check 7 in `claude/scripts/audit-team.sh` and `claude/cobb/kaizen/{history,plan}.md`.
  I reviewed the **final** state of the file, after the mid-review policy change. The username
  check and its name/email exemption are now gone. Only the home-path check (absolute and slug
  forms) and the hostname check remain. teco reworded the policy comments afterwards (comments
  only).
- **U2 (`devops`):** the 29 redacted documents and `opencode/agents/tank/opencode.json`.
- **Not reviewed:** the other session's `docs/plans/connected-code-graph-{p0,p0-2,bundle-spec,egress-spec,…}.md`.

Identifier values are masked throughout as `<home>`, `<user>`, `<host>` and `<winuser>`.

**Verdict: approve with suggestions.** There are no blockers.

- The final check 7 does what the policy now asks.
- U2's redactions change only identifiers, and every hunk pairs line for line.
- The `{env:HOME}` keys resolve correctly. When `HOME` is unset they **fail closed**, not open.
- The one major finding is a scope question for the stakeholder: a Windows-side home path that
  no check covers.

CPG: considered, not relevant — the artifacts are a Bash script, Markdown and a JSON config; no `cpg_*` graph covers `claude/` or `opencode/`.

### How this was verified

- **Full audit, run twice.** I ran `claude/scripts/audit-team.sh` after both revisions. It exits 1.
  The only FAIL is the home-path check, on 5 lines in the excluded
  `docs/plans/connected-code-graph-p0{,-2}.md`.
  - I also ran the `dfa5d537` script against the same tree, with only its `ROOT` line repointed
    to the repo, from a scratchpad copy.
  - With every `repo:` line filtered out, the two outputs are **identical** (121 PASS lines). Checks
    1–6 and 8+ are therefore unchanged.
- **Check 7 in isolation.** I extracted check 7 verbatim into a scratchpad harness and ran it
  against a scratch git repo. Synthetic identities were injected through shim `id`/`hostname`
  binaries, a `HOME` override and the repo-local `git config`. Results are in Appendix A.
- **U2 diffs.** I read them all, with masking applied, and swept the tree for leftover and
  variant forms.
- **Config.** I ran `opencode debug agent tank` under three settings: the real `HOME`, a
  scratchpad `HOME`, and `HOME` unset. I also ran `tests/wrapper-scripts.sh`: 23/23 PASS.

## Findings

### Major

**M1 — The Windows-side home path, `/mnt/c/Users/<winuser>/…` and `C:\Users\<winuser>\…`, is in
no check, and U2 did not touch it. Owner: stakeholder decision (via `teco`), then `devops` and `cobb`.**

- **Evidence (verified by sweep):** 16 lines across 9 tracked files. Among them are
  `falkor-chat/docs/plans/wsl2-memory-diagnostic.md`, `falkor-chat/docs/BACKLOG.md`,
  `falkor-chat/docs/HISTORY.md`, `docs/plans/small-model-benchmarking-coordination.md`, and one
  executable: `salesperson/scripts/qwen_watch.sh:4` (`LMS=/mnt/c/Users/<winuser>/.lmstudio/bin/lms.exe`).
- **Why it matters:** `<winuser>` is the Windows profile name. It differs from `<user>`, so neither
  `$HOME` nor its slug matches it. It is a home path on the same machine and identifies the same
  person. The ledger lists only `/home/<user>` forms, so this falls outside U2's brief. The
  policy's intent ("home path is not acceptable") plausibly covers it.
- **Suggested improvement:** ask the stakeholder whether the Windows profile path is in scope. If
  it is:
  - `devops` redacts the prose occurrences.
  - `devops` makes `qwen_watch.sh` portable rather than blanking it. One option is to derive the
    path at runtime from `cmd.exe /c echo %USERPROFILE%` with `wslpath`. Another is an
    `LMS` env override that keeps the current value as a fallback, kept outside the repo. The
    plans already mention a glob form, `/mnt/c/Users/*/.lmstudio/bin/lms.exe`
    (`docs/plans/small-model-benchmarking.md:235`).
  - `cobb` adds a runtime-derived pattern to check 7. The value is the Windows profile directory,
    resolved through `wslpath` when running under WSL, and the pattern covers both separators.
    The check stays silent off WSL.

### Minor

**m1 — A `$HOME` with a trailing slash silently narrows both home-path patterns. Owner: `cobb`.**
*(Verified in the harness.)*

- With `HOME=/home/maria/`:
  - `cd /home/maria` (end of line) → no hit.
  - `x -home-maria` → no hit.
  - Only `/home/maria/y` is caught.
- The removed username check used to backstop this, so the gap is new in practice.
- **Fix:** normalise once with `h=${HOME%/}`. Use `$h` for both the absolute pattern and the slug
  derivation.
- **Test:** add a harness row with a trailing-slash `HOME` and an end-of-line occurrence, and
  assert that it FAILs.

**m2 — The script comment "a username leaking inside a path is still caught by the home-path
forms" (`claude/scripts/audit-team.sh:33-36`, repeated in the ledger header) overclaims.
Owner: `cobb` (comment); `teco` (ledger).** *(Verified in the harness, Appendix A.)*

- These path forms are **not** caught: `\/home\/<user>\/` (JSON-escaped), `/home//<user>`,
  `%2Fhome%2F<user>`, `~<user>/…`, and `/mnt/c/Users/<winuser>` (M1).
- The live tree holds none of the first four (swept), so nothing leaks today.
- **Fix:**
  - Narrow the comment to "the home path, in its absolute and slug forms".
  - If any escaped form should be gated, add it as a derived pattern, for example
    `${HOME//\//\\/}` for JSON-escaped text. Gate it with a coverage probe over separator × escape
    × case, not a hand-picked list.

**m3 — The policy rationale in `cobb`'s kaizen predates the stakeholder's clarification. Owner:
`cobb`.**

- `claude/cobb/kaizen/history.md` (2026-10-30 entry, "Second narrowing: the username is acceptable
  outright") and `plan.md` K-056 ("username … acceptable") both say the username is acceptable.
- The stakeholder's final position is different. The username is **not** acceptable, but its
  bare-word check is dropped because it coincides with their real name.
- The script comments and the ledger header already carry the corrected wording. The kaizen should
  match, or a later maintainer will "restore" the check on a misread.
- **Fix:** reword both entries.
- Related, for `teco`: the ledger's "Identifier forms in scope" paragraph still lists "the bare
  username as a word …".

### Nits

- **n1 — A Python snippet now reads `sys.path.insert(0, "$HOME/…")`. Owner: `devops`.** The file
  is `model-bench/docs/reviews/small-model-catalog-sweep-impl.md:154-155`.
  - Python does not expand `$HOME`. A copy-paste of the snippet would import from a nonexistent
    directory. The failure is loud (`ImportError`), not silent, so this is a nit.
  - `os.path.expanduser("~/prg/…")` or a repo-relative path would say what it means.
  - The other `$HOME` substitutions are fine:
    - `$ git log -- $HOME/…` (`docs/reviews/cpg-provenance-stamp.md:951`) expands correctly
      in a shell.
    - The `SOURCE_PATH:"$HOME/…"` and `"cwd":"$HOME/…"` lines show recorded values and read
      clearly as redacted.
- **n2 — Inconsistent placeholders. Owner: `devops`, take or leave.** Slug paths became
  `/tmp/claude-<uid>/-home-<user>-<repo>/…`, which redacts the uid and the repo path. Neither is in
  scope. Absolute paths kept `$HOME/prg/graphmind-ai-lab/…`. Both are harmless; one convention
  would read better.
- **n3 — Hostname false-positive surface. Pre-existing, not introduced here. Owner: `cobb`.** The
  real hostname is a very short token, matched case-insensitively as a word. Zero hits today.
  If false positives start, an allowlist is cheaper than loosening the match.

## Policy fit

The outcome matches the ledger's final policy. I checked four things:

- **Name and email:** they are never scanned. A line with `Maria Silva <maria.s@ex.com>` passes
  (Appendix A).
- **Home path and hostname:** they FAIL in every form the ledger lists, including the slug form
  in both the scratch-path and projects-path shapes, all-caps, and an FQDN host.
- **Bare username:** it no longer FAILs, as the clarification requires.
- **Genuine name mentions (the coordinator's explicit check):** U2 removed no genuine mention of
  the stakeholder's name.
  - Of the 49 occurrences of the first-name token on U2's removed lines, 47 sit inside a home
    path or slug.
  - The other two are the "the human (`<user>`)" pair in
    `claude/docs/plans/bypass-permissions-subagent-gap{,-coordination}.md`, which teco ruled
    acceptable.
  - The surname token appears on no changed line.
  - See Open question 2.

## What's solid

- **Masked output.** `leakfail` prints `path:line` plus file and line counts. A human can locate
  every hit, and no identifier reaches the output.
- **`grep -H`.** It closes a real single-file-batch gap.
- **Runtime derivation.** Every pattern is derived at runtime. The `HOME=/` guard and the
  empty-hostname guard both hold (verified).
- **Check 7 had no username/exemption logic left to break.** The removed version had a latent
  column-1 bug: the awk `substr(c, 0, 1)` under gawk returns the first character, so a username at
  the start of a line was never flagged. The removal made it moot. If a bare-word check ever
  returns, guard the boundary read with `s > 1`. I checked that guard against `grep -w` over a
  128-case position × neighbour probe: zero mismatches.
- **U2 is surgical.** It touches 30 files, 55 lines added and 53 removed. Every hunk pairs line for
  line except the README's added 2-line explanation. The only non-path change is a hostname value
  in a JSON excerpt, replaced with `"<host>"`.
- **The `{env:HOME}` config is sound (verified live).**
  - OpenCode 1.18.30 substitutes `{env:VAR}` inside permission **keys**. With a scratchpad `HOME`,
    the three patterns resolve under that directory.
  - With `HOME` unset, they resolve to `/prg/graphmind-ai-lab/…`. No real invocation matches that,
    so the `"*": "deny"` rule governs. The stated silent-empty risk therefore **fails closed**:
    bring-up and tear-down get refused rather than widened.
  - The README documents the clone-location caveat.
  - One limit on the evidence: `tests/wrapper-scripts.sh` (23/23) never reads `opencode.json`, so
    it says nothing about this change. Only the `opencode debug agent` evidence does.

## Open questions

1. **Is the Windows-side home path in scope?** (M1) This needs the stakeholder's call before
   anyone acts on it.
2. **The two "the human (`<user>`)" removals.** teco ruled them harmless. Under the stakeholder's
   clarification ("genuine mentions of their name must never be … removed"), they could be read
   as name mentions. They identify a person in backticks, which is closer to a login than a name.
   I treat them as not a finding. Flagging it so the stakeholder can overrule.

## Pass 2

**Verdict: approve with suggestions.** There are no blockers and no majors.

**Scope:** baseline is still `dfa5d537`. This pass covers:
- `cobb`'s revision of check 7: Windows-side derivation, `${HOME%/}`, and bounded escaped-literal
  EREs;
- `cobb`'s kaizen;
- `devops` U2b: 8 redacted documents, `salesperson/scripts/qwen_watch.sh` and the Python nit.

**Stakeholder decisions applied (2026-10-30):** M1 is in scope. The two "the human" removals stay.

CPG: considered, not relevant — Bash, Markdown and JSON only; no `cpg_*` graph covers these paths.

**Coordinator's probes, re-run with the real identifiers.** I ran them through the extracted
check 7 (`bash -n` OK, real `HOME`, `hostname` and profile directory). Values are masked here.
- `C:\Users\<winuser>\x` → home path. Confirmed.
- `/mnt/c/Users/<winuser>/…` → home path. Confirmed.
- An expanded `$HOME` line → home path. Confirmed.
- The slug form → home path. Confirmed.
- The hostname in all caps → hostname. Confirmed.
- A bare username → PASS. Confirmed.
- Git name plus email → PASS. Confirmed.
- **Whole audit:** exits 1, with exactly one FAIL, the home-path check. It covers 8 lines, all in
  the excluded `docs/plans/connected-code-graph-p0{,-2}.md`. Three of those lines are the
  Windows-profile residue, so the new patterns catch it in the wild. Checks 1–6 and 8+ are still
  identical to the `dfa5d537` script.

### Dispositions of Pass 1 findings

- **M1 — fixed.**
  - **Derivation:** profiles come from the `/mnt/c/Users/*` directories that hold `NTUSER.DAT`.
    Symlinks and `Default` are skipped. The live host derives exactly one profile, in about 0.01 s,
    with no interop spawn.
  - **Separators:** a fixture with several profiles was derived correctly. It included a profile
    with a space and one with metacharacters (`Jo.Ana`, `Ana Lu`, `x(y)+z`, `Other`), and every
    separator form hit:
    - `\`, `\\` and `/`;
    - `\/` (JSON-escaped);
    - `%2F`, upper and lower case.
  - **Skips held:** `Default`, `Public`, a symlinked profile and the `All Users` link were all
    skipped.
  - **U2b:** the residue sweep finds the profile name only in the excluded files.
  - One gap remains: the slug separator. See P2-m1.
- **m1 — fixed.** With `HOME=/home/maria/`, `cd /home/maria` and `x -home-maria` now FAIL.
- **m2 — fixed.**
  - JSON-escaped and URL-encoded forms (upper and lower case) FAIL.
  - The script comment now lists what is not caught (`/home//<user>`, `~<user>/`). I confirmed
    `/home//maria` still passes, as documented.
  - The ledger header's "it stays caught inside paths" is still slightly broad, but the script
    comment is precise. Not re-raised.
- **m3 — fixed.** The `cobb` kaizen now says the username *is* a leak, and that the bare-word
  check is dropped only because the username coincides with the stakeholder's name.
- **n1 — fixed.** The snippet now uses `os.path.expanduser("~/…")` and adds `import os`.
- **n2** (placeholder style) and **n3** (hostname false-positive surface) — not addressed. Both
  were take-or-leave; I don't re-raise them.

**Escaping, the coordinator's focus** (verified in the harness). `ere_lit` escapes
`[]\.*^$+?(){}|/`, and GNU grep 3.11 accepts the redundant `\/` silently. Every case below
behaved correctly:

| `HOME` | Line | Result |
|---|---|---|
| `/home/m.a+r(i)a` | the literal path | FAIL |
| `/home/m.a+r(i)a` | `/home/mXa+r(i)a` | PASS (the `.` is not a wildcard) |
| `/home/m a` (space) | the path, and its slug `-home-m-a` | FAIL |
| `/home/m[1]` | the literal path | FAIL |
| `/home/m\x` | the literal path | FAIL |
| `/home/m\|x` | an unrelated line | PASS (no alternation leak) |

Boundaries also hold against the normalisation and the separators:
- `/home/maria2` and `Users/Jo.Anab` → PASS (end bound).
- `pre-home-maria`, and `pre-root` under `HOME=/root` → PASS (the slug's start bound fixes the
  short-`HOME` false positive from Pass 1).
- `%2Fhome%2Fmaria%2F` → FAIL. `-w` would have missed it, which is why the change to explicit
  bounds was right.

### New findings

**P2-m1 (minor) — The Windows slug separator only works for profile names that are entirely
alphanumeric. Owner: `cobb`.** *(Verified in the harness.)*

- **Evidence:** the `-` separator replaces only the separator, never the name's own characters.
  Claude Code's slug turns *every* non-alphanumeric into `-`, so for `C:\Users\Jo.Ana\…` the slug
  is `C--Users-Jo-Ana-…`.
  - The pattern `Users-Jo\.Ana` misses it: `C--Users-Jo-Ana-x` → PASS.
  - Likewise `C--Users-Ana-Lu-x` → PASS.
- **Why it matters:** the script comment claims "escaped/slug variants" for the Windows side.
  Today's host profile name is alphanumeric only, so there is no live gap. A dotted or spaced
  profile name, which is common, would pass silently.
- **Fix:** build the `-` row from the slugified name, `"Users-${w//[^A-Za-z0-9]/-}"`, and keep the
  other five separators on the raw `$w`.
- **Test:** add the two fixture rows above to the harness as must-FAIL. That is a separator ×
  name-character-class probe, not a single shape.

**P2-n1 (nit) — `qwen_watch.sh` resolves `lms` before it validates its argument. Owner: `devops`.**

- **Order is sound** (verified by sourcing the function in isolation): `$LMS` → `$LMS_CLI` → PATH
  `lms`/`lms.exe` → `cmd.exe` `%USERPROFILE%` → a stderr error with rc=1. The `cmd.exe` spawn,
  which can stall for about 13 s, is correctly last before the error. Every later use of `"$LMS"`
  is quoted, so a profile path containing a space is safe.
- **The nit:** `qwen_watch.sh` with no argument first runs the resolver, possibly including that
  `cmd.exe` stall, and only then prints the `${1:?}` usage error.
- **Fix:** move `OUT="${1:?…}"` above `LMS=$(resolve_lms)`. Optionally wrap the `cmd.exe` call in
  `timeout 20`.

**U2b redaction fidelity.** I read the word-level diffs of all 9 documents, with masking applied.
- Each hunk swaps only the profile name for `<winuser>`. In `salesperson-single-model.md` and the
  `extract-transient-retry` test plan, hunks also complete the Pass 1 slug redactions.
- Hunk pairs are line for line (1/1, 4/4, 5/5, 2/2). The one exception is the deliberate
  `import os` line in the `model-bench` review (3/2).
- No placeholder landed in an executable context. The archived `wsl2-memory-diagnostic.md` edits
  are identifier-only, as the ledger's archived-edit note permits.

## Appendix A — check 7 harness results (Pass 1 script)

Synthetic identity: user `maria`, host `box1`, name `Maria Silva`, email `maria.s@ex.com`,
`HOME=/home/maria`.

| Line content | Result | Line content | Result |
|---|---|---|---|
| `maria` (bare) | PASS | `x -home-maria-prg` (slug) | home path |
| `x maria y` | PASS | `~/.claude/projects/-home-maria-repo/x` | home path |
| `Maria Silva <maria.s@ex.com>` | PASS | `/HOME/MARIA/x` | home path |
| `BOX1` | hostname | `cd /home/maria` | home path |
| `box1.local` | hostname | `%2Fhome%2Fmaria` | PASS (m2) |
| `box10` | PASS | `\/home\/maria\/x` | PASS (m2) |
| `MARIA-PC` (host `maria-pc`) | hostname | `/home//maria` | PASS (m2) |
| `HOME=/home/maria/`: `cd /home/maria` | PASS (m1) | `HOME=/home/maria/`: `x -home-maria` | PASS (m1) |
| `HOME=/`, empty `HOME`, empty host | guarded, PASS | `HOME=/root`: `pre-root thing` | home path (short-`HOME` FP) |
