# Review: audit-team.sh check 7 leak exemptions

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063 follow-ups

Scope: uncommitted `git diff claude/scripts/audit-team.sh` (baseline HEAD `bebfb7bbf73e9ef1d868dc0234a1f617db7dd806`) vs `docs/reviews/audit-team-leaks-triage.md` and the 2026-11-09 decision in `docs/plans/audit-leaks-cleanup-coordination.md`.
CPG: considered, not relevant — one Bash script, no CPG.

**Verdict: approve with suggestions.** One major (header match is not anchored to the real header), rest minor. All checks below were executed (awk body copied verbatim into a scratch tree; full script run old vs new with ROOT pinned).

## Findings

**M1 (major, verified) — a body line `> **Status:** archived` freezes a live doc.** The regex scans all of the first 12 lines, not just the header line under the H1. Planted file with header `> **Status:** active` at line 3 and a quoted `> **Status:** archived` at line 5 was skipped (not scanned). Reviews and plans routinely quote other documents' headers in blockquotes, so a live doc can silently drop out of check 7. Fix: stop at the first line matching `^> \*\*Status:\*\*` (take that line's verdict, frozen or not) instead of scanning all 12; keep 12 as the search bound for that first Status line. Planted-input test: the quote.md case above, must be scanned.

**m1 (minor, verified) — `archived-ish` / `archived-foo` is treated as frozen** (`[^A-Za-z]` accepts `-`). Harmless for real tokens, but `[^A-Za-z-]` would be stricter. `Archived` (capitalised) is not skipped, correctly (not canonical).

**m2 (minor, verified, pre-existing) — an unreadable file is not skipped but is dropped from the scan.** `getline` failure leaves `frozen=0`, so the name is printed; then `grep ... 2>/dev/null` swallows the permission error. No leak-through from the new code, but the check cannot report it. Same for a tracked-but-deleted file. Not introduced by this change.

**m3 (minor, by design) — no coverage-probe for the exemption set.** Nothing fails if `LEAK_EXEMPT_PATHS` stops matching (a directory rename silently re-exposes ~600 lines, which is the safe direction) — acceptable.

## Answers to the five questions

1. **Header-only / 12 lines.** Header `> **Status:** active · ... archived` is NOT skipped (good). Bare body text `Status: archived` is NOT skipped (needs the `> **Status:**` prefix). Boundary: archived on line 12 skipped, line 13 scanned (good). Failure mode is M1 only. CRLF headers and double space after the label are handled.
2. **RS switching.** Correct. Names with spaces, unicode, and an embedded newline all passed through the NUL-delimited list and were skipped/kept correctly; file with 1 line, or no trailing newline, handled; the `RS="\n"` / `RS="\0"` toggle restores before the next record. Unreadable: see m2 (kept in list, not leaked by the awk stage).
3. **Prefix match.** `index($0,P)==1` from the root: `cpg/tests/fixtures/headless_x/a.txt` IS scanned, `cpg/tests/fixtures/headless/b.txt` skipped. Correct. Note the trailing slash in each entry is load-bearing and is present.
4. **Hostname bound.** Host here is `<host>` (3 chars). Old check had exactly one hit, `docs/reviews/typed-decision-models4-impl.md:2254`, a finding ID `<host>` — pure noise, now gone. The `@<host>` form (case-insensitive via `-i`, ended by a non-word char) still misses: `g15:` / `g15$` prompts, `ssh g15`, `g15.local`, `\\g15\share`, `user@<host>` is caught but `<host>` alone in a log line is not. Bound: only the `user@host` shape is detected for hosts under 4 chars. Acceptable: the short-token alternative is a false-positive on every 3-char ID, and the policy already tolerates not scanning the username; but state it in the script comment ("misses non-@ forms") so the PASS is not read as proof. Low risk to the maintainer.
5. **Other checks unchanged.** Ran HEAD script (`git show bebfb7bb:...`) and the working-tree script with ROOT pinned to the repo, outputs diffed excluding hit-location lines: the only differing lines are the check-7 FAIL summaries (home path 139 files/672 lines -> 52 files/74 lines; hostname FAIL gone). Every other PASS/FAIL/NOTE/INFO line is identical. Residual 52 files are code under `cpg/spikes/frontend-fitness/adjudication/tools/` etc., which the decision keeps scanned (U2's scope).

## What's solid
Path exemption prefix semantics, NUL-safe plumbing, location-only output preserved, non-leak checks untouched, header-keyed (not path-keyed) frozen skip as decided.

## Open questions
None blocking. Whether to fix M1 before U1 closes is the owner's call; I recommend yes (one-line change).
