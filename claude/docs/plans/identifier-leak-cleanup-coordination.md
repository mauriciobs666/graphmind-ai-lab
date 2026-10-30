# Identifier-leak cleanup — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Stakeholder ask (2026-10-30): clear `claude/scripts/audit-team.sh` check 7's personal-identifier
leaks. **Git user.name and user.email are acceptable** in committed files; **hostname and home path
are not.** The username is not acceptable as a leak, but it coincides with the stakeholder's real
name, so a bare-word check is dropped to keep genuine name mentions; it stays caught inside paths. Surfaced by `docs/plans/document-size-discipline-coordination.md` (I2).

Identifier forms in scope (both U1 and U2 briefs carry this list): the absolute home path
`/home/<user>` (and `$HOME` expanded); its Claude-Code slug form `-home-<user>-` (inside
`/tmp/claude-<uid>/…` scratch paths and `~/.claude/projects/…` paths); the Windows-side home
path (`/mnt/c/Users/<winuser>/…`, `C:\Users\<winuser>\…`; added 2026-10-30); the hostname as a
word. The bare username is not scanned (see above). Out of scope: git history (no rewrite), the GitHub handle in the
`origin` remote URL, and the other session's uncommitted `docs/plans/connected-code-graph-p0*.md`.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` | `ae050f62ae4af7f15` | accepted | `claude/scripts/audit-team.sh` check 7: drop git name/email from FAIL; catch the slug form; don't flag the username where it is part of the git name; `cobb` kaizen | U3 Passes 1–2 → approve with suggestions; P2-m1 verified by teco probe | 150k tok / 11 tools (last resume; cumulative tool count not reported) |
| U2 | `devops` | `a8e68e868b95df4e1` | accepted | Redact in-scope identifiers from every tracked + untracked-non-ignored file except the excluded ones; list of files changed | U3 | 104k tok / 28 tools |
| U2b | `devops` | `afe02b0e733fdf7ec` | accepted | Redact the Windows-side home path (U3 M1); make `salesperson/scripts/qwen_watch.sh` portable; fix the Python `$HOME` nit | U3 Pass 2 → approve with suggestions | 70k tok / 16 tools |
| U3 | `analyst` | `af797a03a10dd1d02` | accepted | `claude/docs/reviews/identifier-leak-cleanup.md`: static review of U1+U2 diffs | gate on U1+U2(+U2b) → Pass 2 **approve with suggestions** (P2-m1 → `cobb`; P2-n1 fixed by teco) | 150k tok / 58 tools (cumulative) |

## Done-conditions

- **U1** — the amended check 7 FAILs on each in-scope form (proved by a mutation: a scratch file
  carrying each form makes it FAIL) and does not FAIL on name/email alone.
- **U2** — after U1 lands, check 7 shows no FAIL except hits in the excluded files; no config
  file's behaviour changes silently (a path a tool needs at runtime is made portable, not blanked).
- **Close** — `teco` commits by explicit path; reports the excluded files and git history as residue.

## Notes

- 2026-10-30: closed. Residue: the other session's `docs/plans/connected-code-graph-p0{,-2}.md` (8 home-path lines; check 7 still FAILs on them until that session scrubs them); git history; K-059 (checks 2–3 echo `$HOME`). P2-m1 not re-gated: one-pattern change, mutation-proven by `cobb`, re-probed by teco.

- 2026-10-30: stakeholder: Windows-side home path is in scope (U3 M1) → U2b + U1 resumed; the two "the human (`<user>`)" removals stay.

- 2026-10-30: stakeholder: bare username must not be flagged — it coincides with their real name, whose genuine mentions must not be removed (clarified: not that the username is "allowed") → U1 resumed to drop the username check; `analyst` (U3, in flight) told immediately; U2's two existing username redactions stay (harmless). The forms list above is superseded for the username clause.

- 2026-10-30: U1 first delivery: name/email dropped from check 7, slug form + name-exemption added, output masked. teco ruling: a standalone capitalized first name is the (allowed) name → username match made case-sensitive (U1 resumed). `cobb` filed K-059 (checks 2–3 echo `$HOME` in PASS lines) as a follow-up.

- 2026-10-30: redaction edits to `archived` documents are permitted for this cleanup (stakeholder
  ask); they replace identifiers only, no other content.
