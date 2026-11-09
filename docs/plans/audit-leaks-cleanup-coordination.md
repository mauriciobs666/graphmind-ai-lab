# audit-team.sh identifier-leak cleanup — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-063 follow-ups

Source: `docs/reviews/audit-team-leaks-triage.md`. User decision 2026-11-09: approve the exemptions (header-keyed `archived`/`superseded` skip; path exemption for recorded fixtures/results/samples), proceed. Live-document hits are genericized (`$HOME`, `<host>`); hardcoded paths in code are fixed in code.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `devops` | `a25ada8b521cf0572` | accepted | `claude/scripts/audit-team.sh` exemptions + hostname false-positive fix + triage re-check | `analyst` → approve with suggestions (M1 fixed); `eca54ba2` | — |
| U2 | `coder` | `a5769f3fef5c7ecbb` | accepted | hardcoded paths in `cpg/tests/test_eval_tools.py`, spike scripts | `analyst` → approve with suggestions; `786eca12` | — |
| U3 | `analyst` | `a192ed1d444ff7ec1` | accepted | 6 live `docs/reviews/*` files genericized | gate skipped (token swap, diff checked); `bebfb7bb` | — |
| U4 | `qa-engineer` | `a7042939aeff66d87` | accepted | `docs/test-reports/typed-decision-models-harness-probe-report.md` | gate skipped (token swap, diff checked); `486f2d1c` | — |
| U5 | `cobb` | `afddba2d601575ed2` | accepted | `claude/docs/reviews/team-engagement-in-other-repos.md` | gate skipped (one token); `296ea133` | — |
