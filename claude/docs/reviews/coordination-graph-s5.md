# Coordination graph S5 — skill, template and catalog gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** coordination-graph (S5)

## Scope & verdict

Static gate of commit `6bf0ad2f` (S5): `skills/coordination-graph/SKILL.md` + `references/{backlog-format,ids-and-units}.md`, the `skills/agent-maintenance/SKILL.md` §1 `plan.md` template change, the `skills/README.md` catalog row, and the `claude/cobb/kaizen/history.md` entry. Baseline: `claude/docs/plans/coordination-graph.md` §4 S5 row, §3.1, §3.4, §3.6, §3.7; `coordination-graph3.md` §3.1–§3.6, §5; and the real code under `coordination-graph/coordgraph/` (code wins). No root/`claude`/`skills` `REVIEW.md` exists beyond the root one, which I applied.

Checks run (not just read): every built subcommand form via `python3 -m coordgraph` (the component has no `.venv`, so `coordgraph.sh` itself exits 1 with the documented message, as the skill says): `lint`, `lint --count`, `facts --like`, `history-coverage -v`, `migrate --dry-run`, `migrate --help` (never `--apply`; no store touched). Delivery-heading rules were run against `history.delivery_headings` on 19 planted headings; ledger labels/status tokens against `parse.ledger`; example IDs against `wilib.rules.parse`.

**Verdict: approve with suggestions.** No blocker: every recipe and flag the skill gives for a built command exists and behaves as stated; unbuilt behavior is marked planned in three places; the heading/`started`/fact-token conventions match the parsers. Eight minor findings and nits below, two of them (F1, F2) worth fixing before S9 builds on the text.

CPG: considered, not relevant — documentation/skill artifact reviewed against a small Python CLI read directly; no code-level change under review.

## Findings

### F1 (minor) — ledger-`Needs` feature rule omits the "at least one unit" condition
`references/ids-and-units.md` ("Need is **met** when … (feature ID) every ledger carrying it is archived or all its units are `accepted`"). `identity._feature_need` (`identity.py:261-269`) is met only if archived **or** `units >= 1 and accepted == units`; plan3 §3.4 says the same ("has ≥1 unit and all units `accepted`"). Read literally, the skill makes an empty non-archived ledger "met". Fix: add "and has at least one unit".

### F2 (minor) — status aliases wrongly scoped to "archived ledgers only"
`SKILL.md` ledger section and `ids-and-units.md` say `relayed`/`done`/`superseded` are aliases "(archived ledgers only)". `status_token` (`parse/ledger.py:55-71`) applies `ALIASES` to every ledger (verified: `done`→`delivered`, `relayed 2026-01-01`→`paused`, `superseded`→`abandoned`, no archived test). The restriction is the graph note's justification for the table, not a behavior. The skill's "anything else reads `unknown`" is right; the parenthetical misleads a writer into thinking `done` in a live ledger is a cell to fix when it silently maps. Fix: drop the parenthetical, or say "aliases exist for old cells; write teco's seven words".

### F3 (minor) — `migrate --apply` row lacks the S4 guard rails the plan puts around it
`SKILL.md` command table. The row names `--titles-reviewed` and "IDs are permanent" (good, and the one plan↔CLI mismatch is handled correctly), but a loaded skill presents `--apply` as an ordinary recipe. Plan §4 S4 / §3.6 reserve it: one file, one commit per file, owner confirms parking-lot statuses from the dry-run, stakeholder approves each `BACKLOG.md` edit, and the commit stages `.work-items/registry.jsonl` (plan3 §3.5). The CLI flushes internally (`cli.py:55`, `flush=lambda: wi_flush(root)`), so the skill's generic "run `wi flush` after any allocation" is satisfied for `migrate`, but staging the registry file is not mentioned. Fix: one clause on the row, "S4 only — see plan §4 S4; stage `.work-items/registry.jsonl` with the file". No need for a new section.

### F4 (minor) — planner behavior stated as present fact
`backlog-format.md` "Fact tokens" ("Two items overlap (and are never scheduled in parallel) when…") and "Milestone `started`" ("the planner never inserts a milestone ahead of a started one") describe `propose`, which is S7b/unbuilt. The build-status block at the top of `SKILL.md` covers the skill generally, but these reference sentences are read on their own. Fix: append "(`propose`, planned S7b)". Same for `SKILL.md`'s closing "A ledger edit is only done when the hook reports no failure": with no hook registered (S8) it cannot be satisfied; add "(once the hook is registered, S8)" as rule 3 already does.

### F5 (minor) — `query` carries an owner the plan does not give it
`SKILL.md` table: `query` "planned (S7a)". Plan §3.1/§3.4 define `query` and S5 requires a recipe, but the S7a row (§4) lists only `inflight`, `unit`, `planned`; no step builds `query` (this is the known gap). The skill's handling is acceptable (planned, ad-hoc questions meanwhile fall back to reading files), but it asserts an owner it cannot back, and it tells readers to prefer `query` over raw `mcp__cypher__query` with no stated fallback. Fix in the skill: "planned (unowned in plan §4; §3.4)". Route the plan gap to `architect` (assign `query` to S7a or S3).

### F6 (minor) — agent-maintenance template: stale legend and step text
`skills/agent-maintenance/SKILL.md:82` legend still reads "✅ done (then moved to history.md)"; the parsed word is `delivered` (`STATUS_WORDS`, `parse/backlog.py:35`), and `done` is unrecognised so `lint` raises `status`. Step 2 (line ~66) still says "move completed ones out of the active table". Neither matches the new field-block layout. Fix: "✅ delivered" in the legend and "out of the active items" in step 2. Also the new line 104 tells authors `wi new K` but not `--home <agent>` (what `migrate` uses, plan3 §3.5); add it or cite `skills/work-items/SKILL.md`.

### F7 (nit) — example IDs read as real
`SKILL.md` and `backlog-format.md` use `GAL-F0042`, `GAL-F0043`, `GAL-I02-K0042` and `falkor-chat:K-070` with no "illustrative" note (plan §3.6 marks its example so). They parse (`wilib.rules.parse` ok) and the registry holds only `GAL-I02-K0001`, so nothing collides today, but a copied example becomes a dependency on a future real item. One line: "IDs below are illustrative".

### F8 (nit) — duplicated content between `SKILL.md` and `backlog-format.md`
The delivery-heading rule, `started` convention, status words and fact-token rule each appear in both (SKILL.md ~1.2k words, references ~1.5k). Today they agree (checked), and the SKILL.md copy is the short one. Drift is the risk later; consider leaving only the two-line rule in `SKILL.md` with a pointer. Also the orphan sentence "Or drop the dependent's ref." at the end of the delivery bullet is detached from its condition. Not a growth-stops case (a skill is not a plan/requirements doc).

## Conformance checks (what held)

- **Recipes/flags.** `lint [--count] [PATH…]`, `facts [--like]`, `migrate --dry-run [PATH…]`, `migrate --apply PATH [--titles-reviewed]`, `history-coverage [-v]` all match `cli.py`'s parser; all other subcommands are marked planned, none presented as runnable. Exit-1 wording matches (`cmd_lint`). `lint` on the live tree currently reports 38 errors / 1 warning (pre-S4 layout), not a skill defect.
- **Known mismatches, judged.** `wilib-unavailable` is described exactly as coded (`lint.py:88-92`: error when `wilib_expected()`, else warning) with the right consequence ("a pass is then not a pass"). `identity` not built by the CLI: the skill says lint "compares IDs as written" today and the registry join is planned, which matches `cli.py` (no `Identity` constructed). `history-coverage` listed as built although absent from plan §3.1's table: correct.
- **Delivery heading.** Verified against `history.py`: level 2-4, fences skipped, verbs `closed/delivered/fixed/shipped/done` case-insensitive, `done criteria`/`fixed-up` excluded, `✅`+ID anywhere. `K-070/K-071 delivered`, `GAL-F0042 delivered:`, `falkor-chat:K-065/GAL-F0043 closed` deliver all listed IDs; `K-070, K-071`, `K-070,K-071` deliver nothing; `K-070 and/+/ / K-071` and `K-070/ K-071` deliver only `K-071`; `(part 1) delivered` and `K-067: … (mechanism delivered …)` deliver nothing; a level-5 heading and an H1 do not count. Every claim in the skill's list reproduces.
- **`started`.** `lint._milestone_findings` errors on disagreement among **open** items of a milestone and on a `🟡` item without `started`, as the skill states; `started` survives delivery per the parser.
- **Heading regex, field grammar, status words, `Milestone` parts, fact tokens (casefolded, mode mandatory)** match `parse/backlog.py`; unit-label cuts (`U10b Note:`→`U10b`, `U1(a)`→`U1`, `S6 gate` vs `S6` exact-key precedence) match `parse/ledger.py` and `identity.select_units`; registry header strings match `Identity.registry_line`.
- **Planned vs built.** Built list correct; store, views, `propose`, `kb-*`, hook marked planned; the "until S7a read the file" fallback is sensible.
- **Description** is 743 characters (limit 1024); `name` matches the folder.
- **Catalog row / cobb history.** Row text agrees with the skill; the history entry matches the commit, is newest-first, and honestly states what S9 still owns. The `~/.claude/skills/coordination-graph` symlink is absent (checked: `~/.claude/skills/` holds six links, not this one nor `work-items`), as the entry says is a machine step.

## What's solid

Accurate built/planned split; the five never-bend rules map one-to-one to plan §3.3/§3.4/§3.7; the delivery-heading section states the strict slash-join semantics with the failure forms listed, which is what the parsers need; the code-wins header on each reference.

## Open questions

1. Who builds `query`? (plan gap, F5; to `architect`/`teco`).
2. Should the `~/.claude/skills/` symlinks for both `coordination-graph` and `work-items` be made now? The skill is undiscoverable in Claude Code until then (human machine step).

### Proposed REVIEW.md lines
None.
