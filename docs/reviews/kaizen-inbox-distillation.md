# Kaizen inbox distillation — cross-agent review (scope: everything except `analyst`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Extended by:** `docs/reviews/kaizen-inbox-distillation2.md`

## Scope & verdict

Reviewed the uncommitted, in-progress `agent-maintenance` skill §5 learnings-inbox distillation
pass run by `cobb` (teco-coordinated) across the repo, **excluding** the `analyst` agent's own
files (reviewed separately, to avoid a self-review conflict). In scope, as specified:

1. `claude/architect/architect.md` + `kaizen/{history.md,plan.md,inbox.md}`
2. `claude/tdd-engineer/tdd-engineer.md` + `kaizen/{history.md,inbox.md}`
3. `claude/coder/coder.md` (+ its `kaizen/history.md`, read for cross-check)
4. `falkor-chat/docs/{DESIGN.md,BACKLOG.md,HISTORY.md}`, `falkor-chat/AGENTS.md`
5. `skills/python-web-quirks/SKILL.md` + catalog entries in `skills/README.md` and root `AGENTS.md`

Baseline: the live working tree (`git diff` against `HEAD`; nothing in this batch is committed
yet — `git status` shows all of it as modified/untracked). Every load-bearing citation below was
independently re-verified against the current repo state, not taken on the documents' own word;
the `_drive_loop` SHA command was actually executed. Where the review needed context from
`analyst`'s own kaizen files or `cobb`'s kaizen files to determine whether *this* scope's
bookkeeping is self-consistent, I read those files but did not review their content — only used
them as corroborating evidence for findings about the in-scope files.

**Verdict: needs changes.** One blocker: `architect`'s own kaizen bookkeeping (item 1) is stale —
it documents a "held pending follow-up" state for two inbox entries whose follow-up has, in fact,
already landed elsewhere in this same batch, and nothing recorded that. Everything else — the
prompt edits, the `falkor-chat` doc updates, the new skill's content, and the `_drive_loop` SHA
claim — is accurate, well-grounded, and cleanly bookkept.

## Findings

### Blocker — `architect`'s inbox/history were not updated to reflect a follow-up that already landed

`claude/architect/kaizen/inbox.md` carries two entries (2026-08-01, kiro-cli local-agent
discovery and `mcpServers`/`tools` facts), both annotated: *"Disposition decided 2026-08-09:
promote to `skills/agent-standards/kiro.md` ..., queued for consolidated follow-up ... — do not
clear until that lands."* `claude/architect/kaizen/history.md`'s matching 2026-08-09 entry
("Inbox distillation: 13 entries discarded") says the same: *"entries #1, #15, #16 handled
separately ... `kaizen/inbox.md` ... still carries #15/#16 pending a consolidated
`skills/agent-standards/kiro.md` follow-up."*

The brief asked me to check specifically whether this is still-pending-and-expected or
should-have-landed-and-hasn't. It has landed:

- `git diff -- skills/agent-standards/kiro.md` (part of this same uncommitted batch) shows both
  facts written in: the exact-CWD-only local-agent-discovery bullet (kiro.md:46–53) and the
  `mcpServers` key-presence-not-`type` discrimination fact (kiro.md:66–73), each stamped
  "verified 2026-08-01 against `kiro-cli 2.14.1`, re-verified 2026-08-09 against `2.16.2`." The
  file's header `Verified:` block (kiro.md:10–14) was also updated to describe the same edit.
- `claude/analyst/kaizen/history.md`'s own 2026-08-09 entry ("Held entry 28 promoted:
  consolidated Kiro-facts edit landed") independently confirms the same edit happened *today*,
  as a bundle of `analyst`'s held entry 28 (`resources: []` default) plus "two related facts from
  `architect`'s inbox" — i.e., precisely these two entries.

So the condition architect's own inbox explicitly gates clearing on ("do not clear until that
lands") is satisfied, but:
- `claude/architect/kaizen/inbox.md` still carries both entries, unmarked as cleared.
- `claude/architect/kaizen/history.md` has no entry logging the promotion (unlike `analyst`'s
  history, which does).
- `claude/cobb/kaizen/history.md` — the owning agent's history for `skills/agent-standards/*`
  per its own established convention (`agent-maintenance`/`agent-standards` changes are logged
  there; see `skills/README.md`'s Maintenance section and `cobb`'s own K-014 backlog note) — has
  a 2026-08-09 entry for the *other* half of the same day's work (the new `python-web-quirks`
  skill + four `claude-code.md` additions) but **no entry for the `kiro.md` edit at all**, on
  either the `analyst`-inbox side (analyst's own entry does the bookkeeping there) or the
  `architect`-inbox side.

Net effect: the three-way "batched edit, closed from both contributing inboxes" was only half
done. `analyst`'s side is closed out correctly (entry cleared, history entry added). `architect`'s
side is not — its inbox and history now describe a state ("still pending, do not clear") that is
false as of this same batch. A future `cobb` distillation pass, or an `architect` session that
happens to read its own stale inbox, would reasonably re-investigate or re-flag something that is
already done.

**Suggested fix:** `cobb` closes the other half of the same edit it already made in `analyst`'s
kaizen files — clear both entries from `claude/architect/kaizen/inbox.md`, add a matching
2026-08-09 history entry to `claude/architect/kaizen/history.md` (mirroring `analyst`'s "Held
entry 28 promoted" entry, cross-referencing it), and add the `kiro.md` edit to
`claude/cobb/kaizen/history.md` (it currently only has the `python-web-quirks`/`claude-code.md`
half of the day's work).

### Minor — `claude/README.md` catalog rows weren't updated for the new `python-web-quirks` routing clause

`architect`, `coder`, `tdd-engineer` (and `analyst`, out of scope) all gained a `python-web-quirks`
routing clause in their frontmatter `description` this batch. The established precedent for this
exact kind of change — wiring `cpg-analysis` into `architect`/`analyst`/`qa-engineer`'s
descriptions — updated the matching rows in `claude/README.md` to name the skill (e.g.
`architect`'s row: *"For call-graph impact analysis ... uses the `cpg-analysis` skill..."*,
`claude/README.md:9`). `claude/AGENTS.md`'s own maintenance rule says a description/behavior
change updates "the full catalog entry in `README.md` ... in the same change." None of
`architect`'s (`claude/README.md:9`), `coder`'s (`:10`), or `tdd-engineer`'s (`:13`) rows mention
`python-web-quirks` — checked directly, not inferred. This isn't misleading (the frontmatter
`description`, which is what's actually auto-injected, is correct and complete), but it's an
inconsistency with the repo's own convention and its own recent precedent, and it leaves the
human-facing catalog silently out of date.

**Suggested fix:** add a short clause to the three rows, mirroring the `cpg-analysis` pattern —
e.g. for `architect`: *"In a Python web/async codebase, uses the
[`python-web-quirks`](../skills/python-web-quirks/SKILL.md) skill for asyncio/FastAPI/Starlette/
pydantic gotchas."*

### Nit — `falkor-chat/AGENTS.md`'s citation bundles two symbols under one line number

The new "Probing shared graph state without mutating it" subsection cites *"`get_snapshot`/
`_read_subgraph` (`repository.py:1031`)"* — `_read_subgraph` is indeed defined at line 1031, but
`get_snapshot` is a different method, defined at `repository.py:1702` (it calls into
`_read_subgraph`, doesn't sit at the same line). Not misleading in substance — `_read_subgraph`
is genuinely the shared implementation both `get_snapshot` and def-read go through — but the
citation reads as if both symbols live at :1031. Minor precision issue; not worth blocking on.

## What's solid

- **Every re-verifiable claim in this batch checked out.** The `_drive_loop` SHA-reproduction
  command in `falkor-chat/docs/DESIGN.md` (§6.2) was actually run against
  `falkor-chat/server/falkorchat/executor.py` on this commit and reproduced `71055f756280`
  exactly, matching the doc. All `repository.py`/`db.py` line-number citations in the new
  "Probing shared graph state" subsection (`_reference()` :156-158, `reference_graph()` :87-94,
  `_PUBLISH_CYPHER` :992, `_READ_META_CYPHER` :1016, `_read_subgraph` :1031,
  `materialize_snapshot` :1669) matched exactly on `grep -n`. `scripts/bootstrap_schema.sh`'s
  cited `bootstrap_reference` span (37–70, unconditional call at :248, DDL-only body) matched
  exactly. `pytest --collect-only -q` was run live in `falkor-chat/server/` and confirmed
  non-mutating and DB-connection-free (696/697 collected, 1 deselected, no FalkorDB error) —
  exactly the claim `DESIGN.md` §14.7 makes. `test_services.py`'s `FakeRepo`-only, no-live-`conn`
  claim was confirmed by reading the file and `conftest.py`'s fixtures.
- **`skills/python-web-quirks/SKILL.md`'s three technical claims all reproduced.** Ran the exact
  installed-version check (`pydantic 2.13.4`, `fastapi 0.139.0`, `starlette 1.3.1` — matching the
  skill's cited versions exactly) and then reproduced each claim directly: the nested-model
  `exclude_unset` field drop (`Inner.b` vanished from the dump exactly as documented);
  `BackgroundTask.__call__`'s route through `run_in_threadpool` → `anyio.to_thread.run_sync`
  (confirmed via `inspect.getsource`); and the default `anyio` thread limiter's cap (confirmed
  `40` via a live `CapacityLimiter` inspection). The asyncio GC-safety entry is appropriately
  hedged as "real per the docs, didn't reproduce under stress" rather than overclaimed either way.
- **`tdd-engineer`'s five prompt merges are complete, well-placed, and match the history's
  description exactly** — checked each of the five against the actual prompt text (symmetric
  teardown, marker-gated optional tests, expanded "Cover the edges," shared-extractor tolerance
  contract, rewritten step-5 verification); `kaizen/inbox.md` is genuinely empty, not just
  claimed-empty.
- **`architect`'s "13 entries discarded" batch is unusually well re-verified**, not just asserted
  — spot-checked several of its supporting citations (K-034 closure in `falkor-chat/docs/
  HISTORY.md`, the K-031 grouping-key callout in `QUERIES.md:925-941`, the `EXPLAIN`/`PROFILE`
  behavior in `skills/cpg-analysis/SKILL.md:65-75`, `teco.md:67`'s write-guard-scope statement)
  and all matched. The one line-number citation that's now off (`HISTORY.md:60` for K-034, which
  moved to `:77`) drifted only because this *same* batch prepended a new entry to that
  ever-growing log — not a distillation error, just the known fragility of citing line numbers in
  a prepend-style change log (the kind of thing the batch's own new Guardrails bullet, about
  verifying what a hook pattern-matches rather than trusting a description, is adjacent in
  spirit to).
- **`claude/scripts/audit-team.sh` passes clean** against the edited state — boundary-pair
  symmetry, catalog presence, hook wiring, and the no-personal-identifiers scan are all green
  with these edits in place.
- **The new Guardrails bullet on `architect.md`** (verify what a `PreToolUse` hook actually
  pattern-matches, not the intent behind the prompt) is well-placed among the existing Guardrails
  bullets, doesn't duplicate anything, and is traceable to a real, dated incident (the C-311
  `pipeline.sh --reset` bypass) rather than a hypothetical.

## Open questions

- Should `cobb` close the architect-side half of the "consolidated Kiro-facts" edit as a small
  standalone follow-up, or fold it into whatever unit picks up this review's blocker? Either
  works; flagging only because the fix touches three files across two agents' kaizen dirs
  (`architect`'s inbox + history, `cobb`'s own history) and is easy to under-scope if treated as
  "just clear the inbox."
- `claude/architect/kaizen/plan.md`'s parking-lot item on the "live-probe seam check" (parked,
  not promoted, per the 2026-08-09 batch entry) reads as a reasonable call given it's
  single-occurrence — no objection, just noting it wasn't re-litigated here since the brief didn't
  ask for a judgment call on that disposition specifically.
