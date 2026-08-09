# `analyst` learnings-inbox distillation (2026-08-09) — Review

> **Status:** active · **Owner:** `cobb` · **Tracks:** K-014 (cobb kaizen)

## Scope & verdict

Independent second-opinion review of a same-day `agent-maintenance` skill §5 distillation of the
`analyst` agent's learnings inbox, routed to a fresh `cobb` session because the artifact under
review is `analyst`'s own prompt (a self-review conflict for `analyst`). Reviewed against the live
working tree (all changes are currently uncommitted): `claude/analyst/analyst.md`,
`claude/analyst/review-techniques.md` (new), `claude/analyst/kaizen/{history.md,inbox.md}`,
`claude/graph-dba/falkordb-quirks.md`, `skills/agent-standards/claude-code.md`, and
`claude/cobb/kaizen/{history.md,plan.md}`. Falkor-chat-side promotions
(`falkor-chat/AGENTS.md`/`docs/DESIGN.md`/`docs/HISTORY.md`) and the new
`skills/python-web-quirks/` skill were spot-checked as corroborating evidence since they're cited
throughout the `analyst` history entry. Explicitly out of scope, per the brief: the pending
independent safety recheck on review-technique (b) in `review-techniques.md` (a separate session's
job), and the concurrent, unrelated distillations visible in the working tree for `architect` and
`tdd-engineer` (their own inbox passes, not this one — I only checked that the shared
`python-web-quirks` routing clause landed identically in all four consumer agents).

**Verdict: approve with suggestions.** No blockers, no majors. Every technical claim I attempted to
independently reproduce did reproduce, several exactly to the byte. The scope discipline on
`analyst.md` matches what was pre-approved with no bloat. Two minor, non-blocking polish notes
below.

## Independent verification performed

I did not take any citation on faith — for each verifiable claim I re-derived it against the live
system:

| Claim | Method | Result |
|---|---|---|
| `shellcheck` not installed | `command -v shellcheck` | Confirmed (exit 1, not found) |
| Bash tool shadows `find`→`bfs`, `grep`→`ugrep`; not inherited by a subprocess | `type find`, `type grep`, then `bash -c 'type find; type grep'` | Confirmed exactly, including the exact `ARGV0=bfs -S dfs -regextype findutils-default` / `ARGV0=ugrep ... -G --ignore-files --hidden -I ...` flag strings quoted in the doc, and the subprocess-doesn't-inherit claim (`bash -c` shows real `/usr/bin/find`/`/usr/bin/grep`) |
| FastMCP emits `outputSchema` + duplicates payload as `structuredContent` for a plain `-> str` tool; `structured_output=False` suppresses it | Installed `mcp` 1.28.1 in `cpg/mcp/.venv`; wrote and ran a throwaway `FastMCP` tool, called `list_tools()`/`call_tool()` | Confirmed exactly — `outputSchema` present, `call_tool` returns `(TextContent, {'result': 'hello'})` duplicating the payload; `structured_output=False` → `outputSchema: None`. Version (1.28.1) and the `structured_output: bool \| None = None` parameter both matched the doc's citation verbatim |
| `sum(CASE WHEN … THEN 1 ELSE 0 END)` returns `0`, not `NULL`, on zero-row aggregation | Live `GRAPH.QUERY` against the running `falkordb-dev` container (v4.18.11, matching the doc's cited version) for both zero-row and non-empty input | Confirmed the NULL-vs-zero behavior. Could not independently confirm the float-vs-int *type* distinction via `redis-cli`'s text protocol (it doesn't surface RESP type at that layer) — the doc cites `falkordb-py`-level verification for that part, which I didn't re-run; noted as an unverified-by-me detail, not a doubted one |
| `GRAPH.PROFILE` executes writes for real (not read-only) | `GRAPH.PROFILE` a `CREATE` statement live, then queried for the created node | Confirmed — the node existed after profiling |
| `EXPLAIN`/`PROFILE` prefix inside a `GRAPH.QUERY` string is silently ignored | Ran `GRAPH.QUERY ... "EXPLAIN RETURN 1"` live | Confirmed — returned `1`, no plan, no error |
| `audit-team.sh` check 7 (and the whole script) now clean | Ran `claude/scripts/audit-team.sh` | Full `PASS`, including check 7 and the boundary-reciprocity checks |
| Stale "joern agent" string gone from `cpg/mcp/server.py` | `grep -rn "joern agent" cpg/mcp/` | No hits; live string reads "the graph-dba agent's job" |
| C-311 `pipeline.sh --reset` branch exists in the destructive-ops guard | Read `claude/scripts/guard-destructive-ops.sh` | Present, dated 2026-08-08, matches the entry-15 disposition |
| `response_model_exclude_unset` drops defaulted fields on **nested** models | Ran the doc's exact repro against `falkor-chat/server/.venv` (pydantic 2.13.4, FastAPI 0.139.0 — versions matched the citation) | Confirmed exactly — `b` (nested default) missing from `model_dump(exclude_unset=True)` |
| `falkor-chat/AGENTS.md`'s re-verified line numbers (`repository.py:156-158/992/1016/1031/1669`, `db.py:87-94`) | Read each cited line directly | All six citations landed on exactly the described construct |
| `DESIGN.md`'s new line-number-independent SHA-lock extraction command reproduces `71055f756280` | Ran the exact `awk`/`sed`/`sha256sum` pipeline quoted in the doc | Reproduced `71055f756280` verbatim |
| `analyst.md` scope matches the approved shape (1 new Guardrails bullet + clause extensions to one sentence + 2 routing/pointer additions, nothing more) | Full `git diff claude/analyst/analyst.md` | Confirmed — exactly that and nothing else |
| Inbox bookkeeping: 30/31 cleared, entry 28 held with the stated note | Read current `inbox.md` | Matches; entry 28 present with "queued for consolidated follow-up," disposition note dated 2026-08-09; entry-count arithmetic in `history.md` (A–H) is internally consistent — every ID 1–31 is accounted for exactly once (entry 7 legitimately appears in both B and F, explained as a misfiled line vs. its real topic, not a duplicate-count error) |
| `python-web-quirks` skill wired into `coder`/`tdd-engineer`/`architect`/`analyst` frontmatter, cataloged in `skills/README.md` and root `AGENTS.md` | `git diff` on all four agent files + both catalogs | All four descriptions carry an identical routing clause; both catalog entries present and accurate |
| `claude-code.md`'s new "Bash tool environment" section placement | `grep -n "^##\|^###"` over the whole file | Correctly a new top-level section (doesn't fit Subagents/Skills/Memory/Hooks/MCP), placed between Hooks and MCP |
| `cobb`'s own kaizen history/plan honesty | Read `claude/cobb/kaizen/{history.md,plan.md}` against the actual diffs | Accurate — the K-014 update correctly states no `python-web-quirks/kaizen/` dir exists (confirmed: `find skills/python-web-quirks -type f` returns only `SKILL.md`), and the "by-example precedent, not a written rule" framing is honest about the gap it's still tracking |

## Findings

No blockers. No majors.

**Minor — "Evidence over vibes" guardrail bullet is now dense (6 sentences, 5 distinct sub-rules)
after the four clause extensions.** `claude/analyst/analyst.md`, Guardrails section. This was an
explicit stakeholder call (history.md disposition G: "per stakeholder instruction to avoid four new
standalone bullets"), so I'm not relitigating the scope decision — but as a straight
prompt-quality-lint (§7, cognitive-load dimension) observation: a reader scanning the Guardrails
section in one pass now has to parse one long run-on for five separable rules (base evidence rule +
git-grep-as-bound + regex-claim-run-it + guard-glob-crosscheck + no-shellcheck-use-bash-n). A
lightweight sub-list under the same bullet (still one bullet, not four) would preserve the
stakeholder's bullet-count constraint while restoring scannability:
```
- **Evidence over vibes.** Distinguish what you verified from what you infer. Never report a
  suite as green without running it; never claim a bug you didn't trace to a concrete path.
  Also:
  - a `git grep`/`git ls-files` count is a bound, not a fact, when the artifact under review …
  - a suggested regex/glob/pattern fix is a claim — run it before writing it into a review.
  - cross-check a named agent's `PreToolUse` guard globs when a plan assigns it doc-write ownership.
  - `shellcheck` isn't installed here — verify with `bash -n` + live execution, not static analysis.
```
Not blocking; take or leave.

**Minor — the new "Bash tool environment" section isn't reflected in `claude-code.md`'s top-of-file
`Verified:` stamp block.** `skills/agent-standards/claude-code.md:1-15`. Every other section
(Subagents, MCP, `model` field) gets a dated stamp line at the top pointing readers to its
verification status; the new section instead carries its provenance only inline
("Observed graphmind-ai-lab, 2026-07-26 and 2026-08-08"). That's the correct *kind* of citation for
an internally-observed harness quirk (there's no official doc page to verify it against), so this
is purely a discoverability nit — a reader skimming just the header stamp block wouldn't know the
new section exists. A one-line addition ("Bash tool environment: observed 2026-07-26/2026-08-08,
not doc-sourced") would close the gap. Not blocking.

## What's solid

- **Every reproducible technical claim reproduced, several byte-for-byte** — the FastMCP
  `outputSchema`/`structured_output` behavior, the pydantic nested-`exclude_unset` drop, the
  `GRAPH.PROFILE`-executes-writes and `EXPLAIN`-prefix-ignored behaviors, the shell-shadowing
  `ARGV0` strings, and the SHA-lock re-extraction command all matched on live re-run, not just on
  re-reading the citation. This is a genuinely high bar of evidence discipline for a distillation
  pass, not just plausible-sounding prose.
- **Line-number re-verification was real, not copied.** The `falkor-chat/AGENTS.md` promotion
  explicitly states original inbox line numbers had drifted and were corrected against current
  `HEAD` — I checked all six citations independently and every one landed exactly on the described
  function/constant.
- **Scope discipline on `analyst.md` matches the pre-approved shape exactly**: one new Guardrails
  bullet, clause-level extensions to exactly one existing sentence, two routing/pointer additions
  (the `python-web-quirks` description clause, the `review-techniques.md` step-3 pointer) — nothing
  else touched. No scope creep.
- **Bookkeeping is honest and internally consistent.** The inbox went from 31 entries to exactly 1
  held entry with a clear reason; the history entry's A–H disposition groups account for every
  entry ID 1–31 exactly once; `cobb`'s own kaizen log candidly notes what it did and did *not*
  close (K-014 remains open, correctly).
- **The held entry (28) was left alone, correctly.** It wasn't cleared prematurely to make the pass
  look more complete than it is — the "queued for consolidated follow-up" note names the reason
  (avoiding a race with `architect`'s inbox on the same shared file) and is still present, unmodified,
  in the current working tree.
- **`review-techniques.md`'s pending-safety-recheck marker on technique (b) is preserved intact**
  and not touched or re-litigated by this pass, correctly deferring to the separate session
  handling that check — I sanity-checked only the file's structure (consistent headers,
  self-contained sections, technique 1's AST byte-range approach and technique 2(a)'s PEP
  660-editable-install premise are both technically sound) without re-judging (b)'s safety.
- **`python-web-quirks` is a well-scoped new skill** with version-pinned, independently-reproducible
  entries and correct `allowed-tools` (Read/WebFetch/WebSearch only — no write access needed for a
  reference skill), wired consistently into all four consumer agents' `description` fields.

## Open questions

- None that block approval. The two minor items above are take-or-leave polish; neither needs the
  user's or stakeholder's input to resolve.
