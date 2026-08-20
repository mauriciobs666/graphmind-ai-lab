# Kaizen — Learnings Inbox: analyst

> **FROZEN — 2026-08-20.** This file is a historical snapshot only. Its 5 entries (as of this
> date) were imported into the `kaizen_analyst` FalkorDB graph (`claude/cobb/kaizen/history.md`,
> 2026-08-20 entry); `analyst` no longer appends here. New raw learnings are written directly into
> the graph and are immediately queryable by any agent: `mcp__cypher__query(graph='kaizen_analyst',
> cypher='MATCH (e:KaizenEntry) RETURN e.date, e.fact, e.evidence, e.context, e.suggestedHome,
> e.author ORDER BY e.date')`. Content below is preserved for historical reference and will not
> change.

> Append-only capture of durable, non-obvious environment facts the `analyst` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-15 — No `SendMessage` tool means a nested review-gate result can't reach its actual delegator, only whichever ancestor session happens to be live
- **Evidence:** dispatched as `teco`'s delegate (via `cobb` → `teco` → `analyst`) for K-026's Pass 2 re-gate. On completing the review I stated plainly: "I don't have a `SendMessage` tool in this session, so I can't message `teco` directly — reporting the outcome here instead." The result then routed to `cobb` (two levels up, the only actively-listening session) rather than to `teco` (my direct delegator, one level up but dormant at that moment — its own turn had ended right after dispatching me), requiring a manual relay + `SendMessage`-triggered resume from `cobb` before `teco` even knew Pass 2 was done.
- **Context:** K-026 GraphRAG eval harness coordination, `analyst` Pass 2 re-gate of `falkor-chat/docs/plans/graphrag-eval.md`.
- **Suggested home:** unsure, flagged for a proper evaluation rather than a rushed grant — logged at the stakeholder's direction after they proposed giving `SendMessage` broadly "for efficiency." Two things would need confirming first: (1) whether `SendMessage` actually force-resumes a dormant intermediate coordinator the way it visibly did for `teco` when `cobb` used it (`"Resuming agent ..."` in the tool result), which is what would make this actually save the relay rather than just move where the address book lives; (2) `teco`'s own brief template doesn't currently hand delegates its own `agentId` — granting the tool without an address to send to wouldn't help. See the paired entry in `claude/teco/kaizen/inbox.md`. Candidate scope if it pans out: the review-gate reporters specifically (`analyst`, `qa-engineer`, `data-scientist`), not a blanket grant to every agent.

## 2026-08-11 — Reviewing a kaizen-inbox distillation: count removed `## ` entries in each inbox diff and reconcile against the history entry's own claimed count — that arithmetic is where silent drops surface

- **Evidence:** reviewing `cobb`'s 39-file 2026-08-11 distillation, `git diff claude/coder/kaizen/inbox.md | grep -c '^-## '` → 8 while `coder/kaizen/history.md`'s new entry claimed "8 entries routed (6 to …, 1 discarded)" — 6+1≠8, and mapping each removed entry to a stated disposition found two with none at all (both `Suggested home: prompt`, one of them a still-real gap: the skip-count rule exists at `tdd-engineer.md:42` but not at `coder.md:22`). The same reconciliation caught four more unlogged dispositions and four wrong header counts across `graph-dba` (7 removed vs. "5"), `devops` (13 vs. "9"), `tico` (4 vs. "3"), `qa-engineer` (sub-counts don't close). Second trap it exposes: a history entry can list promotions that came from a *different* agent's inbox (coder's listed analyst's/architect's urllib + LM Studio entries), which makes an incomplete list read as complete — so map entry→disposition from the **diff**, never from the history's prose. Third: an inbox entry can be *headless* (a stray `- **Evidence:**` with no `## ` heading, as in `teco`'s 458k one), so a pure heading count under-counts by one there.
- **Context:** diff-scoped review of a full-team kaizen distillation (`docs/reviews/kaizen-distillation-2026-08.md`); this check produced the review's only blocker, and nothing else in the diff would have revealed it.
- **Suggested home:** knowledge base (`claude/analyst/review-techniques.md`) — a short "auditing a kaizen distillation" technique: per inbox, `grep -c '^-## '` the diff, enumerate each removed entry's disposition from the diff text, reconcile against the history header, and check each claimed promotion's *source* inbox.

## 2026-08-11 — Agent prompts deploy by symlink into the working tree, so an *uncommitted* prompt edit under review is already live for the running team — including the reviewing agent itself

- **Evidence:** `ls -la ~/.claude/agents` shows one symlink per agent into `/home/<user>/prg/graphmind-ai-lab/claude/<name>` (and `~/.claude/skills → <repo>/skills`). Reviewing an uncommitted diff that edited `claude/analyst/analyst.md`, the new clause was already present verbatim in this review run's own system prompt. The symlink layout itself is documented (`claude/README.md:61-67`, `claude/AGENTS.md:34`); the review-relevant corollary is not stated anywhere I found.
- **Context:** diff-scoped review of a working-tree-only change to six agent prompts and three new knowledge bases.
- **Suggested home:** prompt or knowledge base (`claude/analyst/review-techniques.md`) — when the artifact under review is an agent prompt/skill under `claude/` or `skills/`, "uncommitted" does not mean "not yet in effect": findings ship immediately rather than at commit, which raises the urgency of a blocker and makes "restore it now, it's still recoverable via `git diff`" a real, time-boxed remedy.

## 2026-08-16 — Verifying a "copied verbatim" text-block claim (a caveat, a prompt, a spec quote) needs a programmatic whitespace-normalized diff, not a read-through — a markdown soft line-break renders as a space in the source but silently vanishes when hand-transcribed into a multi-line Python string-literal concatenation

- **Evidence:** reviewing `falkor-chat/server/tests/eval/generate_report.py`'s `_SAME_MODEL_CAVEAT_TEMPLATE` (claimed "emitted VERBATIM" against a `data-scientist` review's block-quoted recommended caveat text, `docs/reviews/graphrag-eval-ml.md` M-1). A close read-through found no difference. Extracting the Python string literal, `.format()`-rendering it, and diffing it whitespace-normalized against the recommendation surfaced one real, otherwise-invisible discrepancy: the recommendation's markdown source wraps mid-sentence as `"...borderline/\nsubjective calls..."` — a soft line break, which every standard renderer collapses to a single space — while the implementer's Python string concatenation joined the same two lines with no separator at all, producing the literal run-together word `"borderline/subjective"` in the actual generated output. Both read identically on a human eyeball pass (the wrapped source *looks* like it has a space there); only the programmatic diff caught the drop.
- **Context:** K-026 GraphRAG eval harness, Unit 3 code review (`falkor-chat/docs/reviews/graphrag-eval.md`, "Unit 3 code review (judge layer)" section, finding N-2) — confirming a caveat-language conformance requirement the coordinating session explicitly asked to be checked.
- **Suggested home:** knowledge base (`claude/analyst/review-techniques.md`) — a short technique note: when a finding/spec asks you to confirm text was reproduced "verbatim," extract both strings into variables and diff them with whitespace normalized (`re.sub(r'\s+', ' ', s).strip()`) rather than reading them side by side — a line-wrap point in the *source* of the "expected" text is exactly where a hand-transcription is most likely to silently drop a character, and it's the one place a visual read is least likely to notice it (both versions look right individually).

## 2026-08-19 — When verifying a plan's cited "regression floor" pytest baseline, `pytest -k "not live"` and the project's actual `pytest.ini` `addopts = -m "not live"` silently give different pass/deselect splits for the same suite

- **Evidence:** `docs/plans/cpg-mcp-rename.md` cited `cypher-mcp`'s offline baseline as "84 passed, 7
  deselected." Running `.venv/bin/python -m pytest tests/ -k "not live" -q` from `cypher-mcp/`
  produced `83 passed, 8 deselected` — a one-test discrepancy that looked like a stale/wrong plan
  claim. Re-running with no `-k` override (letting `pytest.ini`'s own `addopts = -m "not live"`
  apply) produced `84 passed, 7 deselected`, matching the plan exactly. Root cause: `-k` is a
  substring/keyword filter over test *names*, not the marker-based `-m` deselection `pytest.ini`
  actually documents (`cypher-mcp/pytest.ini`: "`-m \"not live\"` DESELECTS the FalkorDB-dependent
  tests by default... a reachability-skip would not"); one test apparently has "live" somewhere in
  its collected name/id without carrying the `live` marker, so `-k` and `-m` disagree on it by one.
- **Context:** `docs/reviews/cpg-mcp-rename.md`, verifying the plan's §2/§5 offline-suite baseline
  claim before trusting it as a regression floor.
- **Suggested home:** knowledge base (`claude/analyst/review-techniques.md`) — when verifying a
  plan's cited test-count baseline, run the project's own default/documented invocation (check its
  `pytest.ini`/`pyproject.toml` `addopts` first) rather than a hand-written `-k` filter that looks
  equivalent — the two are different filtering mechanisms and can silently disagree on the split
  even when the total is identical.

