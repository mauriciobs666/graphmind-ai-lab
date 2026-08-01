# Kaizen — Learnings Inbox: teco

> Append-only capture of durable, non-obvious environment facts the `teco` agent
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

## 2026-07-29 — Resuming a background subagent after a platform error means a fresh `Agent` call with a self-contained state-recovery brief, not a true session continuation
- **Evidence:** two consecutive background `Agent` calls for the same frontend unit (K-036 Wave 3+4, graphmind-ai-lab) failed with `API Error: 500 Internal server error` mid-run. Task-notification text references resuming via "SendMessage," but no `SendMessage` tool was present in this session's tool list — only `Agent` (which starts a fresh context). Recovery worked by launching a new `Agent` call whose prompt described exactly what the failed agent's own `git status`/`git diff` would show (files already written, what was in progress) and instructed it to verify that state itself before continuing — not by literally resuming the old session.
- **Context:** falkor-chat K-036 web-api-coverage Wave 3+4 delivery, two transient 500s in a row before the third attempt succeeded.
- **Suggested home:** prompt (teco.md's Guardrails/Delegate-with-complete-briefs section) — note explicitly that "continue this agent via SendMessage" in task-notification text is not always backed by a tool actually available to teco; the fallback is a fresh `Agent` call whose brief tells the agent to inspect on-disk/git state and resume from there.

## 2026-07-29 — This dev box has no `node` on `PATH`; subagents needed a workaround to run bare-`node` JS tests
- **Evidence:** `falkor-chat/`'s web test convention (its own architect plan, `docs/plans/web-api-coverage.md` §5.2) requires bare-`node` unit tests for dependency-free JS logic. Two independent subagent sessions (frontend-engineer, then analyst re-reviewing) both had to discover a workaround — one used a Playwright-bundled Node binary, the other found `node.exe` reachable from WSL — before they could run `node web/tests/run-select.test.js`. Neither workaround was mentioned in `falkor-chat/AGENTS.md`.
- **Context:** K-036 Wave 3+4 delivery and its Gate 2 review, graphmind-ai-lab/falkor-chat.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`'s "Key scripts"/environment section) — worth a one-line pointer to a working `node` invocation for this box, so future frontend/analyst briefs in this repo don't re-spend turns rediscovering it.

## 2026-07-31 — A haiku-model doc-close-out delegate can fabricate plausible-sounding implementation details even when told to verify against source
- **Evidence:** delegated a "routine doc-only touch-up" (HISTORY.md entry + BACKLOG.md close-out for falkor-chat K-039 item 3) to a `coder` agent with `model: "haiku"`, brief explicitly said "cite real numbers/paths you verified, don't invent anything" and pointed at the exact source file/line to check. The drafted HISTORY.md entry stated `postSuccess.status` is `"ok"` when rate ≥ 70%, `"degraded"` when 0 < rate < 70% — a specific, confident-sounding threshold that does not exist anywhere in the codebase; the actual logic (`server/falkorchat/services.py:1101-1103`) is `"ok"` iff *every* sampled run posted, `"degraded"` otherwise, no percentage threshold at all. A second smaller error in the same entry misstated the test-count breakdown (5/3/3 vs. the real 6 repository + 5 service + 2 extended-not-added API tests). Both were only caught because teco independently grepped the actual service code before accepting the draft, per the "verify by reading, never accept 'docs updated' as a claim" rule.
- **Context:** graphmind-ai-lab, falkor-chat K-039 item 3 close-out — the doc-close-out unit was the very last step of an otherwise fully double-gated (two analyst reviews) multi-unit build; nothing upstream would have caught this since neither review pass touched HISTORY.md (it didn't exist yet at review time).
- **Suggested home:** prompt — tighten the "For a unit with no design/code-quality stakes... pass model: haiku" routing guidance to note that a haiku-routed doc-close-out summarizing *numeric/logic* details (thresholds, branch conditions, counts) still needs the same independent-verification pass as any other delegate's claim — the cost-cutting is about the model tier, not a license to skip integration-time verification, which should already be assumed but drift happened here.
