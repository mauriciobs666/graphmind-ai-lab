# Anthropic API adoption across modules — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-11-03

## Intent
The stakeholder has received Anthropic API credits and wants to experiment with them. They want the Anthropic API usable from **every module of the repository that calls an LLM**, configured through the default-and-fallback mechanism `falkor-chat` already has, and used as well as possible.

## Problem & current state
- `falkor-chat` resolves models through two hand-edited files and a gateway (`falkor-chat/docs/SERVER.md` §1.8). That gives per-kind defaults, ordered fallback "roles", per-model settings and a per-workspace hard cap.
- The gateway speaks only the OpenAI-compatible wire format. A model reference naming any other protocol (e.g. `anthropic`) is refused at load (`falkor-chat/server/falkorchat/modelconfig.py`, `SUPPORTED_PROTOCOLS`).
- The shipped defaults are all local LM Studio models (qwen3-4b for chat kinds, Qwen3 embedding), constrained by a 6 GB GPU (`docs/requirements/standard-local-model.md`, archived).
- Other modules have no shared model configuration: `model-bench` targets LM Studio only (hosted models were declined for now, "the door stays open": `docs/requirements/small-model-benchmarking.md`), `mcp-monitor` launches an agent CLI, `opencode/agents/tank` has its own `opencode.json`, `decision-models` planned a hosted labeler via `ANTHROPIC_API_KEY` (`docs/plans/typed-decision-models.md`).
- The `extraction` kind is outside what a workspace override can reach (K-080), so an Anthropic-switched workspace still extracts locally during ingestion.
- Prior decisions this touches: `falkor-chat/docs/requirements/llm-provider-config.md` FR-13 (archived) already named Anthropic as a required hosted provider kind; `docs/requirements/typed-decision-models.md` decision log 2026-10-31 chose Claude Code subagents over the API because no key existed.

## User stories
- As the **lab owner**, I want Anthropic models scored in model-bench next to the local ones on four roles and on all three decision points, so that I can compare them and decide where they are worth using.
- As the **lab owner**, I want to switch a chosen falkor-chat workspace (including the storefront demo) to Anthropic without changing anything else, so that I can try it side by side while everything else stays local.
- As the **lab owner**, I want decision-model data generated and labelled through the Anthropic API when a key exists, so that the pending dataset work can proceed without waiting on subagent usage windows.
- As the **lab owner**, I want to see which Anthropic model answered and roughly how many tokens it used, so that I can follow my credits.
- As the **lab owner**, I want the API key kept safe, so that my credits cannot leak.

## Functional requirements
- **FR-1** — Defaults stay local everywhere: with no Anthropic configuration, every module behaves as it does today and spends nothing.
- **FR-2** — falkor-chat: a chosen workspace can be switched to Anthropic for its agent, workflow steps and guard judge (an observable switch, not a code change); the existing per-workspace override is the expected carrier. Ingestion extraction stays local. The storefront demo's workspace is switchable the same way; its pinned local model stays the default.
- **FR-3** — In an Anthropic-switched workspace, a temporarily failed Anthropic call (credits exhausted, API down, rate limit) falls back to the local model, and the step record shows which model answered and that a fallback occurred. A missing, invalid or revoked key does not fall back: the step fails with a clear message.
- **FR-4** — Embeddings stay local (Anthropic offers no embeddings model). Checked by AC-11.
- **FR-5** — model-bench can score an Anthropic model on the chat-responder, tool-caller, nlq-generator and guard-judge roles, stored and compared like any other run. The embedder role is excluded.
- **FR-6** — model-bench can score Anthropic on all three decision points (guard, message type, join-in).
- **FR-7** — decision-models generates and labels data through the Anthropic API when a key is available, with the generator/labeller model split left to design (the labeller stays independent of the generator). Claude Code subagents remain the path when no key exists.
- **FR-8** — `tank` is unchanged and stays on its local model (excluded; see Out of scope).
- **FR-9** — `mcp-monitor` needs no change. Satisfied by AC-1.
- **FR-10** — Each Anthropic result (workflow step, benchmark run, generation/labelling call) records the answering model, approximate token usage and a hosted flag, and nothing else about the call: raw request headers, credentials in URLs, provider error bodies and the Anthropic organization or workspace id are never persisted. No spend cap is required of the software.
- **FR-11** — The API key is stored and handled safely: it is never in the repository, config files, results, logs, graph records or test fixtures (fixtures use obviously fake keys); never passed on a command line, in a prompt or message body, or into the context of any LLM-driven agent or tool; held only by a closed list of components; can be rotated or revoked without editing tracked files; and a missing key fails closed (no silent switch to another provider, except FR-7's explicit subagent path). Error text from a failed call must not echo credentials. The mechanism is for design, with a security review.
- **FR-12** — Where an existing guarantee depends on a local model (model-bench's environment fingerprint, "compared only within a role", reproducible runs), a hosted model's results are marked as hosted so they are never read as locally reproducible. A hosted result names the exact pinned model version, not a moving alias.
- **FR-13** — In a storefront workspace switched to Anthropic, a full multi-turn shopping conversation with tool calls completes, as it does on the local model.
- **FR-14** — Unattended Anthropic-backed runs (generation and labelling campaigns, benchmark sweeps) stop at a token bound set before the run starts, because no spending limit is set on the Anthropic side. The bound limits one run; it is not a budget across runs. Interactive and workflow use in a switched workspace is deliberately unbounded.
- **FR-15** — Only synthetic, demo or benchmark content is sent to Anthropic. The owner flags a workspace as private; a flagged workspace cannot be switched to it. A workspace with no flag can be switched, so existing workspaces are switchable until flagged. Benchmark packs and decision-model campaigns use synthetic content by construction.

## Out of scope
- Anthropic embeddings; any spend cap or budget enforcement.
- Changing any module's local default, or the 2026-10-31 choice of qwen as the standard local chat model.
- Any change to `mcp-monitor`.
- A shared configuration file for all modules: each module is configured by its own existing means.
- Hosted providers other than Anthropic.
- `tank`: it can run shell commands and read files, which cannot be reconciled with keeping the key out of any agent's reach (decision 2026-11-03).

## Acceptance criteria
- **AC-1** — Given no Anthropic configuration, when any module runs, then no call reaches Anthropic and behaviour is unchanged (FR-1).
- **AC-2** — Given a workspace switched to Anthropic, when it runs a workflow step, then the step record shows an Anthropic model as the answerer (FR-2, FR-10).
- **AC-3** — Given that workspace and an Anthropic failure, when a step runs, then the local model answers and the record shows the fallback (FR-3).
- **AC-4** — Given an Anthropic model, when model-bench runs each of the four roles, then each run is stored and appears in a comparison with local models, marked hosted (FR-5, FR-12). The guard-judge role and the guard decision point are separate runs; neither counts as the other.
- **AC-5** — Given an Anthropic model, when model-bench runs the guard, message-type and join-in decision points, then each result is stored the same way (FR-6).
- **AC-6** — Given a key and the generation campaign, when it runs, then data is generated and labelled via the API; given no key, it still runs via subagents (FR-7).
- **AC-7** — Given a run with a canary key, when the working tree (including untracked files), git history, stored results, campaign ledgers and transcripts, server and process logs, graph step records, error and traceback text, test fixtures, and subagent and child-process environments are searched for it, then it appears nowhere; and the same search finds it when deliberately planted in each place. The canary also runs through a successful call, using a local stand-in for the Anthropic endpoint, since a fake key never authenticates against the real one. The search excludes the closed list of components allowed to hold the key (FR-11) (FR-10, FR-11).
- **AC-8** — Given the storefront workspace switched to Anthropic, when a scripted multi-turn shopping conversation that needs tool calls (the script is chosen in design) is run, then it completes and the step records show an Anthropic model (FR-13).
- **AC-9** — Given an unattended Anthropic-backed campaign, when it runs, then it stops at a token bound that was set before it started (FR-14).
- **AC-11** — Given a workspace switched to Anthropic, when embeddings are written or searched, then they use the local embedding model (FR-4).
- **AC-12** — Given a switched workspace and a missing, invalid or revoked key, when a step runs, then it fails with a clear message and the local model does not answer (FR-3, FR-11).
- **AC-10** — Given a workspace flagged as private, when it is switched to Anthropic, then the switch is refused with a clear message; given an unflagged workspace, the switch is allowed (FR-15).

## Open questions
None. The exact Anthropic models (one small, one large) and the decision-models generator/labeller split are left to design.

## Decision log
2026-11-03 — Should an interview be opened on using Anthropic API credits across modules, via falkor-chat's default/fallback configuration? → **Yes, open it.** (scope)
2026-11-03 — What is the main aim for the credits? → **Experiment and compare.** (intent)
2026-11-03 — Which modules should use Anthropic? → **All of them: falkor-chat, the salesperson demo, model-bench, and decision-models / opencode tank / mcp-monitor.** Overturns nothing yet; `model-bench`'s local-only scope and the 2026-10-31 decision-models entry are flagged in Open questions. (scope)
2026-11-03 — What does a finished experiment look like? → **Scoring Anthropic models in model-bench**, side by side with the local ones. (intent; model-bench's LM-Studio-only scope is now in question)
2026-11-03 — For the non-benchmark modules, which default/fallback posture? → **Decide per module**, one by one. (scope)
2026-11-03 — falkor-chat's default posture with Anthropic available? → **Local first, Anthropic as fallback.** (scope; embeddings stay local by nature, to be confirmed)
2026-11-03 — (Reconsidered) falkor-chat's default posture? → **Defaults stay local; Anthropic only on selected workspaces** through the per-workspace override. Overturns the previous entry ("Local first, Anthropic as fallback"). (scope)
2026-11-03 — In an Anthropic workspace, what happens when Anthropic fails? → **Fall back to the local model**, and the step record shows which model answered. (scope)
2026-11-03 — Should the salesperson storefront demo be switchable to Anthropic like any workspace? → **Yes, the same way**; its pinned local model stays the default. (scope)
2026-11-03 — Which model-bench roles get Anthropic models? → **chat-responder, tool-caller, nlq-generator, guard-judge.** The embedder role is not selected. (scope)
2026-11-03 — Should decision-models data generation and labelling use the Anthropic API? → **Yes when a key is available; Claude Code subagents remain the fallback.** Partly overturns the 2026-10-31 entry in `docs/requirements/typed-decision-models.md` ("Through Claude Code subagents, not the API"). Status check: the campaign machinery is built and smoke-tested, but no labelled dataset has been produced. (scope; no-hosted-client invariant in `decision-models/AGENTS.md` becomes a plan change)
2026-11-03 — Should Anthropic also be scored on the message-type and join-in decision points, not just the guard? → **Yes, all three.** (scope)
2026-11-03 — What should change for `opencode`'s `tank`? → **Able to switch to Anthropic through its own config; default stays local.** (scope)
2026-11-03 — What should change for `mcp-monitor`? → **Nothing**; it makes no LLM calls itself. (scope)
2026-11-03 — Help with credit spending? → **Just see what was used:** each result records the answering Anthropic model and approximate token usage; no cap. (scope)
2026-11-03 — How should the API key be supplied? → **"Ensure this is safely stored and handled"**; no specific mechanism chosen, which is left to design with a security review. (FR-11, AC-7)
2026-11-03 — Which Anthropic models? → **One small and one large**; exact models left to design. (FR-5, FR-6)
2026-11-03 — Consult: security-expert on FR-11/AC-7 key handling → tightened FR-10, FR-11, AC-7, FR-8; leak paths for design: unignored decision-models transcripts/ledgers, tracked model-bench results, falkor-chat Bearer header (Anthropic uses `x-api-key`) and error bodies, no `.gitignore` for key files; spend ceiling raised as Open question 4. (FR-8, FR-10, FR-11, AC-7)
2026-11-03 — Must the storefront's multi-turn tool use work on Anthropic from the first release? → **Yes, required.** (FR-13, AC-8)
2026-11-03 — Will a spending limit be set on Anthropic's side? → **Not now**, so unattended runs must be bounded by design. (FR-14, AC-9)
2026-11-03 — Do embeddings stay local? → **Yes.** (FR-4)
2026-11-03 — Keep the opus-generates / sonnet-labels split for decision-models? → **Leave it to design.** (FR-7)
2026-11-03 — Consult: analyst on the whole doc → verdict needs changes (B1 FR-3 vs FR-11 on a failed key, B2 tank AC, M1 FR-14 bound, M2 tank vs key handling, M3 unnamed split change, M4 dangling pointers); non-contested findings folded in, the rest put to the stakeholder. Clarifies the 2026-11-03 "Leave it to design" entry: it loosens the 2026-10-31 "Generator opus, labeller sonnet" entry in `docs/requirements/typed-decision-models.md`, which gains no pointer. Earlier entries' references to "Open questions" refer to the questions answered by this log. (FR-2, FR-4, FR-7, FR-8, FR-9, FR-12, AC-8, AC-10)
2026-11-03 — On a missing, invalid or revoked key in a switched workspace? → **Fail loudly**; only temporary failures fall back to local. Narrows the 2026-11-03 "Fall back to the local model" entry. (FR-3, FR-11)
2026-11-03 — How should `tank` be handled given its shell and file tools? → **Drop tank from this feature.** Overturns the 2026-11-03 entry "Able to switch to Anthropic through its own config". (FR-8, Out of scope)
2026-11-03 — What is the bound on unattended runs counted in? → **Tokens used**, set before each run. (FR-14, AC-9)
2026-11-03 — Is any content barred from going to Anthropic? → **Real or private data is barred;** only synthetic, demo or benchmark content may go. (FR-15, AC-10)
2026-11-03 — How is a workspace known as private? → **The owner flags it as private (mark-list);** unflagged workspaces can be switched. Refines the 2026-11-03 "Real or private data barred" entry. (FR-15, AC-10)
2026-11-03 — Consult: analyst Pass 2 → approve with suggestions; N1–N6 folded in (AC-11, AC-12, canary via local stand-in, K-056 reference removed, FR-14 clarified, duplicates removed). (FR-4, FR-14, AC-7, AC-8, AC-11, AC-12)
2026-11-03 — Readback: is the document ready for design? → **Yes, confirmed**; latest analyst verdict (Pass 2) is approve with suggestions. Status flipped to Ready for design. (all)
