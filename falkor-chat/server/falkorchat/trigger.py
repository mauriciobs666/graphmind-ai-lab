"""The `@mention`→workflow trigger (M3, K-023 — Phase 4 / U11, plan §6).

`WorkflowTrigger.maybe_trigger` is the single background handler the API schedules for a
posted message when the workflow engine is wired. It applies the §6 **ordered rule** so
**exactly one** of {resume, start, responder} acts on a message — the M3 "one handler per
request" guarantee (the trigger *holds* the M2 responder for its no-workflow fall-through,
so an `@mention` can never fire both a workflow and a direct reply):

  1. **loop-guard** — an agent-authored message (`role == "assistant"`) is ignored, exactly
     like the responder's loop-guard: an agent post that mentions the agent must not
     re-trigger forever.
  2. **resume-if-waiting** — if this thread has a `waiting` run, a human reply resumes it
     **without a re-@mention** (§2.4): the reply belongs to that run's intake loop.
     `resume_workflow_run` CASes `waiting→running` single-flight (§12.4) — a concurrent
     loser just no-ops; either way this branch **returns** (never falls through).
  3. **@mention-to-start** — otherwise, if the message @mentions the agent and a def is
     configured, start a fresh run of that def, triggered by this message.
  4. **fall-through** — no workflow applies: hand the message to the held M2 responder,
     which self-decides whether to answer.

Layering (AGENTS.md): the trigger orchestrates the **service layer** only — it holds no
Cypher and no executor reference (the executor is wired behind `services`). Everything is
injected so the ordered rule is unit-testable offline.
"""

from __future__ import annotations

from typing import Any

from .config import CallContext


class WorkflowTrigger:
    """Route a posted message per the §6 ordered rule (see module docstring)."""

    def __init__(
        self,
        services: Any,
        *,
        agent_id: str,
        def_key: str,
        def_version: str,
        responder: Any | None = None,
        trace: bool = False,
    ) -> None:
        self._services = services
        self._agent_id = agent_id
        self._def_key = def_key
        self._def_version = def_version
        self._responder = responder
        self._trace = trace

    def maybe_trigger(
        self, ctx: CallContext, *, thread_id: str, msg_id: str, text: str,
        role: str, mentions: list[str] | None,
    ) -> dict[str, Any] | None:
        """Apply the §6 ordered rule; return the acting handler's result (or `None`).

        `text` is required for the step-4 responder fall-through (`maybe_respond` needs it);
        `start_workflow_run` does not use it — it reads the trigger message from the graph
        by `trigger_msg_id` (`msg_id`).
        """
        # 1. loop-guard — never act on an agent-authored message.
        if role == "assistant":
            return None

        # 2. resume-if-waiting — a reply to a parked run belongs to that run (§2.4). This
        #    branch always returns: a waiting run in this thread owns the reply, whether the
        #    single-flight resume CAS wins (drives) or loses (a concurrent reply won → no-op).
        waiting = self._services.find_waiting_run_for_thread(ctx, thread_id=thread_id)
        if waiting is not None:
            return self._services.resume_workflow_run(ctx, run_id=waiting["runId"])

        # 3. @mention-to-start — start a run of the configured def, triggered by this message.
        if self._def_key and self._agent_id in (mentions or []):
            return self._services.start_workflow_run(
                ctx, def_key=self._def_key, version=self._def_version,
                trigger_msg_id=msg_id, trace=self._trace,
            )

        # 4. fall-through — no workflow applies; hand off to the held M2 responder.
        if self._responder is not None:
            return self._responder.maybe_respond(
                ctx, thread_id=thread_id, msg_id=msg_id, text=text, role=role,
                channel_id=None, mentions=mentions,
            )
        return None
