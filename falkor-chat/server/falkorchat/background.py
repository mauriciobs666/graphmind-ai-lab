"""Out-of-band post-message policy, shared by both transports (K-041).

`_safe_embed`/`_safe_respond`/`_safe_run_workflow` encode the M3 one-handler
guarantee — exactly one of {trigger, responder} handles a posted message,
embedding is independent of both — and must run failure-isolated (never raise
into the caller's scheduling mechanism). Both `api.py` (FastAPI
`BackgroundTasks`) and `mcp.py` (a `threading.Thread` fire-and-forget, since
MCP tools have no per-call background-task object) schedule these same three
functions after a successful `post_message`, so the policy is defined exactly
once here instead of drifting between two copies.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import CallContext

_log = logging.getLogger(__name__)


def _safe_embed(embed_worker: Any, ws: str, msg_id: str, text: str) -> None:
    """Embed a posted message out-of-band, swallowing+logging any failure.

    Runs off-band (DECISION 3, K-008 posture): EVERY posted message is
    embedded so the retrievable corpus grows, but a message is readable
    before its embedding lands and an embedder hiccup must never surface to
    the poster. Embedding is a pure write — it never triggers a response.
    """
    try:
        embed_worker.embed_message(ws, msg_id=msg_id, text=text)
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background embed failed (msgId=%s)", msg_id)


def _safe_respond(responder: Any, ctx: CallContext, posted: dict[str, Any]) -> None:
    """Fire the AI responder out-of-band, swallowing+logging any failure.

    Runs off-band, off the guarded write path. The responder owns the
    trigger policy (@mention + non-agent-authored) and self-no-ops
    otherwise, so each transport stays a thin adapter and never
    re-implements the trigger check. `channel_id` is not carried by the
    message-post path; channel-scoped retrieval in the wired path is a K-014
    follow-up (needs a thread→channel read).
    """
    try:
        responder.maybe_respond(
            ctx,
            thread_id=posted["threadId"],
            msg_id=posted["msgId"],
            text=posted["text"],
            role=posted["role"],
            channel_id=None,
            mentions=posted.get("mentions", []),
        )
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background responder failed (msgId=%s)", posted.get("msgId"))


def _safe_run_workflow(trigger: Any, ctx: CallContext, posted: dict[str, Any]) -> None:
    """Fire the workflow trigger out-of-band, swallowing+logging any failure.

    Runs off-band, off the guarded write path, mirroring `_safe_respond`. The
    trigger applies the §6 ordered rule (resume/start/responder fall-through)
    and **owns** the responder, so this is the *single* handler when a
    trigger is wired — exactly one of trigger-vs-responder is scheduled per
    posted message (the M3 one-handler guarantee). Its latency never blocks
    the poster.
    """
    try:
        trigger.maybe_trigger(
            ctx,
            thread_id=posted["threadId"],
            msg_id=posted["msgId"],
            text=posted["text"],
            role=posted["role"],
            mentions=posted.get("mentions", []),
        )
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception(
            "background workflow trigger failed (msgId=%s)", posted.get("msgId")
        )
