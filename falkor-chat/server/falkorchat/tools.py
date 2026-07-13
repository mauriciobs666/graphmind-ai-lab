"""Node capabilities (FR-5) — the `ToolRegistry`, built-in tools, and the MCP-client seam.

Phase 3 of the M3 LLM-native executor (`docs/plans/m3-executor.md` §4, units U9+U10).
A `type:'agent'` node (executor `_run_agent_node`) is offered **only its granted tool
schemas** and drives the model; when the model calls a tool the executor dispatches it
here. Every tool is a typed callable the registry exposes by name.

**The registry interface is the contract `executor._run_agent_node` already calls:**
`schema(name) -> dict` (the OpenAI function schema offered to the model) and
`dispatch(name, arguments, *, ctx, run) -> str` (run the tool, return the string fed back
to the model — which the executor records as the `tool_result` trace event). This module
does not trace itself: the executor's `_handle_tool_call` traces the call + the returned
result, exactly as the other phases wired it.

**Layering (AGENTS.md).** Tools are domain callables and hold **no Cypher** — `post_message`
and `graphrag_retrieve` go THROUGH `services` (which owns the queries via `repository`). The
`PRODUCED` emission link is `services.link_step_emission` (→ `repository.link_step_emission`,
D2 — distinct from K-013's `EMITTED`).

Built-ins (§4):
  * `post_message` (FR-5a) — post into the run's thread as the workflow agent (guarded §4
    write via `services.post_agent_answer`, role derived `assistant`) and return the posted
    `msgId`. The `StepRun -[:PRODUCED]-> Message` emission link is **not** made here (Option B,
    K-023): no `stepRunId` is resolvable at dispatch time (the StepRun is created after the
    node runs), so the **executor** buffers the returned msgId and links after `_record`. The
    link is the deliberately **two-step, non-atomic** second query (§3/§9): the message is the
    durable artifact, a missing link is a diagnosable/retry-able gap, not a torn thread.
  * `graphrag_retrieve` (FR-5b) — embed the query via the injected `Embedder`, hit
    `services.hybrid_search`, then apply the DS-note **Q2** policy (distance cutoff τ, cap 5 /
    floor 1, **abstain** when nothing passes τ) — deliberately NOT the responder's raw-k=10
    all-seeds anti-pattern.
  * `human_handoff` (FR-5d) — a registered capability that **signals suspend** (raises
    `HumanHandoffSignal`). Present, not exercised: no triage node grants it. The integrated
    executor (Landing 2) catches the signal to park the run pending a human.

MCP-client seam (U10 / FR-5c): `McpToolClient` lists + calls tools on an **external** MCP
server and registers each as an `McpTool` so an MCP-exposed tool is indistinguishable from a
built-in to the node (same `schema`/`dispatch` path). falkor-chat is an MCP *server* today
(DESIGN §15); this adds the *client* direction as a separate seam. Verified against a stub/
in-memory MCP server in tests only — wiring a real external server is deferred (§4).
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from typing import Any, Protocol

from .config import CallContext
from .embedding import Embedder

# ── DS-note Q2 retrieval-to-context policy (calibration starting points, configurable) ──
# `score` from `hybrid_search` is **cosine distance** (0 = identical, ASC). τ keeps seeds
# whose distance ≤ τ (≈ similarity ≥ 1-τ). These are tuning seeds, not shipped constants —
# the coder/QA calibrate τ on the golden set (m3-executor-ml.md Q2); do not treat as final.
DEFAULT_RETRIEVE_TAU: float = 0.5   # distance cutoff — keep seeds with score ≤ τ
DEFAULT_RETRIEVE_CAP: int = 5       # keep at most this many after the cutoff
DEFAULT_RETRIEVE_K: int = 10        # ANN fan-out asked of hybrid_search


# ── the tool seam ─────────────────────────────────────────────────────────────────────

class Tool(Protocol):
    """A node capability: a name, an offered JSON schema, and a synchronous `run`.

    `run` returns the string handed back to the model (the executor records it as the
    `tool_result` trace). A non-string is JSON-encoded by the registry before it is fed back.
    """

    name: str

    @property
    def schema(self) -> dict[str, Any]: ...

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> Any: ...


class UnknownToolError(KeyError):
    """Raised when `schema`/`dispatch` names a tool that is not registered.

    Subclasses `KeyError` so it is consistent with the bare-dict stub registries the
    U8 agent-loop tests use (`self._schemas[name]`). In the live flow the executor
    checks the node's granted set **before** dispatching (AC-6, tested in U8), so a
    truly-unknown name here means a def granted a tool the registry never registered —
    a misconfiguration surfaced loudly rather than silently.
    """


class ToolRegistry:
    """Holds the node capabilities and satisfies the `_run_agent_node` dispatch contract.

    `schema(name)` returns the offered OpenAI function schema (called once per granted tool
    when the node builds its offer); `dispatch(name, arguments, *, ctx, run)` runs the tool
    and returns the string fed back to the model. Both raise `UnknownToolError` for an
    unregistered name.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add (or replace) a tool by its `name`."""
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        """The registered tool names (registration order)."""
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def schema(self, name: str) -> dict[str, Any]:
        try:
            return self._tools[name].schema
        except KeyError:
            raise UnknownToolError(name) from None

    def dispatch(
        self, name: str, arguments: dict[str, Any], *, ctx: CallContext,
        run: dict[str, Any],
    ) -> str:
        try:
            tool = self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None
        result = tool.run(arguments or {}, ctx=ctx, run=run)
        return result if isinstance(result, str) else json.dumps(result, default=str)


def _thread_id_of(run: dict[str, Any]) -> str | None:
    """The run's thread id — the post target (§4 `post_message`).

    Prefers an explicit `run['threadId']`; else parses the serialized run `ctx`
    (`{"threadId": …}`, set by `services.start_workflow_run`). `None` when unbound.
    """
    tid = run.get("threadId")
    if tid:
        return tid
    ctx_raw = run.get("ctx")
    if isinstance(ctx_raw, dict):
        return ctx_raw.get("threadId")
    if isinstance(ctx_raw, str) and ctx_raw:
        try:
            obj = json.loads(ctx_raw)
        except (ValueError, TypeError):
            return None
        if isinstance(obj, dict):
            return obj.get("threadId")
    return None


# ── built-in tools ──────────────────────────────────────────────────────────────────────

class PostMessageTool:
    """FR-5a — post into the run's thread as the workflow agent, then link the emission.

    Reuses the guarded §4 write via `services.post_agent_answer` (actor swapped to the agent
    id so `role` derives to `assistant` in the service, never trusted from the caller). The
    `StepRun -[:PRODUCED]-> Message` audit link is **not** made here (Option B, K-023): at
    dispatch time no `stepRunId` is resolvable (the StepRun is created by `record_step_and_
    advance` *after* the node runs), so the tool returns the posted `msgId` in its result
    envelope and the **executor** buffers it and links after `_record` (`executor._link_
    emissions`). This keeps the tool decoupled — audit linking is the executor's concern.
    """

    name = "post_message"

    def __init__(self, services: Any, *, agent_id: str) -> None:
        self._services = services
        self._agent_id = agent_id

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Post a message into the current thread as the assistant. Use this to "
                    "ask the user a question or to deliver your answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The message body to post.",
                        },
                        "mentions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional member ids to @mention.",
                        },
                    },
                    "required": ["text"],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        text = arguments.get("text", "")
        mentions = arguments.get("mentions") or None
        thread_id = _thread_id_of(run)
        if not thread_id:
            return "error: no thread is bound to this run; cannot post a message"

        agent_ctx = CallContext(ws=ctx.ws, actor=self._agent_id)
        posted = self._services.post_agent_answer(
            agent_ctx, thread_id=thread_id, text=text, mentions=mentions
        )
        msg_id = posted["msgId"]

        # Return the posted msgId in the envelope; the executor buffers it and links
        # StepRun→PRODUCED→Message after `_record` (Option B, K-023). The link is the
        # two-step, non-atomic second query (§3/§9) — distinct from K-013's EMITTED (D2).
        return json.dumps({"posted": msg_id, "threadId": thread_id})


class GraphragRetrieveTool:
    """FR-5b — retrieve grounded context via GraphRAG, with the DS-note Q2 policy.

    The model calls this with a **text** query; `services.hybrid_search` takes a query
    **vector**, so the tool embeds the query via the injected `Embedder` first (mirroring
    the responder's embed step) — then applies the Q2 discipline the responder does NOT:
    a distance cutoff τ, a cap of `cap` seeds after the cutoff, and **abstention** when
    nothing passes τ (returns a "no relevant context found" finding rather than synthesizing
    from noise). τ/cap/k are configurable (calibration seeds, not shipped constants).
    """

    name = "graphrag_retrieve"

    def __init__(
        self, services: Any, embedder: Embedder, *,
        tau: float = DEFAULT_RETRIEVE_TAU, cap: int = DEFAULT_RETRIEVE_CAP,
        k: int = DEFAULT_RETRIEVE_K, channel_id: str | None = None,
    ) -> None:
        self._services = services
        self._embedder = embedder
        self._tau = tau
        self._cap = cap
        self._k = k
        self._channel_id = channel_id

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Retrieve relevant messages from the workspace to ground your answer. "
                    "Returns ranked seeds (msgId, text, score) or a 'no relevant context "
                    "found' finding when nothing is relevant."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural-language query to retrieve context for.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        query = arguments.get("query", "")
        q_vec = self._embedder.embed(query)
        rows = self._services.hybrid_search(
            ctx, q_vec=q_vec, k=self._k, channel_id=self._channel_id
        )
        # `score` is cosine DISTANCE (ASC). Keep only seeds within τ, capped — never raw
        # top-k. Rows come pre-ordered by score ASC, so slicing after the filter preserves rank.
        passing = [r for r in rows if r["score"] <= self._tau][: self._cap]
        if not passing:
            return json.dumps({"seeds": [], "finding": "no relevant context found"})
        seeds = [
            {"msgId": r["msgId"], "text": r["text"], "score": r["score"]}
            for r in passing
        ]
        return json.dumps({"seeds": seeds})


class HumanHandoffSignal(Exception):
    """Control signal raised by `human_handoff`: suspend the run pending a human.

    The integrated executor (Landing 2) catches this to park the run — distinct from the
    intake wait-for-reply (guard-driven suspend, §2.4). Present, not exercised: no triage
    node grants `human_handoff`, so it is never raised inside the proof flow.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason or "human handoff requested")
        self.reason = reason


class HumanHandoffTool:
    """FR-5d — a registered capability that signals suspend. Present, not exercised."""

    name = "human_handoff"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Hand off to a human: suspend the run until a designated person responds. "
                    "Use only when a human decision is required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why a human is needed.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        raise HumanHandoffSignal(arguments.get("reason", ""))


def build_builtin_registry(
    services: Any, embedder: Embedder, *, agent_id: str,
    tau: float = DEFAULT_RETRIEVE_TAU, cap: int = DEFAULT_RETRIEVE_CAP,
    k: int = DEFAULT_RETRIEVE_K, channel_id: str | None = None,
) -> ToolRegistry:
    """Wire the three built-in capabilities into a fresh `ToolRegistry` (§4).

    `human_handoff` is registered (present) but the triage nodes do not grant it — the
    AC-6 fence is per-node `config.tools`, not registry membership.
    """
    return ToolRegistry([
        PostMessageTool(services, agent_id=agent_id),
        GraphragRetrieveTool(
            services, embedder, tau=tau, cap=cap, k=k, channel_id=channel_id
        ),
        HumanHandoffTool(),
    ])


# ── MCP-client seam (U10 / FR-5c) ─────────────────────────────────────────────────────────

def _mcp_tool_schema(tool: Any) -> dict[str, Any]:
    """Convert an MCP `types.Tool` to the OpenAI function-schema shape the node offers."""
    params = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": getattr(tool, "description", None) or "",
            "parameters": params,
        },
    }


def _content_to_text(result: Any) -> str:
    """Flatten an MCP `CallToolResult` into the string fed back to the model.

    Prefers `structuredContent` (JSON-encoded) when the server returned it; otherwise
    concatenates the text of the content blocks.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, default=str)
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


class McpTool:
    """One external MCP tool, presented uniformly with a built-in.

    Holds its offered schema (fetched at registration) and delegates `run` to the owning
    `McpToolClient.call_tool` — so `registry.schema(name)`/`registry.dispatch(name, …)`
    behave identically whether the tool is built-in or MCP-exposed.
    """

    def __init__(self, client: McpToolClient, *, name: str,
                 schema: dict[str, Any]) -> None:
        self._client = client
        self.name = name
        self._schema = schema

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        return self._client.call_tool(self.name, arguments)


class McpToolClient:
    """MCP **client** (FR-5c): list + call tools on an external MCP server, synchronously.

    The executor's tool dispatch is synchronous, but the MCP `ClientSession` is async and
    bound to the event loop it was opened on. So this client owns a **background asyncio
    loop thread** and a persistent, initialized session opened on it; every `list_tools`/
    `call_tool` runs on that loop via `run_coroutine_threadsafe`, giving a fully synchronous
    public API that works from any thread.

    Construct with `connect`: a zero-arg callable returning an async context manager that
    yields an initialized `mcp.ClientSession` — e.g.
    `lambda: create_connected_server_and_client_session(server)` in tests, or a real
    stdio/HTTP transport CM in production. Use as a context manager (`with McpToolClient(...)
    as client:`) or call `start()`/`close()` explicitly.

    **Scope (§4):** verified against a stub/in-memory MCP server in tests; wiring a real
    external server into the proof flow is deferred. This is the *client* direction, a
    separate seam from the MCP *server* front door (DESIGN §15).
    """

    def __init__(self, connect: Any) -> None:
        self._connect = connect
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._closing: asyncio.Event | None = None
        self._serve: Future | None = None

    def start(self) -> McpToolClient:
        # The connect context manager opens an anyio task group whose cancel scope must
        # be entered AND exited in the same task — so the whole session lifetime lives in
        # ONE long-lived `_serve` task on the background loop (open → hold → close),
        # while `list_tools`/`call_tool` schedule ordinary request coroutines onto it.
        if self._session is not None:
            return self
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        ready: Future = Future()

        async def _serve() -> None:
            try:
                async with self._connect() as session:
                    self._session = session
                    self._closing = asyncio.Event()
                    ready.set_result(None)
                    await self._closing.wait()
            except BaseException as exc:  # surface an open failure to start()
                if not ready.done():
                    ready.set_exception(exc)
                raise

        self._serve = asyncio.run_coroutine_threadsafe(_serve(), self._loop)
        ready.result(timeout=30)  # block until the session is open (or raise its failure)
        return self

    def close(self) -> None:
        if self._session is None:
            return
        assert self._loop is not None and self._thread is not None
        # Ask `_serve` to leave its `async with` (same task → clean task-group exit).
        self._loop.call_soon_threadsafe(self._closing.set)
        try:
            if self._serve is not None:
                self._serve.result(timeout=30)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
            self._session = self._closing = self._serve = None
            self._loop = self._thread = None

    def __enter__(self) -> McpToolClient:
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _run(self, coro: Any, *, timeout: float | None = 30.0) -> Any:
        if self._loop is None:  # pragma: no cover — misuse guard
            raise RuntimeError("McpToolClient used before start()")
        # `timeout` is a safety net so a wedged server surfaces loudly rather than
        # hanging the (synchronous) caller forever.
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def list_tools(self) -> list[dict[str, Any]]:
        """The server's tools as OpenAI function schemas (uniform with built-ins)."""
        result = self._run(self._session.list_tools())
        return [_mcp_tool_schema(t) for t in result.tools]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call one server tool and flatten its result to the model-facing string."""
        result = self._run(self._session.call_tool(name, arguments or {}))
        return _content_to_text(result)

    def register_into(self, registry: ToolRegistry) -> list[str]:
        """Register every server tool into `registry` as an `McpTool`; returns their names."""
        names: list[str] = []
        for schema in self.list_tools():
            tool_name = schema["function"]["name"]
            registry.register(McpTool(self, name=tool_name, schema=schema))
            names.append(tool_name)
        return names
