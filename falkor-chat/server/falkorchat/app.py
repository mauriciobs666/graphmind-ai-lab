"""One process, two front doors (plan §3.1, §6).

`create_app` builds the shared `Services` once and mounts both the REST router
(`api.py`) and the MCP Streamable-HTTP app (`mcp.py`) on a single FastAPI/uvicorn
process. The MCP app's lifespan MUST be forwarded to FastAPI or the MCP session
manager never initialises (python-sdk #1367).

Importing this module (and calling `create_app` without injected services)
never touches the network — the default services hold a deferred connection
handle, and the first FalkorDB round-trip happens at lifespan startup, where
an unreachable instance fails fast (DEF-2).

Run:  uvicorn falkorchat.app:app   (agents connect at /mcp; REST under /)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import api, config, db
from . import mcp as mcp_mod
from .config import CallContext
from .guards import WorkflowConfigError
from .llm import extract_own_line_json_object
from .modelconfig import ModelGateway, StaticModelGateway
from .repository import Repository
from .services import (
    ChannelNotFoundError,
    ServiceError,
    Services,
    ThreadNotFoundError,
    WorkflowDefConflictError,
    WorkflowDefNotFoundError,
    WorkflowDefSpecError,
    WorkflowEngineDisabledError,
    WorkflowInputRejectedError,
    WorkflowRunNotFoundError,
    WorkflowRunNotWaitingError,
)


class _McpPathAlias:
    """ASGI shim: rewrite bare ``/mcp`` to ``/mcp/`` (QA DEF-1).

    Starlette's Mount only serves the sub-app under ``/mcp/``; a client POSTing
    to the documented ``/mcp`` got 405 and the MCP python client does not
    auto-append the slash. Rewriting the path serves both spellings without
    relying on clients following redirects.
    """

    def __init__(self, app) -> None:  # noqa: ANN001 - ASGI app
        self._app = app

    async def __call__(self, scope, receive, send):  # noqa: ANN001 - ASGI signature
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
        await self._app(scope, receive, send)

# The minimal browser client (DESIGN §14.5) lives at the repo root `web/`,
# a sibling of `server/`. Served from this process so there is no CORS seam.
_DEFAULT_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _handle_service_error(_request, exc: ServiceError):  # noqa: ANN001
        not_found = isinstance(exc, (ChannelNotFoundError, ThreadNotFoundError))
        return JSONResponse(
            status_code=404 if not_found else 400,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # §11 workflow errors live in `repository` (no import cycle) so they are not
    # `ServiceError` subclasses — map them explicitly: bad spec → 400, absent
    # def → 404, same envelope shape as the service-error handler.
    @app.exception_handler(WorkflowDefSpecError)
    async def _handle_wf_spec_error(_request, exc: WorkflowDefSpecError):  # noqa: ANN001
        return JSONResponse(
            status_code=400,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.exception_handler(WorkflowDefNotFoundError)
    async def _handle_wf_not_found(_request, exc: WorkflowDefNotFoundError):  # noqa: ANN001
        return JSONResponse(
            status_code=404,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # A topology-differing re-publish/re-materialize (K-034) — same "state
    # conflict, nothing written" 409 precedent as `WorkflowRunNotWaitingError`.
    @app.exception_handler(WorkflowDefConflictError)
    async def _handle_wf_def_conflict(_request, exc: WorkflowDefConflictError):  # noqa: ANN001
        return JSONResponse(
            status_code=409,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # §12 run start/input error map (K-024 D-G). Same envelope shape throughout.
    # Note what is deliberately absent: a fault raised *during the drive* never
    # reaches here — `services._drive_or_fault` converts it into a 200/201
    # `{"status":"failed", …}` envelope, because the run is already correctly
    # terminal in the graph and a 500 would misreport it (D-G). What is mapped
    # below are the faults raised **before** anything was written.
    _wf_run_error_codes: tuple[tuple[type[Exception], int], ...] = (
        # the run id does not exist in this workspace
        (WorkflowRunNotFoundError, 404),
        # input for a run that is not parked, or that lost the resume CAS —
        # nothing was written (§12.13's zero-row contract)
        (WorkflowRunNotWaitingError, 409),
        # reserved key / undeclared key / disallowed value (D-H) — a free
        # rejection: nothing written, no step budget consumed
        (WorkflowInputRejectedError, 400),
        # a structurally malformed `cmp` guard, whose dominant source is
        # publish-time validation — an authoring defect, same class as
        # `WorkflowDefSpecError`
        (WorkflowConfigError, 400),
        # the engine is not wired into this deployment — a named RuntimeError
        # subclass so this maps to 503 without a blanket handler masking real bugs
        (WorkflowEngineDisabledError, 503),
    )

    def _register(exc_type: type[Exception], code: int) -> None:
        async def _handler(_request, exc: Exception):  # noqa: ANN001
            return JSONResponse(
                status_code=code,
                content={"error": type(exc).__name__, "detail": str(exc)},
            )

        app.add_exception_handler(exc_type, _handler)

    for _exc_type, _code in _wf_run_error_codes:
        _register(_exc_type, _code)


def create_app(
    services: Services | None = None,
    *,
    context_provider: Callable[[], CallContext] | None = None,
    mount_mcp: bool = True,
    web_dir: Path | None = None,
    responder: object | None = None,
    embed_worker: object | None = None,
    trigger: object | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `services`/`context_provider` are injectable for tests (target `ws:test`).
    `mount_mcp=False` skips the MCP mount — used by REST tests so the FastMCP
    session manager (run-once per instance) isn't started repeatedly.
    `web_dir` overrides where the static browser client is served from; it
    defaults to the repo-root `web/` and is skipped if that directory is absent.

    `responder`/`embed_worker` (K-013) are **opt-in** out-of-band handlers wired
    onto `BackgroundTasks` in the message-post path; both default to `None` so
    building the default app stays network-free and existing tests are untouched.
    A production wiring constructs them explicitly and passes them in, e.g.::

        from falkorchat.embedding import EmbeddingWorker
        from falkorchat.modelconfig import GraphWorkspaceOverrides, ModelGateway
        from falkorchat.responder import AgentResponder

        models = ModelGateway.from_env(workspace_overrides=GraphWorkspaceOverrides(repo))
        worker = EmbeddingWorker(repo, models=models)
        responder = AgentResponder(services, worker=worker, agent_id="bot1", models=models)
        app = create_app(services, responder=responder, embed_worker=worker)
    """
    if services is None:
        # DEF-2: a deferred connection handle — building the app must never
        # touch the network (this function runs at import time via the
        # module-level `app = create_app()` below). The first real connection
        # happens at lifespan startup and fails fast with a clear
        # `db.FalkorDBUnreachableError` (host:port) when FalkorDB is down.
        services = Services(Repository(db.LazyFalkorDB()))

    provider = context_provider or config.get_context

    if mount_mcp:
        mcp_mod.configure(
            services,
            context_provider=context_provider,
            responder=responder,
            embed_worker=embed_worker,
            trigger=trigger,
        )
        mcp_app = mcp_mod.mcp.streamable_http_app()
        # Forward the MCP app's lifespan or the session manager never inits
        # (python-sdk #1367). On this build the lifespan is exposed as the
        # Starlette router's `lifespan_context` (a callable taking the app).
        mcp_lifespan = mcp_app.router.lifespan_context
    else:
        mcp_lifespan = None

    @asynccontextmanager
    async def _lifespan(app_: FastAPI):
        # The §4 write paths anchor on the author node — ensure the configured
        # actor exists before the first write. This is also the first real
        # FalkorDB round-trip (the default services hold a deferred handle):
        # an unreachable instance aborts startup within the connect-timeout
        # budget (db.FalkorDBUnreachableError, names host:port), and an actor
        # id colliding with an existing Agent aborts loudly
        # (MemberIdCollisionError) instead of silently shadowing it (DEF-1).
        services.ensure_actor(provider())
        if mcp_lifespan is not None:
            async with mcp_lifespan(app_):
                yield
        else:
            yield

    app = FastAPI(lifespan=_lifespan)

    if context_provider is not None:
        app.dependency_overrides[api.get_context] = context_provider

    app.include_router(
        api.build_router(
            services, responder=responder, embed_worker=embed_worker, trigger=trigger
        )
    )
    _register_error_handlers(app)

    if mount_mcp:
        app.mount("/mcp", mcp_app)
        # Serve the documented slash-less spelling too (QA DEF-1).
        app.add_middleware(_McpPathAlias)

    # Static UI mounts LAST: "/" is a catch-all, so it must sit behind the REST
    # routes and the /mcp mount (Starlette matches routes in registration order).
    web = _DEFAULT_WEB_DIR if web_dir is None else web_dir
    if web.is_dir():
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")

    return app


def _build_default_app() -> FastAPI:
    """Construct the module-level ASGI app for `uvicorn falkorchat.app:app`.

    Gated on `config.ENABLE_AGENT` (env `FALKORCHAT_ENABLE_AGENT`): when off (the
    default), returns the plain network-free app so importing this module and the
    pytest baseline never touch any provider or read any config file. When on, builds
    **one** `ModelGateway` (K-042 FR-4/FR-15 — the two config files are read once,
    here) and wires the live K-013 loop through it: the embedding worker embeds EVERY
    posted message, and `AgentResponder` replies to an `@mention` of `config.AGENT_ID`
    — sharing one `Repository`/`Services`/`ModelGateway` with the app. Constructing the
    gateway is itself offline (it only parses the two files); the first network call
    happens when a posted message runs its background tasks.

    The served app MUST run at the workspace's embedding dimension
    (`FALKORCHAT_EMBEDDING_DIM=1024` for `ws:acme`, or the overlay's declared `dim` for
    the resolved embedding model, §4.5) or embeddings silently drop out of the ANN index
    — `scripts/start_server.sh` sets it.
    """
    repo = Repository(db.LazyFalkorDB())
    services = Services(repo)
    if not config.ENABLE_AGENT:
        return create_app(services)

    # Imported lazily so the disabled path carries no import-time weight and the
    # dependency surface for offline imports stays minimal.
    from .embedding import EmbeddingWorker
    from .modelconfig import GraphWorkspaceOverrides, ModelGateway
    from .responder import AgentResponder

    # K-042 L2-3 (FR-16/FR-17): the same `repo`/connection every other collaborator
    # here shares, not a second connection — a graph-backed `WorkspaceOverrides`
    # reading the workspace's `WorkspaceConfig` singleton (`-graph.md` §2.2/§2.5) per
    # call for the two consumers with no per-drive `run` dict (`agent`/`embedding`,
    # below); the executor/guard path instead reads once per drive in `_drive` and
    # never touches this port (§2.6).
    models = ModelGateway.from_env(
        workspace_overrides=GraphWorkspaceOverrides(repo)
    )
    worker = EmbeddingWorker(repo, models=models)
    responder = AgentResponder(
        services, worker=worker, agent_id=config.AGENT_ID, models=models
    )

    # M3 workflow engine (K-023): checked AFTER the responder/embedder/LLM exist (the
    # trigger holds the responder for its no-workflow fall-through). Off by default; the
    # imports are lazy so the disabled path carries no weight. Constructing the executor
    # + trigger is offline — the first network call is when a posted message runs its
    # background tasks. The trigger (not the responder) is passed to `create_app`, so the
    # API schedules exactly one handler per request.
    if config.WORKFLOW_ENABLED:
        from .executor import GraphTracer, WorkflowExecutor
        from .tools import build_builtin_registry
        from .trigger import WorkflowTrigger

        registry = build_builtin_registry(services, agent_id=config.AGENT_ID, models=models)
        judge = _build_llm_judge(models)
        executor = WorkflowExecutor(
            services, repo, models=models, guard_judge=judge,
            tool_registry=registry, tracer=GraphTracer(repo),
        )
        services.set_executor(executor)  # late-bind (breaks the services↔executor cycle)
        trigger = WorkflowTrigger(
            services, agent_id=config.AGENT_ID, def_key=config.TRIGGER_DEF_KEY,
            def_version=config.TRIGGER_DEF_VERSION, responder=responder,
        )
        return create_app(services, trigger=trigger, embed_worker=worker)

    return create_app(services, responder=responder, embed_worker=worker)


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict gate deciding whether a workflow may advance. You are given a "
    "CONDITION and evidence about the conversation: a CURRENT STATE (the agent's compact "
    "understanding of the request — request/known/missing) and/or RECENT TURNS (the latest "
    "messages, oldest first, for context only). One of them may be absent; judge the "
    "CONDITION against whatever you are given. Reply with a single JSON object and nothing "
    'else: {"decision": <true|false>, "rationale": "<one short sentence>"}. Answer true '
    "ONLY when the condition is clearly satisfied; when in doubt answer false. The "
    "rationale must state ONLY the evidence supporting your decision: when advancing, say "
    "what the user DID provide — never mention what is absent, missing or unclear."
)

# The assembled judge user message is capped (≈1500 tokens, the DS §Q1 budget). Oldest turns
# are dropped first; 6 turns × 400 chars ≈ 2.4k, so this is a backstop, not a routine path.
JUDGE_USER_MAX_CHARS = 6000


def _render_judge_user(
    condition: object, understanding: object, recent_turns: object
) -> str:
    """Assemble the DS §Q1 judge user message: CONDITION / CURRENT STATE / RECENT TURNS.

    Each evidence block is omitted when empty (the omit rule itself lives upstream in
    `guards.evaluate_guard` — this only renders what it is handed). The result is capped at
    `JUDGE_USER_MAX_CHARS` by dropping **oldest** turns first, so the newest turn — the one
    the condition is usually about — always survives; a hard truncation backstops the rest.
    """
    blocks = [f"CONDITION: {condition}"]
    if understanding:
        state = json.dumps(understanding, indent=2, sort_keys=True, default=str)
        blocks.append(f"CURRENT STATE:\n{state}")

    turns = list(recent_turns) if isinstance(recent_turns, list) else []
    while turns:
        rendered = "\n".join(
            f"{t.get('speaker', 'member')}: {t.get('text', '')}" for t in turns
        )
        candidate = blocks + [f"RECENT TURNS (context only):\n{rendered}"]
        text = "\n\n".join(candidate)
        if len(text) <= JUDGE_USER_MAX_CHARS:
            return text
        turns.pop(0)  # drop the oldest turn and retry

    return "\n\n".join(blocks)[:JUDGE_USER_MAX_CHARS]


class _LlmGuardJudge:
    """The production fuzzy-guard judge (DS §Q1), K-042: a small callable **object**
    (not a closure) so it can carry `accepts_run = True` — the zero-churn capability
    flag `guards.evaluate_guard` checks before forwarding `run=` (§4.10 step 4). Every
    stub judge in the test suite lacks this attribute and is called exactly as before.

    Matches the injected judge shape `guards.evaluate_guard` calls:
    `(condition, *, understanding, recent_turns, ctx, step_output, model=None, run=None)
    -> {decision, rationale}`. Builds the §Q1 prompt from whichever evidence tier
    `guards` selected — the compact `understanding` (primary) or the RECENT-TURNS
    fallback; the omit rule is applied upstream in `guards.evaluate_guard`, so this
    judge is a **dumb renderer** of what it is handed. A non-JSON / malformed reply
    resolves to `{"decision": False, …}` — and `guards._coerce_verdict` applies the
    same bias-to-suspend downstream, so an unreliable judge never falsely advances.
    Calibration (κ / false-advance) is a U14/U15 concern; the wired judge must simply
    exist so an `llm` guard never hits the m-3 "no judge" path in the served flow.

    K-042 FR-4/FR-6/§4.10: resolves kind `guard` through the gateway on every
    evaluation — `model` is the guard's own choice if it declared one, else the kind
    default; `ws` comes off `run["ws"]` (the B-1 carrier `_drive` stamps, since this
    judge otherwise has no `CallContext` — `ctx` here is the *run* ctx dict).

    The reply is parsed with `llm.extract_own_line_json_object(..., require_key="decision")`,
    **not** a bare `json.loads` (K-027 item 1) and **not** the permissive
    `extract_json_object` the tool-call path uses (K-027 gate B-1). A model that wraps its
    verdict in a ```json fence or a sentence of prose used to break *every* verdict silently —
    the D13 probe scored Ministral 26/26 "unparseable judge output", one of them a correct
    `decision:true`. What the parse now accepts, precisely: a reply that is entirely one JSON
    object (bare or fenced), or **exactly one** `decision`-carrying object that owns its lines.

    Everything else — prose with no object, a verdict quoted mid-sentence
    (*"I would answer {"decision": true …} but they did not"*), two candidate objects that
    disagree — resolves to `decision=False`. That asymmetry is the point: a *missed* advance
    costs the run one more turn, a *false* advance passes a gate the workflow relies on, and
    `guards._coerce_verdict` cannot catch a quoted verdict downstream because its rationale
    reads perfectly clean. Tolerance is therefore only ever applied to how a verdict is
    *wrapped*, never to whether one was *asserted*.

    **Residual, declared:** a *hypothetical* or *schema-echo* verdict the model writes on
    its **own line**, with no second object to disambiguate it, is still accepted and
    still advances (pinned in `test_app.py`). Line ownership is the only signal a parser
    has; separating "here is the shape of a verdict" from "here is my verdict" needs the
    model's intent. K-027 item 3 owns the false-advance metric — if calibration finds
    this shape in the count, the stricter rule (accept only a reply that is *entirely*
    one object after fence-stripping) is the documented next narrowing.
    """

    accepts_run = True

    def __init__(self, models: object) -> None:
        self._models = models

    def __call__(
        self, condition, *, understanding, recent_turns, ctx, step_output,
        model=None, run=None,
    ):  # noqa: ANN001
        ws = run.get("ws") if isinstance(run, dict) else None
        # K-042 L2-3 (FR-16/FR-17, B-1's payoff): the per-drive workspace-override
        # read `_drive` stamps onto `run["modelOverrides"]` (§2.6) — the same carrier
        # `ws` already rides, since `evaluate_guard`'s `ctx=` here is the run-ctx
        # dict, not a `CallContext` (§4.10). Passed as `overrides=` so `resolve()`
        # uses the pre-fetched dict rather than a second graph round trip.
        overrides = run.get("modelOverrides") if isinstance(run, dict) else None
        llm = self._models.llm("guard", requested=model, ws=ws, overrides=overrides)
        user = _render_judge_user(condition, understanding, recent_turns)
        text = llm.complete([
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        parsed = extract_own_line_json_object(text, require_key="decision")
        if parsed is None:
            return {"decision": False, "rationale": "unparseable judge output"}
        return parsed


def _as_model_gateway(obj: object) -> object:
    """FR-4 sugar (§3): a raw `llm`-shaped object (has `.complete(...)`) wraps into a
    `StaticModelGateway`; an already-real gateway (`ModelGateway`/`StaticModelGateway`)
    passes through unchanged. Lets `_build_llm_judge` accept either a bare injected
    stub LLM (every existing direct test call) or the production `ModelGateway`."""
    if isinstance(obj, (ModelGateway, StaticModelGateway)):
        return obj
    return StaticModelGateway(llm=obj)


def _build_llm_judge(models: object) -> _LlmGuardJudge:
    """Build the production fuzzy-guard judge (see `_LlmGuardJudge`). `models` is
    either a real `ModelGateway` (production) or a bare LLM client (tests — wrapped
    via `_as_model_gateway`'s FR-4 sugar)."""
    return _LlmGuardJudge(_as_model_gateway(models))


# Default ASGI app for `uvicorn falkorchat.app:app`.
app = _build_default_app()
