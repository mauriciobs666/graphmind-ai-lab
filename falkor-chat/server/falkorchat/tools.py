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

**Layering (AGENTS.md).** Tools are domain callables and hold **no Cypher** — `post_message`,
`graphrag_retrieve`, and the catalog tools below all go THROUGH `services` (which owns the
queries via `repository`). The `PRODUCED` emission link is `services.link_step_emission` (→
`repository.link_step_emission`, D2 — distinct from K-013's `EMITTED`).

Built-ins (§4):
  * `post_message` (FR-5a) — post into the run's thread as the workflow agent (guarded §4
    write via `services.post_agent_answer`, role derived `assistant`) and return the posted
    `msgId`. The `StepRun -[:PRODUCED]-> Message` emission link is **not** made here (Option B,
    K-023): no `stepRunId` is resolvable at dispatch time (the StepRun is created after the
    node runs), so the **executor** buffers the returned msgId and links after `_record`. The
    link is the deliberately **two-step, non-atomic** second query (§3/§9): the message is the
    durable artifact, a missing link is a diagnosable/retry-able gap, not a torn thread.
    An unresolvable `mentions` id (the model reaching for a *display name* it saw in the
    folded thread context) comes back as an `error:` string for the model to correct — a
    hallucinated `@mention` must not end the run.
  * `graphrag_retrieve` (FR-5b) — embed the query via the injected `Embedder`, hit
    `services.hybrid_search`, then apply the DS-note **Q2** policy (distance cutoff τ, cap 5 /
    floor 1, **abstain** when nothing passes τ) — deliberately NOT the responder's raw-k=10
    all-seeds anti-pattern. `services.hybrid_search`'s result rows can be `Message`- or
    `Chunk`-shaped since K-050 M5 Stage 5 (merged, tagged `seedKind`) — the returned seed's
    id is resolved generically, the same way `responder.py` does.
  * `human_handoff` (FR-5d) — a registered capability that **signals suspend** (raises
    `HumanHandoffSignal`). Present, not exercised: no triage node grants it. The integrated
    executor (Landing 2) catches the signal to park the run pending a human.
  * `lookup_product_fact` / `filter_products` (K-052 M6, `docs/plans/workflow-catalog-lookup.md`
    §3.5) — the `salesperson` demo's two catalog-lookup capabilities, disjoint from `triage`'s
    tool set (no def grants both `graphrag_retrieve`/`human_handoff` and the catalog tools
    today) but registered into this same shared `ToolRegistry` — the AC-6 fence is per-node
    `config.tools`, not registry membership, the same "present, registered, only offered where
    granted" posture `human_handoff` already established for `triage`. `lookup_product_fact`
    resolves one named product's category/price via `services.lookup_product`; `filter_products`
    lists products by category and/or price range via `services.filter_products`. Both abstain
    with a JSON `{"found": false}` / `{"items": [], "finding": ...}` shape rather than a
    fabricated answer — the same idiom `graphrag_retrieve`'s "no relevant context found" already
    uses.
  * `view_cart` / `add_to_cart` / `remove_from_cart` / `clear_cart` / `place_order` (K-053 M6,
    `docs/plans/workflow-cart-and-totals.md` §3.3) — the `salesperson` demo's cart/order
    capabilities, going through `services.get_cart`/`add_cart_item`/`remove_cart_item`/
    `clear_cart`/`place_order`, which own the `ensure_customer`/`ensure_cart` write-path
    ordering and the FR-8 pure-arithmetic call (`pricing.compute_line_total`) — no Cypher and
    no arithmetic lives in this module. `add_to_cart`/`remove_from_cart` abstain
    `{"found": false}` on an unknown product name, same idiom as the catalog tools;
    `place_order` on an empty cart returns an explanatory **string**, not a zero-line order.
  * `get_profile` / `save_profile` (K-054 M6, `docs/plans/workflow-durable-profile.md` §3.2) —
    the `salesperson` demo's durable customer-profile capabilities, going through
    `services.get_profile`/`save_profile`, which own the `coalesce()`-guarded partial-update
    write (`repository.upsert_profile`) — no Cypher lives in this module.
    `get_profile` deliberately does **not** use the `{"found": false}` abstention shape: it
    always returns `{"name", "deliveryAddress"}` with absent fields as `null` — "no profile
    yet" is the ordinary first-conversation state here, not an error/abstention case (unlike
    the catalog/cart tools' "unknown name" case). `save_profile`'s two arguments are both
    optional; omitting one leaves that field's stored value unchanged, it never clears it.
  * `query_graph_data` (K-055 M6, `docs/plans/workflow-nl-query-generation.md` §3.1/§3.4) —
    the `salesperson` demo's arbitrarily-phrased structured-query capability. The model's own
    arguments (`question`, `dataset`) are not the answer: this tool makes a second, internal,
    non-agent-loop structured-completion LLM call whose only allowed output shape is
    `querygen.QueryRequest`, which `querygen.compile` turns into a `CompiledQuery` run via
    `services.run_structured_query` → `repository.run_readonly_query` (always `.ro_query`, never
    a write — FR-3/FR-3a's structural, engine-backed non-mutation guarantee). Abstains
    `{"items": [], "finding": "no matching data found"}` on any unparseable/schema-invalid
    completion or empty result — same idiom as `graphrag_retrieve`/the catalog tools, never a
    fabricated answer.

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

from pydantic import ValidationError

from . import querygen
from .config import CallContext
from .embedding import Embedder
from .llm import extract_own_line_json_object
from .modelconfig import StaticModelGateway
from .services import UnknownMemberError

# ── DS-note Q2 retrieval-to-context policy (calibration starting points, configurable) ──
# `score` from `hybrid_search` is **cosine distance** (0 = identical, ASC). τ keeps seeds
# whose distance ≤ τ (≈ similarity ≥ 1-τ). These are tuning seeds, not shipped constants —
# the coder/QA calibrate τ on the golden set (m3-executor-ml.md Q2); do not treat as final.
DEFAULT_RETRIEVE_TAU: float = 0.5   # distance cutoff — keep seeds with score ≤ τ
DEFAULT_RETRIEVE_CAP: int = 5       # keep at most this many after the cutoff
DEFAULT_RETRIEVE_K: int = 10        # ANN fan-out asked of hybrid_search

# K-052 M6: a demo-scale cap on `filter_products`'s result set (`workflow-catalog-
# lookup.md` §3.5) — the real result-set cap is FalkorDB's own `RESULTSET_SIZE`
# default of 10000, irrelevant at this catalog's size (~15 products).
DEFAULT_FILTER_LIMIT: int = 20


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
        try:
            posted = self._services.post_agent_answer(
                agent_ctx, thread_id=thread_id, text=text, mentions=mentions
            )
        except UnknownMemberError as exc:
            # The node's prompt carries the thread transcript as `"{displayName}: {text}"`
            # (executor `_assemble_messages`), so a model reliably reaches for the *name*
            # it can see — `mentions:["alice"]` — while §4 resolves **member ids**. Turn
            # that into an actionable refusal in the tool's own error dialect (like the
            # no-thread case above) so the re-prompt is a correction, not a blind retry.
            # The executor's tool-error net would also survive this generically; naming the
            # id-vs-name confusion here is what actually gets the next turn right.
            return (f"error: unknown member id(s) {exc}; `mentions` takes member ids, not "
                    f"display names — retry with valid member ids or omit `mentions`")
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

    K-050 M5 Stage 5 (FR-2): `services.hybrid_search` merges `Message` and `Chunk` ANN
    pools and tags each row `seedKind`. The returned seed dict here is `{seedId, text,
    score, documentId}` — `seedId` resolves to `msgId`/`chunkId` depending on `seedKind`
    (mirrors `responder.py`'s resolution), `documentId` is populated for a `Chunk` seed
    (already denormalized on the row, no extra hop) and `null` for a `Message` seed.
    """

    name = "graphrag_retrieve"

    def __init__(
        self, services: Any, embedder: Embedder | None = None, *,
        models: Any = None,
        tau: float = DEFAULT_RETRIEVE_TAU, cap: int = DEFAULT_RETRIEVE_CAP,
        k: int = DEFAULT_RETRIEVE_K, channel_id: str | None = None,
    ) -> None:
        self._services = services
        # K-042 M-3: a real FR-4 consumer now, resolving through the gateway inside
        # `run()` (which already has `ctx.ws`) instead of a bound `Embedder` fixed at
        # construction. FR-4 sugar: a directly-injected `embedder=` wraps into a
        # `StaticModelGateway` — every existing `GraphragRetrieveTool(services, stub)`
        # construction keeps working unmodified.
        self._models = models or StaticModelGateway(embedder=embedder)
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
                    "Retrieve relevant context from the workspace to ground your answer — "
                    "chat messages and ingested-document chunks alike. Returns ranked seeds "
                    "(seedId, text, score, documentId — set when the seed is a document "
                    "chunk, null for a chat message) or a 'no relevant context found' "
                    "finding when nothing is relevant."
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
        embedder = self._models.embedder("embedding", ws=ctx.ws)
        q_vec = embedder.embed(query)
        rows = self._services.hybrid_search(
            ctx, q_vec=q_vec, k=self._k, channel_id=self._channel_id
        )
        # `score` is cosine DISTANCE (ASC). Keep only seeds within τ, capped — never raw
        # top-k. Rows come pre-ordered by score ASC, so slicing after the filter preserves rank.
        passing = [r for r in rows if r["score"] <= self._tau][: self._cap]
        if not passing:
            return json.dumps({"seeds": [], "finding": "no relevant context found"})
        # K-050 M5 Stage 5: `rows` can now be `Message`- or `Chunk`-shaped
        # (`services.hybrid_search`'s app-side merge tags each item `seedKind`) —
        # resolve the id generically, the same way `responder.py` does
        # (`s["msgId"] if s["seedKind"] == "Message" else s["chunkId"]`), rather than
        # assuming every row has `msgId`. `documentId` is surfaced (already denormalized
        # on the Chunk row by `repository.search_chunks`, no extra hop needed) so a
        # document-grounded hit still carries source attribution, not just an opaque id.
        seeds = [
            {
                "seedId": r["msgId"] if r["seedKind"] == "Message" else r["chunkId"],
                "text": r["text"],
                "score": r["score"],
                "documentId": r["documentId"] if r["seedKind"] == "Chunk" else None,
            }
            for r in passing
        ]
        return json.dumps({"seeds": seeds})


class LookupProductFactTool:
    """FR-1/FR-4 (K-052 M6) — exact-name catalog fact lookup: one product's category/price.

    Thin dispatch onto `services.lookup_product`, which normalizes the model-supplied
    name and does the exact `=` lookup (`docs/QUERIES.md` §15.1). Abstention shape mirrors
    `GraphragRetrieveTool`'s "no relevant context found" idiom: `{"found": false}` when
    nothing matches, never a fabricated row (AC-3). AC-4's wording tolerance ("how much is
    the X" vs "what's the price of the X") is the calling model's own argument-extraction
    job — both phrasings are expected to extract the same `name` argument.
    """

    name = "lookup_product_fact"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Look up a fact (category, price) about one specific product by name. "
                    "Use this to answer a question about a single named product."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "The product's name, as the customer referred to it."
                            ),
                        },
                    },
                    "required": ["name"],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        row = self._services.lookup_product(ctx, name=arguments.get("name", ""))
        if row is None:
            return json.dumps({"found": False})
        return json.dumps({"found": True, **row})


class FilterProductsTool:
    """FR-2/FR-3 (K-052 M6) — category/price-range catalog filter: list matching products.

    Thin dispatch onto `services.filter_products` (`docs/QUERIES.md` §15.2). Every filter
    argument is optional; an all-omitted call lists the whole catalog up to
    `DEFAULT_FILTER_LIMIT` — acceptable at this catalog's demo scale (~15 products, plan
    §3.5). Abstention shape mirrors `LookupProductFactTool`'s: `{"items": [], "finding":
    "no matching products found"}` when nothing matches (AC-3).
    """

    name = "filter_products"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "List products matching an optional category and/or price range. "
                    "Omit any argument you don't need to filter by; omitting all of them "
                    "lists the whole catalog."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Only products in this category.",
                        },
                        "minPrice": {
                            "type": "number",
                            "description": "Only products priced at or above this amount.",
                        },
                        "maxPrice": {
                            "type": "number",
                            "description": "Only products priced at or below this amount.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        rows = self._services.filter_products(
            ctx,
            category=arguments.get("category"),
            min_price=arguments.get("minPrice"),
            max_price=arguments.get("maxPrice"),
            limit=DEFAULT_FILTER_LIMIT,
        )
        if not rows:
            return json.dumps({"items": [], "finding": "no matching products found"})
        return json.dumps({"items": rows})


class ViewCartTool:
    """FR-1 (K-053 M6) — view the cart: items + a live-computed total.

    Thin dispatch onto `services.get_cart` (`docs/plans/workflow-cart-and-
    totals.md` §3.3) — prices are resolved fresh from the catalog on every
    call (FR-3: never stale), never persisted on the cart line itself. No
    abstention shape: an empty cart is a normal, valid answer
    (`{"items": [], "total": 0.0}`), not a "not found" case.
    """

    name = "view_cart"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "View the customer's current cart: every line item with its "
                    "current price and quantity, and the live total."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        return json.dumps(self._services.get_cart(ctx))


class AddToCartTool:
    """FR-1 (K-053 M6) — add a quantity of a named product to the cart.

    Thin dispatch onto `services.add_cart_item`, which resolves `productName`
    via the same catalog lookup `LookupProductFactTool` uses, then ensures
    the `Customer`/`Cart` anchors and writes the line (plan §3.3, the MAJOR-
    finding fix). Abstention shape mirrors `LookupProductFactTool`'s:
    `{"found": false}` on an unknown product name, since it calls the same
    underlying lookup. `quantity` defaults to `1` when the model omits it.
    """

    name = "add_to_cart"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Add a quantity of a named product to the customer's cart. "
                    "Calling this again for the same product adds more of it "
                    "(quantities accumulate, they don't replace)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "productName": {
                            "type": "string",
                            "description": (
                                "The product's name, as the customer referred to it."
                            ),
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "How many to add. Defaults to 1 if omitted.",
                            "minimum": 1,
                        },
                    },
                    "required": ["productName"],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        row = self._services.add_cart_item(
            ctx, product_name=arguments.get("productName", ""),
            quantity=arguments.get("quantity") or 1,
        )
        if row is None:
            return json.dumps({"found": False})
        return json.dumps({"found": True, **row})


class RemoveFromCartTool:
    """FR-1 (K-053 M6) — remove a quantity of a named product from the cart,
    or the whole line when `quantity` is omitted.

    Thin dispatch onto `services.remove_cart_item` (plan §3.3). Abstention
    shape mirrors `LookupProductFactTool`'s: `{"found": false}` on an unknown
    product name. A known product with no matching cart line (never added,
    or already removed) is a distinct, non-abstaining outcome —
    `{"found": true, "removed": false, ...}` — since the *product* resolved
    fine, there was simply nothing of it in the cart to remove.
    """

    name = "remove_from_cart"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Remove a quantity of a named product from the customer's "
                    "cart. Omit quantity to remove the entire line for that "
                    "product, however many are in the cart."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "productName": {
                            "type": "string",
                            "description": (
                                "The product's name, as the customer referred to it."
                            ),
                        },
                        "quantity": {
                            "type": "integer",
                            "description": (
                                "How many to remove. Omit to remove the whole line."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["productName"],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        row = self._services.remove_cart_item(
            ctx, product_name=arguments.get("productName", ""),
            quantity=arguments.get("quantity"),
        )
        if row is None:
            return json.dumps({"found": False})
        return json.dumps({"found": True, **row})


class ClearCartTool:
    """FR-1 (K-053 M6) — empty the cart entirely.

    Thin dispatch onto `services.clear_cart`. Clearing an already-empty (or
    never-touched) cart is a plain no-op, not an error — the returned shape
    doesn't distinguish the two, matching the repository's own contract
    (graph note §2.4/§2.5).
    """

    name = "clear_cart"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Remove every item from the customer's cart.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        self._services.clear_cart(ctx)
        return json.dumps({"cleared": True})


class PlaceOrderTool:
    """FR-4/FR-5 (K-053 M6) — place an order from the current cart.

    Thin dispatch onto `services.place_order`, which resolves each line's
    *current* price via the batch catalog lookup, computes the frozen total
    via `pricing.compute_line_total` (FR-8 — plain Python, no LLM call), and
    persists the snapshot (AC-5/AC-6). On an empty cart (or one whose every
    product has since vanished from the catalog), returns an explanatory
    **string**, not a JSON envelope — deliberately not `{"found": false}`
    (there is no "product name" here to have not been found; the established
    idiom doesn't fit this abstention shape, plan §3.3).
    """

    name = "place_order"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Place an order for everything currently in the customer's "
                    "cart, at current catalog prices. Clears the cart on success."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        result = self._services.place_order(ctx)
        if result is None:
            return "The cart is empty — add an item before placing an order."
        return json.dumps(result)


class GetProfileTool:
    """FR-1/FR-2 (K-054 M6) — read the stored customer profile, if any.

    Thin dispatch onto `services.get_profile` (plan §3.2) — always returns a
    shape with both fields, defaulting absent fields to `None`. **Not** the
    `{"found": false}` abstention shape the catalog-lookup tools use: "no
    profile yet" is the ordinary first-conversation state, not an
    error/abstention case (plan §3.2).
    """

    name = "get_profile"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Look up this customer's stored name and delivery address, "
                    "if any were saved in an earlier conversation. Missing "
                    "fields come back as null — ask the customer for those, "
                    "don't invent them."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        return json.dumps(self._services.get_profile(ctx))


class SaveProfileTool:
    """FR-1/FR-2/FR-3 (K-054 M6) — save (or update) the customer's name and/or
    delivery address.

    Thin dispatch onto `services.save_profile` (plan §3.2). Both arguments
    are optional — omitting one leaves that field's stored value unchanged
    (AC-2), it never clears it. At least one is expected but not structurally
    enforced (plan §3.2) — a call with neither argument still round-trips
    through the upsert (a no-op write that only bumps `profileUpdatedAt`).
    """

    name = "save_profile"

    def __init__(self, services: Any) -> None:
        self._services = services

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Save (or update) the customer's name and/or delivery "
                    "address for future conversations. Omit whichever field "
                    "you don't have — it leaves the previously stored value "
                    "unchanged, it does not clear it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The customer's name.",
                        },
                        "deliveryAddress": {
                            "type": "string",
                            "description": "The customer's delivery address.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        result = self._services.save_profile(
            ctx, name=arguments.get("name"),
            delivery_address=arguments.get("deliveryAddress"),
        )
        return json.dumps(result)


# ── query_graph_data (K-055 M6) ───────────────────────────────────────────────
#
# `docs/plans/workflow-nl-query-generation.md` §3.1/§3.4. FR-1/FR-2's arbitrary-
# phrasing answer over structured graph data, structurally incapable of a
# mutating query (FR-3/FR-3a — see `querygen`'s own module docstring for the
# full two-layer safety argument this tool's Layer 1 half depends on).

def _describe_dataset_schema(schema: querygen.DatasetSchema) -> str:
    """One line per dataset's labels + registered properties — shared by both
    the outer tool schema's `description` (every dataset, so the routing model
    knows what each holds) and the internal structured-completion system
    prompt (one resolved dataset, so the filling model knows its exact
    allowlist). Generated from `querygen.DATASET_REGISTRY`/`DatasetSchema`
    directly (plan §3.3) — never hand-written prose that could drift from the
    registry."""
    return "; ".join(
        f"{label} (properties: {', '.join(sorted(props))})"
        for label, props in schema.labels.items()
    )


def _describe_all_datasets() -> str:
    return "\n".join(
        f"- {name}: {_describe_dataset_schema(schema)}"
        for name, schema in querygen.DATASET_REGISTRY.items()
    )


# The internal, non-agent-loop structured-completion call's system prompt
# (§3.1 step 2) — mirrors `extraction._SYSTEM_PROMPT`'s own "reply with a
# single JSON object and nothing else, in exactly this shape" discipline,
# parsed the same fence-tolerant way (`llm.extract_own_line_json_object`).
# `dataset` is deliberately NOT part of the shape the model fills: the dataset
# was already selected by the outer tool call's own argument, so this tool
# injects it itself rather than asking the filling model to restate (and
# possibly contradict) it.
_QUERY_REQUEST_INSTRUCTIONS = (
    "You translate a natural-language question into a small, structured "
    "query against ONE graph dataset. Reply with a single JSON object and "
    "nothing else, in exactly this shape:\n"
    '{{"matches": [{{"var": "<short lowercase identifier, e.g. \'p\' or \'e\'>", '
    '"label": "<one of the labels listed below>", '
    '"filters": [{{"property": "<one of the properties listed for that label>", '
    '"op": "<one of = <> < <= > >=>", "value": <a bare JSON number for a '
    'numeric property (e.g. 50, never "50"), a bare JSON string for a text '
    'property, or true/false>}}]}}], '
    '"returns": ["<var>.<property>", or "count(<var>)"/"count(<var>.<property>)"/'
    '"avg(...)"/"min(...)"/"max(...)" the same way], '
    '"order_by": "<var>.<property>" (omit if not sorting), '
    '"order_dir": "ASC" or "DESC" (default ASC), '
    '"limit": <integer between 1 and 50, default 20>}}\n\n'
    "`matches` has exactly one entry. Use ONLY the labels and properties "
    "listed below for this dataset — never invent one. Add a filter ONLY for "
    "a condition the question actually states: a superlative question "
    "(\"cheapest\", \"most expensive\") needs order_by + limit, never an "
    "invented filter with no basis in the question. When the question names "
    "a specific item or entity, filter on its plain `name` property using "
    "the exact text the question uses — never a `*Normalized` property "
    "(e.g. `nameNormalized`), those hold internal lower-cased values you do "
    "not have. When the question asks to list, identify, or classify "
    "entities, return `<var>.name` — never `<var>.entityId`, which is an "
    "internal identifier the reader cannot use. Reply with your best single "
    "JSON object even if you are unsure; never reply with prose.\n\n"
    "Examples (the schema differs per dataset — only the pattern matters):\n"
    '- "How much does the Wireless Charging Pad cost?" -> '
    '{{"matches": [{{"var": "p", "label": "Product", "filters": '
    '[{{"property": "name", "op": "=", "value": "Wireless Charging Pad"}}]}}], '
    '"returns": ["p.price"]}}\n'
    '- "Which products cost less than $50?" -> '
    '{{"matches": [{{"var": "p", "label": "Product", "filters": '
    '[{{"property": "price", "op": "<", "value": 50}}]}}], '
    '"returns": ["p.name"]}}\n'
    '- "Which entities are of type Location?" -> '
    '{{"matches": [{{"var": "e", "label": "Entity", "filters": '
    '[{{"property": "type", "op": "=", "value": "Location"}}]}}], '
    '"returns": ["e.name"]}}\n'
    '- "Which product is the cheapest?" -> '
    '{{"matches": [{{"var": "p", "label": "Product", "filters": []}}], '
    '"returns": ["p.name"], "order_by": "p.price", "order_dir": "ASC", '
    '"limit": 1}}\n\n'
    "This dataset's schema:\n{dataset_schema}"
)


def _build_query_request_system_prompt(schema: querygen.DatasetSchema) -> str:
    return _QUERY_REQUEST_INSTRUCTIONS.format(dataset_schema=_describe_dataset_schema(schema))


class QueryGraphDataTool:
    """FR-1/FR-2/FR-3 (K-055 M6) — answer an arbitrarily-phrased question
    against structured graph data via the constrained `querygen` DSL (plan
    §3.1/§3.4). Use this when the question isn't one of the fixed catalog
    lookups/filters and isn't free-text-retrievable via `graphrag_retrieve`.

    The model's own function-call arguments are **not** the answer — they
    select a `dataset` and restate the `question`. This tool then makes a
    **second, internal, non-agent-loop** structured-completion LLM call
    (resolved through the same `ModelGateway` `step` kind `_run_agent_node`
    already uses, `executor.py`) whose only allowed output shape is
    `querygen.QueryRequest`, parsed via the same fence-tolerant
    `llm.extract_own_line_json_object` helper `extraction.py` already proved
    for a different feature — reused here, never a second,
    independently-written parser.

    Every failure short of a genuine infrastructure fault — an unparseable or
    schema-invalid structured completion, or a `querygen.compile` rejection
    (e.g. the model named a label/property outside the resolved dataset's
    allowlist) — returns the same abstention shape as "no matching data
    found," never a fabricated answer and never a crash: mirrors
    `evaluate_guard`'s "bias to decline" posture and every other lookup
    tool's abstention convention in this module. A genuine model-resolution
    or provider-call failure (`ModelResolutionError`/`ProviderCallError`) is
    **not** caught here — the same posture `GraphragRetrieveTool` already
    has for its own embedder call: that is an infrastructure fault, not a
    bad model answer, and this module's tools never swallow those.
    """

    name = "query_graph_data"

    def __init__(self, services: Any, *, llm: Any = None, models: Any = None) -> None:
        self._services = services
        # Same FR-4 sugar `GraphragRetrieveTool` already uses: a directly-
        # injected `llm=` stub wraps into a `StaticModelGateway`; production
        # passes the real `models=` gateway instead.
        self._models = models or StaticModelGateway(llm=llm)

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Answer an arbitrarily-phrased question against structured "
                    "graph data — not limited to a fixed catalog lookup or "
                    "filter. Use this when the question needs a fact, a filter, "
                    "or an aggregate (e.g. a count) over one of the datasets "
                    "below, and graphrag_retrieve's free-text retrieval isn't "
                    "the right fit. Available datasets:\n"
                    f"{_describe_all_datasets()}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The natural-language question to answer.",
                        },
                        "dataset": {
                            "type": "string",
                            "enum": list(querygen.DATASET_REGISTRY),
                            "description": "Which dataset to query.",
                        },
                    },
                    "required": ["question", "dataset"],
                },
            },
        }

    def run(self, arguments: dict[str, Any], *, ctx: CallContext,
            run: dict[str, Any]) -> str:
        schema = querygen.DATASET_REGISTRY.get(arguments.get("dataset"))
        if schema is None:
            return json.dumps({"items": [], "finding": "unknown dataset"})

        question = arguments.get("question", "")
        llm = self._models.llm("step", ws=ctx.ws)
        reply = llm.complete([
            {"role": "system", "content": _build_query_request_system_prompt(schema)},
            {"role": "user", "content": question},
        ])
        parsed = extract_own_line_json_object(reply, require_key="matches")
        if parsed is None:
            return json.dumps({"items": [], "finding": "no matching data found"})

        try:
            request = querygen.QueryRequest.model_validate(
                {**parsed, "dataset": arguments["dataset"]}
            )
            compiled = querygen.compile(request, schema)
        except (ValidationError, ValueError):
            return json.dumps({"items": [], "finding": "no matching data found"})

        graph_key = schema.graph_key or f"ws:{ctx.ws}"
        rows = self._services.run_structured_query(
            ctx, graph_key, compiled, timeout=querygen.DEFAULT_QUERY_TIMEOUT_MS,
        )
        if not rows:
            return json.dumps({"items": [], "finding": "no matching data found"})
        return json.dumps({"items": rows})


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
    services: Any, embedder: Embedder | None = None, *, agent_id: str,
    models: Any = None,
    tau: float = DEFAULT_RETRIEVE_TAU, cap: int = DEFAULT_RETRIEVE_CAP,
    k: int = DEFAULT_RETRIEVE_K, channel_id: str | None = None,
) -> ToolRegistry:
    """Wire every built-in capability into a fresh `ToolRegistry` (§4, K-052 M6).

    One shared, process-wide registry serves every workflow def — `human_handoff`
    was already registered (present) even though only `triage` nodes might grant
    it, and the K-052 catalog tools (`lookup_product_fact`/`filter_products`, for
    the disjoint `salesperson` demo) follow the identical posture: the AC-6 fence
    is per-node `config.tools`, not registry membership, so a def that never
    grants a given tool simply never offers it to the model — registering it here
    costs nothing and is the only currently-wired path to a live run (there is no
    per-def registry seam in `app.py`). K-053/K-054/K-055 register their own
    `salesperson`-only tools here too, the same way, as their own clusters land.
    `embedder`/`models` follow `GraphragRetrieveTool`'s own FR-4 sugar (a bare
    `embedder=` wraps into a `StaticModelGateway`; production passes the real
    `models=` gateway instead).
    """
    return ToolRegistry([
        PostMessageTool(services, agent_id=agent_id),
        GraphragRetrieveTool(
            services, embedder, models=models, tau=tau, cap=cap, k=k, channel_id=channel_id
        ),
        HumanHandoffTool(),
        LookupProductFactTool(services),
        FilterProductsTool(services),
        ViewCartTool(services),
        AddToCartTool(services),
        RemoveFromCartTool(services),
        ClearCartTool(services),
        PlaceOrderTool(services),
        GetProfileTool(services),
        SaveProfileTool(services),
        QueryGraphDataTool(services, models=models),
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
