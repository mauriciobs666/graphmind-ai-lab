"""Unit + integration tests for the node-capability tools (U9).

`ToolRegistry` satisfies the exact interface `executor._run_agent_node` calls
(`schema(name)` + `dispatch(name, args, *, ctx, run)`). The built-ins:
  * `post_message` posts as the workflow agent via `services.post_agent_answer` (role
    derived `assistant`) and links `StepRun -[:PRODUCED]-> Message` via
    `services.link_step_emission` when a `stepRunId` is on the run (two-step, non-atomic).
  * `graphrag_retrieve` embeds the query (injected `Embedder`) → `services.hybrid_search`,
    then applies the DS-note Q2 policy (distance cutoff τ, cap 5, abstain on no-hit).
  * `human_handoff` raises `HumanHandoffSignal` (present, not exercised).

Threshold/registry logic is tested with stubs (deterministic, no DB); `post_message`'s §4
write + PRODUCED link and `graphrag_retrieve`'s abstention are exercised against the live
`ws:test` graph (conftest bootstraps schema + wipes per test).
"""

from __future__ import annotations

import itertools
import json

import pytest
from conftest import TEST_EMBEDDING_DIM

from falkorchat import querygen, tools
from falkorchat.config import CallContext
from falkorchat.services import Services, UnknownMemberError
from falkorchat.tools import (
    DEFAULT_FILTER_LIMIT,
    AddToCartTool,
    ClearCartTool,
    FilterProductsTool,
    GetProfileTool,
    GraphragRetrieveTool,
    HumanHandoffSignal,
    HumanHandoffTool,
    LookupProductFactTool,
    PlaceOrderTool,
    PostMessageTool,
    QueryGraphDataTool,
    RemoveFromCartTool,
    SaveProfileTool,
    ToolRegistry,
    UnknownToolError,
    ViewCartTool,
    build_builtin_registry,
)

CTX = CallContext(ws="test", actor="u1")
WS = "test"


# ── stubs ────────────────────────────────────────────────────────────────────

_UNSET = object()  # distinguishes "not scripted" from a scripted `None` result


class StubServices:
    """Records `post_agent_answer` / `link_step_emission` / `hybrid_search` calls."""

    def __init__(self, *, search_rows=None, link_ok=True,
                 products_by_name=None, filter_result=None,
                 add_result=_UNSET, remove_result=_UNSET,
                 cart_result=None, order_result=_UNSET,
                 profile_result=None, structured_query_rows=None, catalog=None,
                 has_category=True, categories=None):
        self._search_rows = search_rows or []
        self._link_ok = link_ok
        self.posted: list[dict] = []
        self.linked: list[dict] = []
        self.searched: list[dict] = []
        self._msg_seq = itertools.count(1)
        # §15 product catalog (K-052 M6)
        self._products_by_name = products_by_name or {}
        self._filter_result = filter_result or []
        self.looked_up: list[str] = []
        self.filtered: list[dict] = []
        # K-078 D2: what `catalog_index` returns on a `lookup_product_fact` miss
        self._catalog = catalog or []
        self.catalog_reads: int = 0
        # K-078 id2 D8.1: what `has_category` answers (default: a known category)
        self._has_category = has_category
        self.category_checks: list = []
        # K-086 D6: what `catalog_categories` answers (default: none, so the
        # querygen prompt and every pre-existing query_graph_data result are
        # unchanged)
        self._categories = [] if categories is None else categories
        self.categories_reads: int = 0
        # §16 Cart / Order (K-053 M6) — set the *_result kwargs to script a
        # call's return value; every call is recorded so the tool's
        # argument-mapping into the service (productId/quantity keyword
        # translation, defaulting) is checkable.
        self._add_result = add_result
        self._remove_result = remove_result
        self._cart_result = cart_result if cart_result is not None else {
            "items": [], "total": 0.0,
        }
        self._order_result = order_result
        self.added: list[dict] = []
        self.removed: list[dict] = []
        self.cart_views: int = 0
        self.cleared: int = 0
        self.ordered: int = 0
        # §17 durable customer profile (K-054 M6)
        self._profile_result = profile_result if profile_result is not None else {
            "name": None, "deliveryAddress": None,
        }
        self.profile_reads: int = 0
        self.saved_profiles: list[dict] = []
        # §18 structured natural-language query generation (K-055 M6)
        self._structured_query_rows = (
            [] if structured_query_rows is None else structured_query_rows
        )
        self.structured_query_calls: list[dict] = []

    def post_agent_answer(self, ctx, *, thread_id, text, mentions=None, seeds=None):
        msg_id = f"m{next(self._msg_seq)}"
        self.posted.append(
            {"actor": ctx.actor, "thread_id": thread_id, "text": text,
             "mentions": mentions}
        )
        return {"msgId": msg_id, "threadId": thread_id, "text": text}

    def link_step_emission(self, ctx, *, step_run_id, msg_id):
        self.linked.append({"step_run_id": step_run_id, "msg_id": msg_id})
        return {"stepRunId": step_run_id, "msgId": msg_id} if self._link_ok else None

    def hybrid_search(self, ctx, *, q_vec, k=10, channel_id=None):
        self.searched.append({"q_vec": q_vec, "k": k, "channel_id": channel_id})
        return list(self._search_rows)

    def lookup_product(self, ctx, *, name):
        self.looked_up.append(name)
        return self._products_by_name.get(name)

    def filter_products(self, ctx, *, category, min_price, max_price, limit):
        self.filtered.append(
            {"category": category, "min_price": min_price,
             "max_price": max_price, "limit": limit}
        )
        return list(self._filter_result)

    def catalog_index(self, ctx):
        self.catalog_reads += 1
        return list(self._catalog)

    def has_category(self, ctx, *, category):
        self.category_checks.append(category)
        return self._has_category

    def catalog_categories(self, ctx):
        self.categories_reads += 1
        return list(self._categories)

    def add_cart_item(self, ctx, *, product_id, quantity):
        self.added.append({"product_id": product_id, "quantity": quantity})
        return None if self._add_result is _UNSET else self._add_result

    def remove_cart_item(self, ctx, *, product_id, quantity=None):
        self.removed.append({"product_id": product_id, "quantity": quantity})
        return None if self._remove_result is _UNSET else self._remove_result

    def get_cart(self, ctx):
        self.cart_views += 1
        return self._cart_result

    def clear_cart(self, ctx):
        self.cleared += 1

    def place_order(self, ctx):
        self.ordered += 1
        return None if self._order_result is _UNSET else self._order_result

    def get_profile(self, ctx):
        self.profile_reads += 1
        return self._profile_result

    def save_profile(self, ctx, *, name=None, delivery_address=None):
        result = {
            "name": name if name is not None else self._profile_result.get("name"),
            "deliveryAddress": (
                delivery_address if delivery_address is not None
                else self._profile_result.get("deliveryAddress")
            ),
        }
        self.saved_profiles.append({"name": name, "delivery_address": delivery_address})
        self._profile_result = result
        return result

    def run_structured_query(self, ctx, graph_key, compiled, *, timeout=None):
        self.structured_query_calls.append(
            {"graph_key": graph_key, "compiled": compiled, "timeout": timeout}
        )
        return list(self._structured_query_rows)


class StubEmbedder:
    def __init__(self, vec=None):
        self._vec = vec or [1.0, 0.0, 0.0, 0.0]
        self.calls: list[str] = []

    def embed(self, text):
        self.calls.append(text)
        return list(self._vec)


class StubQueryLLM:
    """Stub LLM returning one canned reply, recording the messages it was
    sent — mirrors `test_extraction.py`'s `_ReplyLLM` (same seam: `.complete`
    is what a `querygen`-filling structured-completion call uses)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self._text


class _FakeTool:
    name = "echo"
    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}

    def run(self, arguments, *, ctx, run):
        return f"echo:{arguments.get('v')}"


# ── ToolRegistry contract (matches _run_agent_node) ──────────────────────────

def test_registry_schema_and_dispatch_roundtrip():
    reg = ToolRegistry([_FakeTool()])
    assert reg.names() == ["echo"]
    assert reg.schema("echo")["function"]["name"] == "echo"
    out = reg.dispatch("echo", {"v": 7}, ctx=CTX, run={})
    assert out == "echo:7"


def test_registry_unknown_tool_raises_unknown_tool_error():
    reg = ToolRegistry([])
    with pytest.raises(UnknownToolError):
        reg.schema("nope")
    with pytest.raises(UnknownToolError):
        reg.dispatch("nope", {}, ctx=CTX, run={})
    # KeyError-compatible so bare-dict stub registries stay consistent (U8)
    assert issubclass(UnknownToolError, KeyError)


def test_registry_jsonifies_nonstring_results():
    class DictTool:
        name = "d"
        schema = {"type": "function", "function": {"name": "d", "parameters": {}}}

        def run(self, arguments, *, ctx, run):
            return {"a": 1}

    reg = ToolRegistry([DictTool()])
    assert json.loads(reg.dispatch("d", {}, ctx=CTX, run={})) == {"a": 1}


def test_build_builtin_registry_registers_all_thirteen():
    # K-052/K-053/K-054/K-055 M6: the salesperson-demo catalog + cart/order +
    # profile + structured-query tools are all registered into this same
    # shared registry (the AC-6 fence is per-node config.tools, not registry
    # membership — the same "present, only offered where granted" posture
    # human_handoff already established for triage).
    reg = build_builtin_registry(StubServices(), StubEmbedder(), agent_id="assistant")
    assert set(reg.names()) == {
        "post_message", "graphrag_retrieve", "human_handoff",
        "lookup_product_fact", "filter_products",
        "view_cart", "add_to_cart", "remove_from_cart", "clear_cart", "place_order",
        "get_profile", "save_profile", "query_graph_data",
    }


# ── post_message (unit, stub services) ───────────────────────────────────────

def test_post_message_posts_as_agent_and_returns_posted_msg_id():
    # Option B (K-023): the tool posts as the agent and returns the msgId in its
    # envelope; it does NOT link the emission (the executor owns PRODUCED linking
    # after the StepRun exists).
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    run = {"runId": "r1", "ctx": json.dumps({"threadId": "t1"})}

    out = json.loads(tool.run({"text": "hello"}, ctx=CTX, run=run))

    # posted as the agent (actor swapped), role derives assistant in the service
    assert svc.posted == [
        {"actor": "assistant", "thread_id": "t1", "text": "hello", "mentions": None}
    ]
    # the tool no longer links — that is the executor's job (Option B)
    assert svc.linked == []
    assert out["posted"]                 # the posted msgId, for the executor to link
    assert out["threadId"] == "t1"
    assert "linked" not in out           # no linked flag — linking moved to the executor


def test_post_message_does_not_link_even_with_step_run_id_on_run():
    # Even if a stepRunId happens to be on the run dict, the tool never links
    # (Option B decouples the tool from audit linking).
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    run = {"ctx": json.dumps({"threadId": "t1"}), "stepRunId": "sr1"}

    tool.run({"text": "hi"}, ctx=CTX, run=run)

    assert len(svc.posted) == 1
    assert svc.linked == []


def test_post_message_reads_thread_from_explicit_run_key():
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    tool.run({"text": "x"}, ctx=CTX, run={"threadId": "tX"})
    assert svc.posted[0]["thread_id"] == "tX"


def test_post_message_errors_without_a_bound_thread():
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    out = tool.run({"text": "x"}, ctx=CTX, run={"runId": "r1"})
    assert "no thread" in out
    assert svc.posted == []


def test_post_message_forwards_mentions():
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    tool.run({"text": "hey", "mentions": ["u2"]}, ctx=CTX,
             run={"threadId": "t1"})
    assert svc.posted[0]["mentions"] == ["u2"]


def test_post_message_returns_actionable_error_for_unknown_mentions():
    # Defect B at the source: the model reads `Alice: …` off the folded thread context
    # and passes the *displayName* as a mention id. The §4 write rejects it with
    # UnknownMemberError. The tool turns that into an actionable error string — the
    # same shape as its no-thread refusal — so the model can retry, and so the failure
    # never leaves the tool as an exception at all (defense in depth with the executor's
    # own tool-error net).
    class _RejectingServices(StubServices):
        def post_agent_answer(self, ctx, *, thread_id, text, mentions=None, seeds=None):
            raise UnknownMemberError(["alice"])

    svc = _RejectingServices()
    tool = PostMessageTool(svc, agent_id="assistant")

    out = tool.run({"text": "hey", "mentions": ["alice"]}, ctx=CTX,
                   run={"threadId": "t1"})

    assert out.startswith("error:")
    assert "alice" in out
    # the model is told WHY, so the re-prompt is a correction and not a blind retry
    assert "member id" in out


def test_post_message_lets_unrelated_service_errors_propagate():
    # Only the mention-resolution failure is translated here. Anything else keeps
    # escaping the tool — the executor's tool-error net (or the M-1 net) classifies it.
    class _BoomServices(StubServices):
        def post_agent_answer(self, ctx, *, thread_id, text, mentions=None, seeds=None):
            raise RuntimeError("engine exploded")

    tool = PostMessageTool(_BoomServices(), agent_id="assistant")
    with pytest.raises(RuntimeError, match="engine exploded"):
        tool.run({"text": "hey"}, ctx=CTX, run={"threadId": "t1"})


# ── graphrag_retrieve (unit, stub services + embedder) ───────────────────────

def _row(msg_id, score, text="ctx"):
    return {"msgId": msg_id, "text": text, "role": "user", "score": score,
            "relatedContext": [], "seedKind": "Message"}


def _chunk_row(chunk_id, score, text="chunk ctx", document_id="d1"):
    """A `Chunk`-shaped merged row, mirroring `services.hybrid_search`'s own
    tagging (K-050 M5 Stage 5) — no `msgId`/`role` key, `seedKind: 'Chunk'`.
    """
    return {"chunkId": chunk_id, "text": text, "documentId": document_id,
            "seq": 0, "score": score, "relatedContext": [], "seedKind": "Chunk"}


def test_graphrag_retrieve_embeds_then_applies_tau_cutoff_and_cap():
    rows = [_row("a", 0.0), _row("b", 0.4), _row("c", 0.6), _row("d", 0.9)]
    emb = StubEmbedder([0.5, 0.5, 0.0, 0.0])
    svc = StubServices(search_rows=rows)
    tool = GraphragRetrieveTool(svc, emb, tau=0.5, cap=5)

    out = json.loads(tool.run({"query": "reset password"}, ctx=CTX, run={}))

    assert emb.calls == ["reset password"]            # query embedded, not a raw vector
    assert svc.searched[0]["q_vec"] == [0.5, 0.5, 0.0, 0.0]
    # only seeds within τ=0.5 survive (a,b), ordering preserved; c,d cut
    assert [s["seedId"] for s in out["seeds"]] == ["a", "b"]
    assert "finding" not in out


def test_graphrag_retrieve_caps_at_five():
    rows = [_row(str(i), 0.1) for i in range(8)]  # all pass τ
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.5, cap=5)
    out = json.loads(tool.run({"query": "q"}, ctx=CTX, run={}))
    assert len(out["seeds"]) == 5


def test_graphrag_retrieve_abstains_when_nothing_passes_tau():
    rows = [_row("a", 0.7), _row("b", 0.9)]  # all beyond τ
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.5)
    out = json.loads(tool.run({"query": "off topic"}, ctx=CTX, run={}))
    assert out["seeds"] == []
    assert out["finding"] == "no relevant context found"


def test_graphrag_retrieve_configurable_tau():
    rows = [_row("a", 0.7)]
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.8)  # looser cutoff keeps 0.7
    out = json.loads(tool.run({"query": "q"}, ctx=CTX, run={}))
    assert [s["seedId"] for s in out["seeds"]] == ["a"]


def test_graphrag_retrieve_resolves_chunk_seed_id_and_document_id():
    """The BLOCKER regression this pins: `services.hybrid_search` can return a
    `Chunk`-shaped row (no `msgId` key at all) — the tool must resolve
    `seedId` via `chunkId` instead of crashing with `KeyError: 'msgId'`, and
    surface `documentId` for it.
    """
    rows = [_chunk_row("ch1", 0.1, text="chunk text", document_id="d1")]
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.5)
    out = json.loads(tool.run({"query": "q"}, ctx=CTX, run={}))
    assert out["seeds"] == [
        {"seedId": "ch1", "text": "chunk text", "score": 0.1, "documentId": "d1"}
    ]


def test_graphrag_retrieve_mixed_message_and_chunk_seeds():
    rows = [_row("m1", 0.0, text="msg text"), _chunk_row("ch1", 0.1, text="chunk text")]
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.5)
    out = json.loads(tool.run({"query": "q"}, ctx=CTX, run={}))
    assert [(s["seedId"], s["documentId"]) for s in out["seeds"]] == [
        ("m1", None), ("ch1", "d1"),
    ]


# ── K-042 code review Major 2 (the M-3 fix itself): resolution happens INSIDE run() ──
#
# Every construction above passes a bare `embedder=` — the FR-4 *sugar* path
# (`StaticModelGateway`), which is indistinguishable from the pre-K-042
# bound-at-construction `Embedder` unless something actually calls `.embedder()`
# through the gateway seam. This drives `GraphragRetrieveTool` with a recording
# `models=` double instead, so a regression back to "resolve once at __init__" (the
# exact bug M-3 exists to prevent) would fail here: the double hands back a
# DIFFERENT embedder on each call, keyed by `ctx.ws`, and only an inside-`run()`
# resolution can possibly pick the right one per call.

class RecordingGateway:
    """A minimal `ModelGateway`-shaped double: records every `.embedder()` call's
    `(kind, ws)` and returns a `StubEmbedder` keyed by `ws` — so two calls with two
    different workspaces are provably resolved independently, never cached from
    construction."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self._embedders: dict[str, StubEmbedder] = {}

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        self.calls.append((kind, ws))
        if ws not in self._embedders:
            self._embedders[ws] = StubEmbedder([float(len(self._embedders) + 1), 0.0, 0.0, 0.0])
        return self._embedders[ws]


def test_graphrag_retrieve_resolves_the_embedder_inside_run_not_at_construction():
    rows = [_row("a", 0.0)]
    svc = StubServices(search_rows=rows)
    gateway = RecordingGateway()
    tool = GraphragRetrieveTool(svc, models=gateway, tau=0.5)

    # Nothing resolved yet — construction alone must not touch the gateway.
    assert gateway.calls == []

    tool.run({"query": "q1"}, ctx=CTX, run={})
    assert gateway.calls == [("embedding", CTX.ws)]

    other_ctx = CallContext(ws="other-ws", actor="u1")
    tool.run({"query": "q2"}, ctx=other_ctx, run={})

    # A second call, with a different ctx.ws, resolves again (and independently) —
    # exactly what a bound-at-construction embedder could never do.
    assert gateway.calls == [("embedding", CTX.ws), ("embedding", "other-ws")]
    assert svc.searched[0]["q_vec"] == [1.0, 0.0, 0.0, 0.0]
    assert svc.searched[1]["q_vec"] == [2.0, 0.0, 0.0, 0.0]


# ── human_handoff (present, not exercised) ───────────────────────────────────

def test_human_handoff_dispatch_signals_suspend():
    reg = ToolRegistry([HumanHandoffTool()])
    with pytest.raises(HumanHandoffSignal) as exc:
        reg.dispatch("human_handoff", {"reason": "needs a person"}, ctx=CTX, run={})
    assert exc.value.reason == "needs a person"


# ── lookup_product_fact / filter_products (K-052 M6, unit, stub services) ────

def test_lookup_product_fact_returns_found_row():
    svc = StubServices(products_by_name={
        "Bluetooth Speaker": {"productId": "bluetooth-speaker",
                              "name": "Bluetooth Speaker", "category": "audio",
                              "price": 89.99},
    })
    tool = LookupProductFactTool(svc)

    out = json.loads(tool.run({"name": "Bluetooth Speaker"}, ctx=CTX, run={}))

    assert out == {"found": True, "productId": "bluetooth-speaker",
                   "name": "Bluetooth Speaker", "category": "audio", "price": 89.99}
    assert svc.looked_up == ["Bluetooth Speaker"]
    assert svc.catalog_reads == 0  # a hit never lists the catalog


def test_lookup_product_fact_miss_lists_the_catalog_with_a_hint():
    """K-078 D2: a miss returns `found: false`, a non-empty `hint`, and the
    catalog as `{productId, name}` rows only — never a price."""
    catalog = [
        {"productId": "bluetooth-speaker", "name": "Bluetooth Speaker"},
        {"productId": "wireless-mouse-pro", "name": "Wireless Mouse Pro"},
    ]
    svc = StubServices(catalog=catalog)
    tool = LookupProductFactTool(svc)

    out = json.loads(tool.run({"name": "Mouse sem fio Pro"}, ctx=CTX, run={}))

    assert out["found"] is False
    assert isinstance(out["hint"], str) and out["hint"]
    assert "lookup_product_fact" in out["hint"]
    assert out["catalog"] == catalog
    assert all(set(row) == {"productId", "name"} for row in out["catalog"])
    assert set(out) == {"found", "hint", "catalog"}


def test_lookup_product_fact_defaults_missing_name_to_empty_string():
    svc = StubServices()
    tool = LookupProductFactTool(svc)
    tool.run({}, ctx=CTX, run={})
    assert svc.looked_up == [""]


def test_filter_products_returns_items():
    rows = [{"name": "Bluetooth Speaker", "category": "audio", "price": 89.99}]
    svc = StubServices(filter_result=rows)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run(
        {"category": "audio", "minPrice": 50.0, "maxPrice": 150.0}, ctx=CTX, run={},
    ))

    assert out == {"items": rows}
    assert svc.filtered == [
        {"category": "audio", "min_price": 50.0, "max_price": 150.0,
         "limit": DEFAULT_FILTER_LIMIT}
    ]
    # id2 plan §8 R-E: a non-empty result never pays the category re-check
    # or the catalog listing (impl review m1)
    assert svc.category_checks == []
    assert svc.catalog_reads == 0


# K-078 id2 D8.2: an empty result whose `category` the catalog doesn't know
# carries D2's recovery (a filter-specific hint and the price-free catalog).
# This pair replaces the old `test_filter_products_abstains_when_nothing_matches`,
# whose `"nonexistent"` category is now the miss branch.

_MISS_CATALOG = [
    {"productId": "bluetooth-speaker", "name": "Bluetooth Speaker"},
    {"productId": "wireless-mouse-pro", "name": "Wireless Mouse Pro"},
]


def _assert_category_miss(out, svc):
    assert out == {
        "items": [],
        "finding": tools.FILTER_CATEGORY_MISS_FINDING,
        "hint": tools.FILTER_CATEGORY_MISS_HINT,
        "catalog": _MISS_CATALOG,
    }
    assert all(set(row) == {"productId", "name"} for row in out["catalog"])
    assert svc.catalog_reads == 1


def test_filter_products_unknown_category_lists_the_catalog_with_a_hint():
    """id2 §6.1 #30, the DEF-2 reproduction: S7's pt-BR call shape — a
    translated product name sent as `category`, with price bounds."""
    svc = StubServices(filter_result=[], has_category=False, catalog=_MISS_CATALOG)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run(
        {"category": "Mouse sem fio Pro", "minPrice": 0, "maxPrice": 100},
        ctx=CTX, run={},
    ))

    _assert_category_miss(out, svc)
    assert svc.category_checks == ["Mouse sem fio Pro"]


def test_filter_products_unknown_category_without_price_bounds_is_a_miss_too():
    """id2 §6.1 #31."""
    svc = StubServices(filter_result=[], has_category=False, catalog=_MISS_CATALOG)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run({"category": "Mouse sem fio"}, ctx=CTX, run={}))

    _assert_category_miss(out, svc)
    assert svc.category_checks == ["Mouse sem fio"]


def test_filter_products_known_category_emptied_by_price_keeps_the_old_abstention():
    """id2 §6.1 #32 (the rewritten half of the old abstention test): a known
    category with nothing in range is byte-identical to the pre-id2 result."""
    svc = StubServices(filter_result=[], has_category=True, catalog=_MISS_CATALOG)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run(
        {"category": "Peripherals", "maxPrice": 1}, ctx=CTX, run={},
    ))

    assert out == {"items": [], "finding": "no matching products found"}
    assert svc.category_checks == ["Peripherals"]
    assert svc.catalog_reads == 0


def test_filter_products_no_or_blank_category_keeps_the_old_abstention():
    """id2 §6.1 #33: a price-only miss never asks `has_category` (contract
    pin, N6); a blank category reaches it raw, the repository answers it as
    omitted (`True`), and the old shape stays."""
    svc = StubServices(filter_result=[], has_category=False, catalog=_MISS_CATALOG)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run({"maxPrice": 1}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching products found"}
    assert svc.category_checks == []

    svc = StubServices(filter_result=[], has_category=True, catalog=_MISS_CATALOG)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run({"category": ""}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching products found"}
    assert svc.category_checks == [""]


def test_filter_category_miss_hint_is_written_for_filter_products():
    """id2 §6.1 #34: names the pick (`productId`) and the price route
    (`lookup_product_fact`), never says "again" (no lookup preceded it), and
    carries no price (K-079)."""
    hint = tools.FILTER_CATEGORY_MISS_HINT

    assert "lookup_product_fact" in hint and "productId" in hint
    assert "again" not in hint
    assert "$" not in hint and not any(c.isdigit() for c in hint)
    assert hint != tools.LOOKUP_MISS_HINT


def test_filter_products_all_arguments_optional():
    rows = [{"name": "Wireless Mouse", "category": "accessories", "price": 25.0}]
    svc = StubServices(filter_result=rows)
    tool = FilterProductsTool(svc)

    out = json.loads(tool.run({}, ctx=CTX, run={}))

    assert out == {"items": rows}
    assert svc.filtered == [
        {"category": None, "min_price": None, "max_price": None,
         "limit": DEFAULT_FILTER_LIMIT}
    ]


def test_filter_products_blank_price_bounds_are_treated_as_omitted():
    """Sibling of `salesperson-fabricated-prices-rca.md` CF-1: a model that
    sends `""` for "no category" can send `""` for "no price bound". A string
    reaching `p.price >= coalesce($minPrice, -1.0)` is not null, so `coalesce`
    keeps it and the number-vs-string comparison matches nothing (live-checked:
    0 of 15 catalog rows). A blank/whitespace-only price string must reach
    services as `None`, exactly like an omitted bound.
    """
    for blank in ("", " ", "\t", " \n "):
        svc = StubServices(filter_result=[])
        tool = FilterProductsTool(svc)

        tool.run({"minPrice": blank, "maxPrice": blank}, ctx=CTX, run={})

        assert svc.filtered == [
            {"category": None, "min_price": None, "max_price": None,
             "limit": DEFAULT_FILTER_LIMIT}
        ], f"price bound {blank!r} did not behave as omitted"


def test_filter_products_numeric_string_price_bounds_become_numbers():
    """`docs/reviews/salesperson-fabricated-prices-impl.md` m1: `maxPrice: "50"`
    reaches `coalesce('50', …)` and matches 0 rows where the number 50 matches 6
    (live-verified). A string that parses as a finite number must reach services
    as that `float`.
    """
    for raw, expected in (("50", 50.0), (" 49.99 ", 49.99), ("0", 0.0), ("1e2", 100.0)):
        svc = StubServices(filter_result=[])
        tool = FilterProductsTool(svc)

        tool.run({"minPrice": raw, "maxPrice": raw}, ctx=CTX, run={})

        call = svc.filtered[0]
        assert (call["min_price"], call["max_price"]) == (expected, expected), raw
        assert isinstance(call["min_price"], float) and isinstance(call["max_price"], float)


def test_filter_products_non_numeric_price_bound_passes_through_unchanged():
    """m1's other half: no new error path. A string that is not a finite number
    (incl. `"nan"`/`"inf"`, which `float()` would accept) is left as is; numbers
    are untouched.
    """
    for raw in ("cheap", "$50", "nan", "inf", 50, 49.99):
        svc = StubServices(filter_result=[])
        tool = FilterProductsTool(svc)

        tool.run({"maxPrice": raw}, ctx=CTX, run={})

        assert svc.filtered[0]["max_price"] is raw, raw


# ── view_cart / add_to_cart / remove_from_cart / clear_cart / place_order ────
# (K-053 M6, unit, stub services)

def test_view_cart_returns_the_services_cart_shape():
    cart = {"items": [{"productId": "prod2", "name": "Bluetooth Speaker",
                        "price": 89.99, "quantity": 2, "lineTotal": 179.98}],
            "total": 179.98}
    svc = StubServices(cart_result=cart)
    tool = ViewCartTool(svc)

    out = json.loads(tool.run({}, ctx=CTX, run={}))

    assert out == cart
    assert svc.cart_views == 1


def test_add_to_cart_returns_found_row():
    svc = StubServices(add_result={
        "productId": "prod2", "name": "Bluetooth Speaker", "price": 89.99,
        "quantity": 2,
    })
    tool = AddToCartTool(svc)

    out = json.loads(tool.run(
        {"productId": "prod2", "quantity": 2}, ctx=CTX, run={},
    ))

    assert out == {
        "found": True, "productId": "prod2", "name": "Bluetooth Speaker",
        "price": 89.99, "quantity": 2,
    }
    assert svc.added == [{"product_id": "prod2", "quantity": 2}]


def test_add_to_cart_abstains_with_a_reason_when_product_unknown():
    svc = StubServices()  # add_result unset -> add_cart_item returns None
    tool = AddToCartTool(svc)

    out = json.loads(tool.run(
        {"productId": "nonexistent-gadget", "quantity": 1}, ctx=CTX, run={},
    ))

    assert out == {
        "found": False,
        "reason": "no catalog product has productId 'nonexistent-gadget'",
    }


def test_add_to_cart_ignores_an_extra_product_name():
    """K-078 D1: `productName` is gone; a stray one next to a `productId` is
    ignored — only the ID reaches the service."""
    svc = StubServices(add_result={"productId": "p", "name": "X", "price": 1.0,
                                   "quantity": 1})
    tool = AddToCartTool(svc)

    tool.run({"productId": "p", "productName": "Something Else"}, ctx=CTX, run={})

    assert svc.added == [{"product_id": "p", "quantity": 1}]


def test_add_to_cart_defaults_missing_quantity_to_one():
    svc = StubServices(add_result={"productId": "p", "name": "X", "price": 1.0,
                                    "quantity": 1})
    tool = AddToCartTool(svc)

    tool.run({"productId": "p"}, ctx=CTX, run={})

    assert svc.added == [{"product_id": "p", "quantity": 1}]


def test_remove_from_cart_partial_quantity():
    svc = StubServices(remove_result={
        "removed": False, "productId": "prod2", "name": "Bluetooth Speaker",
        "quantity": 3,
    })
    tool = RemoveFromCartTool(svc)

    out = json.loads(tool.run(
        {"productId": "prod2", "quantity": 2}, ctx=CTX, run={},
    ))

    assert out == {
        "found": True, "removed": False, "productId": "prod2",
        "name": "Bluetooth Speaker", "quantity": 3,
    }
    assert svc.removed == [{"product_id": "prod2", "quantity": 2}]


def test_remove_from_cart_omitted_quantity_passes_none_through():
    svc = StubServices(remove_result={
        "removed": True, "productId": "prod2", "name": "Bluetooth Speaker",
    })
    tool = RemoveFromCartTool(svc)

    tool.run({"productId": "prod2"}, ctx=CTX, run={})

    assert svc.removed == [{"product_id": "prod2", "quantity": None}]


def test_remove_from_cart_abstains_with_a_reason_when_product_unknown():
    svc = StubServices()  # remove_result unset -> remove_cart_item returns None
    tool = RemoveFromCartTool(svc)

    out = json.loads(tool.run({"productId": "nonexistent-gadget"}, ctx=CTX, run={}))

    assert out == {
        "found": False,
        "reason": "no catalog product has productId 'nonexistent-gadget'",
    }


def _schema_text(tool) -> str:
    """Every model-visible string in a tool schema: its description plus the
    `productId` parameter's description (K-078 review F3)."""
    fn = tool.schema["function"]
    return fn["description"] + " " + fn["parameters"]["properties"]["productId"][
        "description"
    ]


@pytest.mark.parametrize("tool_cls", [AddToCartTool, RemoveFromCartTool])
def test_cart_write_schemas_require_product_id_and_have_no_product_name(tool_cls):
    params = tool_cls(StubServices()).schema["function"]["parameters"]

    assert params["required"] == ["productId"]
    assert params["properties"]["productId"]["type"] == "string"
    assert "productName" not in params["properties"]
    assert set(params["properties"]) == {"productId", "quantity"}


def test_add_to_cart_schema_never_names_view_cart_as_a_source():
    """K-078 D1/D3: a `view_cart` ID does not ground an add, so the add
    schema must not tell the model it does; the remove schema does."""
    add_text = _schema_text(AddToCartTool(StubServices()))
    remove_text = _schema_text(RemoveFromCartTool(StubServices()))

    assert "view_cart" not in add_text
    assert "lookup_product_fact" in add_text and "filter_products" in add_text
    assert "view_cart" in remove_text
    assert "lookup_product_fact" in remove_text and "filter_products" in remove_text


def test_clear_cart_calls_services_and_confirms():
    svc = StubServices()
    tool = ClearCartTool(svc)

    out = json.loads(tool.run({}, ctx=CTX, run={}))

    assert out == {"cleared": True}
    assert svc.cleared == 1


def test_place_order_returns_json_receipt_on_success():
    receipt = {"orderId": "o1", "created": True, "lineCount": 1,
               "lines": [{"productId": "prod2", "name": "Bluetooth Speaker",
                          "unitPrice": 89.99, "quantity": 1, "lineTotal": 89.99}],
               "total": 89.99}
    svc = StubServices(order_result=receipt)
    tool = PlaceOrderTool(svc)

    out = json.loads(tool.run({}, ctx=CTX, run={}))

    assert out == receipt
    assert svc.ordered == 1


def test_place_order_on_empty_cart_returns_an_explanatory_string_not_json():
    svc = StubServices()  # order_result unset -> place_order returns None
    tool = PlaceOrderTool(svc)

    out = tool.run({}, ctx=CTX, run={})

    assert isinstance(out, str)
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


# ── get_profile / save_profile (K-054 M6, unit, stub services) ──────────────

def test_get_profile_returns_both_fields_none_when_no_profile_yet():
    svc = StubServices()  # profile_result defaults to {"name": None, "deliveryAddress": None}
    tool = GetProfileTool(svc)

    out = json.loads(tool.run({}, ctx=CTX, run={}))

    assert out == {"name": None, "deliveryAddress": None}
    assert svc.profile_reads == 1


def test_get_profile_returns_the_services_stored_shape():
    svc = StubServices(profile_result={
        "name": "Alice", "deliveryAddress": "123 Main St",
    })
    tool = GetProfileTool(svc)

    out = json.loads(tool.run({}, ctx=CTX, run={}))

    assert out == {"name": "Alice", "deliveryAddress": "123 Main St"}


def test_save_profile_full_write_passes_both_fields_through():
    svc = StubServices()
    tool = SaveProfileTool(svc)

    out = json.loads(tool.run(
        {"name": "Alice", "deliveryAddress": "123 Main St"}, ctx=CTX, run={},
    ))

    assert out == {"name": "Alice", "deliveryAddress": "123 Main St"}
    assert svc.saved_profiles == [
        {"name": "Alice", "delivery_address": "123 Main St"},
    ]


def test_save_profile_omitted_field_is_passed_through_as_none_not_dropped():
    """AC-2, at the tool layer: an omitted argument must reach `services.save_profile`
    as `None` (its "not provided, leave unchanged" signal), not be silently
    coerced to `""` or dropped from the call entirely."""
    svc = StubServices(profile_result={
        "name": "Alice", "deliveryAddress": "123 Main St",
    })
    tool = SaveProfileTool(svc)

    out = json.loads(tool.run({"deliveryAddress": "456 New Ave"}, ctx=CTX, run={}))

    assert out == {"name": "Alice", "deliveryAddress": "456 New Ave"}
    assert svc.saved_profiles == [
        {"name": None, "delivery_address": "456 New Ave"},
    ]


# ── query_graph_data (K-055 M6, unit, stub services + stub LLM) ─────────────

_CATALOG_FILTER_REPLY = (
    '{"matches": [{"var": "p", "label": "Product", '
    '"filters": [{"property": "category", "op": "=", "value": "Audio"}]}], '
    '"returns": ["p.name", "p.price"]}'
)


def test_query_graph_data_unknown_dataset_abstains_without_calling_the_llm():
    svc = StubServices()
    llm = StubQueryLLM(_CATALOG_FILTER_REPLY)
    tool = QueryGraphDataTool(svc, llm=llm)

    out = json.loads(
        tool.run({"question": "what audio products?", "dataset": "nope"}, ctx=CTX, run={})
    )

    assert out == {"items": [], "finding": "unknown dataset"}
    assert llm.calls == []  # never resolves/calls the model for an unknown dataset


def test_query_graph_data_happy_path_compiles_and_returns_items():
    rows = [{"p.name": "Bluetooth Speaker", "p.price": 49.99}]
    svc = StubServices(structured_query_rows=rows)
    llm = StubQueryLLM(_CATALOG_FILTER_REPLY)
    tool = QueryGraphDataTool(svc, llm=llm)

    out = json.loads(
        tool.run({"question": "what audio products do you have?", "dataset": "catalog"},
                 ctx=CTX, run={})
    )

    assert out == {"items": rows}
    # the internal structured-completion call actually happened
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == {"role": "user", "content": "what audio products do you have?"}
    # compiled against the resolved (reference-graph) catalog schema, default timeout
    [call] = svc.structured_query_calls
    assert call["graph_key"] == "reference"
    assert call["timeout"] == querygen.DEFAULT_QUERY_TIMEOUT_MS
    assert call["compiled"].cypher == (
        "MATCH (p:Product) WHERE p.category = $p0 RETURN DISTINCT p.name, p.price LIMIT $limit"
    )
    assert call["compiled"].params == {"p0": "Audio", "limit": 20}


def test_query_graph_data_workspace_scoped_dataset_resolves_graph_key_from_ctx():
    rows = [{"e.type": "Organization"}]
    svc = StubServices(structured_query_rows=rows)
    reply = (
        '{"matches": [{"var": "e", "label": "Entity", '
        '"filters": [{"property": "nameNormalized", "op": "=", "value": "acme corp"}]}], '
        '"returns": ["e.type"]}'
    )
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(
        tool.run({"question": "what type of entity is Acme Corp?", "dataset": "knowledge_base"},
                 ctx=CTX, run={})
    )

    assert out == {"items": rows}
    [call] = svc.structured_query_calls
    assert call["graph_key"] == f"ws:{CTX.ws}"  # knowledge_base is workspace-scoped


def test_query_graph_data_abstains_on_unparseable_reply():
    svc = StubServices()
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM("sorry, I'm not sure how to answer that"))

    out = json.loads(tool.run({"question": "anything", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []  # never reaches compile/execute


def test_query_graph_data_abstains_on_pydantic_validation_failure():
    # `extra="forbid"` rejection — a smuggled/unexpected field.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "filters": []}], '
        '"returns": ["p.name"], "raw_cypher": "MATCH (n) DETACH DELETE n"}'
    )
    svc = StubServices()
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "anything", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


def test_query_graph_data_abstains_on_compile_rejection_for_unregistered_label():
    # Valid at the Pydantic layer (label is a plain str there — no per-dataset
    # allowlist until `querygen.compile`), but not a label `CATALOG_SCHEMA`
    # registers — `compile()`'s own ValueError must be absorbed here, never
    # propagate as a crash.
    reply = '{"matches": [{"var": "p", "label": "SecretInternalTable", "filters": []}], "returns": ["p.name"]}'
    svc = StubServices()
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "anything", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


def test_query_graph_data_abstains_on_compile_rejection_for_duplicate_returns():
    # analyst review MAJOR 2: a duplicate `returns` expression (a plausible
    # small-model completion) used to compile successfully and only fail at
    # FalkorDB execution time, propagating uncaught through this tool and
    # failing the whole workflow run — contradicting this tool's own "never a
    # crash" contract. `querygen.compile`'s new uniqueness guard raises
    # `ValueError` before execution, which this try/except already absorbs.
    reply = '{"matches": [{"var": "p", "label": "Product", "filters": []}], "returns": ["p.name", "p.name"]}'
    svc = StubServices()
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "anything", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []  # never reaches execution


def test_query_graph_data_abstains_when_rows_come_back_empty():
    svc = StubServices(structured_query_rows=[])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(_CATALOG_FILTER_REPLY))

    out = json.loads(
        tool.run({"question": "what audio products?", "dataset": "catalog"}, ctx=CTX, run={})
    )

    assert out == {"items": [], "finding": "no matching data found"}


def test_query_graph_data_ignores_a_dataset_field_the_model_supplied_and_uses_the_arguments_one():
    # The internal prompt never asks the model to restate `dataset`, but a
    # tolerant parse could still see one if the model added it anyway — the
    # tool must force the outer call's own `dataset` argument regardless.
    reply = (
        '{"dataset": "knowledge_base", "matches": [{"var": "p", "label": "Product", '
        '"filters": []}], "returns": ["p.name"]}'
    )
    rows = [{"p.name": "Widget"}]
    svc = StubServices(structured_query_rows=rows)
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "anything", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": rows}
    [call] = svc.structured_query_calls
    assert call["graph_key"] == "reference"  # catalog's graph_key, not knowledge_base's


# K-066 Q2 (`docs/test-reports/salesperson-single-model-report.md`): ministral's
# verbatim completion for golden item `nlq-12` — a two-condition filter spelled
# as the `filters` key REPEATED inside one match object, fenced, no `returns`.
_NLQ12_MINISTRAL_COMPLETION = """```json
{
  "matches": [
    {
      "var": "p",
      "label": "Product",
      "filters": [{"property": "category", "op": "=", "value": "Wearables"}],
      "filters": [{"property": "price", "op": "<", "value": 200}]
    }
  ]
}
```"""


def test_query_graph_data_ands_repeated_filters_keys_from_the_verbatim_nlq12_completion():
    rows = [{"p.name": "Fitness Tracker Band"}]
    svc = StubServices(structured_query_rows=rows)
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(_NLQ12_MINISTRAL_COMPLETION))

    out = json.loads(tool.run(
        {"question": "Show me Wearables products that cost under $200.", "dataset": "catalog"},
        ctx=CTX, run={},
    ))

    assert out == {"items": rows}
    [call] = svc.structured_query_calls
    # Both repeated `filters` arrays survive, ANDed — the category condition is
    # not dropped by last-wins — and the absent `returns` defaults to the name.
    assert call["compiled"].cypher == (
        "MATCH (p:Product) WHERE p.category = $p0 AND p.price < $p1 "
        "RETURN DISTINCT p.name LIMIT $limit"
    )
    assert call["compiled"].params == {"p0": "Wearables", "p1": 200.0, "limit": 20}


def test_query_graph_data_ands_repeated_filters_keys_when_returns_is_present():
    # Isolates the duplicate-key merge from the `returns` default.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", '
        '"filters": [{"property": "category", "op": "=", "value": "Wearables"}], '
        '"filters": [{"property": "price", "op": "<", "value": 200}]}], '
        '"returns": ["p.name"]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "Fitness Tracker Band"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert "p.category = $p0 AND p.price < $p1" in call["compiled"].cypher
    assert call["compiled"].params["p0"] == "Wearables"


def test_query_graph_data_defaults_absent_returns_to_the_match_name():
    # Isolates the `returns` default from the duplicate-key merge.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", '
        '"filters": [{"property": "category", "op": "=", "value": "Audio"}]}]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "Bluetooth Speaker Mini"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert call["compiled"].cypher.endswith("RETURN DISTINCT p.name LIMIT $limit")


def test_query_graph_data_still_abstains_on_absent_returns_when_the_label_has_no_name():
    # `Chunk` registers no `name` property — no default exists to fall back
    # on, so the absent `returns` stays a validation failure (abstention).
    reply = '{"matches": [{"var": "c", "label": "Chunk", "filters": []}]}'
    svc = StubServices(structured_query_rows=[{"c.text": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "knowledge_base"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


def test_query_graph_data_abstains_not_crashes_on_absent_returns_with_a_non_string_label():
    # The `returns` default must not trip over a malformed label (an
    # unhashable list) — that is validation's job, ending in abstention.
    reply = '{"matches": [{"var": "p", "label": ["Product"], "filters": []}]}'
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}


def test_query_graph_data_abstains_on_a_conflicting_repeated_scalar_key():
    # A repeated scalar with two different values has no unambiguous merge —
    # rejected (abstention), never a silent last-wins pick. Fixture chosen so
    # last-wins would SUCCEED (`var` "p" matches `returns`), so only the
    # rejection rule can make this abstain.
    reply = (
        '{"matches": [{"var": "q", "var": "p", "label": "Product", '
        '"filters": []}], "returns": ["p.name"]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


def test_query_graph_data_keeps_an_identically_repeated_returns_list_once():
    # analyst Pass 5 P5-1: an identical repeat is kept once, never
    # concatenated — `["p.name", "p.name"]` would be rejected by `compile()`'s
    # duplicate-expression guard, a new false abstention.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "filters": []}], '
        '"returns": ["p.name"], "returns": ["p.name"]}'
    )
    rows = [{"p.name": "x"}]
    svc = StubServices(structured_query_rows=rows)
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": rows}


def test_query_graph_data_keeps_an_identically_repeated_filters_list_once():
    # P5-1: an identical repeated `filters` compiles to ONE condition, not a
    # redundant `p.category = $p0 AND p.category = $p1`.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", '
        '"filters": [{"property": "category", "op": "=", "value": "Audio"}], '
        '"filters": [{"property": "category", "op": "=", "value": "Audio"}]}], '
        '"returns": ["p.name"]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert call["compiled"].cypher == (
        "MATCH (p:Product) WHERE p.category = $p0 RETURN DISTINCT p.name LIMIT $limit"
    )


_RA = '"p.name"'
_RB = '"p.price"'
_FA = '{"property": "category", "op": "=", "value": "Audio"}'
_FB = '{"property": "price", "op": "<", "value": 50}'
_REPEAT_SHAPES = {  # (first list, repeated list) — every shape's union is {A, B}
    "identical": ("[{a}, {b}]", "[{a}, {b}]"),
    "reordered": ("[{a}, {b}]", "[{b}, {a}]"),
    "partial-overlap": ("[{a}]", "[{a}, {b}]"),
    "disjoint": ("[{a}]", "[{b}]"),
}


@pytest.mark.parametrize("shape", list(_REPEAT_SHAPES))
def test_query_graph_data_dedups_a_repeated_returns_list_per_element(shape):
    # analyst Pass 6 P6-1: a reordered/overlapping repeat must not produce a
    # duplicate `returns` expression, which `compile()` would reject (a false
    # abstention).
    first, second = (s.format(a=_RA, b=_RB) for s in _REPEAT_SHAPES[shape])
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "filters": []}], '
        f'"returns": {first}, "returns": {second}}}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x", "p.price": 1.0}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert call["compiled"].cypher == (
        "MATCH (p:Product) RETURN DISTINCT p.name, p.price LIMIT $limit"
    )


@pytest.mark.parametrize("shape", list(_REPEAT_SHAPES))
def test_query_graph_data_dedups_a_repeated_filters_list_per_element(shape):
    # P6-1: overlapping repeats AND each distinct condition exactly once — a
    # redundant `X AND X` would count toward `QueryMatch.filters`' max of 4.
    first, second = (s.format(a=_FA, b=_FB) for s in _REPEAT_SHAPES[shape])
    reply = (
        '{"matches": [{"var": "p", "label": "Product", '
        f'"filters": {first}, "filters": {second}}}], "returns": ["p.name"]}}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert call["compiled"].cypher == (
        "MATCH (p:Product) WHERE p.category = $p0 AND p.price < $p1 "
        "RETURN DISTINCT p.name LIMIT $limit"
    )


def test_query_graph_data_defaults_returns_for_a_name_exclusion_list_question():
    # analyst Pass 6 P6-2: only an EQUALITY on the name is a single-entity
    # lookup; `name <> X` ("everything except X") is a list question, so the
    # name default applies.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "filters": '
        '[{"property": "name", "op": "<>", "value": "Wireless Charging Pad"}]}]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert call["compiled"].cypher.endswith("RETURN DISTINCT p.name LIMIT $limit")


def test_query_graph_data_abstains_on_two_different_repeated_matches_lists():
    # Two DIFFERENT `matches` lists name two different queried entities — an
    # ambiguity like a conflicting scalar, so the tool abstains (the
    # concatenation fails the one-match limit) instead of last-wins
    # silently answering about the second one.
    reply = (
        '{"matches": [{"var": "e", "label": "Entity", "filters": []}], '
        '"matches": [{"var": "p", "label": "Product", "filters": []}], '
        '"returns": ["p.name"]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


@pytest.mark.parametrize("name_prop", ["name", "nameNormalized"])
def test_query_graph_data_does_not_default_returns_for_a_single_entity_name_lookup(name_prop):
    # analyst Pass 5 P5-2: "How much does X cost?" with `returns` omitted
    # must abstain, not hand back X's bare name with no price (a hand-off
    # that invites the agent to fabricate the price).
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "filters": '
        f'[{{"property": "{name_prop}", "op": "=", "value": "Wireless Charging Pad"}}]}}]}}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "Wireless Charging Pad"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run(
        {"question": "How much does the Wireless Charging Pad cost?", "dataset": "catalog"},
        ctx=CTX, run={},
    ))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


def test_query_graph_data_does_not_default_returns_for_an_entity_id_lookup():
    # K-079 F2 D3 (`docs/plans/salesperson-query-graph-data.md` §3.1): the
    # single-entity-lookup test is the label's declared `identity`, so an `=`
    # on `Entity.entityId` is a lookup too — echoing back the looked-up
    # entity's bare name is the fabrication bait `default_returns` excludes.
    reply = (
        '{"matches": [{"var": "e", "label": "Entity", "filters": '
        '[{"property": "entityId", "op": "=", "value": "ent-1"}]}]}'
    )
    svc = StubServices(structured_query_rows=[{"e.name": "Acme Corp"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "knowledge_base"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


def test_query_graph_data_abstains_not_crashes_on_an_unhashable_lookup_property():
    # Review m5: an `=` filter whose `property` is a list, with `returns`
    # absent, reaches `default_returns`' lookup test before validation. It
    # must abstain like any other malformed completion, never raise
    # `TypeError` past `run`'s `(ValidationError, ValueError)` net.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", '
        '"filters": [{"property": ["name"], "op": "=", "value": "x"}]}]}'
    )
    svc = StubServices(structured_query_rows=[{"p.name": "x"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": [], "finding": "no matching data found"}
    assert svc.structured_query_calls == []


# ── K-079 F2: row identity (`docs/plans/salesperson-query-graph-data.md` §3.1, §6.2) ──


def _compiled_cypher(reply: str, dataset: str = "catalog") -> str:
    svc = StubServices(structured_query_rows=[{"x": 1}])
    QueryGraphDataTool(svc, llm=StubQueryLLM(reply)).run(
        {"question": "q", "dataset": dataset}, ctx=CTX, run={},
    )
    [call] = svc.structured_query_calls
    return call["compiled"].cypher


def test_query_graph_data_prepends_name_to_the_rcas_ordered_price_list():
    # T-A1: the RCA's verbatim second completion
    # (`docs/reviews/salesperson-fabricated-prices-rca.md` §3). Once `p.name`
    # is projected, the order key is too, so the plain branch is taken.
    reply = (
        '{"matches": [{"var": "p", "label": "Product"}], '
        '"returns": ["p.price"], "order_by": "p.name"}'
    )

    assert _compiled_cypher(reply) == (
        "MATCH (p:Product) RETURN DISTINCT p.name, p.price ORDER BY p.name ASC LIMIT $limit"
    )


def test_query_graph_data_prepends_name_to_an_unordered_price_list():
    # T-A2: the t36/t45 shape — `DISTINCT p.price` alone collapsed the two
    # $89.99 products into one row.
    reply = '{"matches": [{"var": "p", "label": "Product"}], "returns": ["p.price"]}'

    assert _compiled_cypher(reply) == (
        "MATCH (p:Product) RETURN DISTINCT p.name, p.price LIMIT $limit"
    )


@pytest.mark.parametrize("group_key", [
    "p.category",
    # A non-facet grouping key: with the aggregate exemption gone, the facet
    # exemption would still cover `p.category` alone, so only this case pins
    # the aggregate exemption itself at the tool level (mutant MA1).
    "p.price",
])
def test_query_graph_data_leaves_a_grouped_aggregate_unchanged(group_key):
    # T-A3: the non-aggregate column of a grouped aggregate is its grouping
    # key — prepending `p.name` would regroup per product and change every count.
    reply = (
        '{"matches": [{"var": "p", "label": "Product"}], '
        f'"returns": ["{group_key}", "count(p)"]}}'
    )

    assert _compiled_cypher(reply) == (
        f"MATCH (p:Product) RETURN {group_key}, count(p) LIMIT $limit"
    )


@pytest.mark.parametrize("name_prop", ["name", "nameNormalized"])
def test_query_graph_data_leaves_a_single_entity_lookup_unchanged(name_prop):
    # T-A4: the `nlq-01` shape — one fact about an entity the question names.
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "filters": '
        f'[{{"property": "{name_prop}", "op": "=", "value": "usb-c hub"}}]}}], '
        '"returns": ["p.price"]}'
    )

    assert _compiled_cypher(reply) == (
        f"MATCH (p:Product) WHERE p.{name_prop} = $p0 RETURN DISTINCT p.price LIMIT $limit"
    )


@pytest.mark.parametrize(("reply", "expected"), [
    pytest.param(
        '{"matches": [{"var": "e", "label": "Entity", "filters": '
        '[{"property": "nameNormalized", "op": "=", "value": "acme corp"}]}], '
        '"returns": ["e.type"]}',
        "MATCH (e:Entity) WHERE e.nameNormalized = $p0 RETURN DISTINCT e.type LIMIT $limit",
        id="name-lookup",
    ),
    pytest.param(
        '{"matches": [{"var": "e", "label": "Entity"}], "returns": ["e.type"]}',
        "MATCH (e:Entity) RETURN DISTINCT e.type LIMIT $limit",
        id="facet-enumeration",
    ),
])
def test_query_graph_data_leaves_kb_entity_lookup_and_facet_enumeration_unchanged(reply, expected):
    # T-A5.
    assert _compiled_cypher(reply, dataset="knowledge_base") == expected


@pytest.mark.parametrize(("order_by", "expected"), [
    pytest.param(
        "p.name",
        "MATCH (p:Product) RETURN DISTINCT p.name, p.category ORDER BY p.name ASC LIMIT $limit",
        id="ordered-outside-returns-is-per-row",
    ),
    pytest.param(
        "p.category",
        "MATCH (p:Product) RETURN DISTINCT p.category ORDER BY p.category ASC LIMIT $limit",
        id="ordered-by-the-facet-enumerates",
    ),
])
def test_query_graph_data_identifies_a_facet_list_only_when_ordered_per_row(order_by, expected):
    # T-A6 (review B1): `order_by` outside `returns` takes compile()'s
    # tuple-distinct branch, which yields one unkeyed category row per
    # product — RC-1's shape with `category` in place of `price`.
    reply = (
        '{"matches": [{"var": "p", "label": "Product"}], "returns": ["p.category"], '
        f'"order_by": "{order_by}"}}'
    )

    assert _compiled_cypher(reply) == expected


def test_query_graph_data_ands_repeated_filters_in_a_prose_wrapped_own_line_reply():
    # P5-3: the verbatim nlq-12 reply is fenced, so it parses whole; this one
    # is prose-wrapped and only the own-line scan can find it — the hook must
    # reach that path too.
    reply = (
        "Here is the query:\n"
        '{"matches": [{"var": "p", "label": "Product", '
        '"filters": [{"property": "category", "op": "=", "value": "Wearables"}], '
        '"filters": [{"property": "price", "op": "<", "value": 200}]}], '
        '"returns": ["p.name"]}\n'
        "Hope that helps."
    )
    svc = StubServices(structured_query_rows=[{"p.name": "Fitness Tracker Band"}])
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    [call] = svc.structured_query_calls
    assert "p.category = $p0 AND p.price < $p1" in call["compiled"].cypher


def test_query_graph_data_tolerates_an_identically_repeated_scalar_key():
    reply = (
        '{"matches": [{"var": "p", "label": "Product", "label": "Product", '
        '"filters": []}], "returns": ["p.name"]}'
    )
    rows = [{"p.name": "x"}]
    svc = StubServices(structured_query_rows=rows)
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))

    out = json.loads(tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={}))

    assert out == {"items": rows}


def test_query_graph_data_schema_description_is_generated_from_the_dataset_registry():
    tool = QueryGraphDataTool(StubServices(), llm=StubQueryLLM(""))
    description = tool.schema["function"]["description"]
    for name, schema in querygen.DATASET_REGISTRY.items():
        assert name in description
        for label in schema.labels:
            assert label in description


# ── query_graph_data inherits the calling step's model (K-066) ───────────────
# `docs/plans/salesperson-single-model.md` §3.1/§5 T1–T3: the tool's internal
# `step` call resolves with the invoking node's `config.model` (`run["stepModel"]`,
# stamped by `_run_agent_node`) and the drive's override snapshot
# (`run["modelOverrides"]`), and with neither (`run={}`, the NLQ eval runner)
# exactly as before.

_MINISTRAL = "lmstudio/mistralai/ministral-3-3b"
# The `Repository.read_model_overrides` shape: all four keys, all unset.
_NO_OVERRIDES = {
    "agentModel": None, "guardModel": None,
    "embeddingModel": None, "responderModel": None,
}


class RecordingStepGateway:
    """A `ModelGateway`-shaped double for the tool's `.llm()` seam: records every
    call's full `(kind, requested, ws, overrides)` and answers with one
    `StubQueryLLM`. (The module's `RecordingGateway` above is embedder-only.)"""

    def __init__(self, reply: str = _CATALOG_FILTER_REPLY) -> None:
        self.calls: list[dict] = []
        self.client = StubQueryLLM(reply)

    def llm(self, kind, *, requested=None, ws=None, overrides=None):
        self.calls.append(
            {"kind": kind, "requested": requested, "ws": ws, "overrides": overrides}
        )
        return self.client


def _run_catalog_query(run: dict) -> RecordingStepGateway:
    gateway = RecordingStepGateway()
    tool = QueryGraphDataTool(StubServices(structured_query_rows=[{"p.name": "x"}]),
                              models=gateway)
    tool.run({"question": "what audio products?", "dataset": "catalog"}, ctx=CTX, run=run)
    return gateway


def test_query_graph_data_resolves_step_with_the_calling_nodes_model_and_overrides():
    gateway = _run_catalog_query({"stepModel": _MINISTRAL, "modelOverrides": _NO_OVERRIDES})

    assert gateway.calls == [
        {"kind": "step", "requested": _MINISTRAL, "ws": CTX.ws, "overrides": _NO_OVERRIDES}
    ]
    assert len(gateway.client.calls) == 1


def test_query_graph_data_passes_a_workspace_override_snapshot_through_untouched():
    overrides = {**_NO_OVERRIDES, "agentModel": "lmstudio/some/override"}

    gateway = _run_catalog_query({"stepModel": _MINISTRAL, "modelOverrides": overrides})

    [call] = gateway.calls
    # the exact snapshot object — the gateway's own precedence decides, the tool
    # neither drops it nor re-reads the workspace
    assert call["overrides"] is overrides
    assert call["requested"] == _MINISTRAL


def test_query_graph_data_with_an_empty_run_requests_the_step_default_as_before():
    gateway = _run_catalog_query({})

    assert gateway.calls == [
        {"kind": "step", "requested": None, "ws": CTX.ws, "overrides": None}
    ]


# ── U29f fix D: prompt hardening (`_QUERY_REQUEST_INSTRUCTIONS`) ────────────


def test_query_request_instructions_clarifies_bare_json_numbers():
    # RCA category A defense-in-depth: a rule + example steering the model
    # away from serializing a numeric filter value as a quoted string.
    text = tools._QUERY_REQUEST_INSTRUCTIONS
    assert "bare JSON number" in text
    assert 'never "50"' in text


def test_query_request_instructions_warns_normalized_properties_are_internal():
    # RCA category B defense-in-depth: the model should target the plain
    # `name` property, never a `*Normalized` one it can't see the value of.
    text = tools._QUERY_REQUEST_INSTRUCTIONS
    assert "Normalized" in text
    assert "name" in text


def test_query_request_instructions_has_name_not_entityid_projection_rule():
    # RCA category D fix: list/classify questions should project `name`, not
    # an internal `entityId`.
    text = tools._QUERY_REQUEST_INSTRUCTIONS
    assert "entityId" in text


def test_query_request_instructions_has_no_invented_filter_rule_for_superlatives():
    # The nlq-16 compounding defect: a superlative question ("cheapest") must
    # not get an invented filter alongside its order_by/limit.
    text = tools._QUERY_REQUEST_INSTRUCTIONS
    assert "cheapest" in text.lower()
    assert "invented filter" in text.lower() or "never an invented" in text.lower()


def test_query_request_instructions_preserves_existing_constraints():
    # Every pre-existing constraint (matches cardinality, allowed operators,
    # JSON-only reply) must survive the rewrite unchanged in substance.
    text = tools._QUERY_REQUEST_INSTRUCTIONS
    assert "exactly one entry" in text
    assert "= <> < <= > >=" in text
    assert "single JSON object" in text
    assert "never reply with prose" in text.lower()


def test_query_request_instructions_still_formats_with_dataset_schema():
    # The prompt-building seam itself must survive the rewrite: the schema
    # placeholder is substituted, not left as a literal "{dataset_schema}".
    prompt = tools._build_query_request_system_prompt(querygen.CATALOG_SCHEMA)
    assert "{dataset_schema}" not in prompt
    assert "Product" in prompt
    assert "price" in prompt


# ── K-086 D6: the catalog's categories in the querygen prompt (plan §4.1) ────

_DEMO_CATEGORIES = [
    "Accessories", "Audio", "Cameras", "Displays",
    "Peripherals", "Smart Home", "Storage", "Wearables",
]


def _categories_block(categories):
    # D6's wording, typed here from the plan rather than imported, so a
    # drifted module constant cannot certify itself.
    return (
        f"Product.category is exactly one of: {', '.join(categories)}. When the "
        "question names one of these categories, in any language, filter on "
        "category with the listed value exactly as written here, never on name "
        "and never with the question's own word. When the question names a "
        "specific product, keep filtering on its name."
    )


def _no_categories_prompt(schema):
    return tools._QUERY_REQUEST_INSTRUCTIONS.format(
        dataset_schema=tools._describe_dataset_schema(schema)
    )


def test_query_request_prompt_without_categories_is_byte_identical_to_before():
    # T-B3, first half: the knowledge_base path's pin.
    assert (tools._build_query_request_system_prompt(querygen.CATALOG_SCHEMA)
            == _no_categories_prompt(querygen.CATALOG_SCHEMA))


def test_query_request_prompt_with_categories_ends_with_the_category_block():
    # T-B3, second half.
    prompt = tools._build_query_request_system_prompt(
        querygen.CATALOG_SCHEMA, categories=["Audio", "Peripherals"],
    )

    assert prompt.startswith(_no_categories_prompt(querygen.CATALOG_SCHEMA))
    assert prompt.endswith(_categories_block(["Audio", "Peripherals"]))


def test_query_graph_data_catalog_prompt_lists_the_catalogs_categories_once():
    # T-B4.
    svc = StubServices(structured_query_rows=[{"p.name": "x"}], categories=_DEMO_CATEGORIES)
    llm = StubQueryLLM(_CATALOG_FILTER_REPLY)
    tool = QueryGraphDataTool(svc, llm=llm)

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    system = llm.calls[0][0]
    assert system["role"] == "system"
    assert system["content"].endswith(_categories_block(_DEMO_CATEGORIES))
    assert svc.categories_reads == 1


def test_query_graph_data_knowledge_base_prompt_never_reads_categories():
    # T-B5: listing `Entity.type` would move KB Layer 1 for a dataset no
    # DEF-C customer touches (§4.1).
    svc = StubServices(structured_query_rows=[{"e.name": "x"}], categories=_DEMO_CATEGORIES)
    llm = StubQueryLLM('{"matches": [{"var": "e", "label": "Entity"}], "returns": ["e.name"]}')
    tool = QueryGraphDataTool(svc, llm=llm)

    tool.run({"question": "q", "dataset": "knowledge_base"}, ctx=CTX, run={})

    assert svc.categories_reads == 0
    assert llm.calls[0][0]["content"] == _no_categories_prompt(querygen.KNOWLEDGE_BASE_SCHEMA)


def test_query_graph_data_empty_catalog_categories_leave_the_prompt_unchanged():
    # §4.1 call site: an empty list (empty catalog) passes `None`, never a
    # "exactly one of: ." block.
    svc = StubServices(structured_query_rows=[{"p.name": "x"}], categories=[])
    llm = StubQueryLLM(_CATALOG_FILTER_REPLY)
    tool = QueryGraphDataTool(svc, llm=llm)

    tool.run({"question": "q", "dataset": "catalog"}, ctx=CTX, run={})

    assert llm.calls[0][0]["content"] == _no_categories_prompt(querygen.CATALOG_SCHEMA)


# ── K-086 D7: an empty catalog result naming an unknown category or product ──
# (plan §4.2). The stub's `lookup_product` is an exact-key `dict.get`; a test
# that needs the repository's normalising lookup keys BOTH the raw and the
# normalised name to one row whose `name` is the real one (review Pass 2 N2).

_MOUSE_ROW = {"productId": "wireless-mouse-pro", "name": "Wireless Mouse Pro",
              "category": "Peripherals", "price": 29.99}
_MOUSE_BOTH_KEYS = {"Wireless Mouse Pro": _MOUSE_ROW, "wireless mouse pro": _MOUSE_ROW}
_ABSTAIN = {"items": [], "finding": "no matching data found"}


def _catalog_reply(*filters):
    return json.dumps({
        "matches": [{"var": "p", "label": "Product", "filters": [
            {"property": p, "op": op, "value": v} for p, op, v in filters
        ]}],
        "returns": ["p.name"],
    })


def _recovery(categories=_DEMO_CATEGORIES):
    return {"items": [], "finding": tools.QUERY_CATALOG_MISS_FINDING,
            "hint": tools.QUERY_CATALOG_MISS_HINT, "categories": categories}


def _run(svc, reply, dataset="catalog"):
    tool = QueryGraphDataTool(svc, llm=StubQueryLLM(reply))
    return json.loads(tool.run({"question": "q", "dataset": dataset}, ctx=CTX, run={}))


def test_query_graph_data_recovers_on_the_es_translated_category():
    # T-B6: DEF-C's es completion, verbatim shape.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("category", "=", "periféricos")))

    assert out == _recovery()
    assert svc.categories_reads == 1  # D6's read, reused — never a second one
    assert svc.category_checks == []  # never `has_category` (review M1)


def test_query_graph_data_recovers_on_an_unknown_category_in_any_filter_position():
    # T-B6b (not in §6.2): the miss is the second filter, so a trigger that
    # read only `filters[0]` would fall through to the dead end.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("price", "<", 100), ("category", "=", "periféricos")))

    assert out == _recovery()


def test_query_graph_data_recovers_on_a_category_word_sent_as_a_name():
    # T-B7: DEF-C's pt-BR completion.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("name", "=", "periféricos")))

    assert out == _recovery()
    assert svc.looked_up == ["periféricos"]


def test_query_graph_data_recovers_on_an_unknown_normalized_name():
    # T-B8.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("nameNormalized", "=", "periféricos")))

    assert out == _recovery()


def test_query_graph_data_keeps_the_abstention_for_a_known_category_emptied_by_price():
    # T-B9.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("category", "=", "Peripherals"), ("price", "<", 1)))

    assert out == _ABSTAIN


def test_query_graph_data_keeps_the_abstention_for_a_known_exact_name_emptied_by_price():
    # T-B10: the raw key hits and `row["name"] == value`.
    svc = StubServices(categories=_DEMO_CATEGORIES, products_by_name=_MOUSE_BOTH_KEYS)

    out = _run(svc, _catalog_reply(("name", "=", "Wireless Mouse Pro"), ("price", "<", 1)))

    assert out == _ABSTAIN
    assert svc.looked_up == ["Wireless Mouse Pro"]


@pytest.mark.parametrize("filters", [(), (("price", "<", 1),)], ids=["no-filter", "range-only"])
def test_query_graph_data_keeps_the_abstention_with_no_category_or_name_filter(filters):
    # T-B11.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(*filters))

    assert out == _ABSTAIN
    assert svc.looked_up == []


def test_query_graph_data_unparseable_reply_runs_no_recovery_check():
    # T-B12.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, "sorry, no idea")

    assert out == _ABSTAIN
    assert svc.looked_up == [] and svc.category_checks == []


def test_query_graph_data_knowledge_base_gets_no_recovery():
    # T-B13: the KB dataset gets F2, not K-086 (§1).
    svc = StubServices(categories=_DEMO_CATEGORIES)
    reply = json.dumps({"matches": [{"var": "e", "label": "Entity", "filters": [
        {"property": "name", "op": "=", "value": "periféricos"}]}], "returns": ["e.type"]})

    out = _run(svc, reply, dataset="knowledge_base")

    assert out == _ABSTAIN
    assert svc.looked_up == [] and svc.category_checks == []


def test_query_graph_data_non_empty_result_runs_no_recovery_check():
    # T-B14.
    rows = [{"p.name": "Wireless Mouse Pro"}]
    svc = StubServices(structured_query_rows=rows, categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("name", "=", "periféricos")))

    assert out == {"items": rows}
    assert svc.looked_up == [] and svc.category_checks == []


def test_query_catalog_miss_text_avoids_the_none_phrase_and_routes_to_both_tools():
    # T-B15: "no matching" is the phrase models turn into "we have none"
    # (K-078 R2).
    assert "no matching" not in tools.QUERY_CATALOG_MISS_FINDING
    assert "filter_products" in tools.QUERY_CATALOG_MISS_HINT
    assert "lookup_product_fact" in tools.QUERY_CATALOG_MISS_HINT


def test_query_graph_data_recovers_on_a_case_variant_of_a_real_category():
    # T-B16 (review M1): `compile()` matched `category` raw, so a normalising
    # check would call "peripherals" known and recreate DEF-C.
    svc = StubServices(categories=_DEMO_CATEGORIES)

    out = _run(svc, _catalog_reply(("category", "=", "peripherals")))

    assert out == _recovery()


def test_query_graph_data_recovers_on_a_case_variant_of_a_real_name():
    # T-B17 (review M1): the normalising lookup hits, but the raw `=` failed.
    svc = StubServices(categories=_DEMO_CATEGORIES, products_by_name=_MOUSE_BOTH_KEYS)

    out = _run(svc, _catalog_reply(("name", "=", "wireless mouse pro")))

    assert out == _recovery()


def test_query_graph_data_keeps_the_abstention_for_a_known_normalized_name_emptied_by_price():
    # T-B18 (review m6): both matchers normalise, so the product exists and
    # another filter emptied the result.
    svc = StubServices(categories=_DEMO_CATEGORIES, products_by_name=_MOUSE_BOTH_KEYS)

    out = _run(svc, _catalog_reply(("nameNormalized", "=", "wireless mouse pro"),
                                   ("price", "<", 1)))

    assert out == _ABSTAIN


def test_query_graph_data_empty_catalog_gets_no_recovery():
    # T-B19 (review n3): "these are all the catalog's categories" over [] is false.
    svc = StubServices(categories=[])

    out = _run(svc, _catalog_reply(("category", "=", "periféricos")))

    assert out == _ABSTAIN


def test_query_request_instructions_includes_a_name_projection_example():
    # A worked example pairing a "list/classify entities" question with a
    # `.name` projection (fixes nlq-25/26's entityId-over-name defect).
    text = tools._QUERY_REQUEST_INSTRUCTIONS
    assert '"e.name"' in text or "'e.name'" in text


def test_query_graph_data_offers_both_registered_datasets_in_the_enum():
    tool = QueryGraphDataTool(StubServices(), llm=StubQueryLLM(""))
    params = tool.schema["function"]["parameters"]
    assert set(params["properties"]["dataset"]["enum"]) == set(querygen.DATASET_REGISTRY)
    assert params["required"] == ["question", "dataset"]


# ── integration: post_message writes the agent message via §4 (durable artifact) ─
# The PRODUCED audit edge is asserted at executor altitude now (Option B — the
# executor links after the StepRun exists); see test_executor_produced.py.

def _configure_services(repo, *, actor="u1"):
    clock = itertools.count(1000)
    ids = (f"id{n}" for n in itertools.count(1))
    return Services(repo, clock=lambda: next(clock), id_gen=lambda: next(ids))


def test_post_message_writes_agent_message_live(repo, conn):
    repo.ensure_user(WS, user_id="u1")
    repo.ensure_agent(WS, agent_id="assistant", name="Bot")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    repo.post_first_message(
        WS, thread_id="t1", msg_id="trigger", author_id="u1",
        text="please help", role="user", created_at=10,
    )

    svc = _configure_services(repo)
    tool = PostMessageTool(svc, agent_id="assistant")
    run = {"runId": "r1", "ctx": json.dumps({"threadId": "t1"})}

    out = json.loads(tool.run({"text": "here is your answer"}, ctx=CTX, run=run))

    posted = repo.get_message(WS, msg_id=out["posted"])
    assert posted["text"] == "here is your answer"
    assert posted["role"] == "assistant"            # role derived, agent author
    assert posted["authorId"] == "assistant"


# ── integration: graphrag_retrieve threshold + abstention over seeded ws:test ─

def _pad(head):
    return (head + [0.0] * TEST_EMBEDDING_DIM)[:TEST_EMBEDDING_DIM]


def _seed_embedded_thread(repo):
    repo.ensure_user(WS, user_id="u1")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    msgs = [("m1", "about cats", _pad([1.0, 0.0])),
            ("m2", "about dogs", _pad([0.0, 1.0]))]
    first = True
    ts = 10
    for msg_id, text, vec in msgs:
        write = repo.post_first_message if first else repo.post_subsequent_message
        write(WS, thread_id="t1", msg_id=msg_id, author_id="u1",
              text=text, role="user", created_at=ts)
        repo.set_embedding(WS, msg_id=msg_id, embedding=vec, expected_dim=TEST_EMBEDDING_DIM)
        first = False
        ts += 1


def _seed_embedded_document(repo, *, chunk_id, vec, document_id="d1", text="chunk text"):
    """Ingest a `Document`+`Chunk` (K-050 M5 Stage 1/2 write path) with an
    embedding, so `services.hybrid_search`'s real merge (Stage 5) has a
    `Chunk` pool to draw from — requires `u1` (ensured by `_seed_embedded_thread`).
    """
    repo.create_document(
        WS, document_id=document_id, title="Doc", text=text,
        source_format="text", ingested_by="u1", created_at=1,
        chunks=[{"chunkId": chunk_id, "text": text, "seq": 0}],
    )
    repo.set_chunk_embedding(
        WS, chunk_id=chunk_id, embedding=vec, expected_dim=TEST_EMBEDDING_DIM
    )


def test_graphrag_retrieve_returns_near_seed_live(repo, fresh_vector_index):
    _seed_embedded_thread(repo)
    # A Chunk seed near the query too — the BLOCKER regression this pins: before
    # the fix, a Chunk row ranking within τ crashed with KeyError('msgId') instead
    # of resolving via `chunkId` (real `services.hybrid_search`, not a stub).
    _seed_embedded_document(repo, chunk_id="ch1", vec=_pad([1.0, 0.0]), text="cats in a doc")
    svc = _configure_services(repo)
    # query vector identical to m1/ch1 → distance ~0 (≤ τ); m2 is orthogonal (distance ~1)
    tool = GraphragRetrieveTool(svc, StubEmbedder(_pad([1.0, 0.0])), tau=0.5)
    out = json.loads(tool.run({"query": "cats"}, ctx=CTX, run={}))
    ids = [s["seedId"] for s in out["seeds"]]
    assert "m1" in ids
    assert "m2" not in ids                # orthogonal seed cut by τ
    assert "ch1" in ids                   # the Chunk seed, resolved without crashing
    chunk_seed = next(s for s in out["seeds"] if s["seedId"] == "ch1")
    assert chunk_seed["documentId"] == "d1"
    msg_seed = next(s for s in out["seeds"] if s["seedId"] == "m1")
    assert msg_seed["documentId"] is None


def test_graphrag_retrieve_abstains_when_all_seeds_distant_live(repo, fresh_vector_index):
    _seed_embedded_thread(repo)
    # A Chunk in the workspace too, also distant — abstention must still fire
    # correctly with a Chunk pool present, not just when it's empty.
    _seed_embedded_document(repo, chunk_id="ch1", vec=_pad([0.0, 1.0]), text="dogs in a doc")
    svc = _configure_services(repo)
    # a query orthogonal to every seed (Message AND Chunk) → abstain
    tool = GraphragRetrieveTool(svc, StubEmbedder(_pad([0.0, 0.0, 1.0])), tau=0.5)
    out = json.loads(tool.run({"query": "unrelated"}, ctx=CTX, run={}))
    assert out["seeds"] == []
    assert out["finding"] == "no relevant context found"
