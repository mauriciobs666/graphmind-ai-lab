"""Unit tests for the service layer.

Services own the invariants (write-variant dispatch, mention validation, RO/RW
read dispatch, `cursorId` construction, id/clock generation). They are tested
against a fake repository so the logic is pinned without a live database.
"""

from __future__ import annotations

import itertools
import json

import pytest
from redis.exceptions import ResponseError

from falkorchat import config
from falkorchat.config import CallContext
from falkorchat.modelconfig import ModelResolutionError
from falkorchat.repository import DocumentWriteStatus, MessageWriteStatus
from falkorchat.schemas import MAX_BATCH_SIZE, MAX_DIFF_PREVIEW, MAX_KEY_LEN
from falkorchat.services import (
    DEMO_EXPECTED_DEFS,
    POST_SUCCESS_SAMPLE_SIZE,
    RAG_QUERY_TIMEOUT_MS,
    BatchTooLargeError,
    ChannelNotFoundError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidSearchQueryError,
    MatchNotFoundError,
    MemberIdCollisionError,
    SearchNotAvailableError,
    Services,
    ThreadNotFoundError,
    UnknownActorError,
    UnknownMemberError,
    UnknownOrderTransitionError,
    WorkflowDefConflictError,
    WorkflowRunNotFoundError,
    _diff_structures,
    _structural_diffs,
)

CTX = CallContext(ws="test", actor="u1")

OK = MessageWriteStatus(written=True, had_head=False, dup_msg=False, author_found=True)
DUP = MessageWriteStatus(written=False, had_head=False, dup_msg=True, author_found=True)
HAD_HEAD = MessageWriteStatus(written=False, had_head=True, dup_msg=False, author_found=True)


class FakeRepo:
    """Records calls and simulates the small amount of state services depend on."""

    def __init__(self):
        self.channels: set[str] = set()
        self.threads: set[str] = set()
        self.heads: set[str] = set()          # threads that already have a HEAD
        self.members: set[str] = set()        # known User ids
        self.agents: set[str] = set()         # known Agent ids
        # cursorId -> (lastReadAt, lastReadMsgId) composite pair
        self.cursors: dict[str, tuple[int, str | None]] = {}
        self.calls: list[tuple] = []
        self.since_rows: list[dict] = []
        # scripted §4 v2 status rows (popped first); default behavior otherwise
        self.first_status: list[MessageWriteStatus | None] = []
        self.subseq_status: list[MessageWriteStatus | None] = []
        self.agent_first_status: list[MessageWriteStatus | None] = []
        self.agent_subseq_status: list[MessageWriteStatus | None] = []
        # §11 workflow state
        self.published: list[dict] = []       # publish_def kwargs, in order
        self.materialized: list[dict] = []     # materialize_snapshot kwargs, in order
        self.defs: dict[tuple, dict] = {}      # (key, version) -> def subgraph/meta
        self.snapshots: dict[tuple, dict] = {}  # (key, version) -> snapshot subgraph/meta
        # K-031 structure reads — repository shape `{name,kind,start_keys,steps,transitions}`
        self.def_structures: dict[tuple, dict] = {}
        self.snapshot_structures: dict[tuple, dict] = {}
        # §12 workflow-run state
        self.messages: dict[str, dict] = {}    # msgId -> message (for get_message)
        self.started_runs: list[dict] = []     # start_run kwargs, in order
        self.start_run_result = _UNSET        # override to None to simulate a miss
        self.runs: dict[str, dict] = {}        # runId -> run state (for get_run)
        self.step_runs: dict[str, list] = {}   # runId -> step-run trail
        self.trace: dict[str, list] = {}       # runId -> trace events
        self.waiting_runs: dict[str, dict] = {}  # threadId -> waiting run (resume lookup)
        self.runs_by_thread: dict[str, list[dict]] = {}  # threadId -> run history (§12.14)
        self.participants: dict[str, list[dict]] = {}  # threadId -> raw participant rows
        self.post_success_result = _UNSET  # override to script read_recent_post_success (§12.15)
        self.documents: dict[str, dict] = {}  # documentId -> created_document (K-050)
        # `hybrid_search`/`search_chunks` return `since_rows` by default (back-
        # compat with pre-Stage-5 tests exercising just one of the two pools);
        # set either explicitly to drive `services.hybrid_search`'s merge with
        # independent Message/Chunk pools (K-050 M5 Stage 5).
        self.hybrid_rows: list[dict] | None = None
        self.chunk_rows: list[dict] | None = None
        self.matches: dict[str, dict] = {}  # matchId -> match state (K-050 M5 Stage 4)
        # §15 product catalog (K-052 M6) — keyed by nameNormalized for lookup_product;
        # `products` (a flat list of {name, category, price} dicts) backs filter_products.
        self.products_by_name: dict[str, dict] = {}
        self.products: list[dict] = []
        # §16 Cart / Order (K-053 M6) — see the methods below for the exact
        # (simplified but faithful) semantics each mirrors from `repository.py`.
        self.customers: set[str] = set()
        self.customer_created_at: dict[str, int] = {}
        self.carts: set[str] = set()
        self.cart_created_at: dict[str, int] = {}
        self.cart_items: dict[str, dict[str, dict]] = {}  # customerId -> {productId: line}
        self.products_by_id: dict[str, dict] = {}  # productId -> {"name", "price"}
        self.orders: dict[str, dict] = {}  # orderId -> order state
        # §17 durable customer profile (K-054 M6) — customerId -> {"name",
        # "deliveryAddress", "profileUpdatedAt"}; absent key means no
        # `Customer` node at all (mirrors repository.get_profile's `None`).
        self.profiles: dict[str, dict] = {}

    # writes / lookups used by services
    def create_channel(self, ws, *, channel_id, name, created_at):
        self.channels.add(channel_id)
        self.calls.append(("create_channel", ws, channel_id, name, created_at))

    def channel_exists(self, ws, *, channel_id):
        return channel_id in self.channels

    def create_thread(self, ws, *, channel_id, thread_id, title, created_at):
        self.threads.add(thread_id)
        self.calls.append(("create_thread", ws, channel_id, thread_id, title, created_at))

    def thread_exists(self, ws, *, thread_id):
        return thread_id in self.threads

    def thread_has_head(self, ws, *, thread_id):
        return thread_id in self.heads

    def resolve_member_kinds(self, ws, *, ids):
        def kind(i):
            if i in self.agents:
                return "Agent"
            if i in self.members:
                return "User"
            return None

        return {i: kind(i) for i in ids}

    def post_first_message(self, ws, **kw):
        self.calls.append(("post_first_message", kw))
        if self.first_status:
            return self.first_status.pop(0)
        self.heads.add(kw["thread_id"])
        return OK

    def post_subsequent_message(self, ws, **kw):
        self.calls.append(("post_subsequent_message", kw))
        if self.subseq_status:
            return self.subseq_status.pop(0)
        return OK

    def post_agent_answer(self, ws, **kw):
        self.calls.append(("post_agent_answer", kw))
        if self.agent_subseq_status:
            return self.agent_subseq_status.pop(0)
        return OK

    def post_agent_answer_first(self, ws, **kw):
        self.calls.append(("post_agent_answer_first", kw))
        if self.agent_first_status:
            return self.agent_first_status.pop(0)
        self.heads.add(kw["thread_id"])
        return OK

    def get_cursor(self, ws, *, cursor_id):
        return self.cursors.get(cursor_id)

    def advance_cursor(self, ws, *, me_id, thread_id, cursor_id, now, now_msg_id):
        prev = self.cursors.get(cursor_id, (0, ""))
        self.cursors[cursor_id] = max(prev, (now, now_msg_id))
        self.calls.append(("advance_cursor", cursor_id, now, now_msg_id))
        return self.cursors[cursor_id]

    def read_thread_since(self, ws, *, thread_id, me_id, since, since_msg_id=None,
                          limit=50):
        self.calls.append(
            ("read_thread_since", thread_id, me_id, since, since_msg_id, limit)
        )
        return self.since_rows

    def read_ws_since(self, ws, *, me_id, since, since_msg_id=None, limit=50):
        self.calls.append(("read_ws_since", me_id, since, since_msg_id, limit))
        return self.since_rows

    def search_messages(self, ws, *, query, limit=50):
        self.calls.append(("search_messages", query, limit))
        if isinstance(self.since_rows, Exception):
            raise self.since_rows
        return self.since_rows

    def hybrid_search(self, ws, *, q_vec, k, limit, channel_id=None, timeout=None):
        self.calls.append(("hybrid_search", ws, tuple(q_vec), k, limit, channel_id, timeout))
        return self.since_rows if self.hybrid_rows is None else self.hybrid_rows

    def ensure_user(self, ws, *, user_id, display_name=None, email=None):
        # mirrors the §2 v2 guarded ensure: an id held by an Agent is refused
        if user_id in self.agents:
            raise MemberIdCollisionError(user_id)
        self.members.add(user_id)
        self.calls.append(("ensure_user", ws, user_id))

    # ── §14 Documents & Chunks (K-050 M5 Stage 1) ─────────────────────────────────

    def create_document(
        self, ws, *, document_id, title, text, source_format,
        ingested_by, created_at, chunks,
    ):
        self.calls.append(("create_document", ws, document_id, ingested_by, chunks))
        if ingested_by in self.agents:
            kind, actor_label = "agent", "Agent"
        elif ingested_by in self.members:
            kind, actor_label = "document", "User"
        else:
            return DocumentWriteStatus(written=False, ingestor_found=False)
        self.documents[document_id] = {
            "documentId": document_id, "title": title, "text": text,
            "sourceFormat": source_format, "sourceKind": kind,
            "status": "processing", "createdAt": created_at,
            "ingestedByKind": actor_label, "ingestedById": ingested_by,
            "chunks": chunks,
        }
        return DocumentWriteStatus(written=True, ingestor_found=True)

    def get_document(self, ws, *, document_id):
        doc = self.documents.get(document_id)
        if doc is None:
            return None
        return {k: v for k, v in doc.items() if k != "chunks"}

    def list_document_chunks(self, ws, *, document_id):
        self.calls.append(("list_document_chunks", ws, document_id))
        doc = self.documents.get(document_id)
        if doc is None:
            return []
        return [{"chunkId": c["chunkId"], "text": c["text"]} for c in doc["chunks"]]

    def search_chunks(self, ws, *, q_vec, k, limit, timeout=None):
        self.calls.append(("search_chunks", ws, tuple(q_vec), k, limit, timeout))
        return self.since_rows if self.chunk_rows is None else self.chunk_rows

    # ── §14.6 Entity fusion — SAME_AS (K-050 M5 Stage 4) ──────────────────────────
    # matchId -> {"status", "entityA", "entityB", ...} — seed via `self.matches`
    # directly in a test, mirroring `self.documents`'s dict-of-state idiom above.

    def confirm_match(self, ws, *, match_id, decided_by, decided_at):
        self.calls.append(("confirm_match", ws, match_id, decided_by, decided_at))
        match = self.matches.get(match_id)
        if match is None:
            return None
        match["status"] = "confirmed"
        match["decidedBy"] = decided_by
        match["decidedAt"] = decided_at
        return {
            "matchId": match_id, "status": "confirmed",
            "entityA": match["entityA"], "entityB": match["entityB"],
        }

    def reject_match(self, ws, *, match_id, decided_by, decided_at):
        self.calls.append(("reject_match", ws, match_id, decided_by, decided_at))
        match = self.matches.get(match_id)
        if match is None:
            return None
        match["status"] = "rejected"
        match["decidedBy"] = decided_by
        match["decidedAt"] = decided_at
        return {
            "matchId": match_id, "status": "rejected",
            "entityA": match["entityA"], "entityB": match["entityB"],
        }

    def recheck_match(self, ws, *, match_id, at):
        self.calls.append(("recheck_match", ws, match_id, at))
        match = self.matches.get(match_id)
        if match is None or match["status"] != "rejected":
            return None
        match["status"] = "pending"
        return {
            "matchId": match_id, "status": "pending",
            "entityA": match["entityA"], "entityB": match["entityB"],
        }

    def _match_row(self, match_id):
        m = self.matches[match_id]
        return {
            "matchId": match_id, "entityA": m["entityA"], "nameA": m.get("nameA", ""),
            "entityB": m["entityB"], "nameB": m.get("nameB", ""),
            "status": m["status"], "confidence": m.get("confidence", 1.0),
            "technique": m.get("technique", "exact"), "createdAt": m.get("createdAt", 0),
        }

    def list_pending_matches(self, ws, *, limit=50):
        self.calls.append(("list_pending_matches", ws, limit))
        rows = [
            self._match_row(mid) for mid, m in self.matches.items()
            if m["status"] == "pending"
        ]
        return rows[:limit]

    def list_matches(self, ws, *, status=None, limit=50):
        self.calls.append(("list_matches", ws, status, limit))
        rows = [
            self._match_row(mid) for mid, m in self.matches.items()
            if status is None or m["status"] == status
        ]
        return rows[:limit]

    # ── §11 workflow defs (reference) + snapshots (workspace) ────────────────────

    def publish_def(self, *, key, version, name, kind, start_key, steps, transitions):
        self.published.append({
            "key": key, "version": version, "name": name, "kind": kind,
            "start_key": start_key, "steps": steps, "transitions": transitions,
        })
        return {
            "key": key, "version": version,
            "stepCount": len(steps), "transitionCount": len(transitions),
        }

    def read_def_subgraph(self, *, key, version):
        self.calls.append(("read_def_subgraph", key, version))
        return self.defs.get((key, version))

    def get_def(self, *, key, version=None):
        self.calls.append(("get_def", key, version))
        return self.defs.get((key, version))

    def list_defs(self, *, limit=50):
        self.calls.append(("list_defs", limit))
        return list(self.defs.values())

    def materialize_snapshot(self, ws, *, key, version, name, kind, start_key,
                             steps, transitions):
        self.materialized.append({
            "ws": ws, "key": key, "version": version, "name": name, "kind": kind,
            "start_key": start_key, "steps": steps, "transitions": transitions,
        })
        return {
            "key": key, "version": version,
            "stepCount": len(steps), "transitionCount": len(transitions),
        }

    def read_def_structure(self, *, key, version):
        self.calls.append(("read_def_structure", key, version))
        return self.def_structures.get((key, version))

    def get_snapshot(self, ws, *, key, version):
        self.calls.append(("get_snapshot", ws, key, version))
        return self.snapshots.get((key, version))

    def read_snapshot_structure(self, ws, *, key, version):
        self.calls.append(("read_snapshot_structure", ws, key, version))
        return self.snapshot_structures.get((key, version))

    def list_snapshots(self, ws, *, limit=50):
        self.calls.append(("list_snapshots", ws, limit))
        return list(self.snapshots.values())

    # ── §12 workflow runs ────────────────────────────────────────────────────

    def get_message(self, ws, *, msg_id):
        self.calls.append(("get_message", ws, msg_id))
        return self.messages.get(msg_id)

    def start_run(self, ws, *, run_id, def_key, def_version, started_at,
                  trigger_msg_id, ctx, trace, max_steps):
        self.started_runs.append({
            "ws": ws, "run_id": run_id, "def_key": def_key,
            "def_version": def_version, "started_at": started_at,
            "trigger_msg_id": trigger_msg_id, "ctx": ctx, "trace": trace,
            "max_steps": max_steps,
        })
        if self.start_run_result is _UNSET:
            return {"runId": run_id, "startKey": "intake", "status": "running",
                    "stepCount": 0}
        return self.start_run_result

    def get_run(self, ws, *, run_id):
        self.calls.append(("get_run", ws, run_id))
        return self.runs.get(run_id)

    def read_step_runs(self, ws, *, run_id):
        self.calls.append(("read_step_runs", ws, run_id))
        return self.step_runs.get(run_id, [])

    def read_trace(self, ws, *, run_id):
        self.calls.append(("read_trace", ws, run_id))
        return self.trace.get(run_id, [])

    def find_waiting_run_for_thread(self, ws, *, thread_id):
        self.calls.append(("find_waiting_run_for_thread", ws, thread_id))
        return self.waiting_runs.get(thread_id)

    def find_runs_for_thread(self, ws, *, thread_id, limit=10):
        self.calls.append(("find_runs_for_thread", ws, thread_id, limit))
        return self.runs_by_thread.get(thread_id, [])[:limit]

    def read_recent_post_success(self, ws, *, def_key, def_version, limit):
        self.calls.append(("read_recent_post_success", ws, def_key, def_version, limit))
        if self.post_success_result is _UNSET:
            return {"sampleSize": 0, "postedCount": 0}
        return self.post_success_result

    def list_thread_participants(self, ws, *, thread_id):
        self.calls.append(("list_thread_participants", ws, thread_id))
        return self.participants.get(thread_id, [])

    # ── §15 product catalog (K-052 M6, reference) ─────────────────────────────

    def lookup_product(self, *, name_normalized):
        self.calls.append(("lookup_product", name_normalized))
        return self.products_by_name.get(name_normalized)

    def filter_products(self, *, category, min_price, max_price, limit=20):
        self.calls.append(("filter_products", category, min_price, max_price, limit))
        rows = [
            p for p in self.products
            if (category is None or p["category"] == category)
            and (min_price is None or p["price"] >= min_price)
            and (max_price is None or p["price"] <= max_price)
        ]
        return sorted(rows, key=lambda p: p["price"])[:limit]

    # ── §16 Cart / Order (K-053 M6) ────────────────────────────────────────────
    #
    # Simplified but semantically faithful to `repository.py` §16 (mirrors the
    # graph note's `[verified]` Cypher shapes): `ensure_cart`/`add_to_cart`
    # return `None` with no `Customer`/`Cart` respectively; `place_order`
    # raises with no `Customer` (never a silent no-op); a guarded lifecycle
    # CAS returns `None` on a status mismatch.

    def ensure_customer(self, ws, *, customer_id, now):
        self.calls.append(("ensure_customer", customer_id, now))
        if customer_id not in self.customers:
            self.customers.add(customer_id)
            self.customer_created_at[customer_id] = now
        return {"customerId": customer_id, "createdAt": self.customer_created_at[customer_id]}

    def ensure_cart(self, ws, *, customer_id, now):
        self.calls.append(("ensure_cart", customer_id, now))
        if customer_id not in self.customers:
            return None
        if customer_id not in self.carts:
            self.carts.add(customer_id)
            self.cart_created_at[customer_id] = now
            self.cart_items.setdefault(customer_id, {})
        return {"customerId": customer_id, "createdAt": self.cart_created_at[customer_id]}

    def add_to_cart(self, ws, *, customer_id, product_id, qty, now):
        self.calls.append(("add_to_cart", customer_id, product_id, qty, now))
        if customer_id not in self.carts:
            return None
        items = self.cart_items.setdefault(customer_id, {})
        line = items.get(product_id)
        if line is None:
            items[product_id] = {"quantity": qty, "addedAt": now, "updatedAt": now}
        else:
            line["quantity"] += qty
            line["updatedAt"] = now
        return {"productId": product_id, "quantity": items[product_id]["quantity"]}

    def adjust_cart_item(self, ws, *, customer_id, product_id, qty, now):
        self.calls.append(("adjust_cart_item", customer_id, product_id, qty, now))
        items = self.cart_items.get(customer_id, {})
        line = items.get(product_id)
        if line is None:
            return None
        new_qty = line["quantity"] - qty
        if new_qty > 0:
            line["quantity"] = new_qty
            line["updatedAt"] = now
            return {"quantity": new_qty, "removed": False}
        del items[product_id]
        return {"quantity": new_qty, "removed": True}

    def read_cart(self, ws, *, customer_id):
        self.calls.append(("read_cart", customer_id))
        items = self.cart_items.get(customer_id, {})
        rows = [
            {"productId": pid, "quantity": line["quantity"], "addedAt": line["addedAt"]}
            for pid, line in items.items()
        ]
        return sorted(rows, key=lambda r: r["addedAt"])

    def clear_cart(self, ws, *, customer_id):
        self.calls.append(("clear_cart", customer_id))
        self.cart_items[customer_id] = {}

    def lookup_products_by_id(self, *, product_ids):
        self.calls.append(("lookup_products_by_id", tuple(product_ids)))
        return [
            {"productId": pid, **self.products_by_id[pid]}
            for pid in product_ids if pid in self.products_by_id
        ]

    def place_order(self, ws, *, customer_id, order_id, now, lines):
        self.calls.append(("place_order", customer_id, order_id, now, lines))
        if customer_id not in self.customers:
            raise RuntimeError(
                f"place_order was a no-op — customer not found ({customer_id!r})"
            )
        if order_id in self.orders:
            return {"created": False, "lineCount": len(self.orders[order_id]["lines"])}
        self.orders[order_id] = {
            "customerId": customer_id, "status": "placed",
            "placedAt": now, "updatedAt": now, "lines": list(lines),
        }
        items = self.cart_items.get(customer_id)
        if items is not None:
            items.clear()
        return {"created": True, "lineCount": len(lines)}

    def get_order(self, ws, *, order_id):
        self.calls.append(("get_order", order_id))
        order = self.orders.get(order_id)
        if order is None:
            return None
        total = sum(line["lineTotal"] for line in order["lines"])
        return {
            "orderId": order_id, "status": order["status"], "placedAt": order["placedAt"],
            "updatedAt": order["updatedAt"], "lines": list(order["lines"]), "total": total,
        }

    def _order_cas(self, order_id, expected, new_status, now):
        self.calls.append(("order_cas", order_id, expected, new_status, now))
        order = self.orders.get(order_id)
        if order is None or order["status"] != expected:
            return None
        order["status"] = new_status
        order["updatedAt"] = now
        return {"orderId": order_id, "status": new_status}

    def fulfill_order(self, ws, *, order_id, now):
        return self._order_cas(order_id, "placed", "fulfilled", now)

    def deliver_order(self, ws, *, order_id, now):
        return self._order_cas(order_id, "fulfilled", "delivered", now)

    def cancel_order(self, ws, *, order_id, now):
        return self._order_cas(order_id, "placed", "cancelled", now)

    # ── §17 Durable customer profile (K-054 M6) ───────────────────────────────
    #
    # Faithful to `repository.py` §17's `coalesce()`-per-field semantics: an
    # omitted (`None`) field on `upsert_profile` leaves the stored value
    # unchanged, never clears it.

    def get_profile(self, ws, *, customer_id):
        self.calls.append(("get_profile", customer_id))
        return self.profiles.get(customer_id)

    def upsert_profile(self, ws, *, customer_id, name, delivery_address, now):
        self.calls.append(("upsert_profile", customer_id, name, delivery_address, now))
        existing = self.profiles.get(customer_id, {"name": None, "deliveryAddress": None})
        stored = {
            "name": name if name is not None else existing["name"],
            "deliveryAddress": (
                delivery_address if delivery_address is not None
                else existing["deliveryAddress"]
            ),
            "profileUpdatedAt": now,
        }
        self.profiles[customer_id] = stored
        return {
            "customerId": customer_id, "name": stored["name"],
            "deliveryAddress": stored["deliveryAddress"],
        }


_UNSET = object()


class StubExecutor:
    """Records `run`/`resume` calls and returns scripted statuses (U5 tests the
    service orchestration in isolation; the real engine is covered in U4)."""

    def __init__(self, *, step_budget=12, run_status="waiting", resume_status="done"):
        self.step_budget = step_budget
        self._run_status = run_status
        self._resume_status = resume_status
        self.run_calls: list[str] = []
        self.resume_calls: list[str] = []

    def run(self, ctx, *, run_id):
        self.run_calls.append(run_id)
        return self._run_status

    def resume(self, ctx, *, run_id):
        self.resume_calls.append(run_id)
        return self._resume_status


def make_service(repo, *, now=1000, executor=None, models=None):
    ids = (f"id{n}" for n in itertools.count(1))
    return Services(repo, clock=lambda: now, id_gen=lambda: next(ids),
                    executor=executor, models=models)


# ── create_channel / create_thread ─────────────────────────────────────────────


def test_create_channel_generates_id_and_time():
    repo = FakeRepo()
    svc = make_service(repo, now=1000)

    ch = svc.create_channel(CTX, name="general")

    assert ch["channelId"] == "id1"
    assert ch["name"] == "general"
    assert ch["createdAt"] == 1000
    assert "id1" in repo.channels


def test_create_thread_requires_existing_channel():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(ChannelNotFoundError):
        svc.create_thread(CTX, channel_id="missing", title="hi")


def test_create_thread_creates_when_channel_exists():
    repo = FakeRepo()
    repo.channels.add("c1")
    svc = make_service(repo, now=1000)

    th = svc.create_thread(CTX, channel_id="c1", title="hi")

    assert th["threadId"] == "id1"
    assert th["channelId"] == "c1"
    assert th["createdAt"] == 1000


# ── ingest_document / get_document (K-050 M5 Stage 1) ───────────────────────────


def test_ingest_document_mints_id_splits_and_returns_processing_status():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo, now=1000)

    result = svc.ingest_document(CTX, text="hello world", title="My Doc")

    assert result == {"documentId": "id1", "chunkCount": 1, "status": "processing"}
    doc = repo.documents["id1"]
    assert doc["title"] == "My Doc"
    assert doc["text"] == "hello world"
    assert doc["chunks"] == [{"chunkId": "id2", "text": "hello world", "seq": 0}]
    assert doc["createdAt"] == 1000


def test_ingest_document_chunk_count_matches_split(monkeypatch):
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)
    monkeypatch.setattr(
        "falkorchat.services.chunking.split_into_chunks",
        lambda text, **kw: ["a", "b", "c"],
    )

    result = svc.ingest_document(CTX, text="whatever")

    assert result["chunkCount"] == 3
    chunks = repo.documents[result["documentId"]]["chunks"]
    assert [c["seq"] for c in chunks] == [0, 1, 2]
    assert [c["text"] for c in chunks] == ["a", "b", "c"]
    # chunk ids are distinct server-minted ids, not derived from position
    assert len({c["chunkId"] for c in chunks}) == 3


def test_ingest_document_defaults_title_to_source_label_then_empty():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    with_label = svc.ingest_document(CTX, text="x", source_label="report.txt")
    assert repo.documents[with_label["documentId"]]["title"] == "report.txt"

    with_neither = svc.ingest_document(CTX, text="x")
    assert repo.documents[with_neither["documentId"]]["title"] == ""


def test_ingest_document_rejects_whitespace_only_text():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    with pytest.raises(EmptyDocumentError):
        svc.ingest_document(CTX, text="   \n\n\t  ")

    assert repo.documents == {}  # rejected before any split/repo call


def test_ingest_document_rejects_empty_text():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    with pytest.raises(EmptyDocumentError):
        svc.ingest_document(CTX, text="")

    assert repo.documents == {}


def test_ingest_document_rejects_oversized_text():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    with pytest.raises(DocumentTooLargeError):
        svc.ingest_document(CTX, text="x" * 500_001)

    assert repo.documents == {}  # nothing written — rejected before any split/call


def test_ingest_document_at_exactly_the_limit_is_accepted():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    result = svc.ingest_document(CTX, text="x" * 500_000)

    assert result["status"] == "processing"


def test_ingest_document_unknown_actor_raises_instead_of_silent_write():
    repo = FakeRepo()  # actor "u1" not registered as a User or Agent
    svc = make_service(repo)

    with pytest.raises(UnknownActorError):
        svc.ingest_document(CTX, text="hello")

    assert repo.documents == {}


def test_ingest_document_known_agent_actor_source_kind_agent():
    repo = FakeRepo()
    repo.agents.add("bot1")
    svc = make_service(repo)
    ctx = CallContext(ws="test", actor="bot1")

    result = svc.ingest_document(ctx, text="hello")

    assert repo.documents[result["documentId"]]["sourceKind"] == "agent"


# ── ingest_documents (K-050 M5 Stage 6a, FR-11 bulk ingestion) ──────────────────


def test_ingest_documents_returns_one_receipt_per_item_in_order():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX,
        documents=[
            {"text": "first document", "title": "First"},
            {"text": "second document", "title": "Second"},
        ],
    )

    assert len(results) == 2
    assert all(r["status"] == "processing" for r in results)
    ids = [r["documentId"] for r in results]
    assert len(set(ids)) == 2  # distinct documents, not one item overwriting another
    assert repo.documents[ids[0]]["title"] == "First"
    assert repo.documents[ids[1]]["title"] == "Second"


def test_ingest_documents_defaults_mirror_ingest_document():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    [result] = svc.ingest_documents(CTX, documents=[{"text": "x"}])

    doc = repo.documents[result["documentId"]]
    assert doc["title"] == ""
    assert doc["sourceFormat"] == "text"


def test_ingest_documents_isolates_a_per_item_failure_into_its_own_receipt():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX,
        documents=[
            {"text": "good document one"},
            {"text": "   "},  # whitespace-only — rejected
            {"text": "good document two"},
        ],
    )

    assert len(results) == 3
    assert results[0]["status"] == "processing"
    assert results[1]["status"] == "error"
    assert results[1]["errorType"] == "EmptyDocumentError"
    assert "error" in results[1]
    assert results[2]["status"] == "processing"
    # the two good documents were still written, distinct from each other
    good_ids = {results[0]["documentId"], results[2]["documentId"]}
    assert len(good_ids) == 2
    assert good_ids <= set(repo.documents)


def test_ingest_documents_isolates_a_malformed_item_missing_text_key():
    # Pass 6 review BLOCKER: a missing "text" key used to raise a bare
    # KeyError BEFORE the per-item try could catch it, aborting the whole
    # batch (and losing the receipt of any already-ingested sibling ahead of
    # it, even though its Document was already written).
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX,
        documents=[
            {"text": "good document one"},
            {"title": "no text key at all"},
            {"text": "good document two"},
        ],
    )

    assert len(results) == 3
    assert results[0]["status"] == "processing"
    assert results[1]["status"] == "error"
    assert results[1]["errorType"] == "MalformedItemError"
    assert results[2]["status"] == "processing"
    good_ids = {results[0]["documentId"], results[2]["documentId"]}
    assert len(good_ids) == 2
    assert good_ids <= set(repo.documents)


def test_ingest_documents_isolates_a_non_string_text_item():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX, documents=[{"text": "good document"}, {"text": 12345}],
    )

    assert results[0]["status"] == "processing"
    assert results[1]["status"] == "error"
    assert results[1]["errorType"] == "MalformedItemError"


def test_ingest_documents_isolates_a_non_dict_item():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX, documents=[{"text": "good document"}, "not a dict"],
    )

    assert results[0]["status"] == "processing"
    assert results[1]["status"] == "error"
    assert results[1]["errorType"] == "MalformedItemError"


def test_ingest_documents_isolates_an_oversized_item_failure():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX, documents=[{"text": "x" * 500_001}, {"text": "fine"}],
    )

    assert results[0]["status"] == "error"
    assert results[0]["errorType"] == "DocumentTooLargeError"
    assert results[1]["status"] == "processing"


def test_ingest_documents_isolates_an_unknown_actor_failure():
    repo = FakeRepo()  # "u1" not registered as a User or Agent
    svc = make_service(repo)

    [result] = svc.ingest_documents(CTX, documents=[{"text": "hello"}])

    assert result["status"] == "error"
    assert result["errorType"] == "UnknownActorError"
    assert repo.documents == {}


def test_ingest_documents_rejects_batch_over_max_size():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    with pytest.raises(BatchTooLargeError):
        svc.ingest_documents(
            CTX, documents=[{"text": "x"} for _ in range(MAX_BATCH_SIZE + 1)],
        )

    assert repo.documents == {}  # nothing written — rejected before any item runs


def test_ingest_documents_at_exactly_the_batch_limit_is_accepted():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    results = svc.ingest_documents(
        CTX, documents=[{"text": "x"} for _ in range(MAX_BATCH_SIZE)],
    )

    assert len(results) == MAX_BATCH_SIZE
    assert all(r["status"] == "processing" for r in results)


def test_ingest_documents_empty_batch_returns_empty_list():
    repo = FakeRepo()
    repo.members.add("u1")
    svc = make_service(repo)

    assert svc.ingest_documents(CTX, documents=[]) == []


def test_get_document_passes_through():
    repo = FakeRepo()
    repo.documents["d1"] = {"documentId": "d1", "text": "hi", "chunks": []}
    svc = make_service(repo)

    assert svc.get_document(CTX, document_id="d1") == {"documentId": "d1", "text": "hi"}


def test_get_document_none_when_absent():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.get_document(CTX, document_id="nope") is None


# ── list_document_chunks / search_documents (K-050 M5 Stage 2) ──────────────────


def test_list_document_chunks_returns_chunkid_and_text_only():
    repo = FakeRepo()
    repo.documents["d1"] = {
        "documentId": "d1",
        "chunks": [
            {"chunkId": "c1", "text": "a", "seq": 0},
            {"chunkId": "c2", "text": "b", "seq": 1},
        ],
    }
    svc = make_service(repo)

    rows = svc.list_document_chunks(CTX, document_id="d1")

    assert rows == [{"chunkId": "c1", "text": "a"}, {"chunkId": "c2", "text": "b"}]


def test_list_document_chunks_empty_when_document_absent():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.list_document_chunks(CTX, document_id="nope") == []


class FakeEmbeddingGateway:
    """Minimal `ModelGateway`-shaped double for `search_documents`: only
    `.embedder(kind, *, ws=None)` is exercised (unlike `FakeModels` above,
    which only implements `.resolve` for the publish-time check)."""

    def __init__(self, vector):
        self._vector = vector
        self.embedder_calls: list[tuple] = []
        self.embedded: list[str] = []

    def embedder(self, kind, *, ws=None):
        self.embedder_calls.append((kind, ws))
        return self

    def embed(self, text):
        self.embedded.append(text)
        return list(self._vector)


def test_search_documents_embeds_the_query_then_searches_chunks():
    repo = FakeRepo()
    repo.since_rows = [
        {"chunkId": "c1", "text": "x", "documentId": "d1", "seq": 0, "score": 0.1},
    ]
    models = FakeEmbeddingGateway([1.0, 0.0])
    svc = make_service(repo, models=models)

    rows = svc.search_documents(CTX, query="hello", limit=5)

    assert rows == repo.since_rows
    assert models.embedded == ["hello"]
    assert models.embedder_calls == [("embedding", "test")]
    call = next(c for c in repo.calls if c[0] == "search_chunks")
    assert call == ("search_chunks", "test", (1.0, 0.0), 5, 5, RAG_QUERY_TIMEOUT_MS)


def test_search_documents_defaults_limit_to_20():
    repo = FakeRepo()
    models = FakeEmbeddingGateway([1.0])
    svc = make_service(repo, models=models)

    svc.search_documents(CTX, query="hello")

    call = next(c for c in repo.calls if c[0] == "search_chunks")
    assert call[3:5] == (20, 20)  # k, limit both default to 20


def test_search_documents_raises_when_no_models_wired():
    repo = FakeRepo()
    svc = make_service(repo, models=None)

    with pytest.raises(SearchNotAvailableError):
        svc.search_documents(CTX, query="hello")

    assert repo.calls == []  # never reaches search_chunks


# ── §14.6 Entity fusion — SAME_AS review surface (K-050 M5 Stage 4) ─────────────


def _seed_service_match(repo, *, match_id="m1", status="pending"):
    repo.matches[match_id] = {
        "status": status, "entityA": "e1", "entityB": "e2",
        "nameA": "Acme", "nameB": "Acme Co", "confidence": 2.0,
        "technique": "fuzzy_fulltext", "createdAt": 100,
    }


def test_confirm_match_stamps_the_calling_actor_never_system():
    repo = FakeRepo()
    _seed_service_match(repo)
    svc = make_service(repo, now=555)

    result = svc.confirm_match(CTX, match_id="m1")

    assert result["status"] == "confirmed"
    assert repo.calls[-1] == ("confirm_match", "test", "m1", CTX.actor, 555)


def test_confirm_match_raises_not_found_for_an_unknown_match_id():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(MatchNotFoundError):
        svc.confirm_match(CTX, match_id="nope")


def test_reject_match_stamps_the_calling_actor():
    repo = FakeRepo()
    _seed_service_match(repo)
    svc = make_service(repo, now=555)

    result = svc.reject_match(CTX, match_id="m1")

    assert result["status"] == "rejected"
    assert repo.calls[-1] == ("reject_match", "test", "m1", CTX.actor, 555)


def test_reject_match_raises_not_found_for_an_unknown_match_id():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(MatchNotFoundError):
        svc.reject_match(CTX, match_id="nope")


def test_recheck_match_flips_a_rejected_match_back_to_pending():
    repo = FakeRepo()
    _seed_service_match(repo, status="rejected")
    svc = make_service(repo, now=999)

    result = svc.recheck_match(CTX, match_id="m1")

    assert result == {"matchId": "m1", "status": "pending", "entityA": "e1", "entityB": "e2"}


def test_recheck_match_returns_none_instead_of_raising_for_a_pending_match():
    # Mirrors the repository's own inability to distinguish "unknown matchId"
    # from "exists but isn't rejected" — the service does not raise either way.
    repo = FakeRepo()
    _seed_service_match(repo, status="pending")
    svc = make_service(repo)

    assert svc.recheck_match(CTX, match_id="m1") is None


def test_recheck_match_returns_none_for_an_unknown_match_id():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.recheck_match(CTX, match_id="nope") is None


def test_list_pending_matches_passes_through_the_repository_rows():
    repo = FakeRepo()
    _seed_service_match(repo, match_id="m1", status="pending")
    _seed_service_match(repo, match_id="m2", status="confirmed")
    svc = make_service(repo)

    rows = svc.list_pending_matches(CTX, limit=10)

    assert [r["matchId"] for r in rows] == ["m1"]
    assert repo.calls[-1] == ("list_pending_matches", "test", 10)


def test_list_matches_passes_status_and_limit_through():
    repo = FakeRepo()
    _seed_service_match(repo, match_id="m1", status="confirmed")
    svc = make_service(repo)

    rows = svc.list_matches(CTX, status="confirmed", limit=25)

    assert [r["matchId"] for r in rows] == ["m1"]
    assert repo.calls[-1] == ("list_matches", "test", "confirmed", 25)


def test_list_matches_with_no_status_lists_every_tier():
    repo = FakeRepo()
    _seed_service_match(repo, match_id="m1", status="pending")
    _seed_service_match(repo, match_id="m2", status="confirmed")
    svc = make_service(repo)

    rows = svc.list_matches(CTX)

    assert {r["matchId"] for r in rows} == {"m1", "m2"}
    assert repo.calls[-1] == ("list_matches", "test", None, 50)


# ── post_message: dispatch + validation ─────────────────────────────────────────


def test_post_message_missing_thread_errors():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(ThreadNotFoundError):
        svc.post_message(CTX, thread_id="nope", text="hi")


def test_post_message_unknown_actor_errors_instead_of_silent_noop():
    repo = FakeRepo()
    repo.threads.add("t1")  # thread exists but the actor u1 is not a member
    svc = make_service(repo)

    with pytest.raises(UnknownActorError):
        svc.post_message(CTX, thread_id="t1", text="hi")

    assert not any(c[0].startswith("post_") for c in repo.calls)


def test_ensure_actor_projects_context_actor_as_user():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.ensure_actor(CTX)

    assert ("ensure_user", "test", "u1") in repo.calls


def test_ensure_actor_propagates_member_id_collision():
    """DEF-1: an actor id held by an Agent must surface, never silently shadow."""
    repo = FakeRepo()
    repo.agents.add("u1")
    svc = make_service(repo)

    with pytest.raises(MemberIdCollisionError):
        svc.ensure_actor(CTX)


# ── list_thread_participants (K-036 — web-api-coverage U5/FR-8) ─────────────


def test_list_thread_participants_normalizes_kind_from_type_list():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.threads.add("t1")
    repo.participants["t1"] = [
        {"memberId": "u1", "displayName": "Alice", "type": ["User"]},
        {"memberId": "a1", "displayName": None, "type": ["Agent"]},
    ]

    got = svc.list_thread_participants(CTX, thread_id="t1")

    assert got == [
        {"memberId": "u1", "displayName": "Alice", "kind": "User"},
        {"memberId": "a1", "displayName": None, "kind": "Agent"},
    ]
    assert ("list_thread_participants", "test", "t1") in repo.calls


def test_list_thread_participants_missing_thread_errors():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(ThreadNotFoundError):
        svc.list_thread_participants(CTX, thread_id="ghost")


def test_list_thread_participants_empty_when_no_members():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.threads.add("t1")

    assert svc.list_thread_participants(CTX, thread_id="t1") == []


def test_post_message_first_uses_first_write_path():
    repo = FakeRepo()
    repo.threads.add("t1")  # exists, no head yet
    repo.members.add("u1")
    svc = make_service(repo, now=1000)

    msg = svc.post_message(CTX, thread_id="t1", text="hello")

    assert repo.calls[-1][0] == "post_first_message"
    assert msg["msgId"] == "id1"
    assert msg["authorId"] == "u1"
    assert msg["role"] == "user"
    assert msg["createdAt"] == 1000


def test_post_message_subsequent_uses_append_path():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")  # already has a head
    repo.members.add("u1")
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="second")

    assert repo.calls[-1][0] == "post_subsequent_message"


def test_post_message_rejects_unknown_mention():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.update({"u1", "u2"})
    svc = make_service(repo)

    with pytest.raises(UnknownMemberError):
        svc.post_message(CTX, thread_id="t1", text="hi", mentions=["u2", "ghost"])

    # nothing written when validation fails
    assert not any(c[0].startswith("post_") for c in repo.calls)


def test_post_message_dedups_mentions_before_write():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.update({"u1", "u2"})
    svc = make_service(repo)

    msg = svc.post_message(CTX, thread_id="t1", text="hi", mentions=["u2", "u2"])

    kw = repo.calls[-1][1]
    assert kw["mentions"] == ["u2"]
    assert msg["mentions"] == ["u2"]


# ── post_message: v2 status dispatch + role derivation (K-007) ──────────────────


def test_post_message_agent_actor_gets_assistant_role():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.agents.add("a1")
    svc = make_service(repo)

    msg = svc.post_message(CallContext(ws="test", actor="a1"), thread_id="t1", text="hi")

    assert msg["role"] == "assistant"
    assert repo.calls[-1][1]["role"] == "assistant"  # derived, never trusted


def test_post_message_dup_msg_is_idempotent_success():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")
    repo.members.add("u1")
    repo.subseq_status = [DUP]  # retry replay of our own write
    svc = make_service(repo, now=1000)

    msg = svc.post_message(CTX, thread_id="t1", text="hi")

    assert msg["msgId"] == "id1"  # returned as success, no error
    assert msg["role"] == "user"


def test_post_message_had_head_redispatches_as_subsequent():
    repo = FakeRepo()
    repo.threads.add("t1")  # no HEAD seen at dispatch time…
    repo.members.add("u1")
    repo.first_status = [HAD_HEAD]  # …but another writer won the first-post race
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="hi")

    kinds = [c[0] for c in repo.calls if c[0].startswith("post_")]
    assert kinds == ["post_first_message", "post_subsequent_message"]


def test_post_message_tailless_subsequent_redispatches_as_first():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")  # HEAD seen at dispatch time…
    repo.members.add("u1")
    repo.subseq_status = [None]  # …but the anchor missed (no TAIL yet)
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="hi")

    kinds = [c[0] for c in repo.calls if c[0].startswith("post_")]
    assert kinds == ["post_subsequent_message", "post_first_message"]


def test_post_message_dispatch_loop_is_bounded():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.add("u1")
    # impossible-by-contract ping-pong: first says "had head", subsequent says "no TAIL"
    repo.first_status = [HAD_HEAD] * 10
    repo.subseq_status = [None] * 10
    svc = make_service(repo)

    with pytest.raises(RuntimeError):
        svc.post_message(CTX, thread_id="t1", text="hi")


def test_post_message_created_at_is_strictly_increasing_under_fixed_clock():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.add("u1")
    svc = make_service(repo, now=1000)  # frozen wall clock

    m1 = svc.post_message(CTX, thread_id="t1", text="one")
    m2 = svc.post_message(CTX, thread_id="t1", text="two")

    # monotonic per-process clock: same-ms ties are impossible at the source
    assert m1["createdAt"] == 1000
    assert m2["createdAt"] == 1001
    assert m2["createdAt"] > m1["createdAt"]


# ── read_messages: RO/RW dispatch ───────────────────────────────────────────────


def test_read_messages_explicit_since_is_pure_read_no_advance():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.read_messages(CTX, thread_id="t1", since=50, advance=True)

    # explicit since → plain `>` semantics (since_msg_id=None), cursor untouched
    assert ("read_thread_since", "t1", "u1", 50, None, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_thread_uses_cursor_and_advances_to_last_returned():
    repo = FakeRepo()
    repo.cursors["u1:t1"] = (200, "m0")
    repo.since_rows = [
        {"msgId": "m1", "createdAt": 300},
        {"msgId": "m2", "createdAt": 450},
    ]
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=True)

    assert ("read_thread_since", "t1", "u1", 200, "m0", 50) in repo.calls
    # cursor moves to the newest row actually delivered — NOT the server clock,
    # which would skip rows a `limit` truncated (and race concurrent posts)
    assert ("advance_cursor", "u1:t1", 450, "m2") in repo.calls


def test_read_messages_empty_page_does_not_advance():
    repo = FakeRepo()
    repo.cursors["u1:t1"] = (200, "m0")
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=True)  # since_rows is []

    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_no_cursor_defaults_since_zero():
    repo = FakeRepo()
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=False)

    # no cursor yet → epoch base with the composite '' msgId convention
    assert ("read_thread_since", "t1", "u1", 0, "", 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_room_wide_requires_no_thread_and_no_advance():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.read_messages(CTX, since=None, advance=True)  # no thread_id → room-wide, since 0

    assert ("read_ws_since", "u1", 0, None, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_cursor_pair_round_trips_as_composite_since():
    repo = FakeRepo()
    repo.since_rows = [
        {"msgId": "m2", "createdAt": 300},
        {"msgId": "m3", "createdAt": 300},  # millisecond tie — last row is max pair
    ]
    svc = make_service(repo)

    svc.read_messages(CTX, thread_id="t1", advance=True)

    # advanced to the last returned (createdAt, msgId) pair
    assert repo.cursors["u1:t1"] == (300, "m3")

    repo.since_rows = []
    svc.read_messages(CTX, thread_id="t1", advance=True)

    # the stored pair is fed back as the composite since — tied siblings with a
    # larger msgId would still be delivered, nothing re-delivered
    assert ("read_thread_since", "t1", "u1", 300, "m3", 50) in repo.calls


# ── search: thin passthrough ────────────────────────────────────────────────────


def test_search_messages_passes_query_and_limit_through():
    repo = FakeRepo()
    repo.since_rows = [{"msgId": "m1", "text": "hello", "createdAt": 120, "score": 1.5}]
    svc = make_service(repo)

    hits = svc.search_messages(CTX, query="hello", limit=10)

    assert ("search_messages", "hello", 10) in repo.calls
    assert hits == repo.since_rows


def test_search_messages_maps_syntax_error_to_service_error():
    repo = FakeRepo()
    repo.since_rows = ResponseError("RediSearch: Syntax error at offset 6")
    svc = make_service(repo)

    with pytest.raises(InvalidSearchQueryError):
        svc.search_messages(CTX, query='hello"unbalanced')


# ── hybrid_search (GraphRAG, Message + Chunk merge — K-050 M5 Stage 5) ──────────


def test_hybrid_search_applies_rag_timeout_constant():
    from falkorchat.services import RAG_QUERY_TIMEOUT_MS

    repo = FakeRepo()
    repo.hybrid_rows = [
        {"msgId": "m1", "text": "cats", "role": "user", "score": 0.0, "relatedContext": []}
    ]
    repo.chunk_rows = []
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0, 0.0], k=10, limit=5)

    assert hits == [{**repo.hybrid_rows[0], "seedKind": "Message"}]
    call = next(c for c in repo.calls if c[0] == "hybrid_search")
    # (name, ws, q_vec, k, limit, channel_id, timeout)
    assert call[1] == "test"
    assert call[3] == 10 and call[4] == 5
    assert call[5] is None
    assert call[6] == RAG_QUERY_TIMEOUT_MS
    # Chunk pool is queried with the same q_vec/k/timeout, no channel scope
    chunk_call = next(c for c in repo.calls if c[0] == "search_chunks")
    assert chunk_call == ("search_chunks", "test", (1.0, 0.0), 10, 5, RAG_QUERY_TIMEOUT_MS)


def test_hybrid_search_forwards_channel_scope_to_message_pool_only():
    repo = FakeRepo()
    repo.hybrid_rows = []
    repo.chunk_rows = []
    svc = make_service(repo)

    svc.hybrid_search(CTX, q_vec=[1.0], k=3, limit=3, channel_id="c1")

    call = next(c for c in repo.calls if c[0] == "hybrid_search")
    assert call[5] == "c1"
    # search_chunks has no channel_id parameter at all — the Chunk pool is
    # never scoped, confirming the documented "unscoped even when the
    # Message pool is channel-scoped" behavior.
    chunk_call = next(c for c in repo.calls if c[0] == "search_chunks")
    assert chunk_call == ("search_chunks", "test", (1.0,), 3, 3, RAG_QUERY_TIMEOUT_MS)


def test_hybrid_search_merges_message_and_chunk_pools_by_score_ascending():
    repo = FakeRepo()
    repo.hybrid_rows = [
        {"msgId": "m1", "text": "m1", "role": "user", "score": 0.5, "relatedContext": []},
        {"msgId": "m2", "text": "m2", "role": "user", "score": 0.1, "relatedContext": []},
    ]
    repo.chunk_rows = [
        {"chunkId": "c1", "text": "c1", "documentId": "d1", "seq": 0, "score": 0.3},
    ]
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0], k=10, limit=10)

    assert [h.get("msgId") or h.get("chunkId") for h in hits] == ["m2", "c1", "m1"]
    assert [h["score"] for h in hits] == [0.1, 0.3, 0.5]
    assert [h["seedKind"] for h in hits] == ["Message", "Chunk", "Message"]
    # Chunk items get the same dormant relatedContext:[] convention as Message
    assert hits[1]["relatedContext"] == []


def test_hybrid_search_truncates_merged_results_to_limit():
    repo = FakeRepo()
    repo.hybrid_rows = [
        {"msgId": "m1", "text": "m1", "role": "user", "score": 0.1, "relatedContext": []},
        {"msgId": "m2", "text": "m2", "role": "user", "score": 0.4, "relatedContext": []},
    ]
    repo.chunk_rows = [
        {"chunkId": "c1", "text": "c1", "documentId": "d1", "seq": 0, "score": 0.2},
        {"chunkId": "c2", "text": "c2", "documentId": "d1", "seq": 1, "score": 0.3},
    ]
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0], k=10, limit=2)

    assert len(hits) == 2
    assert [h.get("msgId") or h.get("chunkId") for h in hits] == ["m1", "c1"]


def test_hybrid_search_fewer_than_limit_combined_results():
    repo = FakeRepo()
    repo.hybrid_rows = [
        {"msgId": "m1", "text": "m1", "role": "user", "score": 0.1, "relatedContext": []},
    ]
    repo.chunk_rows = []
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0], k=10, limit=10)

    assert len(hits) == 1
    assert hits[0]["seedKind"] == "Message"


def test_hybrid_search_all_chunk_results_when_message_ann_empty():
    repo = FakeRepo()
    repo.hybrid_rows = []
    repo.chunk_rows = [
        {"chunkId": "c1", "text": "c1", "documentId": "d1", "seq": 0, "score": 0.2},
    ]
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0], k=10, limit=10)

    assert len(hits) == 1
    assert hits[0]["seedKind"] == "Chunk"
    assert hits[0]["chunkId"] == "c1"


def test_hybrid_search_all_message_results_when_chunk_ann_empty():
    repo = FakeRepo()
    repo.hybrid_rows = [
        {"msgId": "m1", "text": "m1", "role": "user", "score": 0.2, "relatedContext": []},
    ]
    repo.chunk_rows = []
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0], k=10, limit=10)

    assert len(hits) == 1
    assert hits[0]["seedKind"] == "Message"
    assert hits[0]["msgId"] == "m1"


# ── post_agent_answer: agent-authored answer + EMITTED provenance (K-013) ───────

CTX_AGENT = CallContext(ws="test", actor="bot1")


def _agent_svc(repo, *, now=1000):
    repo.threads.add("t1")
    repo.heads.add("t1")  # realistic path: trigger message is the HEAD → subsequent
    repo.agents.add("bot1")
    return make_service(repo, now=now)


def test_post_agent_answer_posts_as_agent_with_role_assistant_and_seeds():
    repo = FakeRepo()
    svc = _agent_svc(repo)

    out = svc.post_agent_answer(
        CTX_AGENT, thread_id="t1", text="the answer",
        seeds=[("s1", 0.1), ("s2", 0.2)],
    )

    assert out["role"] == "assistant"       # derived from the Agent actor label
    assert out["authorId"] == "bot1"
    assert out["text"] == "the answer"
    assert out["seeds"] == [("s1", 0.1), ("s2", 0.2)]
    call = next(c for c in repo.calls if c[0] == "post_agent_answer")
    assert call[1]["role"] == "assistant"
    assert call[1]["author_id"] == "bot1"
    assert call[1]["seeds"] == [("s1", 0.1), ("s2", 0.2)]


def test_post_agent_answer_missing_thread_errors():
    repo = FakeRepo()
    repo.agents.add("bot1")
    svc = make_service(repo)

    with pytest.raises(ThreadNotFoundError):
        svc.post_agent_answer(CTX_AGENT, thread_id="nope", text="hi", seeds=[])


def test_post_agent_answer_unknown_actor_errors():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")
    svc = make_service(repo)  # bot1 not registered

    with pytest.raises(UnknownActorError):
        svc.post_agent_answer(CTX_AGENT, thread_id="t1", text="hi", seeds=[])
    assert not any(c[0].startswith("post_agent") for c in repo.calls)


def test_post_agent_answer_validates_mentions():
    repo = FakeRepo()
    svc = _agent_svc(repo)

    with pytest.raises(UnknownMemberError):
        svc.post_agent_answer(
            CTX_AGENT, thread_id="t1", text="hi", mentions=["ghost"], seeds=[]
        )


def test_post_agent_answer_retries_as_first_when_no_tail():
    repo = FakeRepo()
    svc = _agent_svc(repo)
    repo.agent_subseq_status = [None]  # subsequent path: no TAIL → dispatch to first

    out = svc.post_agent_answer(CTX_AGENT, thread_id="t1", text="hi", seeds=[])

    assert out["role"] == "assistant"
    names = [c[0] for c in repo.calls if c[0].startswith("post_agent")]
    assert names == ["post_agent_answer", "post_agent_answer_first"]


def test_post_agent_answer_dup_msg_is_idempotent_success():
    repo = FakeRepo()
    svc = _agent_svc(repo)
    repo.agent_subseq_status = [DUP]

    out = svc.post_agent_answer(CTX_AGENT, thread_id="t1", text="hi", seeds=[])
    assert out["msgId"]  # returned cleanly, no raise


# ── §11 Workflow definitions & snapshots (M3 Slice 1) ───────────────────────────
#
# publish_workflow_def validates the spec BEFORE any write (plan §B5): unknown
# kind/step-type, duplicate step keys, a start marker that isn't exactly one
# declared step, and dangling transition endpoints all raise WorkflowDefSpecError
# and write nothing. A step marks itself the start with `start: True` (exactly one
# required — that step's key becomes the repo `start_key`). config/guard are
# serialized to opaque strings. materialize_def is two-phase: read the def from
# `reference`, write the snapshot into ctx.ws. Def authoring/reading is global
# (repo omits ws); only materialization + snapshot reads consume ctx.ws.

from falkorchat.services import (  # noqa: E402
    WorkflowDefNotFoundError,
    WorkflowDefSpecError,
)

VALID_STEPS = [
    # `waitsForHuman` is mandatory on a `human`/`wait` step (K-024 U2 publish invariant).
    {"key": "start", "type": "human", "config": {"a": 1, "waitsForHuman": True},
     "start": True},
    {"key": "review", "type": "decision", "config": "raw-string"},
    {"key": "done", "type": "message"},  # no config → serializes to ""
]
VALID_TRANSITIONS = [
    {"from": "start", "to": "review", "on": "submitted", "order": 0},  # no guard → ""
    {"from": "review", "to": "done", "on": "approved",
     "guard": {"expr": "x>0"}, "order": 0},
]

# The repository-shaped ("already stored") structure that a first successful
# `_publish(svc, repo)` of VALID_STEPS/VALID_TRANSITIONS would leave behind —
# serialized `config`/`guard`, `start` flag stripped, matches
# `test_publish_workflow_def_derives_start_and_serializes_config_and_guard`'s
# own assertions byte-for-byte. Seeds `repo.def_structures`/`snapshot_structures`
# to simulate "this (key, version) is already published" for the K-034 gate tests.
EXISTING_ONBOARDING_STRUCTURE = {
    "name": "Onboarding", "kind": "process", "start_keys": ["start"],
    "steps": [
        {"key": "start", "type": "human", "config": '{"a":1,"waitsForHuman":true}'},
        {"key": "review", "type": "decision", "config": "raw-string"},
        {"key": "done", "type": "message", "config": ""},
    ],
    "transitions": [
        {"from": "start", "to": "review", "on": "submitted", "order": 0, "guard": ""},
        {"from": "review", "to": "done", "on": "approved", "order": 0,
         "guard": '{"expr":"x>0"}'},
    ],
}

# Same topology as `EXISTING_ONBOARDING_STRUCTURE`, but in `read_def_subgraph`'s
# single-`start_key` shape (`repo.defs`) — the `materialize_def` read source.
EXISTING_ONBOARDING_SUBGRAPH = {
    "name": "Onboarding", "kind": "process", "start_key": "start",
    "steps": [
        {"key": "start", "type": "human", "config": '{"a":1,"waitsForHuman":true}'},
        {"key": "review", "type": "decision", "config": "raw-string"},
        {"key": "done", "type": "message", "config": ""},
    ],
    "transitions": [
        {"from": "start", "to": "review", "on": "submitted", "order": 0, "guard": ""},
        {"from": "review", "to": "done", "on": "approved", "order": 0,
         "guard": '{"expr":"x>0"}'},
    ],
}


def _publish(svc, repo, *, kind="process", steps=None, transitions=None):
    return svc.publish_workflow_def(
        CTX, key="onboarding", version="1", name="Onboarding", kind=kind,
        steps=steps if steps is not None else VALID_STEPS,
        transitions=transitions if transitions is not None else VALID_TRANSITIONS,
    )


def test_publish_workflow_def_derives_start_and_serializes_config_and_guard():
    repo = FakeRepo()
    svc = make_service(repo)

    _publish(svc, repo)

    assert len(repo.published) == 1
    pub = repo.published[0]
    assert pub["start_key"] == "start"                 # derived from start:True
    # steps handed to the repo carry only {key,type,config}; config is a string
    by_key = {s["key"]: s for s in pub["steps"]}
    # dict → compact JSON, stable key order
    assert by_key["start"]["config"] == '{"a":1,"waitsForHuman":true}'
    assert by_key["review"]["config"] == "raw-string"  # str passthrough
    assert by_key["done"]["config"] == ""              # missing → ""
    assert "start" not in by_key["start"]              # start flag stripped
    # transition guards serialized to strings
    trs = {(t["from"], t["to"]): t for t in pub["transitions"]}
    assert trs[("start", "review")]["guard"] == ""         # missing → ""
    assert trs[("review", "done")]["guard"] == '{"expr":"x>0"}'


def test_publish_workflow_def_returns_repo_result():
    repo = FakeRepo()
    svc = make_service(repo)

    out = _publish(svc, repo)

    assert out["key"] == "onboarding"
    assert out["version"] == "1"
    assert out["stepCount"] == 3
    assert out["transitionCount"] == 2


def test_publish_workflow_def_invalid_kind_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(WorkflowDefSpecError):
        _publish(svc, repo, kind="chatbot")  # not conversation|process

    assert repo.published == []


def test_publish_workflow_def_conversation_kind_allowed():
    repo = FakeRepo()
    svc = make_service(repo)

    _publish(svc, repo, kind="conversation")

    assert repo.published[0]["kind"] == "conversation"


def test_publish_workflow_def_invalid_step_type_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    bad = [
        {"key": "start", "type": "bogus", "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError):
        _publish(svc, repo, steps=bad, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_duplicate_step_key_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    dup = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "start", "type": "message"},
    ]

    # `match=` is load-bearing (B-2): the K-024 U2 invariants were added AFTER this
    # check, so a type-only assertion would pass even if the duplicate-key check were
    # deleted and some later invariant raised instead. Asserting the message keeps this
    # test a real regression net for the rule it names.
    with pytest.raises(WorkflowDefSpecError, match="duplicate step key"):
        _publish(svc, repo, steps=dup, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_no_start_step_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    no_start = [
        {"key": "a", "type": "human", "config": {"waitsForHuman": True}},
        {"key": "b", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="exactly one start step"):
        _publish(svc, repo, steps=no_start, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_multiple_start_steps_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    two_starts = [
        {"key": "a", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "b", "type": "message", "start": True},
    ]

    with pytest.raises(WorkflowDefSpecError, match="exactly one start step"):
        _publish(svc, repo, steps=two_starts, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_zero_transitions_raises_nothing_written():
    # O-6 (U4b M-B): `POST /workflow-defs` with `"transitions": []` is schema-legal, and
    # `_PUBLISH_CYPHER`'s trailing `UNWIND $transitions` collapses the row stream AFTER
    # the def, its steps and the START edge are MERGEd — a 500 plus a `(key, version)`
    # that `MERGE … ON CREATE SET` can never repair. Reject before any write instead.
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(WorkflowDefSpecError, match="at least one transition"):
        _publish(svc, repo, transitions=[])

    assert repo.published == []


def test_the_zero_transition_rule_runs_last_and_cannot_mask_an_older_check():
    # Ordering pin, same contract as the U2 invariants' pin: this spec violates BOTH
    # the start-count rule and the transitions rule, and must fail for the start count.
    repo = FakeRepo()
    svc = make_service(repo)
    no_start = [
        {"key": "a", "type": "human", "config": {"waitsForHuman": True}},
        {"key": "b", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="exactly one start step"):
        _publish(svc, repo, steps=no_start, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_dangling_transition_from_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    bad_tr = [{"from": "ghost", "to": "done", "on": "x", "order": 0}]
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="from 'ghost' is not a declared"):
        _publish(svc, repo, steps=steps, transitions=bad_tr)

    assert repo.published == []


def test_publish_workflow_def_dangling_transition_to_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    bad_tr = [{"from": "start", "to": "ghost", "on": "x", "order": 0}]
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="to 'ghost' is not a declared"):
        _publish(svc, repo, steps=steps, transitions=bad_tr)

    assert repo.published == []


# ── K-049 — oversized indexed-property guard ─────────────────────────────────
#
# FalkorDB v4.18.11 SIGSEGVs the whole redis-server process when a CREATE/MERGE
# commits a value >4096 bytes into a UNIQUE-constrained property
# (docs/reviews/unique-constraint-oversized-value-crash-rca.md). pydantic
# bounds the REST front door (MAX_KEY_LEN=200 on schemas.py's WorkflowStepIn.key
# etc.), but every non-REST caller (tests, scripts, a future MCP tool) goes
# through `Services` directly and bypassed that entirely until
# `_validate_key_lengths` (docs/plans/oversized-indexed-property-guard-graph.md
# §5) was added. These are offline service-layer tests — nothing here ever
# writes an oversized value to a live FalkorDB; that risk was retired via the
# RCA's disposable-container repro, never a standing test.

OVERSIZED_KEY = "x" * (MAX_KEY_LEN + 1)
# An isolated step carrying the oversized key, declared but never referenced by
# any transition — deliberately, so the step-key-length case below can't be
# accidentally caught by the *dangling-transition* structural check instead of
# the length guard (both raise `WorkflowDefSpecError`, so `pytest.raises` alone
# would pass either way; `match=` below pins the case to the length reason).
# Not used for the transition-endpoint cases: `_validate_key_lengths` runs
# checks in `key, version, steps, transitions` order and raises on the FIRST
# violation, so declaring the oversized value as a step too would make the
# step-key check fire first and mask the transition-endpoint check entirely —
# those cases leave the oversized value out of `steps`, on purpose.
OVERSIZED_ISOLATED_STEP = {"key": OVERSIZED_KEY, "type": "message"}


def _spec(**overrides):
    spec = {
        "key": "onboarding", "version": "1", "name": "Onboarding", "kind": "process",
        "steps": VALID_STEPS, "transitions": VALID_TRANSITIONS,
    }
    spec.update(overrides)
    return spec


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # `match=` pins each case to the length guard's OWN wording
        # ("over the N-character bound"), never a bare word like "key" or
        # "from" that a coincidentally-raised structural error (e.g. the
        # dangling-transition check) could also satisfy — load-bearing for the
        # mutation test: disabling the guard call must make every one of these
        # fail, not silently pass because some other check happened to raise
        # a `WorkflowDefSpecError` too.
        ({"key": OVERSIZED_KEY}, r"^key would be \d+ characters, over the"),
        ({"version": OVERSIZED_KEY}, r"^version would be \d+ characters, over the"),
        (
            {"steps": [*VALID_STEPS, OVERSIZED_ISOLATED_STEP]},
            r"^step key would be \d+ characters, over the",
        ),
        (
            {
                "transitions": [
                    {"from": OVERSIZED_KEY, "to": "done", "on": "x", "order": 0},
                    *VALID_TRANSITIONS,
                ],
            },
            r"^transition 'from' would be \d+ characters, over the",
        ),
        (
            {
                "transitions": [
                    {"from": "start", "to": OVERSIZED_KEY, "on": "x", "order": 0},
                    *VALID_TRANSITIONS,
                ],
            },
            r"^transition 'to' would be \d+ characters, over the",
        ),
    ],
    ids=["oversized-key", "oversized-version", "oversized-step-key",
         "oversized-transition-from", "oversized-transition-to"],
)
def test_publish_workflow_def_oversized_value_raises_nothing_written(overrides, match):
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(WorkflowDefSpecError, match=match):
        svc.publish_workflow_def(CTX, **_spec(**overrides))

    assert repo.published == []


def test_publish_workflow_def_oversized_transition_on_is_not_bounded_by_this_guard():
    # Deliberate, per the design doc (§5.1): `on` never feeds `Step.stepUid`'s
    # MERGE key the way `from`/`to` do, and this schema has zero
    # RELATIONSHIP-type constraints on TRANSITION — so it isn't at risk of the
    # crash class this guard defends against. Only pydantic's own
    # `Field(max_length=MAX_KEY_LEN)` bounds it, and only at the REST boundary.
    # A direct (non-REST) `Services` call with an oversized `on` must NOT be
    # rejected by `_validate_key_lengths` — proving the guard's scope stays
    # exactly as narrow as designed, not scope-creeping onto a field that was
    # deliberately left out.
    repo = FakeRepo()
    svc = make_service(repo)
    transitions = [
        {**VALID_TRANSITIONS[0], "on": OVERSIZED_KEY},
        VALID_TRANSITIONS[1],
    ]

    svc.publish_workflow_def(CTX, **_spec(transitions=transitions))

    assert len(repo.published) == 1


# ── K-034 — topology-equality gate on re-publish ─────────────────────────────
#
# `_check_no_structural_conflict`, wired into `publish_workflow_def` right before
# the repository write. §6.1 cases 1-3 (no existing / byte-identical / property-
# only) must stay `201`-equivalent (succeeds, exactly one `repo.published` entry);
# cases 4-7 (new step key / changed transition `to` / changed start key / removed
# transition) must raise `WorkflowDefConflictError` with nothing written.


def test_publish_workflow_def_no_existing_structure_succeeds_unaffected():
    repo = FakeRepo()  # repo.def_structures empty — first-time publish
    svc = make_service(repo)

    _publish(svc, repo)

    assert len(repo.published) == 1


def test_publish_workflow_def_identical_resubmission_succeeds():
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = dict(EXISTING_ONBOARDING_STRUCTURE)
    svc = make_service(repo)

    _publish(svc, repo)  # byte-identical to what's "already stored"

    assert len(repo.published) == 1


def test_publish_workflow_def_property_only_difference_succeeds():
    # mirrors the K-031-pinned API test's exact scenario: stored name/kind/step
    # config differ, topology (step keys, transition identities, start) is the same.
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = {
        "name": "Old Name", "kind": "conversation", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"different":true}'},
            {"key": "review", "type": "decision", "config": "old-raw-string"},
            {"key": "done", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "review", "on": "submitted", "order": 0,
             "guard": ""},
            {"from": "review", "to": "done", "on": "approved", "order": 0,
             "guard": '{"expr":"old"}'},
        ],
    }
    svc = make_service(repo)

    _publish(svc, repo)  # kind="process" (default) differs from stored "conversation"

    assert len(repo.published) == 1


def test_publish_workflow_def_new_step_key_raises_nothing_written():
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "go", "order": 0, "guard": ""},
        ],
    }
    svc = make_service(repo)
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
        {"key": "extra", "type": "message"},   # topology grows — new step key
    ]
    transitions = [{"from": "start", "to": "done", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefConflictError):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []


def test_publish_workflow_def_changed_transition_to_raises_nothing_written():
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
            {"key": "other", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "go", "order": 0, "guard": ""},
        ],
    }
    svc = make_service(repo)
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
        {"key": "other", "type": "message"},
    ]
    # same (from, on, order) as stored, but a different `to` — a parallel edge,
    # not an update (§2.1's MERGE-key analysis)
    transitions = [{"from": "start", "to": "other", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefConflictError):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []


def test_publish_workflow_def_changed_start_key_raises_nothing_written():
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = dict(EXISTING_ONBOARDING_STRUCTURE)
    svc = make_service(repo)
    # same step set/transitions as VALID_STEPS/VALID_TRANSITIONS, but `start:True`
    # moved from "start" to "review" — only the start key differs
    moved_start_steps = [
        {"key": "start", "type": "human", "config": {"a": 1, "waitsForHuman": True}},
        {"key": "review", "type": "decision", "config": "raw-string", "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefConflictError):
        _publish(svc, repo, steps=moved_start_steps)

    assert repo.published == []


def test_publish_workflow_def_removed_transition_raises_nothing_written():
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = dict(EXISTING_ONBOARDING_STRUCTURE)
    svc = make_service(repo)
    # drop review->done — candidate has fewer transitions than stored
    fewer_transitions = [
        {"from": "start", "to": "review", "on": "submitted", "order": 0},
    ]

    with pytest.raises(WorkflowDefConflictError):
        _publish(svc, repo, transitions=fewer_transitions)

    assert repo.published == []


# ── K-042 L2-4 — publish-time model-resolvability check (FR-9) ──────────────────
#
# `_check_models_resolvable` runs LAST inside `publish_workflow_def`, immediately
# before the repository write and AFTER `_check_no_structural_conflict` (§2.7,
# M-4): a candidate that both changes topology and names an unresolvable model
# must return the K-034 409, never this check's 400. A `Services` built without a
# gateway (`models=None`, the default) skips the pass entirely but WARNs when the
# def declares a model (m-7) — the skip must never itself block a publish.


class FakeModels:
    """Minimal `ModelGateway`-shaped double: `.resolve(kind, *, requested=None,
    ws=None, overrides=None)` records every call and raises `ModelResolutionError`
    for any `requested` not in `known`. Also asserts `ws=`/`overrides=` are never
    passed — publish-time validation must never see workspace state
    (`-graph.md` §6.3), and this double is the enforcement point for that rule."""

    def __init__(self, known=()):
        self.known = set(known)
        self.calls: list[tuple[str, str | None]] = []

    def resolve(self, kind, *, requested=None, ws=None, overrides=None):
        assert ws is None and overrides is None, (
            "L2-4 must resolve with no ws=/overrides= — a def is published to the "
            "global reference graph and must never see a workspace override"
        )
        self.calls.append((kind, requested))
        if requested not in self.known:
            raise ModelResolutionError(f"unknown ref {requested!r}")
        return object()


def _agent_step(key, *, model=None, start=False):
    step = {"key": key, "type": "agent", "config": {"model": model} if model else {}}
    if start:
        step["start"] = True
    return step


def test_publish_workflow_def_rejects_unresolvable_step_model():
    repo = FakeRepo()
    models = FakeModels(known=set())
    svc = make_service(repo, models=models)
    steps = [
        _agent_step("start", model="nope/thing", start=True),
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "start", "to": "done", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefSpecError, match=r"step 'start'.*nope/thing"):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []
    assert models.calls == [("step", "nope/thing")]


def test_publish_workflow_def_rejects_unresolvable_guard_model():
    repo = FakeRepo()
    models = FakeModels(known=set())
    svc = make_service(repo, models=models)
    steps = [
        {"key": "start", "type": "decision", "start": True},
        {"key": "done", "type": "message"},
    ]
    transitions = [
        {"from": "start", "to": "done", "on": "go", "order": 0,
         "guard": {"kind": "llm", "text": "is it ready?", "model": "nope/thing"}},
    ]

    with pytest.raises(
        WorkflowDefSpecError, match=r"transition 'start'->'done' on 'go'.*nope/thing"
    ):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []
    assert models.calls == [("guard", "nope/thing")]


def test_publish_workflow_def_non_llm_guard_with_model_key_is_not_checked():
    # a `{"kind":"cmp", ...}` (or any non-"llm") guard's stray "model" key, if one
    # ever appeared, is not this check's concern — only `{"kind":"llm"}` guards
    # that actually name a model are checked.
    repo = FakeRepo()
    models = FakeModels(known=set())
    svc = make_service(repo, models=models)
    steps = [
        {"key": "start", "type": "decision", "start": True},
        {"key": "done", "type": "message"},
    ]
    transitions = [
        {"from": "start", "to": "done", "on": "go", "order": 0,
         "guard": {"kind": "expr", "expr": "x>0", "model": "nope/thing"}},
    ]

    _publish(svc, repo, steps=steps, transitions=transitions)

    assert len(repo.published) == 1
    assert models.calls == []


def test_publish_workflow_def_with_valid_role_succeeds():
    repo = FakeRepo()
    models = FakeModels(known={"nightly"})
    svc = make_service(repo, models=models)
    steps = [
        _agent_step("start", model="nightly", start=True),
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "start", "to": "done", "on": "go", "order": 0}]

    _publish(svc, repo, steps=steps, transitions=transitions)

    assert len(repo.published) == 1
    assert models.calls == [("step", "nightly")]


def test_publish_workflow_def_with_no_declared_models_never_calls_gateway():
    repo = FakeRepo()
    models = FakeModels(known=set())
    svc = make_service(repo, models=models)

    _publish(svc, repo)  # VALID_STEPS/VALID_TRANSITIONS name no model

    assert len(repo.published) == 1
    assert models.calls == []


def test_publish_workflow_def_topology_conflict_wins_over_model_resolution_m4():
    """M-4 regression (§2.7): a candidate that both changes topology (K-034) and
    names an unresolvable model must fail with the 409 (`WorkflowDefConflictError`),
    never the 400 the model check would raise — and the model check must never even
    run. Proven by asserting the gateway was never called, not just by the
    exception type, so this test actually exercises the ordering."""
    repo = FakeRepo()
    repo.def_structures[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "go", "order": 0, "guard": ""},
        ],
    }
    models = FakeModels(known=set())  # "nope/thing" would fail resolution too
    svc = make_service(repo, models=models)
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
        _agent_step("extra", model="nope/thing"),  # new step key AND bad model
    ]
    transitions = [{"from": "start", "to": "done", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefConflictError):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []
    assert models.calls == []  # the model check never ran — ordering, proven


def test_publish_workflow_def_without_gateway_skips_check_and_warns(caplog):
    repo = FakeRepo()
    svc = make_service(repo, models=None)  # no gateway wired (m-7)
    steps = [
        _agent_step("start", model="nope/thing", start=True),
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "start", "to": "done", "on": "go", "order": 0}]

    with caplog.at_level("WARNING", logger="falkorchat.services"):
        _publish(svc, repo, steps=steps, transitions=transitions)

    # the skip must never itself block publish — that would be worse than the
    # silence m-7 exists to fix.
    assert len(repo.published) == 1
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "onboarding" in r.getMessage() and "nope/thing" in r.getMessage()
        for r in warnings
    )


def test_publish_workflow_def_without_gateway_and_no_declared_model_is_silent(caplog):
    repo = FakeRepo()
    svc = make_service(repo, models=None)

    with caplog.at_level("WARNING", logger="falkorchat.services"):
        _publish(svc, repo)  # VALID_STEPS/VALID_TRANSITIONS name no model

    assert len(repo.published) == 1
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_materialize_def_two_phase_reads_reference_then_writes_workspace():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": "start",
        "steps": [{"key": "start", "type": "human", "config": ""}],
        "transitions": [],
    }

    out = svc.materialize_def(CTX, key="onboarding", version="1")

    assert len(repo.materialized) == 1
    mat = repo.materialized[0]
    assert mat["ws"] == "test"                 # writes into ctx.ws
    assert mat["key"] == "onboarding"
    assert mat["name"] == "Onboarding"
    assert mat["start_key"] == "start"
    assert out["key"] == "onboarding"


def test_materialize_def_not_found_raises_nothing_materialized():
    repo = FakeRepo()
    svc = make_service(repo)  # repo.defs is empty

    with pytest.raises(WorkflowDefNotFoundError):
        svc.materialize_def(CTX, key="ghost", version="1")

    assert repo.materialized == []


def test_materialize_def_oversized_stored_step_key_raises_nothing_materialized():
    # K-049 §5.3: `publish_workflow_def`'s own guard cannot protect a def that
    # reached `reference` by some OTHER means (a hand-edited seed, a future
    # non-REST publish path, direct repository write) — simulate exactly that
    # threat model by writing past the guard straight into `repo.defs`, the
    # same fixture shape the K-034 tests below use for "already stored".
    repo = FakeRepo()
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": OVERSIZED_KEY,
        "steps": [{"key": OVERSIZED_KEY, "type": "human", "config": ""}],
        "transitions": [],
    }
    svc = make_service(repo)

    with pytest.raises(WorkflowDefSpecError, match=r"step key would be \d+ characters"):
        svc.materialize_def(CTX, key="onboarding", version="1")

    assert repo.materialized == []


# ── K-034 — topology-equality gate on re-materialize ─────────────────────────
#
# Mirrors the publish-side block above, against `ctx.ws`'s snapshot instead of
# `reference`: `repo.defs` seeds the `read_def_subgraph` source (what's about to
# be materialized), `repo.snapshot_structures` seeds "what's already in ctx.ws".


def test_materialize_def_no_existing_snapshot_succeeds_unaffected():
    repo = FakeRepo()  # repo.snapshot_structures empty — first-time materialize
    repo.defs[("onboarding", "1")] = dict(EXISTING_ONBOARDING_SUBGRAPH)
    svc = make_service(repo)

    svc.materialize_def(CTX, key="onboarding", version="1")

    assert len(repo.materialized) == 1


def test_materialize_def_identical_resubmission_succeeds():
    repo = FakeRepo()
    repo.defs[("onboarding", "1")] = dict(EXISTING_ONBOARDING_SUBGRAPH)
    repo.snapshot_structures[("onboarding", "1")] = dict(EXISTING_ONBOARDING_STRUCTURE)
    svc = make_service(repo)

    svc.materialize_def(CTX, key="onboarding", version="1")

    assert len(repo.materialized) == 1


def test_materialize_def_property_only_difference_succeeds():
    repo = FakeRepo()
    repo.defs[("onboarding", "1")] = dict(EXISTING_ONBOARDING_SUBGRAPH)
    repo.snapshot_structures[("onboarding", "1")] = {
        "name": "Old Name", "kind": "conversation", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"different":true}'},
            {"key": "review", "type": "decision", "config": "old-raw-string"},
            {"key": "done", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "review", "on": "submitted", "order": 0,
             "guard": ""},
            {"from": "review", "to": "done", "on": "approved", "order": 0,
             "guard": '{"expr":"old"}'},
        ],
    }
    svc = make_service(repo)

    svc.materialize_def(CTX, key="onboarding", version="1")

    assert len(repo.materialized) == 1


def test_materialize_def_new_step_key_raises_nothing_materialized():
    repo = FakeRepo()
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": "start",
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
            {"key": "extra", "type": "message", "config": ""},  # topology grows
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "go", "order": 0, "guard": ""},
        ],
    }
    repo.snapshot_structures[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "go", "order": 0, "guard": ""},
        ],
    }
    svc = make_service(repo)

    with pytest.raises(WorkflowDefConflictError):
        svc.materialize_def(CTX, key="onboarding", version="1")

    assert repo.materialized == []


def test_materialize_def_changed_transition_to_raises_nothing_materialized():
    repo = FakeRepo()
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": "start",
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
            {"key": "other", "type": "message", "config": ""},
        ],
        # same (from, on, order) as stored, but a different `to`
        "transitions": [
            {"from": "start", "to": "other", "on": "go", "order": 0, "guard": ""},
        ],
    }
    repo.snapshot_structures[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_keys": ["start"],
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman":true}'},
            {"key": "done", "type": "message", "config": ""},
            {"key": "other", "type": "message", "config": ""},
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "go", "order": 0, "guard": ""},
        ],
    }
    svc = make_service(repo)

    with pytest.raises(WorkflowDefConflictError):
        svc.materialize_def(CTX, key="onboarding", version="1")

    assert repo.materialized == []


def test_materialize_def_changed_start_key_raises_nothing_materialized():
    repo = FakeRepo()
    # same step set/transitions as EXISTING_ONBOARDING_SUBGRAPH, but the start
    # key moved from "start" to "review"
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": "review",
        "steps": EXISTING_ONBOARDING_SUBGRAPH["steps"],
        "transitions": EXISTING_ONBOARDING_SUBGRAPH["transitions"],
    }
    repo.snapshot_structures[("onboarding", "1")] = dict(EXISTING_ONBOARDING_STRUCTURE)
    svc = make_service(repo)

    with pytest.raises(WorkflowDefConflictError):
        svc.materialize_def(CTX, key="onboarding", version="1")

    assert repo.materialized == []


def test_materialize_def_removed_transition_raises_nothing_materialized():
    repo = FakeRepo()
    # drop review->done — the def subgraph about to be materialized has fewer
    # transitions than what's already in the workspace snapshot
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": "start",
        "steps": EXISTING_ONBOARDING_SUBGRAPH["steps"],
        "transitions": [
            {"from": "start", "to": "review", "on": "submitted", "order": 0,
             "guard": ""},
        ],
    }
    repo.snapshot_structures[("onboarding", "1")] = dict(EXISTING_ONBOARDING_STRUCTURE)
    svc = make_service(repo)

    with pytest.raises(WorkflowDefConflictError):
        svc.materialize_def(CTX, key="onboarding", version="1")

    assert repo.materialized == []


def test_get_workflow_def_passthrough_is_global_no_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.defs[("onboarding", "1")] = {
        "key": "onboarding", "version": "1", "name": "Onboarding", "kind": "process",
    }

    got = svc.get_workflow_def(CTX, key="onboarding", version="1")

    assert got["name"] == "Onboarding"
    assert ("get_def", "onboarding", "1") in repo.calls


def test_list_workflow_defs_passthrough():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.defs[("a", "1")] = {"key": "a", "version": "1", "name": "A", "kind": "process"}

    out = svc.list_workflow_defs(CTX)

    assert out and out[0]["key"] == "a"


def test_get_snapshot_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.snapshots[("a", "1")] = {
        "name": "A", "kind": "process", "start_key": "s", "steps": [], "transitions": [],
    }

    got = svc.get_snapshot(CTX, key="a", version="1")

    assert got["name"] == "A"
    assert ("get_snapshot", "test", "a", "1") in repo.calls


def test_list_snapshots_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.snapshots[("a", "1")] = {"key": "a", "version": "1", "name": "A", "kind": "process"}

    out = svc.list_snapshots(CTX)

    assert out and out[0]["key"] == "a"
    assert ("list_snapshots", "test", 50) in repo.calls


# ── §11 def/snapshot structure reads + diff (K-031) ─────────────────────────────
#
# Canonical ordering, camelCase renaming and the comparator are pure service
# logic (the repository stays a 1:1 mirror of QUERIES.md and returns steps and
# transitions unordered by design, F6), so they are pinned here against FakeRepo.


def _raw_structure(*, name="A", kind="process", start_keys=("s",), steps=None,
                   transitions=None):
    """A repository-shaped structure read (pre-canonicalization)."""
    return {
        "name": name,
        "kind": kind,
        "start_keys": list(start_keys),
        "steps": list(steps if steps is not None else [
            {"key": "s", "type": "human", "config": '{"waitsForHuman":true}'},
        ]),
        "transitions": list(transitions if transitions is not None else []),
    }


def test_def_structure_canonical_ordering_is_deterministic():
    repo = FakeRepo()
    svc = make_service(repo)
    # deliberately shuffled at the source — the graph returns no order (F6)
    repo.def_structures[("a", "1")] = _raw_structure(
        steps=[
            {"key": "zeta", "type": "message", "config": ""},
            {"key": "alpha", "type": "human", "config": "{}"},
            {"key": "mid", "type": "decision", "config": ""},
        ],
        transitions=[
            {"from": "mid", "to": "zeta", "on": "b", "order": 1, "guard": ""},
            {"from": "alpha", "to": "mid", "on": "go", "order": 0, "guard": "g"},
            {"from": "mid", "to": "alpha", "on": "a", "order": 0, "guard": ""},
        ],
    )

    out = svc.get_workflow_def_structure(CTX, key="a", version="1")

    assert [s["key"] for s in out["steps"]] == ["alpha", "mid", "zeta"]
    assert [(t["from"], t["order"], t["to"]) for t in out["transitions"]] == [
        ("alpha", 0, "mid"), ("mid", 0, "alpha"), ("mid", 1, "zeta"),
    ]
    assert out["stepCount"] == 3
    assert out["transitionCount"] == 3
    assert out["source"] == "reference"


def test_def_structure_start_keys_omitted_for_one_present_for_two():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.def_structures[("one", "1")] = _raw_structure(start_keys=("s",))
    # unsorted at the source — canonicalization sorts, so `startKey` is stable
    repo.def_structures[("two", "1")] = _raw_structure(start_keys=("z", "a"))

    one = svc.get_workflow_def_structure(CTX, key="one", version="1")
    two = svc.get_workflow_def_structure(CTX, key="two", version="1")

    assert one["startKey"] == "s"
    assert "startKeys" not in one
    assert two["startKeys"] == ["a", "z"]
    assert two["startKey"] == "a"


def test_structure_reads_use_the_ctx_ws_seam_only_for_the_snapshot():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.def_structures[("a", "1")] = _raw_structure()
    repo.snapshot_structures[("a", "1")] = _raw_structure()

    svc.get_workflow_def_structure(CTX, key="a", version="1")
    snap = svc.get_snapshot_structure(CTX, key="a", version="1")

    # the def read is global — no workspace argument at all
    assert ("read_def_structure", "a", "1") in repo.calls
    assert ("read_snapshot_structure", "test", "a", "1") in repo.calls
    assert snap["source"] == "workspace"


def test_snapshot_structure_absent_returns_none():
    svc = make_service(FakeRepo())

    assert svc.get_snapshot_structure(CTX, key="ghost", version="1") is None


def test_def_structure_absent_raises_not_found():
    svc = make_service(FakeRepo())

    with pytest.raises(WorkflowDefNotFoundError):
        svc.get_workflow_def_structure(CTX, key="ghost", version="1")


def _canon(**kwargs):
    from falkorchat.services import _canonical_structure

    return _canonical_structure(
        _raw_structure(**kwargs), source="reference", key="a", version="1"
    )


def test_diff_structures_identical_is_empty():
    assert _diff_structures(_canon(), _canon()) == []


@pytest.mark.parametrize(
    ("left", "right", "path", "def_v", "snap_v"),
    [
        # meta fields
        (dict(name="A"), dict(name="B"), "meta.name", "A", "B"),
        (dict(kind="process"), dict(kind="conversation"),
         "meta.kind", "process", "conversation"),
        (dict(start_keys=("s",)), dict(start_keys=("t",)), "meta.startKey", "s", "t"),
        (dict(start_keys=("s",)), dict(start_keys=("s", "t")),
         "meta.startKeys", None, "s,t"),
        # step presence, both directions
        (dict(steps=[{"key": "x", "type": "message", "config": ""}]),
         dict(steps=[]), "steps[x]", "present", "absent"),
        (dict(steps=[]),
         dict(steps=[{"key": "x", "type": "message", "config": ""}]),
         "steps[x]", "absent", "present"),
        # step payload
        (dict(steps=[{"key": "x", "type": "message", "config": ""}]),
         dict(steps=[{"key": "x", "type": "decision", "config": ""}]),
         "steps[x].type", "message", "decision"),
        (dict(steps=[{"key": "x", "type": "message", "config": "{}"}]),
         dict(steps=[{"key": "x", "type": "message", "config": '{"a":1}'}]),
         "steps[x].config", "{}", '{"a":1}'),
        # transition presence (identity = from,to,on,order) and payload
        (dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": ""}]),
         dict(transitions=[]),
         "transitions[s->x@go#0]", "present", "absent"),
        (dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": "g1"}]),
         dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": "g2"}]),
         "transitions[s->x@go#0].guard", "g1", "g2"),
    ],
)
def test_diff_structures_one_class_at_a_time(left, right, path, def_v, snap_v):
    diffs = _diff_structures(_canon(**left), _canon(**right))

    assert len(diffs) == 1, diffs
    assert diffs[0] == {"path": path, "def": def_v, "snapshot": snap_v}


def test_diff_structures_changed_transition_endpoint_reads_as_two_presences():
    # `_PUBLISH_CYPHER` MERGEs on (on, order) with the endpoints in the pattern,
    # so a changed `to` is a *different* transition — two presence rows, never a
    # single "modified" row (which is what a naive (from,to) keying would report).
    diffs = _diff_structures(
        _canon(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                             "guard": ""}]),
        _canon(transitions=[{"from": "s", "to": "y", "on": "go", "order": 0,
                             "guard": ""}]),
    )

    assert [(d["path"], d["def"], d["snapshot"]) for d in diffs] == [
        ("transitions[s->x@go#0]", "present", "absent"),
        ("transitions[s->y@go#0]", "absent", "present"),
    ]


# ── `_structural_diffs` — topology-only filter over `_diff_structures` (K-034) ──
#
# Mirrors `test_diff_structures_one_class_at_a_time`'s cases: property-only paths
# (`meta.name`/`meta.kind`, `.type`/`.config`/`.guard`) must be filtered OUT;
# presence-shaped paths (`meta.startKey`/`meta.startKeys`, a bare `steps[...]`/
# `transitions[...]` row) must survive.


@pytest.mark.parametrize(
    ("left", "right", "survives"),
    [
        # property-only — filtered out
        (dict(name="A"), dict(name="B"), False),
        (dict(kind="process"), dict(kind="conversation"), False),
        (dict(steps=[{"key": "x", "type": "message", "config": ""}]),
         dict(steps=[{"key": "x", "type": "decision", "config": ""}]), False),
        (dict(steps=[{"key": "x", "type": "message", "config": "{}"}]),
         dict(steps=[{"key": "x", "type": "message", "config": '{"a":1}'}]), False),
        (dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": "g1"}]),
         dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": "g2"}]), False),
        # structural — survives
        (dict(start_keys=("s",)), dict(start_keys=("t",)), True),
        (dict(start_keys=("s",)), dict(start_keys=("s", "t")), True),
        (dict(steps=[{"key": "x", "type": "message", "config": ""}]),
         dict(steps=[]), True),
        (dict(steps=[]),
         dict(steps=[{"key": "x", "type": "message", "config": ""}]), True),
        (dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": ""}]),
         dict(transitions=[]), True),
        (dict(transitions=[]),
         dict(transitions=[{"from": "s", "to": "x", "on": "go", "order": 0,
                            "guard": ""}]), True),
    ],
)
def test_structural_diffs_one_class_at_a_time(left, right, survives):
    diffs = _diff_structures(_canon(**left), _canon(**right))
    assert len(diffs) == 1, diffs

    survivors = _structural_diffs(diffs)

    assert (len(survivors) == 1) is survives


def test_structural_diffs_new_step_key_survives_alongside_a_filtered_property_diff():
    # the class of case §3.2 calls out explicitly: a topology change (new step)
    # must reject even when it arrives bundled with a property-only diff.
    diffs = _diff_structures(
        _canon(name="A", steps=[{"key": "s", "type": "human", "config": ""}]),
        _canon(name="B", steps=[
            {"key": "s", "type": "human", "config": ""},
            {"key": "extra", "type": "message", "config": ""},
        ]),
    )
    assert len(diffs) == 2  # meta.name (property) + steps[extra] (structural)

    survivors = _structural_diffs(diffs)

    assert [d["path"] for d in survivors] == ["steps[extra]"]


def test_diff_preview_truncates_long_opaque_values():
    long_a, long_b = "a" * 500, "b" * 500

    diffs = _diff_structures(
        _canon(steps=[{"key": "x", "type": "message", "config": long_a}]),
        _canon(steps=[{"key": "x", "type": "message", "config": long_b}]),
    )

    assert diffs[0]["def"] == "a" * MAX_DIFF_PREVIEW + "…"
    assert diffs[0]["snapshot"] == "b" * MAX_DIFF_PREVIEW + "…"


def test_diff_def_snapshot_in_sync_and_one_sided():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.def_structures[("a", "1")] = _raw_structure()
    repo.snapshot_structures[("a", "1")] = _raw_structure()

    same = svc.diff_def_snapshot(CTX, key="a", version="1")
    assert same["inSync"] is True
    assert same["differences"] == []
    assert same["differenceCount"] == 0

    # the documented post-`pytest` trap: `reference` wiped, snapshot survives
    del repo.def_structures[("a", "1")]
    one_sided = svc.diff_def_snapshot(CTX, key="a", version="1")
    assert one_sided["defPresent"] is False
    assert one_sided["snapshotPresent"] is True
    assert one_sided["inSync"] is False
    assert one_sided["differences"] == []


def test_diff_def_snapshot_both_absent_raises_not_found():
    svc = make_service(FakeRepo())

    with pytest.raises(WorkflowDefNotFoundError):
        svc.diff_def_snapshot(CTX, key="ghost", version="1")


# ── FR-10 workspace readiness (web-api-coverage plan §3.1c / U2) ────────────────
#
# `check_demo_readiness` is the HTTP form of `scripts/verify_workflows.sh`:
# same `DEMO_EXPECTED_DEFS` pair, same presence/sync/startKeys checks, same
# problem-string wording. These tests mirror the script's own fixture pattern
# (an `ABSENT`-shaped read for a def/snapshot that was never even attempted).


def _seed_pair(repo, key, version, *, def_start=("s",), snap_start=("s",)):
    """Populate both `def_structures`/`snapshot_structures` for one demo pair,
    identical apart from `startKey`(s), so the two sides stay `inSync` unless
    the caller deliberately diverges them."""
    repo.def_structures[(key, version)] = _raw_structure(start_keys=def_start)
    repo.snapshot_structures[(key, version)] = _raw_structure(start_keys=snap_start)


def test_check_demo_readiness_all_present_and_synced_is_ready():
    repo = FakeRepo()
    for key, version in DEMO_EXPECTED_DEFS:
        _seed_pair(repo, key, version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is True
    assert [d["key"] for d in out["defs"]] == [k for k, _ in DEMO_EXPECTED_DEFS]
    for d in out["defs"]:
        assert d["defPresent"] is True
        assert d["snapshotPresent"] is True
        assert d["inSync"] is True
        assert d["problems"] == []


def test_check_demo_readiness_missing_def_names_the_offender():
    key, version = DEMO_EXPECTED_DEFS[0]
    other_key, other_version = DEMO_EXPECTED_DEFS[1]
    repo = FakeRepo()
    # only the snapshot side exists for the first pair
    repo.snapshot_structures[(key, version)] = _raw_structure()
    _seed_pair(repo, other_key, other_version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is False
    entry = out["defs"][0]
    assert entry["defPresent"] is False
    assert entry["snapshotPresent"] is True
    assert entry["inSync"] is False
    assert entry["problems"] == [
        f"{key}@{version}: not published in `reference` at this version"
    ]


def test_check_demo_readiness_missing_snapshot_names_the_offender():
    key, version = DEMO_EXPECTED_DEFS[0]
    other_key, other_version = DEMO_EXPECTED_DEFS[1]
    repo = FakeRepo()
    # only the reference def exists for the first pair
    repo.def_structures[(key, version)] = _raw_structure()
    _seed_pair(repo, other_key, other_version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is False
    entry = out["defs"][0]
    assert entry["defPresent"] is True
    assert entry["snapshotPresent"] is False
    assert entry["inSync"] is False
    assert entry["problems"] == [
        f"{key}@{version}: not materialized into ws:{CTX.ws} at this version"
    ]


def test_check_demo_readiness_both_absent_is_not_ready_not_an_error():
    # both sides absent (the def was never even published) mirrors the
    # script's `read()` ABSENT substitution: `diff_def_snapshot` would raise
    # `WorkflowDefNotFoundError` here — `check_demo_readiness` must catch it,
    # never let it escape as a 500.
    key, version = DEMO_EXPECTED_DEFS[0]
    other_key, other_version = DEMO_EXPECTED_DEFS[1]
    repo = FakeRepo()
    _seed_pair(repo, other_key, other_version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is False
    entry = out["defs"][0]
    assert (entry["defPresent"], entry["snapshotPresent"], entry["inSync"]) == (
        False, False, False,
    )
    assert entry["problems"] == [
        f"{key}@{version}: not published in `reference` at this version",
        f"{key}@{version}: not materialized into ws:{CTX.ws} at this version",
    ]


def test_check_demo_readiness_diverging_names_the_offender_with_count():
    key, version = DEMO_EXPECTED_DEFS[0]
    other_key, other_version = DEMO_EXPECTED_DEFS[1]
    repo = FakeRepo()
    repo.def_structures[(key, version)] = _raw_structure(name="A")
    repo.snapshot_structures[(key, version)] = _raw_structure(name="B")
    _seed_pair(repo, other_key, other_version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is False
    entry = out["defs"][0]
    assert entry["defPresent"] is True
    assert entry["snapshotPresent"] is True
    assert entry["inSync"] is False
    assert entry["problems"] == [
        f"{key}@{version}: reference def and ws:{CTX.ws} snapshot diverge "
        f"(1 differences)"
    ]


def test_check_demo_readiness_flags_multi_start_tripwire():
    # Finding-3: `startKeys` present means more than one `START` edge (K-034).
    # Kept identical on both sides so the pair stays otherwise `inSync` — this
    # isolates the tripwire from the ordinary divergence check.
    key, version = DEMO_EXPECTED_DEFS[0]
    other_key, other_version = DEMO_EXPECTED_DEFS[1]
    repo = FakeRepo()
    _seed_pair(repo, key, version, def_start=("a", "b"), snap_start=("a", "b"))
    _seed_pair(repo, other_key, other_version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is False
    entry = out["defs"][0]
    assert entry["inSync"] is True
    assert entry["problems"] == [
        f"{key}@{version}: reference def has 2 START edges (a, b) — see K-034",
        f"{key}@{version}: ws:{CTX.ws} snapshot has 2 START edges (a, b) — see K-034",
    ]


# ── K-039 item 3: `postSuccess` (recent triage post-success signal) ─────────────
#
# Purely informational — must never affect `ready`/`defs` (plan §3.3). The
# existing `test_check_demo_readiness_*` tests above assert `ready`/`defs` and
# stay green unmodified: they don't script `post_success_result`, so `FakeRepo`'s
# default (`{"sampleSize": 0, "postedCount": 0}`, i.e. "no-data") applies and
# `postSuccess` is simply an extra key those tests don't look at.


def test_check_demo_readiness_post_success_ok_when_all_sampled_runs_posted():
    repo = FakeRepo()
    for key, version in DEMO_EXPECTED_DEFS:
        _seed_pair(repo, key, version)
    repo.post_success_result = {"sampleSize": 5, "postedCount": 5}
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["postSuccess"] == {
        "defKey": config.TRIGGER_DEF_KEY, "defVersion": config.TRIGGER_DEF_VERSION,
        "sampleSize": 5, "postedCount": 5, "rate": 1.0, "status": "ok",
    }
    # informational only — never mixed into ready/defs
    assert out["ready"] is True


def test_check_demo_readiness_post_success_degraded_when_some_unposted():
    repo = FakeRepo()
    for key, version in DEMO_EXPECTED_DEFS:
        _seed_pair(repo, key, version)
    repo.post_success_result = {"sampleSize": 4, "postedCount": 1}
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["postSuccess"]["status"] == "degraded"
    assert out["postSuccess"]["rate"] == 1 / 4
    assert out["postSuccess"]["sampleSize"] == 4
    assert out["postSuccess"]["postedCount"] == 1
    # still purely informational — a degraded post-success rate does not flip ready
    assert out["ready"] is True


def test_check_demo_readiness_post_success_no_data_when_sample_empty():
    repo = FakeRepo()
    for key, version in DEMO_EXPECTED_DEFS:
        _seed_pair(repo, key, version)
    repo.post_success_result = {"sampleSize": 0, "postedCount": 0}
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["postSuccess"]["status"] == "no-data"
    assert out["postSuccess"]["rate"] is None  # never 0% — that would read as unhealthy


def test_check_demo_readiness_post_success_uses_trigger_def_key_and_version():
    repo = FakeRepo()
    for key, version in DEMO_EXPECTED_DEFS:
        _seed_pair(repo, key, version)
    svc = make_service(repo)

    svc.check_demo_readiness(CTX)

    assert (
        "read_recent_post_success", CTX.ws, config.TRIGGER_DEF_KEY,
        config.TRIGGER_DEF_VERSION, POST_SUCCESS_SAMPLE_SIZE,
    ) in repo.calls


def test_check_demo_readiness_existing_ready_and_defs_assertions_are_regression_pinned():
    # a literal re-run of the all-present-and-synced case, confirming `ready`/
    # `defs` are byte-for-byte what they were before `postSuccess` landed —
    # the plan's explicit regression check (§5 item 2).
    repo = FakeRepo()
    for key, version in DEMO_EXPECTED_DEFS:
        _seed_pair(repo, key, version)
    svc = make_service(repo)

    out = svc.check_demo_readiness(CTX)

    assert out["ready"] is True
    assert [d["key"] for d in out["defs"]] == [k for k, _ in DEMO_EXPECTED_DEFS]
    for d in out["defs"]:
        assert d["defPresent"] is True
        assert d["snapshotPresent"] is True
        assert d["inSync"] is True
        assert d["problems"] == []
    assert "postSuccess" in out  # additive, not a replacement


# ── §12 workflow-run orchestration (U5) ─────────────────────────────────────────
#
# The service mints the run id + start timestamp (server clock), resolves the
# trigger message's thread into the run `ctx` (so a suspend can denorm it for the
# resume lookup), starts the run via the repository, then hands off to the injected
# executor. Reads are thin, ctx.ws-scoped pass-throughs. The engine itself is U4.


def test_start_workflow_run_mints_run_seeds_thread_ctx_and_drives():
    repo = FakeRepo()
    repo.messages["trig1"] = {"msgId": "trig1", "threadId": "t1"}
    ex = StubExecutor(step_budget=12, run_status="waiting")
    svc = make_service(repo, executor=ex)

    out = svc.start_workflow_run(
        CTX, def_key="triage", version="1", trigger_msg_id="trig1", trace=True
    )

    # minted a run id + started it at the snapshot, using the executor's budget
    started = repo.started_runs[0]
    assert started["def_key"] == "triage"
    assert started["def_version"] == "1"
    assert started["trigger_msg_id"] == "trig1"
    assert started["trace"] is True
    assert started["max_steps"] == 12
    # the trigger's thread is seeded into ctx for the resume denorm (§2.4)
    assert json.loads(started["ctx"]) == {"threadId": "t1"}
    # then drove the engine and returned its status
    assert ex.run_calls == [out["runId"]]
    assert out["status"] == "waiting"


def test_start_workflow_run_missing_anchor_raises_nothing_driven():
    repo = FakeRepo()
    repo.messages["trig1"] = {"msgId": "trig1", "threadId": "t1"}
    repo.start_run_result = None  # snapshot/trigger anchor missed
    ex = StubExecutor()
    svc = make_service(repo, executor=ex)

    with pytest.raises(WorkflowRunNotFoundError):
        svc.start_workflow_run(
            CTX, def_key="ghost", version="1", trigger_msg_id="trig1"
        )
    assert ex.run_calls == []  # never handed to the engine


def test_start_workflow_run_without_executor_raises():
    repo = FakeRepo()
    svc = make_service(repo)  # no executor wired
    with pytest.raises(RuntimeError):
        svc.start_workflow_run(
            CTX, def_key="triage", version="1", trigger_msg_id="trig1"
        )


def test_resume_workflow_run_delegates_to_executor():
    repo = FakeRepo()
    ex = StubExecutor(resume_status="done")
    svc = make_service(repo, executor=ex)

    out = svc.resume_workflow_run(CTX, run_id="r1")

    assert ex.resume_calls == ["r1"]
    assert out == {"runId": "r1", "status": "done"}


def test_get_workflow_run_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.runs["r1"] = {"runId": "r1", "status": "running", "atStepKey": "intake"}

    got = svc.get_workflow_run(CTX, run_id="r1")

    assert got["status"] == "running"
    assert ("get_run", "test", "r1") in repo.calls


def test_read_workflow_step_runs_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.step_runs["r1"] = [{"stepRunId": "sr1", "stepKey": "intake"}]

    out = svc.read_workflow_step_runs(CTX, run_id="r1")

    assert out and out[0]["stepKey"] == "intake"
    assert ("read_step_runs", "test", "r1") in repo.calls


def test_read_workflow_trace_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.trace["r1"] = [{"traceId": "te1", "kind": "guard_judgment"}]

    out = svc.read_workflow_trace(CTX, run_id="r1")

    assert out and out[0]["kind"] == "guard_judgment"
    assert ("read_trace", "test", "r1") in repo.calls


def test_find_waiting_run_for_thread_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.waiting_runs["t1"] = {"runId": "r1", "status": "waiting"}

    got = svc.find_waiting_run_for_thread(CTX, thread_id="t1")

    assert got["runId"] == "r1"
    assert ("find_waiting_run_for_thread", "test", "t1") in repo.calls


def test_find_waiting_run_for_thread_returns_none_when_nothing_parked():
    repo = FakeRepo()
    svc = make_service(repo)
    assert svc.find_waiting_run_for_thread(CTX, thread_id="t1") is None


# ── list_workflow_runs_for_thread (K-036 — web-api-coverage U4/FR-2) ────────


def test_list_workflow_runs_for_thread_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.threads.add("t1")
    repo.runs_by_thread["t1"] = [
        {"runId": "r2", "status": "running", "defKey": "triage", "defVersion": "1",
         "startedAt": 2000, "endedAt": None},
        {"runId": "r1", "status": "done", "defKey": "triage", "defVersion": "1",
         "startedAt": 1000, "endedAt": 1500},
    ]

    got = svc.list_workflow_runs_for_thread(CTX, thread_id="t1")

    assert [r["runId"] for r in got] == ["r2", "r1"]
    assert ("find_runs_for_thread", "test", "t1", 10) in repo.calls


def test_list_workflow_runs_for_thread_missing_thread_errors():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(ThreadNotFoundError):
        svc.list_workflow_runs_for_thread(CTX, thread_id="ghost")


def test_list_workflow_runs_for_thread_empty_when_no_runs():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.threads.add("t1")

    assert svc.list_workflow_runs_for_thread(CTX, thread_id="t1") == []


def test_list_workflow_runs_for_thread_passes_limit_through():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.threads.add("t1")

    svc.list_workflow_runs_for_thread(CTX, thread_id="t1", limit=5)

    assert ("find_runs_for_thread", "test", "t1", 5) in repo.calls


def test_agent_step_type_is_accepted_by_publish_validation():
    # the LLM-native node kind (§3): STEP_TYPES gains 'agent' so a triage def
    # (type:'agent' steps) validates and publishes
    repo = FakeRepo()
    svc = make_service(repo)

    # ≥1 transition (O-6) — a def that publishes successfully must carry one, so this
    # fixture declares a second step and the transition into it.
    svc.publish_workflow_def(
        CTX, key="triage", version="1", name="Triage", kind="conversation",
        steps=[{"key": "intake", "type": "agent", "start": True,
                "config": {"waitsForHuman": True}},
               {"key": "answer", "type": "agent"}],
        transitions=[{"from": "intake", "to": "answer", "on": "done", "order": 0}],
    )

    assert repo.published[0]["key"] == "triage"
    assert repo.published[0]["steps"][0]["type"] == "agent"


# K-027 item 2 -- must-post engine contract publish invariant.
# docs/plans/must-post-engine-contract.md section 3.4/9/10 (tests 9-12): a fourth
# "deliberately LAST" invariant alongside waitsForHuman -- config.requiredTools,
# when present on a step, must be a list of strings, only on a type:'agent' step,
# and a subset of that step's own config.tools.

def test_publish_workflow_def_required_tool_not_granted_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    steps = [
        {"key": "intake", "type": "agent", "start": True,
         "config": {"tools": ["post_message"], "requiredTools": ["x"]}},
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "intake", "to": "done", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefSpecError, match=r"requiredTools.*\['x'\]"):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []


def test_publish_workflow_def_required_tools_on_non_agent_step_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    steps = [
        # `tools` deliberately already contains `post_message` -- isolates the
        # type=="agent" check from the separate ⊆ config.tools check below.
        {"key": "intake", "type": "human", "start": True,
         "config": {"waitsForHuman": True, "tools": ["post_message"],
                    "requiredTools": ["post_message"]}},
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "intake", "to": "done", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefSpecError, match="requiredTools"):
        _publish(svc, repo, steps=steps, transitions=transitions)

    assert repo.published == []


def test_publish_workflow_def_required_tools_non_list_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    steps_string = [
        {"key": "intake", "type": "agent", "start": True,
         "config": {"tools": ["post_message"], "requiredTools": "post_message"}},
        {"key": "done", "type": "message"},
    ]
    steps_non_string_item = [
        {"key": "intake", "type": "agent", "start": True,
         "config": {"tools": ["post_message"], "requiredTools": [1]}},
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "intake", "to": "done", "on": "go", "order": 0}]

    with pytest.raises(WorkflowDefSpecError, match="list of strings"):
        _publish(svc, repo, steps=steps_string, transitions=transitions)
    assert repo.published == []

    with pytest.raises(WorkflowDefSpecError, match="list of strings"):
        _publish(svc, repo, steps=steps_non_string_item, transitions=transitions)
    assert repo.published == []


def test_publish_workflow_def_required_tools_subset_of_granted_succeeds():
    repo = FakeRepo()
    svc = make_service(repo)
    steps = [
        {"key": "intake", "type": "agent", "start": True,
         "config": {"tools": ["post_message", "graphrag_retrieve"],
                    "requiredTools": ["post_message"]}},
        {"key": "done", "type": "message"},
    ]
    transitions = [{"from": "intake", "to": "done", "on": "go", "order": 0}]

    _publish(svc, repo, steps=steps, transitions=transitions)

    assert len(repo.published) == 1
    by_key = {s["key"]: s for s in repo.published[0]["steps"]}
    assert by_key["intake"]["config"] == (
        '{"requiredTools":["post_message"],"tools":["post_message","graphrag_retrieve"]}'
    )


# ── §15 product catalog (K-052 M6) ────────────────────────────────────────────

def test_lookup_product_normalizes_name_before_repo_call():
    repo = FakeRepo()
    repo.products_by_name["bluetooth speaker"] = {
        "name": "Bluetooth Speaker", "category": "audio", "price": 89.99,
    }
    svc = make_service(repo)

    # Mixed case + irregular whitespace — extraction.normalize_name must
    # case-fold and whitespace-collapse before the repository call, exactly the
    # way `Entity.nameNormalized` does (§14.5).
    row = svc.lookup_product(CTX, name="  Bluetooth   Speaker ")

    assert row == {"name": "Bluetooth Speaker", "category": "audio", "price": 89.99}
    assert repo.calls == [("lookup_product", "bluetooth speaker")]


def test_lookup_product_abstains_when_repo_finds_nothing():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.lookup_product(CTX, name="nonexistent gadget") is None
    assert repo.calls == [("lookup_product", "nonexistent gadget")]


def test_filter_products_passes_arguments_through_to_repo():
    repo = FakeRepo()
    repo.products = [
        {"name": "Wireless Mouse", "category": "accessories", "price": 25.0},
        {"name": "Bluetooth Speaker", "category": "audio", "price": 89.99},
    ]
    svc = make_service(repo)

    rows = svc.filter_products(
        CTX, category="audio", min_price=50.0, max_price=150.0, limit=5,
    )

    assert rows == [{"name": "Bluetooth Speaker", "category": "audio", "price": 89.99}]
    assert repo.calls == [("filter_products", "audio", 50.0, 150.0, 5)]


def test_filter_products_all_omitted_lists_everything_up_to_default_limit():
    repo = FakeRepo()
    repo.products = [
        {"name": "Wireless Mouse", "category": "accessories", "price": 25.0},
        {"name": "Bluetooth Speaker", "category": "audio", "price": 89.99},
    ]
    svc = make_service(repo)

    rows = svc.filter_products(
        CTX, category=None, min_price=None, max_price=None, limit=20,
    )

    assert [r["name"] for r in rows] == ["Wireless Mouse", "Bluetooth Speaker"]


def test_filter_products_abstains_when_nothing_matches():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.filter_products(
        CTX, category="nonexistent", min_price=None, max_price=None, limit=20,
    ) == []


# ── §16 Cart / Order (K-053 M6) ────────────────────────────────────────────────
#
# `docs/plans/workflow-cart-and-totals.md` §3.3 — the explicit `ensure_customer`/
# `ensure_cart` ownership that closes `analyst`'s MAJOR finding. `ctx.actor`
# ("u1") is `customerId` throughout.


def _seed_speaker(repo):
    repo.products_by_name["bluetooth speaker"] = {
        "productId": "prod2", "name": "Bluetooth Speaker",
        "category": "audio", "price": 89.99,
    }
    repo.products_by_id["prod2"] = {"name": "Bluetooth Speaker", "price": 89.99}


def _seed_mouse(repo):
    repo.products_by_name["wireless mouse"] = {
        "productId": "prod1", "name": "Wireless Mouse",
        "category": "accessories", "price": 25.0,
    }
    repo.products_by_id["prod1"] = {"name": "Wireless Mouse", "price": 25.0}


# ── add_cart_item ───────────────────────────────────────────────────────────────

def test_add_cart_item_brand_new_customer_ensures_customer_and_cart_first():
    """Regression for `analyst`'s MAJOR finding (`docs/reviews/workflow-cart-and-
    totals.md`): a brand-new `customerId` with no prior `Customer`/`Cart` node
    must not silently no-op — `ensure_customer` -> `ensure_cart` -> `add_to_cart`,
    in that order, before the item is actually persisted."""
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo, now=1000)

    row = svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=2)

    assert row == {
        "productId": "prod2", "name": "Bluetooth Speaker", "price": 89.99, "quantity": 2,
    }
    assert "u1" in repo.customers
    assert "u1" in repo.carts
    assert repo.cart_items["u1"]["prod2"]["quantity"] == 2
    # ordering: ensure_customer, then ensure_cart, then add_to_cart — in that order
    kinds = [c[0] for c in repo.calls]
    assert kinds.index("ensure_customer") < kinds.index("ensure_cart") < kinds.index(
        "add_to_cart"
    )


def test_add_cart_item_repeated_call_accumulates_quantity():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo, now=1000)

    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=2)
    row = svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=3)

    assert row["quantity"] == 5


def test_add_cart_item_unknown_product_abstains_without_writing_anything():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.add_cart_item(CTX, product_name="nonexistent gadget", quantity=1) is None
    assert repo.customers == set()
    assert repo.carts == set()


def test_add_cart_item_normalizes_name_via_lookup_product():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)

    row = svc.add_cart_item(CTX, product_name="  Bluetooth   Speaker ", quantity=1)

    assert row["productId"] == "prod2"


# ── get_cart ─────────────────────────────────────────────────────────────────────

def test_get_cart_empty_for_never_touched_customer_no_ensure_calls():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.get_cart(CTX) == {"items": [], "total": 0.0}
    assert repo.customers == set()  # no ensure_customer/ensure_cart side effect
    assert repo.carts == set()


def test_get_cart_reflects_added_items_with_live_price_and_total():
    repo = FakeRepo()
    _seed_speaker(repo)
    _seed_mouse(repo)
    svc = make_service(repo, now=1000)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=2)
    svc.add_cart_item(CTX, product_name="Wireless Mouse", quantity=1)

    cart = svc.get_cart(CTX)

    assert cart["items"] == [
        {"productId": "prod2", "name": "Bluetooth Speaker", "price": 89.99,
         "quantity": 2, "lineTotal": 179.98},
        {"productId": "prod1", "name": "Wireless Mouse", "price": 25.0,
         "quantity": 1, "lineTotal": 25.0},
    ]
    assert cart["total"] == pytest.approx(204.98)


def test_get_cart_uses_current_catalog_price_not_a_stale_one():
    """AC-3/FR-3: the cart never stores a price — a catalog price change
    between add-to-cart and view must be reflected immediately."""
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)

    repo.products_by_id["prod2"]["price"] = 79.99  # catalog price change

    cart = svc.get_cart(CTX)
    assert cart["items"][0]["price"] == 79.99
    assert cart["total"] == 79.99


def test_get_cart_drops_a_line_whose_product_vanished_from_the_catalog():
    """Graph note §8: a `CartItem` referencing a since-deleted product isn't
    addressed by any AC — silently excluded rather than raised on."""
    repo = FakeRepo()
    _seed_speaker(repo)
    _seed_mouse(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    svc.add_cart_item(CTX, product_name="Wireless Mouse", quantity=1)

    del repo.products_by_id["prod1"]  # "Wireless Mouse" deleted from the catalog

    cart = svc.get_cart(CTX)
    assert [item["productId"] for item in cart["items"]] == ["prod2"]
    assert cart["total"] == 89.99


# ── remove_cart_item ─────────────────────────────────────────────────────────────

def test_remove_cart_item_partial_quantity_decrements_in_place():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=5)

    row = svc.remove_cart_item(CTX, product_name="Bluetooth Speaker", quantity=2)

    assert row == {
        "removed": False, "productId": "prod2", "name": "Bluetooth Speaker",
        "quantity": 3,
    }
    assert repo.cart_items["u1"]["prod2"]["quantity"] == 3


def test_remove_cart_item_omitted_quantity_removes_the_whole_line():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=5)

    row = svc.remove_cart_item(CTX, product_name="Bluetooth Speaker")

    assert row["removed"] is True
    assert "prod2" not in repo.cart_items.get("u1", {})


def test_remove_cart_item_no_ensure_calls_against_a_never_touched_cart():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)

    row = svc.remove_cart_item(CTX, product_name="Bluetooth Speaker")

    assert row == {"removed": False, "productId": "prod2", "name": "Bluetooth Speaker"}
    assert repo.customers == set()
    assert repo.carts == set()


def test_remove_cart_item_unknown_product_abstains():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.remove_cart_item(CTX, product_name="nonexistent gadget") is None


def test_remove_cart_item_known_product_not_in_cart_is_a_noop():
    repo = FakeRepo()
    _seed_speaker(repo)
    _seed_mouse(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)

    row = svc.remove_cart_item(CTX, product_name="Wireless Mouse")

    assert row == {"removed": False, "productId": "prod1", "name": "Wireless Mouse"}


# ── clear_cart ───────────────────────────────────────────────────────────────────

def test_clear_cart_removes_every_item_no_ensure_calls():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=3)

    svc.clear_cart(CTX)

    assert svc.get_cart(CTX) == {"items": [], "total": 0.0}


def test_clear_cart_on_never_touched_customer_is_a_plain_noop():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.clear_cart(CTX)  # must not raise

    assert repo.customers == set()
    assert repo.carts == set()


# ── place_order ──────────────────────────────────────────────────────────────────

def test_place_order_snapshots_current_prices_and_clears_the_cart():
    repo = FakeRepo()
    _seed_speaker(repo)
    _seed_mouse(repo)
    svc = make_service(repo, now=1000)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=2)
    svc.add_cart_item(CTX, product_name="Wireless Mouse", quantity=1)

    calls_before = len(repo.calls)
    result = svc.place_order(CTX)

    assert result["created"] is True
    assert result["lineCount"] == 2
    assert result["total"] == pytest.approx(204.98)
    assert result["lines"] == [
        {"productId": "prod2", "name": "Bluetooth Speaker", "unitPrice": 89.99,
         "quantity": 2, "lineTotal": 179.98},
        {"productId": "prod1", "name": "Wireless Mouse", "unitPrice": 25.0,
         "quantity": 1, "lineTotal": 25.0},
    ]
    # cart is cleared as part of the same operation (AC-5)
    assert svc.get_cart(CTX) == {"items": [], "total": 0.0}
    # ensure_customer called defensively, before the read/write (§3.3)
    assert repo.calls[calls_before][0] == "ensure_customer"


def test_place_order_snapshot_survives_a_later_catalog_price_change_ac6():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo, now=1000)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)

    result = svc.place_order(CTX)
    order_id = result["orderId"]

    repo.products_by_id["prod2"]["price"] = 999.0  # catalog price change, after placement

    order = repo.get_order(ws="test", order_id=order_id)
    assert order["lines"][0]["unitPrice"] == 89.99  # unchanged
    assert order["total"] == 89.99


def test_place_order_empty_cart_abstains():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.place_order(CTX) is None


def test_place_order_every_line_products_vanished_is_treated_as_empty():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)

    del repo.products_by_id["prod2"]

    assert svc.place_order(CTX) is None


def test_place_order_mints_a_fresh_order_id_each_call():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    first = svc.place_order(CTX)

    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    second = svc.place_order(CTX)

    assert first["orderId"] != second["orderId"]


# ── get_order_status / advance_order ──────────────────────────────────────────────

def test_get_order_status_returns_current_status():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    result = svc.place_order(CTX)

    assert svc.get_order_status(CTX, order_id=result["orderId"]) == "placed"


def test_get_order_status_none_for_unknown_order():
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.get_order_status(CTX, order_id="nope") is None


def test_advance_order_fulfill_then_deliver_lifecycle():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    order_id = svc.place_order(CTX)["orderId"]

    fulfilled = svc.advance_order(CTX, order_id=order_id, transition="fulfill")
    assert fulfilled == {"orderId": order_id, "status": "fulfilled"}
    assert svc.get_order_status(CTX, order_id=order_id) == "fulfilled"

    delivered = svc.advance_order(CTX, order_id=order_id, transition="deliver")
    assert delivered == {"orderId": order_id, "status": "delivered"}
    assert svc.get_order_status(CTX, order_id=order_id) == "delivered"


def test_advance_order_cancel_before_fulfillment_succeeds():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    order_id = svc.place_order(CTX)["orderId"]

    cancelled = svc.advance_order(CTX, order_id=order_id, transition="cancel")
    assert cancelled == {"orderId": order_id, "status": "cancelled"}


def test_advance_order_cancel_after_fulfilled_is_blocked_ac8():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    order_id = svc.place_order(CTX)["orderId"]
    svc.advance_order(CTX, order_id=order_id, transition="fulfill")

    assert svc.advance_order(CTX, order_id=order_id, transition="cancel") is None
    assert svc.get_order_status(CTX, order_id=order_id) == "fulfilled"  # unchanged


def test_advance_order_stale_transition_is_a_noop_returning_none():
    repo = FakeRepo()
    _seed_speaker(repo)
    svc = make_service(repo)
    svc.add_cart_item(CTX, product_name="Bluetooth Speaker", quantity=1)
    order_id = svc.place_order(CTX)["orderId"]

    # deliver before fulfill — the guard requires 'fulfilled', order is 'placed'
    assert svc.advance_order(CTX, order_id=order_id, transition="deliver") is None


def test_advance_order_unknown_transition_raises():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(UnknownOrderTransitionError):
        svc.advance_order(CTX, order_id="anything", transition="frobnicate")


# ── §17 Durable customer profile (K-054 M6) ─────────────────────────────────────
#
# `docs/plans/workflow-durable-profile.md` §3.2. `ctx.actor` ("u1") is
# `customerId` throughout, same identity anchor §16's cart/order tests use.


def test_get_profile_returns_both_fields_none_when_no_customer_yet():
    """Plan §3.2: `get_profile` always returns a shape with both fields — not
    a `{"found": false}` abstention shape — even for a brand-new customer."""
    repo = FakeRepo()
    svc = make_service(repo)

    assert svc.get_profile(CTX) == {"name": None, "deliveryAddress": None}


def test_save_profile_full_write_then_get_profile_round_trips():
    repo = FakeRepo()
    svc = make_service(repo, now=1000)

    result = svc.save_profile(CTX, name="Alice", delivery_address="123 Main St")

    assert result == {"name": "Alice", "deliveryAddress": "123 Main St"}
    assert svc.get_profile(CTX) == {"name": "Alice", "deliveryAddress": "123 Main St"}


def test_profile_persists_across_a_fresh_thread_same_actor_and_ws_ac1():
    """AC-1: a saved profile is retrievable from what stands in for "a later,
    separate conversation" at the service layer — a fresh `CallContext`
    carrying the same `(ws, actor)`, exercised against the same `Services`/
    repository (durability is the repository's job; this pins that the
    service layer never caches around it)."""
    repo = FakeRepo()
    svc = make_service(repo, now=1000)
    svc.save_profile(CTX, name="Alice", delivery_address="123 Main St")

    fresh_ctx = CallContext(ws="test", actor="u1")

    assert svc.get_profile(fresh_ctx) == {
        "name": "Alice", "deliveryAddress": "123 Main St",
    }


def test_save_profile_partial_update_omitted_name_leaves_name_unchanged_ac2():
    """AC-2, the exact case the graph note's own BLOCKER-fix targets: a second
    `save_profile` call that supplies only `delivery_address` must update the
    address and leave the already-stored `name` unchanged — never null it."""
    repo = FakeRepo()
    svc = make_service(repo, now=1000)
    svc.save_profile(CTX, name="Alice", delivery_address="123 Main St")

    result = svc.save_profile(CTX, delivery_address="456 New Ave")

    assert result == {"name": "Alice", "deliveryAddress": "456 New Ave"}
    assert svc.get_profile(CTX) == {"name": "Alice", "deliveryAddress": "456 New Ave"}


def test_save_profile_partial_update_omitted_address_leaves_address_unchanged():
    """The symmetric AC-2 case: omitting `delivery_address` this time must
    leave it unchanged while `name` updates."""
    repo = FakeRepo()
    svc = make_service(repo, now=1000)
    svc.save_profile(CTX, name="Alice", delivery_address="123 Main St")

    result = svc.save_profile(CTX, name="Bob")

    assert result == {"name": "Bob", "deliveryAddress": "123 Main St"}
    assert svc.get_profile(CTX) == {"name": "Bob", "deliveryAddress": "123 Main St"}
