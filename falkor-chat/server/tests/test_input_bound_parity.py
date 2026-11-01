"""REST <-> MCP input-bound parity guard (K-101, `docs/reviews/mcp-input-bounds-rca.md` §5.3).

No database. Two independent declarations are compared:

* **REST** — every constraint keyword (`CONSTRAINT_KEYS`) FastAPI publishes in
  its OpenAPI document for `api.build_router(Services(FakeRepo()))`, on
  parameters and request bodies.
* **MCP** — every argument of every tool FastMCP registers, and every
  constraint keyword its pydantic-built `inputSchema` carries.

Both sides go through one collector, `schema_constraints`, which reaches every
position a bound can sit at: `$ref`, `anyOf`/`oneOf`/`allOf`, nested
properties (`.name`), list elements (`[]`), tuple positions (`[i]`) and map
values (`{}`). Its reach is itself tested: the "collector coverage probe"
section plants one constrained field per location, wrapping and spelling in a
synthetic router and tool, and fails when the collector misses one. A
recursive model is walked once, so a bound inside it is reported where the
model first appears.

Both are checked against one hand-written table each, `MAPPING` (one entry per
REST constraint) and `MCP_ARGS` (one entry per MCP tool argument). The test
fails when:

  (i)   a constrained REST field has no `MAPPING` entry (someone added a
        `Field(max_length=...)`), or a `MAPPING` key no longer exists in REST;
  (ii)  an MCP tool argument has no `MCP_ARGS` entry (someone added an
        `@mcp.tool()` or a parameter), or an `MCP_ARGS` key no longer exists;
  (iii) a constraint mapped to the `Mcp` layer is missing from, or differs in,
        the twin argument's `inputSchema` — or the `inputSchema` carries a
        constraint the table doesn't map to `Mcp`;
  (iv)  a constraint mapped to the `Service` layer is not refused by the named
        error one step past REST's limit, or is refused at REST's limit
        itself — called through `Services(FakeRepo())`.

`KnownGap` marks a bound neither the service nor MCP enforces yet; each one is
a §5.2 fix still to land, named by its target layer.

**Blind spot:** custom `field_validator`s (`schemas._bounded_flat_dict`, the
storefront's `_nonblank`) are invisible to OpenAPI, so this guard cannot see
them. None sits on an MCP-twinned model today: `_bounded_flat_dict` is on
`StartWorkflowRunIn`/`SubmitWorkflowInputIn` (workflow routes, no MCP twin; the
service already mirrors the merged-ctx bound), and the storefront router is not
built here at all. The service's whitespace-only message refusal (K-101
stakeholder decision) is likewise stricter than anything OpenAPI states, so it
is pinned by literal tests in `test_services.py`, not here.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from fastapi import APIRouter, FastAPI, Path, Query
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from test_services import FakeRepo

from falkorchat import api
from falkorchat import mcp as mcp_mod
from falkorchat.config import CallContext
from falkorchat.services import (
    BatchTooLargeError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InputBoundError,
    MessageTooLongError,
    Services,
)

CONSTRAINT_KEYS = (
    "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "enum", "const",
)


# ── mapping vocabulary ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Service:
    """Enforced in `Services` for every front door; `probe` names a `PROBES` entry."""

    probe: str
    error: type[Exception]


@dataclass(frozen=True)
class Mcp:
    """Declared on the MCP tool signature (`Annotated[..., Field(...)]`)."""

    tool: str
    arg: str


@dataclass(frozen=True)
class NotCopied:
    """Deliberately not mirrored off REST — RCA §5.2 item 3."""

    reason: str


@dataclass(frozen=True)
class KnownGap:
    """Not enforced off REST yet; `layer` is where the §5.2 fix lands."""

    layer: str  # "service" | "mcp"


@dataclass(frozen=True)
class Twin:
    """An MCP argument mirrors this REST field (route, field path)."""

    route: str
    field: str


@dataclass(frozen=True)
class NoBound:
    """An MCP argument with no bounded REST twin (REST declares no bound either)."""

    reason: str


# Routes with no MCP tool at all: every constrained field on them is out of the
# parity class (RCA scope). Their K-049-relevant key bounds are mirrored in the
# service by `_validate_key_lengths`, tested in `test_services.py`.
NO_MCP_TWIN_ROUTES = frozenset({
    "GET /threads/{thread_id}/workflow-runs",
    "POST /workflow-defs",
    "GET /workflow-defs",
    "GET /workflow-defs/{key}/versions/{version}",
    "POST /workflow-defs/{key}/versions/{version}/materialize",
    "POST /workflow-runs",
    "POST /workflow-runs/{run_id}/input",
    "POST /workflow-runs/due",
    "GET /workflow-runs/{run_id}",
    "GET /workflow-runs/{run_id}/step-runs",
    "GET /workflow-runs/{run_id}/trace",
    "GET /workspaces/{ws}/snapshots",
    "GET /workspaces/{ws}/snapshots/{key}/versions/{version}",
    "GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff",
})

_PATH_ID = NotCopied("path id is lookup-only; an unknown id is None/NotFound, no K-049 path")
_LIMIT_GAP = (KnownGap("mcp"),)

MSG = "POST /threads/{thread_id}/messages"
DOC = "POST /documents"
BATCH = "POST /documents/batch"

# One entry per REST constraint outside `NO_MCP_TWIN_ROUTES`:
# (route, field, constraint) -> tuple of layers.
MAPPING: dict[tuple[str, str, str], tuple[Any, ...]] = {
    # ── messages ──
    (MSG, "body.text", "minLength"): (
        Service("post_message.text", InputBoundError), KnownGap("mcp"),
    ),
    (MSG, "body.text", "maxLength"): (
        Service("post_message.text", MessageTooLongError), KnownGap("mcp"),
    ),
    (MSG, "body.mentions", "maxItems"): (
        Service("post_message.mentions", InputBoundError), KnownGap("mcp"),
    ),
    ("GET /threads/{thread_id}/messages", "query.since", "minimum"): _LIMIT_GAP,
    ("GET /threads/{thread_id}/messages", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /threads/{thread_id}/messages", "query.limit", "maximum"): _LIMIT_GAP,
    ("GET /search", "query.q", "minLength"): _LIMIT_GAP,
    ("GET /search", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /search", "query.limit", "maximum"): _LIMIT_GAP,
    # ── channels / threads ──
    ("POST /channels", "body.name", "minLength"): (
        Service("create_channel.name", InputBoundError), KnownGap("mcp"),
    ),
    ("POST /channels", "body.name", "maxLength"): (
        Service("create_channel.name", InputBoundError), KnownGap("mcp"),
    ),
    ("GET /channels", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /channels", "query.limit", "maximum"): _LIMIT_GAP,
    ("POST /channels/{channel_id}/threads", "body.title", "minLength"): (
        Service("create_thread.title", InputBoundError), KnownGap("mcp"),
    ),
    ("POST /channels/{channel_id}/threads", "body.title", "maxLength"): (
        Service("create_thread.title", InputBoundError), KnownGap("mcp"),
    ),
    ("GET /channels/{channel_id}/threads", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /channels/{channel_id}/threads", "query.limit", "maximum"): _LIMIT_GAP,
    # ── documents: single ingest ──
    (DOC, "body.text", "minLength"): (Service("ingest_document.text", EmptyDocumentError),),
    (DOC, "body.text", "maxLength"): (Service("ingest_document.text", DocumentTooLargeError),),
    (DOC, "body.title", "maxLength"): (
        Service("ingest_document.title", InputBoundError), KnownGap("mcp"),
    ),
    (DOC, "body.sourceFormat", "minLength"): (
        Service("ingest_document.source_format", InputBoundError), KnownGap("mcp"),
    ),
    (DOC, "body.sourceFormat", "maxLength"): (
        Service("ingest_document.source_format", InputBoundError), KnownGap("mcp"),
    ),
    (DOC, "body.sourceLabel", "maxLength"): (
        Service("ingest_document.source_label", InputBoundError), KnownGap("mcp"),
    ),
    (DOC, "body.producedBy", "maxLength"): (
        NotCopied("lookup-only; an unknown producer is already AgentNotFoundError"),
    ),
    # ── documents: batch (per-item violations become that item's receipt) ──
    (BATCH, "body.documents", "minItems"): (
        NotCopied("an empty batch's correct answer is [] and it writes nothing"),
    ),
    (BATCH, "body.documents", "maxItems"): (
        Service("ingest_documents.documents", BatchTooLargeError),
    ),
    (BATCH, "body.documents[].text", "minLength"): (
        Service("ingest_documents.item.text", EmptyDocumentError),
    ),
    (BATCH, "body.documents[].text", "maxLength"): (
        Service("ingest_documents.item.text", DocumentTooLargeError),
    ),
    (BATCH, "body.documents[].title", "maxLength"): (
        Service("ingest_documents.item.title", InputBoundError),
    ),
    (BATCH, "body.documents[].sourceFormat", "minLength"): (
        Service("ingest_documents.item.source_format", InputBoundError),
    ),
    (BATCH, "body.documents[].sourceFormat", "maxLength"): (
        Service("ingest_documents.item.source_format", InputBoundError),
    ),
    (BATCH, "body.documents[].sourceLabel", "maxLength"): (
        Service("ingest_documents.item.source_label", InputBoundError),
    ),
    (BATCH, "body.documents[].producedBy", "maxLength"): (
        NotCopied("lookup-only; an unknown producer is already AgentNotFoundError"),
    ),
    # ── documents: reads ──
    ("GET /documents", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /documents", "query.limit", "maximum"): _LIMIT_GAP,
    ("GET /documents/search", "query.q", "minLength"): _LIMIT_GAP,
    ("GET /documents/search", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /documents/search", "query.limit", "maximum"): _LIMIT_GAP,
    ("GET /document-updates/pending", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /document-updates/pending", "query.limit", "maximum"): _LIMIT_GAP,
    ("GET /document-updates", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /document-updates", "query.limit", "maximum"): _LIMIT_GAP,
    ("GET /matches/pending", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /matches/pending", "query.limit", "maximum"): _LIMIT_GAP,
    ("GET /matches", "query.limit", "minimum"): _LIMIT_GAP,
    ("GET /matches", "query.limit", "maximum"): _LIMIT_GAP,
    # ── path ids (not copied, RCA §5.2 item 3) ──
    **{
        (route, f"path.{pid}", c): (_PATH_ID,)
        for route, pid in (
            ("DELETE /documents/{document_id}", "document_id"),
            ("GET /documents/{document_id}/deletion", "document_id"),
            ("GET /documents/{document_id}/history", "document_id"),
            ("POST /document-updates/{match_id}/confirm", "match_id"),
            ("POST /document-updates/{match_id}/reject", "match_id"),
            ("POST /document-updates/{match_id}/recheck", "match_id"),
            ("POST /matches/{match_id}/confirm", "match_id"),
            ("POST /matches/{match_id}/reject", "match_id"),
            ("POST /matches/{match_id}/recheck", "match_id"),
        )
        for c in ("minLength", "maxLength")
    },
}

_UNBOUNDED_PATH = NoBound("REST path parameter is unbounded too")
_NO_TWIN = NoBound("no REST twin")

# One entry per MCP tool argument.
MCP_ARGS: dict[tuple[str, str], Twin | NoBound] = {
    ("send_message", "body"): Twin(MSG, "body.text"),
    ("send_message", "mentions"): Twin(MSG, "body.mentions"),
    ("send_message", "re"): _UNBOUNDED_PATH,
    ("send_message", "frm"): NoBound("reserved/ignored in M1"),
    ("read_messages", "re"): _UNBOUNDED_PATH,
    ("read_messages", "since"): Twin("GET /threads/{thread_id}/messages", "query.since"),
    ("read_messages", "limit"): Twin("GET /threads/{thread_id}/messages", "query.limit"),
    ("read_messages", "advance"): _NO_TWIN,
    ("create_thread", "channel_id"): _UNBOUNDED_PATH,
    ("create_thread", "title"): Twin("POST /channels/{channel_id}/threads", "body.title"),
    ("search_messages", "query"): Twin("GET /search", "query.q"),
    ("search_messages", "limit"): Twin("GET /search", "query.limit"),
    ("create_channel", "name"): Twin("POST /channels", "body.name"),
    ("list_channels", "limit"): Twin("GET /channels", "query.limit"),
    ("list_threads", "channel_id"): _UNBOUNDED_PATH,
    ("list_threads", "limit"): Twin("GET /channels/{channel_id}/threads", "query.limit"),
    ("ingest_document", "text"): Twin(DOC, "body.text"),
    ("ingest_document", "title"): Twin(DOC, "body.title"),
    ("ingest_document", "source_format"): Twin(DOC, "body.sourceFormat"),
    ("ingest_document", "source_label"): Twin(DOC, "body.sourceLabel"),
    ("ingest_document", "produced_by"): Twin(DOC, "body.producedBy"),
    # Untyped on purpose: typing it would make one bad item fail the whole call
    # and lose `ingest_documents`' per-item receipts (RCA §5.1 option B).
    ("ingest_documents", "items"): Twin(BATCH, "body.documents"),
    ("get_document", "document_id"): _UNBOUNDED_PATH,
    ("search_documents", "query"): Twin("GET /documents/search", "query.q"),
    ("search_documents", "limit"): Twin("GET /documents/search", "query.limit"),
    ("delete_document", "document_id"): Twin("DELETE /documents/{document_id}", "path.document_id"),
    ("list_documents", "current_only"): NoBound("bool"),
    ("list_documents", "limit"): Twin("GET /documents", "query.limit"),
    ("get_document_deletion", "document_id"): Twin(
        "GET /documents/{document_id}/deletion", "path.document_id",
    ),
    ("get_document_history", "document_id"): Twin(
        "GET /documents/{document_id}/history", "path.document_id",
    ),
    ("list_pending_document_updates", "limit"): Twin(
        "GET /document-updates/pending", "query.limit",
    ),
    ("list_document_updates", "status"): NoBound("REST Query(None), no enum"),
    ("list_document_updates", "limit"): Twin("GET /document-updates", "query.limit"),
    ("confirm_document_update", "match_id"): Twin(
        "POST /document-updates/{match_id}/confirm", "path.match_id",
    ),
    ("reject_document_update", "match_id"): Twin(
        "POST /document-updates/{match_id}/reject", "path.match_id",
    ),
    ("recheck_document_update", "match_id"): Twin(
        "POST /document-updates/{match_id}/recheck", "path.match_id",
    ),
    ("list_pending_matches", "limit"): Twin("GET /matches/pending", "query.limit"),
    ("list_matches", "status"): NoBound("REST Query(None), no enum"),
    ("list_matches", "limit"): Twin("GET /matches", "query.limit"),
    ("confirm_match", "match_id"): Twin("POST /matches/{match_id}/confirm", "path.match_id"),
    ("reject_match", "match_id"): Twin("POST /matches/{match_id}/reject", "path.match_id"),
    ("recheck_match", "match_id"): Twin("POST /matches/{match_id}/recheck", "path.match_id"),
}


# ── enumeration ─────────────────────────────────────────────────────────────────


class ConflictingConstraintError(ValueError):
    """Two schema branches give one position different values for a keyword."""


def _own(node: dict[str, Any]) -> dict[str, Any]:
    """Constraint keywords on this one node (floats that are integers normalized)."""
    return {
        k: int(v) if isinstance(v, float) and v.is_integer() else v
        for k, v in node.items() if k in CONSTRAINT_KEYS
    }


def _pointer(root: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a local JSON pointer (`#/components/schemas/X`, `#/$defs/X`)."""
    if not ref.startswith("#/"):
        raise ValueError(f"non-local $ref {ref!r}: the collector cannot see it")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def schema_constraints(
    node: dict[str, Any], root: dict[str, Any], path: str,
    out: dict[tuple[str, str], Any], _refs: tuple[str, ...] = (),
) -> None:
    """Record `out[(position, keyword)] = value` for every constraint reachable
    from `node`. The one collector both front doors share (review P1-1).

    Recurses through every JSON-Schema position that can carry a bound:
    `$ref` (with its sibling keywords, a recursive model stopping at the second
    visit), `anyOf`/`oneOf`/`allOf` (same position), `properties` (`.name`),
    `items` (`[]`, or `[i]` for the array form), `prefixItems` (`[i]`) and
    `additionalProperties`/`patternProperties` (`{}`). `not` is skipped on
    purpose: a negated keyword is not a bound to mirror. Two branches giving
    one position different values raise `ConflictingConstraintError` instead
    of letting the last one silently win.
    """
    ref = node.get("$ref")
    if ref is not None and ref not in _refs:
        schema_constraints(_pointer(root, ref), root, path, out, (*_refs, ref))
    for key, value in _own(node).items():
        prior = out.setdefault((path, key), value)
        if prior != value:
            raise ConflictingConstraintError(f"{path} {key}: {prior!r} vs {value!r}")
    for combinator in ("anyOf", "oneOf", "allOf"):
        for branch in node.get(combinator, []):
            schema_constraints(branch, root, path, out, _refs)
    for name, prop in node.get("properties", {}).items():
        schema_constraints(prop, root, f"{path}.{name}" if path else name, out, _refs)
    items = node.get("items")
    if isinstance(items, dict):
        schema_constraints(items, root, f"{path}[]", out, _refs)
    elif isinstance(items, list):
        for i, item in enumerate(items):
            schema_constraints(item, root, f"{path}[{i}]", out, _refs)
    for i, item in enumerate(node.get("prefixItems", [])):
        schema_constraints(item, root, f"{path}[{i}]", out, _refs)
    extra = node.get("additionalProperties")
    if isinstance(extra, dict):
        schema_constraints(extra, root, f"{path}{{}}", out, _refs)
    for pattern_schema in node.get("patternProperties", {}).values():
        schema_constraints(pattern_schema, root, f"{path}{{}}", out, _refs)


def rest_constraints(app: FastAPI) -> dict[tuple[str, str, str], Any]:
    """`(route, field path, keyword) -> value` for every constraint `app` publishes."""
    spec = app.openapi()
    found: dict[tuple[str, str, str], Any] = {}
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            route = f"{method.upper()} {path}"
            positions: dict[tuple[str, str], Any] = {}
            for param in op.get("parameters", []):
                if "$ref" in param:
                    param = _pointer(spec, param["$ref"])
                schema_constraints(
                    param.get("schema", {}), spec, f"{param['in']}.{param['name']}", positions,
                )
            body = op.get("requestBody")
            if body:
                if "$ref" in body:
                    body = _pointer(spec, body["$ref"])
                for media in body.get("content", {}).values():
                    schema_constraints(media.get("schema", {}), spec, "body", positions)
            for (field, key), value in positions.items():
                found[(route, field, key)] = value
    return found


def mcp_arguments(server) -> set[tuple[str, str]]:
    """Every `(tool, argument)` pair `server` registers."""
    return {
        (t.name, arg)
        for t in asyncio.run(server.list_tools())
        for arg in t.inputSchema.get("properties", {})
    }


def mcp_constraints(server) -> dict[tuple[str, str, str], Any]:
    """`(tool, argument path, keyword) -> value` for every constraint `server` advertises."""
    found: dict[tuple[str, str, str], Any] = {}
    for tool in asyncio.run(server.list_tools()):
        positions: dict[tuple[str, str], Any] = {}
        schema_constraints(tool.inputSchema, tool.inputSchema, "", positions)
        for (arg, key), value in positions.items():
            found[(tool.name, arg, key)] = value
    return found


_HEAD = re.compile(r"^[^.\[{]+")


def _split(path: str) -> tuple[str, str]:
    """`"mentions[]"` -> `("mentions", "[]")`: the top-level name and its suffix."""
    head = _HEAD.match(path).group(0)
    return head, path[len(head):]


def _live_rest_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api.build_router(Services(FakeRepo())))
    return app


REST = rest_constraints(_live_rest_app())
MCP_ARGUMENTS = mcp_arguments(mcp_mod.mcp)
MCP = mcp_constraints(mcp_mod.mcp)
PARITY_REST = {k: v for k, v in REST.items() if k[0] not in NO_MCP_TWIN_ROUTES}


# ── (i) / (ii): every declaration has exactly one mapping entry ──────────────────


def test_every_constrained_rest_field_has_a_mapping_entry():
    unmapped = sorted(set(PARITY_REST) - set(MAPPING))
    assert unmapped == [], f"REST constraints with no MAPPING entry: {unmapped}"


def test_every_mapping_entry_names_a_live_rest_constraint():
    stale = sorted(set(MAPPING) - set(PARITY_REST))
    assert stale == [], f"MAPPING entries REST no longer declares: {stale}"


def test_no_mcp_twin_routes_all_exist():
    live_routes = {route for route, _f, _c in REST}
    assert sorted(NO_MCP_TWIN_ROUTES - live_routes) == []


def test_every_mcp_argument_has_a_mapping_entry():
    unmapped = sorted(MCP_ARGUMENTS - set(MCP_ARGS))
    assert unmapped == [], f"MCP tool arguments with no MCP_ARGS entry: {unmapped}"


def test_every_mcp_args_entry_names_a_live_tool_argument():
    stale = sorted(set(MCP_ARGS) - MCP_ARGUMENTS)
    assert stale == [], f"MCP_ARGS entries no tool declares: {stale}"


def test_every_twin_points_at_a_constrained_parity_field():
    """A twin names a field that carries a constraint itself or holds one
    below it (a list element, a nested model's field)."""
    def constrained(route: str, field: str) -> bool:
        return any(
            r == route and _split(f)[0] == _split(field)[0] and f.startswith(field)
            for r, f, _c in PARITY_REST
        )

    dangling = sorted(
        (key, twin) for key, twin in MCP_ARGS.items()
        if isinstance(twin, Twin) and not constrained(twin.route, twin.field)
    )
    assert dangling == []


def test_mapping_entries_are_well_formed():
    for key, layers in MAPPING.items():
        assert layers, key
        solo = [x for x in layers if isinstance(x, NotCopied)]
        assert not solo or len(layers) == 1, f"{key}: NotCopied must stand alone"
        assert len(set(layers)) == len(layers), key
        for layer in layers:
            assert isinstance(layer, (Service, Mcp, NotCopied, KnownGap)), (key, layer)
            if isinstance(layer, KnownGap):
                assert layer.layer in {"service", "mcp"}, key
            if isinstance(layer, Service):
                assert layer.probe in PROBES, (key, layer.probe)
        assert not (
            any(isinstance(x, Service) for x in layers) and KnownGap("service") in layers
        ), key
        mcp_layers = [x for x in layers if isinstance(x, Mcp)]
        assert len(mcp_layers) <= 1, f"{key}: one MCP twin per constraint"
        assert not (mcp_layers and KnownGap("mcp") in layers), (
            f"{key}: an Mcp entry closes the MCP gap; drop KnownGap('mcp')"
        )


def test_no_service_layer_gap_remains():
    """K-101 Round 2 ratchet: every stored-value bound is a service invariant.
    Only the MCP-signature half (`KnownGap("mcp")`) may still be open."""
    gaps = sorted(k for k, layers in MAPPING.items() if KnownGap("service") in layers)
    assert gaps == []


# ── (iii): the Mcp layer matches REST exactly, in both directions ────────────────


def _mcp_layers():
    for key, layers in MAPPING.items():
        for layer in layers:
            if isinstance(layer, Mcp):
                yield key, layer


def test_mcp_layer_constraints_match_rest():
    """One test, not a parametrization: with no `Mcp` entry yet an empty
    parameter set would read as a skip rather than as a vacuous pass."""
    mismatches = []
    for key, layer in _mcp_layers():
        route, field, constraint = key
        arg, suffix = _split(layer.arg)
        twin = MCP_ARGS.get((layer.tool, arg))
        if not isinstance(twin, Twin) or (twin.route, twin.field + suffix) != (route, field):
            mismatches.append(f"{layer} is not mapped as the twin of {route} {field}")
            continue
        advertised = MCP.get((layer.tool, layer.arg, constraint))
        if advertised != PARITY_REST[key]:
            mismatches.append(
                f"{layer.tool}({layer.arg}) advertises {constraint}={advertised!r}, "
                f"REST declares {PARITY_REST[key]!r}"
            )
    assert mismatches == []


def test_mcp_advertises_no_constraint_the_mapping_does_not_assign_it():
    assigned = {
        (layer.tool, layer.arg, key[2]) for key, layer in _mcp_layers()
    }
    extra = sorted(key for key in MCP if key not in assigned)
    assert extra == [], f"MCP constraints with no Mcp mapping entry: {extra}"


# ── (iv): the Service layer refuses one step past REST's limit, not at it ───────

CTX = CallContext(ws="test", actor="u1")


def _world() -> tuple[Services, FakeRepo]:
    repo = FakeRepo()
    repo.channels.add("c1")
    repo.threads.add("t1")
    repo.members.add("u1")
    repo.members.update(f"m{i}" for i in range(100))
    ids = (f"id{n}" for n in itertools.count(1))
    return Services(repo, clock=lambda: 1000, id_gen=lambda: next(ids)), repo


class _ItemRefused(Exception):
    def __init__(self, error_type: str):
        super().__init__(error_type)
        self.error_type = error_type


def _batch_item(**item):
    svc, _ = _world()
    item.setdefault("text", "a document")
    [receipt] = svc.ingest_documents(CTX, documents=[item])
    if receipt["status"] == "error":
        raise _ItemRefused(receipt["errorType"])


def _x(n: int) -> str:
    return "x" * n


PROBES: dict[str, Any] = {
    "post_message.text": lambda n: _world()[0].post_message(CTX, thread_id="t1", text=_x(n)),
    "post_message.mentions": lambda n: _world()[0].post_message(
        CTX, thread_id="t1", text="hi", mentions=[f"m{i}" for i in range(n)],
    ),
    "create_channel.name": lambda n: _world()[0].create_channel(CTX, name=_x(n)),
    "create_thread.title": lambda n: _world()[0].create_thread(
        CTX, channel_id="c1", title=_x(n),
    ),
    "ingest_document.text": lambda n: _world()[0].ingest_document(CTX, text=_x(n)),
    "ingest_document.title": lambda n: _world()[0].ingest_document(
        CTX, text="a document", title=_x(n),
    ),
    "ingest_document.source_format": lambda n: _world()[0].ingest_document(
        CTX, text="a document", source_format=_x(n),
    ),
    "ingest_document.source_label": lambda n: _world()[0].ingest_document(
        CTX, text="a document", source_label=_x(n),
    ),
    "ingest_documents.documents": lambda n: _world()[0].ingest_documents(
        CTX, documents=[{"text": f"document {i}"} for i in range(n)],
    ),
    "ingest_documents.item.text": lambda n: _batch_item(text=_x(n)),
    "ingest_documents.item.title": lambda n: _batch_item(title=_x(n)),
    "ingest_documents.item.source_format": lambda n: _batch_item(source_format=_x(n)),
    "ingest_documents.item.source_label": lambda n: _batch_item(source_label=_x(n)),
}


def _refusal(probe, n: int) -> str | None:
    try:
        probe(n)
    except _ItemRefused as exc:
        return exc.error_type
    except Exception as exc:  # noqa: BLE001 — the class name is the assertion
        return type(exc).__name__
    return None


def _service_layers():
    for key, layers in MAPPING.items():
        for layer in layers:
            if isinstance(layer, Service):
                yield key, layer


@pytest.mark.parametrize(
    "key,layer", list(_service_layers()),
    ids=lambda v: "|".join(v) if isinstance(v, tuple) else "",
)
def test_service_layer_refuses_past_rest_limit_and_accepts_at_it(key, layer):
    _route, _field, constraint = key
    limit = PARITY_REST[key]
    if constraint in ("maxLength", "maxItems"):
        past = limit + 1
    elif constraint in ("minLength", "minItems"):
        past = limit - 1
    else:  # pragma: no cover — no numeric/pattern bound is service-layer today
        pytest.fail(f"no service probe semantics for {constraint}")
    probe = PROBES[layer.probe]
    assert _refusal(probe, limit) is None, f"{layer.probe} refused REST's own limit {limit}"
    assert _refusal(probe, past) == layer.error.__name__, (
        f"{layer.probe} at {past}: expected {layer.error.__name__}"
    )


# ── collector coverage probe (K-101 review P1-1) ────────────────────────────────
# The guard above is only as good as the two collectors' reach. A walker that
# recursed into today's shapes only would let the *next* bound (a list
# element, an Enum-class query param, an MCP-only element bound) through every
# check silently. So the collectors are gated here on synthetic declarations:
# one constrained field per value of each axis the two frameworks can publish —
# location (path, query, body / tool argument), wrapping (bare, Optional, list
# element, Optional list element, dict value, tuple position, nested model,
# list of nested models) and spelling (inline `Field`/`Query`/`Path`,
# `Annotated`, `Literal`, `Enum` class). Expected keys are written by hand from
# JSON-Schema semantics, never read back from the collector.


class _Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class _Inner(BaseModel):
    x: str = Field(max_length=10)


class _ProbeBody(BaseModel):
    bare: str = Field(max_length=1)
    annotated: Annotated[str, Field(min_length=2)]
    optional: str | None = Field(None, max_length=3)
    element: list[Annotated[str, Field(max_length=4)]] = []
    optional_element: list[Annotated[str, Field(max_length=5)]] | None = None
    dict_value: dict[str, Annotated[int, Field(le=6)]] = {}
    pair: tuple[Annotated[str, Field(max_length=7)], int] = ("a", 1)
    inner: _Inner
    inners: list[_Inner] = []
    kind: Literal["k1", "k2"] = "k1"
    color: _Color = _Color.RED
    optional_color: _Color | None = None


def _probe_rest_app() -> FastAPI:
    router = APIRouter()

    @router.post("/probe/{pid}")
    def probe(
        body: _ProbeBody,
        pid: Annotated[str, Path(max_length=11)],
        q: str = Query("q", max_length=12),
        oq: str | None = Query(None, min_length=13),
        lq: list[Annotated[str, Field(max_length=14)]] = Query([]),
        lit: Literal["a", "b"] = "a",
        en: _Color = _Color.RED,
        oen: _Color | None = None,
    ):
        return None

    app = FastAPI()
    app.include_router(router)
    return app


_COLORS = ["red", "blue"]
PROBE = "POST /probe/{pid}"
EXPECTED_REST_PROBE = {
    (PROBE, "path.pid", "maxLength"): 11,
    (PROBE, "query.q", "maxLength"): 12,
    (PROBE, "query.oq", "minLength"): 13,
    (PROBE, "query.lq[]", "maxLength"): 14,
    (PROBE, "query.lit", "enum"): ["a", "b"],
    (PROBE, "query.en", "enum"): _COLORS,
    (PROBE, "query.oen", "enum"): _COLORS,
    (PROBE, "body.bare", "maxLength"): 1,
    (PROBE, "body.annotated", "minLength"): 2,
    (PROBE, "body.optional", "maxLength"): 3,
    (PROBE, "body.element[]", "maxLength"): 4,
    (PROBE, "body.optional_element[]", "maxLength"): 5,
    (PROBE, "body.dict_value{}", "maximum"): 6,
    (PROBE, "body.pair[0]", "maxLength"): 7,
    (PROBE, "body.inner.x", "maxLength"): 10,
    (PROBE, "body.inners[].x", "maxLength"): 10,
    (PROBE, "body.kind", "enum"): ["k1", "k2"],
    (PROBE, "body.color", "enum"): _COLORS,
    (PROBE, "body.optional_color", "enum"): _COLORS,
}


def _probe_mcp_server() -> FastMCP:
    server = FastMCP("bound-probe")

    @server.tool()
    def probe_tool(
        annotated: Annotated[str, Field(max_length=21)],
        inner: _Inner,
        optional: Annotated[str | None, Field(max_length=22)] = None,
        element: list[Annotated[str, Field(max_length=23)]] | None = None,
        dict_value: dict[str, Annotated[int, Field(ge=24)]] | None = None,
        inners: list[_Inner] | None = None,
        kind: Literal["k1", "k2"] = "k1",
        color: _Color = _Color.RED,
        optional_color: _Color | None = None,
    ) -> None:
        return None

    return server


EXPECTED_MCP_PROBE = {
    ("probe_tool", "annotated", "maxLength"): 21,
    ("probe_tool", "optional", "maxLength"): 22,
    ("probe_tool", "element[]", "maxLength"): 23,
    ("probe_tool", "dict_value{}", "minimum"): 24,
    ("probe_tool", "inner.x", "maxLength"): 10,
    ("probe_tool", "inners[].x", "maxLength"): 10,
    ("probe_tool", "kind", "enum"): ["k1", "k2"],
    ("probe_tool", "color", "enum"): _COLORS,
    ("probe_tool", "optional_color", "enum"): _COLORS,
}


def test_rest_collector_reports_every_planted_shape():
    found = rest_constraints(_probe_rest_app())
    missing = {k: v for k, v in EXPECTED_REST_PROBE.items() if found.get(k) != v}
    assert missing == {}, f"planted REST constraints the collector missed: {missing}"


def test_mcp_collector_reports_every_planted_shape():
    found = mcp_constraints(_probe_mcp_server())
    missing = {k: v for k, v in EXPECTED_MCP_PROBE.items() if found.get(k) != v}
    assert missing == {}, f"planted MCP constraints the collector missed: {missing}"


def test_schema_collector_covers_composition_axes_the_frameworks_do_not_emit_today():
    """`allOf`/`oneOf`, array-form `items`, `patternProperties`, a `$ref`
    chain with sibling keywords and a recursive model: shapes a future
    FastAPI/pydantic/FastMCP release may publish, planted directly."""
    root = {
        "$defs": {
            "A": {"$ref": "#/$defs/B", "maxLength": 31},
            "B": {"minLength": 32},
            "Node": {
                "type": "object",
                "properties": {
                    "label": {"maxLength": 33},
                    "child": {"$ref": "#/$defs/Node"},
                },
            },
        },
        "type": "object",
        "properties": {
            "all": {"allOf": [{"maxLength": 34}, {"minLength": 35}]},
            "one": {"oneOf": [{"type": "null"}, {"maximum": 36}]},
            "tuple": {"type": "array", "items": [{"maxLength": 37}, {"minimum": 38}]},
            "pattern": {"patternProperties": {"^x": {"maxLength": 39}}},
            "chained": {"$ref": "#/$defs/A"},
            "tree": {"$ref": "#/$defs/Node"},
            "negated": {"not": {"maxLength": 40}},
        },
    }
    found: dict[tuple[str, str], Any] = {}
    schema_constraints(root, root, "", found)
    assert found == {
        ("all", "maxLength"): 34, ("all", "minLength"): 35,
        ("one", "maximum"): 36,
        ("tuple[0]", "maxLength"): 37, ("tuple[1]", "minimum"): 38,
        ("pattern{}", "maxLength"): 39,
        ("chained", "maxLength"): 31, ("chained", "minLength"): 32,
        # a recursive model is walked once: its second visit stops, so the
        # bound is reported where the model first appears, not at every depth
        ("tree.label", "maxLength"): 33,
    }


def test_schema_collector_refuses_conflicting_branches_and_foreign_refs():
    conflicting = {"anyOf": [{"maxLength": 3}, {"maxLength": 5}]}
    with pytest.raises(ConflictingConstraintError):
        schema_constraints(conflicting, conflicting, "x", {})
    foreign = {"$ref": "https://example.com/schema.json"}
    with pytest.raises(ValueError, match="non-local"):
        schema_constraints(foreign, foreign, "x", {})


def test_rest_collector_resolves_referenced_parameters_and_request_bodies():
    """OpenAPI allows a parameter object and a request body to be `$ref`s into
    `components`; FastAPI does not emit either today, so they are planted in a
    hand-written document."""
    spec = {
        "paths": {"/r": {"post": {
            "parameters": [{"$ref": "#/components/parameters/P"}],
            "requestBody": {"$ref": "#/components/requestBodies/B"},
        }}},
        "components": {
            "parameters": {"P": {"in": "query", "name": "p", "schema": {"maxLength": 41}}},
            "requestBodies": {"B": {"content": {"application/json": {
                "schema": {"properties": {"f": {"minLength": 42}}},
            }}}},
        },
    }

    class _SpecApp:
        def openapi(self):
            return spec

    assert rest_constraints(_SpecApp()) == {
        ("POST /r", "query.p", "maxLength"): 41,
        ("POST /r", "body.f", "minLength"): 42,
    }
