#!/usr/bin/env bash
# seed_catalog.sh — seed the fixed, seed-script-write-only electronics catalog
# (K-052 M6, docs/plans/workflow-catalog-lookup.md §3.2) into the GLOBAL
# `reference` graph. Backs the `salesperson` demo agent's `lookup_product_fact`/
# `filter_products` tools (docs/QUERIES.md §15).
#
# Usage:
#   ./scripts/seed_catalog.sh [<wsId>]
#
# The `<wsId>` argument is accepted ONLY for CLI convention parity with
# `seed_demo.sh`/`seed_workflows.sh` (all three scripts take an optional
# workspace id positionally) — the catalog itself is workspace-INDEPENDENT: it
# writes ~15 `Product` nodes into `reference`, never into `ws:<id>` (FR-6, per
# `docs/DESIGN.md` §3/§4: a catalog that is only *looked up* by property key,
# never traversed from a workspace node, has no business being materialized
# per-workspace — unlike `WorkflowDef`). The argument is otherwise unused here.
#
# WHY a Python one-shot over the SERVICE LAYER (mirrors seed_workflows.sh's own
# rationale, not seed_demo.sh's raw redis-cli style): the write is one
# `UNWIND $rows AS row MERGE (p:Product {productId: row.productId}) ON CREATE
# SET ...` — a single parameterized list-of-maps write that is awkward to embed
# in a `CYPHER key=value` redis-cli preamble, but trivial from Python via the
# same `db.reference_graph(db.connect())` accessor `Repository._reference()`
# uses. There is no `services.*` method for writing a `Product` (it is not a
# `WorkflowDef` — no publish-time validation applies to it), so this talks to
# the graph directly, the same posture `scripts/seed_eval_corpus.py` takes for
# its own non-service writes.
#
# IDEMPOTENT BY DESIGN (plan §3.2): `productId` is a DETERMINISTIC slug derived
# from the product's name (e.g. "Wireless Mouse Pro" -> "wireless-mouse-pro"),
# NOT a `uuid4()` minted fresh per run. `scripts/test_queries.sh`'s teardown
# `GRAPH.DELETE`s the entire `reference` graph (schema included) — a naive
# random-id reseed after that wipe would silently duplicate/diverge the catalog
# under fresh ids. A deterministic slug means a re-seed after a wipe
# reconstructs BYTE-IDENTICAL data, and the `MERGE ... ON CREATE SET` (backed by
# the `Product.productId` UNIQUE constraint bootstrap_schema.sh already
# creates) makes a re-seed against a NON-wiped `reference` a clean no-op too.
#
# ORDERING — run this AFTER ./scripts/bootstrap_schema.sh <wsId> (creates the
# `Product` index-then-constraint pair this MERGE relies on). Independent of
# seed_demo.sh/seed_workflows.sh — no ordering constraint against either.
#
# Env vars (all optional):
#   FALKORDB_HOST (default: 127.0.0.1)
#   FALKORDB_PORT (default: 6379)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV_PY="$SERVER_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: server venv not found at $VENV_PY" >&2
  echo "       Create it first:  cd server && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}

echo "── seeding product catalog into reference ──"

FALKORDB_HOST="$HOST" FALKORDB_PORT="$PORT" \
"$VENV_PY" - <<'PY'
import re

from falkorchat import db
from falkorchat.extraction import normalize_name

# The fixed ~15-product consumer-electronics catalog (plan §3.2: "a Python or
# shell-embedded literal, not fetched from anywhere"). name/category/price only
# (FR-1/FR-2/FR-3) — no attribute set beyond this is in scope.
CATALOG = [
    ("Wireless Mouse Pro", "Peripherals", 29.99),
    ("Mechanical Keyboard K200", "Peripherals", 89.99),
    ("Gaming Mouse Pad XL", "Peripherals", 19.99),
    ("Webcam HD 1080p", "Peripherals", 59.99),
    ("27-inch 4K Monitor", "Displays", 349.99),
    ("USB-C Hub 7-in-1", "Accessories", 39.99),
    ("Laptop Stand Aluminum", "Accessories", 34.99),
    ("Wireless Charging Pad", "Accessories", 24.99),
    ("Noise Cancelling Headphones X3", "Audio", 199.99),
    ("Bluetooth Speaker Mini", "Audio", 49.99),
    ("Portable SSD 1TB", "Storage", 109.99),
    ("Smartwatch Series 5", "Wearables", 249.99),
    ("Fitness Tracker Band", "Wearables", 79.99),
    ("Action Camera 4K", "Cameras", 179.99),
    ("Smart Home Hub", "Smart Home", 89.99),
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    """A stable, deterministic `productId` derived from the product name (plan
    §3.2) — NOT `uuid4()` — so a re-seed after a `reference` wipe (test_queries.sh's
    teardown) reconstructs byte-identical ids. Lowercase, non-alnum runs collapsed
    to a single '-', no leading/trailing '-'."""
    return _SLUG_RE.sub("-", name.lower()).strip("-")


rows = [
    {
        "productId": _slugify(name),
        "name": name,
        "nameNormalized": normalize_name(name),
        "category": category,
        "categoryNormalized": normalize_name(category),
        "price": price,
    }
    for name, category, price in CATALOG
]

graph = db.reference_graph(db.connect())
res = graph.query(
    "UNWIND $rows AS row "
    "MERGE (p:Product {productId: row.productId}) "
    "ON CREATE SET p.name = row.name, p.nameNormalized = row.nameNormalized, "
    "              p.category = row.category, "
    "              p.categoryNormalized = row.categoryNormalized, "
    "              p.price = row.price "
    "RETURN count(p) AS n",
    {"rows": rows},
)
seen = res.result_set[0][0]

count_res = graph.ro_query("MATCH (p:Product) RETURN count(p) AS n")
total = count_res.result_set[0][0]

print(f"  {seen} product row(s) processed ({len(rows)} declared in this script)")
print(f"  {total} Product node(s) now in reference")
if total < len(rows):
    print(
        "WARNING: fewer Product nodes than declared rows — check for a "
        "productId collision (two different names slugifying the same)",
    )
PY

echo ""
echo "Catalog seeded (idempotent, create-only). Verify with:"
echo "  ./scripts/verify_catalog.sh"
