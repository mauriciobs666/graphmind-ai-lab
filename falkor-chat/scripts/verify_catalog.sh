#!/usr/bin/env bash
# verify_catalog.sh — check that the product catalog (scripts/seed_catalog.sh,
# K-052 M6) is present in `reference` as expected.
#
# Usage:
#   ./scripts/verify_catalog.sh
#
# ⚠️ STRICTLY READ-ONLY (mirrors verify_workflows.sh's posture) — GRAPH.RO_QUERY
# only, never GRAPH.QUERY. Publishes/seeds/deletes nothing. When something is
# missing it prints the seed command and exits non-zero; running that command is
# the operator's decision, not this script's.
#
# Checks:
#   1. the expected Product COUNT (exactly the catalog's declared row count);
#   2. a couple of NAMED products exist with the expected category/price
#      (a real assertion beyond just "some rows exist" — catches a partial or
#      corrupted seed, not just an absent one).
#
# Exit 0 = all green. Exit 1 = missing or unexpected.
#
# Env vars (all optional):
#   FALKORDB_HOST (default: 127.0.0.1)
#   FALKORDB_PORT (default: 6379)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"

# Pinned to seed_catalog.sh's own CATALOG literal — 15 declared rows, and two
# spot-checked here by name (first and last of the list, so a truncated seed —
# e.g. the write erroring out partway — is caught either way).
EXPECTED_COUNT=15

echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}

echo "── verifying product catalog in reference ──"

rq() {
  # Read-only GRAPH.RO_QUERY — never GRAPH.QUERY (this script must never write).
  redis-cli -h "$HOST" -p "$PORT" GRAPH.RO_QUERY reference "$1" --compact
}

failures=0

# ── 1. expected count ──────────────────────────────────────────────────────
count_out="$(rq 'MATCH (p:Product) RETURN count(p)' 2>&1)" || {
  echo "  Product count : ERROR ($count_out)"
  echo "    (reference may not exist yet, or the schema was never bootstrapped)"
  failures=1
  count_out=""
}
count="$(printf '%s\n' "$count_out" | grep -Eo '^[0-9]+$' | tail -1)"
if [ -z "${count:-}" ]; then
  count=0
fi
if [ "$count" -eq "$EXPECTED_COUNT" ]; then
  echo "  Product count : $count (expected $EXPECTED_COUNT) — OK"
else
  echo "  Product count : $count (expected $EXPECTED_COUNT) — MISMATCH"
  failures=1
fi

# ── 2. a couple of named products, with expected category/price ───────────
check_product() {
  local name="$1" expected_category="$2" expected_price="$3"
  local out
  out="$(rq "MATCH (p:Product {name: '${name}'}) RETURN p.category, p.price" 2>&1)" || {
    echo "  ${name} : ERROR ($out)"
    failures=1
    return
  }
  if ! printf '%s\n' "$out" | grep -q "$expected_category"; then
    echo "  ${name} : MISSING or category mismatch (expected ${expected_category}, got: $out)"
    failures=1
    return
  fi
  if ! printf '%s\n' "$out" | grep -q "$expected_price"; then
    echo "  ${name} : price mismatch (expected ${expected_price}, got: $out)"
    failures=1
    return
  fi
  echo "  ${name} : category=${expected_category} price=${expected_price} — OK"
}

check_product "Wireless Mouse Pro" "Peripherals" "29.99"
check_product "Smart Home Hub" "Smart Home" "89.99"

echo ""
if [ "$failures" -ne 0 ]; then
  echo "RESULT: FAIL"
  echo ""
  echo "If the catalog is missing or incomplete, seed it:"
  echo "  ./scripts/seed_catalog.sh"
  exit 1
fi

echo "RESULT: OK — product catalog in sync ($count products)"
