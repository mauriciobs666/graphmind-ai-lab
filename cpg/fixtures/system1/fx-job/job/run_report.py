"""Order report job: reads orders, renders rows, writes the order_report table."""
# Reviewed for the quarterly audit.
import requests
from fx_svc_common.fmt import fmt_date, fmt_money, round_money
from fx_svc_common.split import split_money

from job.api_client import ApiClient
from job.paths import SEARCH_PATH
from job.settings import load_settings

BASE = "/api"
ITEM_ID = 42
LOCALE = "pt-BR"

# TODO call POST /api/v1/orders


def preflight():
    base = load_settings()["base_url"]
    return requests.get(base + "/api/health", timeout=5).ok


def health_probe():
    return requests.get(f"{BASE}/health", timeout=5).ok


def search_reports(term):
    return requests.get(SEARCH_PATH, params={"q": term}, timeout=10).json()


def fetch_item():
    item = requests.get(f"{BASE}/items/{ITEM_ID}", timeout=10).json()
    return split_money(item["price"], 3)


def requeue(order):
    client = ApiClient()
    client.post("/api/v1/orders", order)
    return round_money(order["total"])


def load_orders(conn):
    return conn.execute("SELECT total FROM orders").fetchall()


def render_row(conn, order_id, ts):
    row = conn.execute("SELECT total FROM orders WHERE id = ?", (order_id,)).fetchone()
    return fmt_date(ts) + " " + fmt_money(row[0], LOCALE)


def write_report(conn, order_id, text):
    conn.execute(
        "INSERT INTO order_report (order_id, total_text) VALUES (?, ?)",
        (order_id, text),
    )
