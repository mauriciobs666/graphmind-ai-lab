"""Order persistence."""
import os
import sqlite3

from fx_svc.db.settings import settings

MEMBER_RATE = 0.4
_RATES = {1: 0.4, 2: 0.8}


def connect():
    pool = [sqlite3.connect(os.environ["FX_DB_URL"]) for _ in range(settings.db_pool_size)]
    return pool[0]


def load_rate(item_id):
    return _RATES.get(item_id, MEMBER_RATE)


def insert_order(conn, order):
    conn.execute("INSERT INTO orders (id, total) VALUES (?, ?)", (order.id, order.total))


def archive_orders(conn, before):
    conn.execute("DELETE FROM orders WHERE created < ?", (before,))
