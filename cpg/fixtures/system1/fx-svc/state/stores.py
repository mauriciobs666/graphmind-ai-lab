"""Item stores behind one protocol."""
from typing import Protocol


class Store(Protocol):
    def load(self, item_id): ...


class DbStore:
    def __init__(self, session):
        self.session = session

    def load(self, item_id):
        return self.session.fetch(item_id)


class CacheStore:
    def __init__(self, items):
        self.items = items

    def load(self, item_id):
        return self.items[item_id]


def get_store() -> Store:
    return DbStore(session=None)
