"""Order repositories behind one protocol."""
from typing import Protocol


class Repo(Protocol):
    def save(self, o): ...


class SqlRepo:
    def __init__(self, session):
        self.session = session

    def save(self, o):
        self.session.add(o)
        self.session.commit()


class MemRepo:
    def __init__(self):
        self.items = {}

    def save(self, o):
        self.items[o.id] = o


def get_repo() -> Repo:
    return SqlRepo(session=None)
