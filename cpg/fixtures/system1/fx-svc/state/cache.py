"""In-memory order cache."""
from types import SimpleNamespace


class OrderCache:
    def __init__(self):
        self.entries = {}
        self.a = SimpleNamespace(b=SimpleNamespace(c=SimpleNamespace(d=0)))

    def put(self, k, v):
        self.entries[k] = v

    def get(self, k):
        return self.entries.get(k)

    def evict(self, k):
        del self.entries[k]

    def limit_deep(self):
        return self.a.b.c.d

    def limit_edge(self):
        return self.a.b.c
