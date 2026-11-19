"""Test doubles for the plain-language front (P1-26): :class:`FakeStore`, an in-memory ``Reader`` answering the anchor reads the resolver uses
(``repo_exists``, ``file_by_path``, ``skipped_by_path``, ``symbols_of_file``, ``symbol_by_qname``, ``symbol_by_name``, ``config_by_path``, ``cp_by_key``, ``cp_keys``,
``provides_in``, ``file_by_suffix``, ``skipped_by_suffix``, ``snapshot``) in the row shapes of ``ccg.engine.reader`` (``-p1-graph.md`` §5), and :class:`FrontReader`, the engine's ``AnswerReader`` plus the same
location reads so a question can be resolved and built end to end."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from engine_answer_helpers import AnswerReader

from ccg import answer, ids
from ccg.answer import EXT_LANG
from ccg.engine import config
from ccg.engine.anchors import name_norm

SNAP = "5" * 32


def hid(*parts) -> str:
    return ids.tuple_id(["test", *parts])


def by_suffix(rows, repos, sep, suffix):
    """The two suffix reads' semantics (``-p1-graph.md`` §5) over ``rows`` of ``(repo, path, *rest)``: ``(sep + path)`` ends with ``suffix``, ``repoId`` in ``repos``,
    ordered by ``(repoId, path)``, at most ``OPTIONS_MAX + 1`` rows."""
    hit = sorted((r for r in rows if r[0] in repos and (sep + r[1]).endswith(suffix)), key=lambda r: (r[0], r[1]))
    return hit[:config.OPTIONS_MAX + 1]


@dataclass(frozen=True)
class Sym:
    qname: str
    path: str | None
    start: int = 1
    end: int = 10
    repo: str = "graphmind-ai-lab"
    unit: str | None = None
    regions: tuple | None = None
    depth: int = 0

    @property
    def sid(self) -> str:
        """The real id where the validator recomputes one (V15: a path whose extension maps to a language), so asking documents over this store validate."""
        lang = EXT_LANG.get(PurePosixPath(self.path or "").suffix)
        return ids.symbol_id(lang, self.repo, self.qname) if lang else hid("sym", self.repo, self.qname)

    @property
    def unit_id(self) -> str:
        return self.unit or hid("unit", self.repo)

    @property
    def name_norm(self) -> str:
        return name_norm(self.qname.rpartition("::")[2].rpartition(".")[2])


@dataclass(frozen=True)
class Cp:
    key: str
    kind: str
    provider: str | None = None          # the qname of the providing symbol

    @property
    def cp_id(self) -> str:
        return hid("cp", self.kind, self.key, self.provider or "")


class FakeStore:
    def __init__(self, symbols=(), *, repos=("graphmind-ai-lab",), skipped=(), cps=(), config=(), extra_files=(), snapshot=SNAP):
        self.symbols, self.repos, self.cps, self.config = list(symbols), tuple(repos), list(cps), list(config)
        self.skipped = {(r, p): (u, why) for r, p, u, why in skipped}
        self.files = {(s.repo, s.path): s.unit_id for s in self.symbols if s.path} | {(r, p): hid("unit", r) for r, p in extra_files}
        self.snapshot, self.calls = snapshot, []

    def read(self, name, params=None):
        p = dict(params or {})
        self.calls.append((name, p))
        return [tuple(r) for r in getattr(self, "_" + name)(**p)]

    def names(self):
        return [c[0] for c in self.calls]

    def recipe_queries(self):
        return []

    def _repo_exists(self, repo):
        return [(repo,)] if repo in self.repos else []

    def _file_by_path(self, repo, path):
        return [(hid("file", repo, path), self.files[(repo, path)])] if (repo, path) in self.files else []

    def _skipped_by_path(self, repo, path):
        return [self.skipped[(repo, path)]] if (repo, path) in self.skipped else []

    def _file_by_suffix(self, repos, sep, suffix):
        return by_suffix([(r, p, hid("file", r, p), u) for (r, p), u in self.files.items()], repos, sep, suffix)

    def _skipped_by_suffix(self, repos, sep, suffix):
        return by_suffix([(r, p, u, why) for (r, p), (u, why) in self.skipped.items()], repos, sep, suffix)

    def _symbols_of_file(self, repo, path):
        return [(s.sid, s.qname, s.start, s.end, s.regions, s.depth) for s in self.symbols if s.repo == repo and s.path == path]

    def _symbol_by_qname(self, repo, qname):
        return [(s.sid, s.unit_id, s.path, s.start, s.end, s.regions) for s in self.symbols if s.repo == repo and s.qname == qname]

    def _symbol_by_name(self, name):
        return [(s.sid, s.repo, s.unit_id, s.qname, s.path, s.start) for s in self.symbols if s.name_norm == name]

    def _config_by_path(self, repo, path):
        return [(hid("cfg", repo, path, kp), hid("unit", repo), line, kp, None) for r, pth, kp, line in self.config if r == repo and pth == path]

    def _cp_by_key(self, key):
        return [(c.cp_id, c.kind, c.key) for c in self.cps if c.key == key]

    def _cp_keys(self, kind):
        return [(c.cp_id, c.key) for c in self.cps if c.kind == kind]

    def _provides_in(self, ids):
        out = []
        for c in self.cps:
            if c.cp_id in ids and c.provider:
                s = next(s for s in self.symbols if s.qname == c.provider)
                out.append((c.cp_id, s.sid, s.repo, s.unit_id, s.qname, s.path, s.start, "confirmed", None))
        return out

    def _snapshot(self):
        return [(self.snapshot, 0, answer.EMPTY_VERDICT_LOG_HASH, "1.1", "1.0")]


class FrontReader(AnswerReader):
    """The engine's ``AnswerReader`` over a ``tests/test_engine_answer`` world, plus the location reads (a symbol's ``path`` is its file) and ``repo_exists``."""

    def __init__(self, g, snapshot_rows=None):
        super().__init__(g, snapshot_rows)
        self.g = g

    def _syms(self):
        return list(self._nodes.get("Symbol", {}).values())

    def _x_repo_exists(self, repo):
        return [(repo,)] if any(s.get("repoId") == repo for s in self._syms()) else []

    def _x_file_by_path(self, repo, path):
        units = {s.get("unitId") for s in self._syms() if s.get("repoId") == repo and s.get("path") == path}
        return [(hid("file", repo, path), u) for u in sorted(units)[:1]]

    def _x_skipped_by_path(self, repo, path):
        return []

    def _x_file_by_suffix(self, repos, sep, suffix):
        files = {(s["repoId"], s["path"]): s.get("unitId") for s in self._syms() if s.get("path")}
        return by_suffix([(r, p, hid("file", r, p), u) for (r, p), u in files.items()], repos, sep, suffix)

    def _x_skipped_by_suffix(self, repos, sep, suffix):
        return []

    def _x_symbols_of_file(self, repo, path):
        return [(s["symbolId"], s["qname"], s["lineStart"], s["lineEnd"], s.get("ownedRegions"), s.get("depth"))
                for s in self._syms() if s.get("repoId") == repo and s.get("path") == path]

    def _x_symbol_by_name(self, name):
        return [(s["symbolId"], s["repoId"], s.get("unitId"), s["qname"], s.get("path"), s["lineStart"])
                for s in self._syms() if name_norm(s["qname"].rpartition("::")[2].rpartition(".")[2]) == name]
