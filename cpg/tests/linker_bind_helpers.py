"""Builders for the binding tests (P1-17 dispatch 2): hand-made bundles (header, files, ``configEntry``, ``launchFact``), the launch
facts the pack runtime emits, and a ``Binder`` over them.

Unit ids are readable words (``svc``, ``job``); a never-built unit and an ``Unmapped.ref`` are 32-hex (``hid``). The system layout is the
fixture system's: manifest repos ``fx-svc``, ``fx-job`` and ``fx-web`` are siblings at the system root, so a compose file at
``fx-svc/deploy/compose.yaml`` reaches ``fx-job`` through ``../../fx-job`` (``-p0-compose-binding.md``).
"""
from __future__ import annotations

from ccg import atrest, valuehash
from ccg.linker import Binder, link
from ccg.packrt import PackFacts
from linker_helpers import cons, hid, prov, sym_in
from packrt_helpers import pack

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
REPOS = {"fx-svc": "fx-svc", "fx-job": "fx-job", "fx-web": "fx-web"}


def fastmcp():
    """The shipped ``fastmcp`` pack, or, under the leaf-mutation harness, the (mutated) block of the spec (``packrt_helpers.pack``)."""
    return pack("fastmcp")


def unit_bundle(unit: str, *, repo: str = "fx-svc", root: str = ".", lang: str = "python", check: bytes | str | None = None,
                records=(), files=()) -> list[dict]:
    """A bundle: header (``bindKeyCheck`` of ``check`` when it is a key, the text itself when a string), file records
    ``(path, role)`` and ``records`` (symbols, ``configEntry``, ``launchFact``...)."""
    header = {"type": "header", "unitId": unit, "unit": {"repoId": repo, "rootPath": root, "lang": lang}}
    if isinstance(check, bytes):
        header["bindKeyCheck"] = atrest.bind_key_check(check)
    elif isinstance(check, str):
        header["bindKeyCheck"] = check
    out = [header]
    for path, role in files:
        out.append({"type": "file", "id": f"file-{unit}-{path}", "repoId": repo, "unitId": unit, "path": path, "role": role})
    return out + list(records)


def config_entry(eid: str, path: str, key_path: str, *, repo: str = "fx-svc", fmt: str = "compose", bind: str | None = None,
                 parts: dict | None = None, example: bool = False, line: int = 1) -> dict:
    r = {"type": "configEntry", "id": eid, "repoId": repo, "path": path, "line": line, "format": fmt, "keyPath": key_path, "example": example}
    if bind is not None:
        r["bindValue"] = bind
    if parts is not None:
        r["bindParts"] = parts
    return r


def launch_fact(lid: str, path: str, kind: str, *, service: str | None = None, origin: str = "explicit", line: int = 1,
                ordinal: int = 0, **kw) -> dict:
    r = {"type": "launchFact", "id": lid, "path": path, "line": line, "ordinal": ordinal, "kind": kind, "origin": origin, **kw}
    if service is not None:
        r["service"] = service
    if kind == "command":
        r.setdefault("args", [])
    return r


def port_fact(unit: str, port: int, origin: str, ids_: list[str], pack: str = "fastmcp") -> dict:
    return {"fact": "port", "packId": pack, "status": "confirmed", "reason": None, "unitId": unit, "port": port, "origin": origin,
            "protocol": "http", "launchIds": sorted(ids_)}


def launch_unit(launch_id: str, unit: str) -> dict:
    return {"fact": "launchUnit", "packId": "fastmcp", "status": "confirmed", "reason": None, "launchId": launch_id, "unitId": unit,
            "target": "x:app", "path": "x"}


def mount_fact(unit: str, path: str, match: str = "prefix", pack: str = "fastmcp") -> dict:
    return {"fact": "mountPath", "packId": pack, "status": "confirmed", "reason": None, "unitId": unit, "path": path, "pathMatch": match,
            "siteId": "mount-site", "line": 1, "conditions": []}


def pack_facts(*, providers=(), consumers=(), lifted=(), launch=()) -> PackFacts:
    return PackFacts(list(providers), list(consumers), [], list(lifted), list(launch), [], [])


def url_lifted(site: str, symbol: str, *urls: str, pack: str = "fetch") -> dict:
    """A resolved lifted leaf (an ``http`` consumer's own URL alternatives)."""
    return {"packId": pack, "lifting": "fetch-url", "siteId": site, "symbolId": symbol, "path": "x.py", "line": 1, "urlPath": "a0",
            "alternatives": list(urls), "resolved": True, "leaf": True, "hops": [], "verbs": None, "partialAlternatives": False,
            "weakenedBy": None, "status": "confirmed", "reason": "lifted-role"}


def http_consumer_world(url: str, *, key: str = "GET /api/x"):
    """A web unit with one ``http`` consumer site ``s1`` fetching ``url``, and its fact list."""
    return sym_in("C", "web", path="web/c.py"), cons("C", "http", key, "s1"), url_lifted("s1", "C", url)


def make_binder(facts: PackFacts, bundles, *, key: bytes = KEY, repos=None, packs=None, units=(), verify_ids: bool = False) -> Binder:
    """Hand-made records carry readable ids, so id verification is off unless a test asks for it (``verify_ids=True``)."""
    return Binder(facts, bundles, key=key, repos=REPOS if repos is None else repos, packs=[fastmcp()] if packs is None else packs, units=units,
                  verify_ids=verify_ids)


def run_link(facts: PackFacts, bundles, **kw):
    b = make_binder(facts, bundles, **kw)
    return link(facts, bundles, bindings=b), b


def parts_of(url: str, declared=("api", "job", "worker"), key: bytes = KEY) -> dict:
    got = valuehash.bind_parts(key, url, declared)
    assert got is not None, url
    return got
