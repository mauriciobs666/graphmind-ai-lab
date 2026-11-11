"""A minimal SCIP (protobuf) *encoder* for the P1-15 tests; the production side only ever decodes.

Field numbers follow ``scip.proto``: Index{metadata=1, documents=2}, Metadata{tool_info=2, project_root=3},
ToolInfo{name=1, version=2}, Document{relative_path=1, occurrences=2, symbols=3, position_encoding=6},
Occurrence{range=1, symbol=2, symbol_roles=3}, SymbolInformation{symbol=1}.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# --- the tiny project the golden index describes -----------------------------------------------------------------
HELPER = "scip-python python proj 0.1 `pkg.a`/helper()."
BOX = "scip-python python proj 0.1 `pkg.a`/Box#"
BOX_PUT = "scip-python python proj 0.1 `pkg.a`/Box#put()."
RUN = "scip-python python proj 0.1 `pkg.b`/run()."
BOX_INIT = "scip-python python proj 0.1 `pkg.a`/Box#__init__()."
OTHER = "scip-python python proj 0.1 `pkg.a`/other()."
LEN = "scip-python python python-stdlib 3.11 builtins/len()."
LOCAL = "local 3"

DEF, IMPORT, READ = 1, 2, 8


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def ld(num: int, payload: bytes) -> bytes:
    return varint(num << 3 | 2) + varint(len(payload)) + payload


def vi(num: int, n: int) -> bytes:
    return varint(num << 3) + varint(n)


def occurrence(rng: Sequence[int], symbol: str, roles: int = 0, *, packed: bool = True) -> bytes:
    if packed:
        body = ld(1, b"".join(varint(x) for x in rng))
    else:
        body = b"".join(vi(1, x) for x in rng)
    body += ld(2, symbol.encode())
    if roles:
        body += vi(3, roles)
    return body


def document(path: str, occs: Iterable[bytes] = (), symbols: Iterable[str] = (), encoding: int | None = 1) -> bytes:
    body = ld(1, path.encode())
    for o in occs:
        body += ld(2, o)
    for s in symbols:
        body += ld(3, ld(1, s.encode()))
    if encoding is not None:
        body += vi(6, encoding)
    return body


def index(docs: Iterable[bytes], tool: tuple[str, str] = ("scip-python", "0.6.8"), project_root: str = "file:///src") -> bytes:
    meta = ld(2, ld(1, tool[0].encode()) + ld(2, tool[1].encode())) + ld(3, project_root.encode())
    return ld(1, meta) + b"".join(ld(2, d) for d in docs)


def tiny_index() -> bytes:
    """The stored golden (``tests/data/scip/tiny.scip``), reproduced byte for byte by this function."""
    a = document("pkg/a.py", [occurrence([0, 4, 10], HELPER, DEF), occurrence([4, 6, 9], BOX, DEF),
                              occurrence([5, 8, 11], BOX_PUT, DEF)], [HELPER, BOX, BOX_PUT])
    b = document("pkg/b.py", [occurrence([0, 5, 11], HELPER, IMPORT), occurrence([2, 4, 7], RUN, DEF),
                              occurrence([3, 4, 10], HELPER), occurrence([6, 8, 11], BOX_PUT, READ),
                              occurrence([7, 4, 7], LEN), occurrence([8, 0, 1], LOCAL),
                              occurrence([9, 4, 7], BOX), occurrence([10, 4, 10, 9], HELPER, packed=False)], [RUN])
    return index([a, b])
