"""Store DDL for the connected-code-graph (S17). ``ddl_v1.cypher`` is the source; :func:`parse_ddl` reads it."""
import re
from pathlib import Path

DDL_V1 = Path(__file__).with_name("ddl_v1.cypher")
_IDX = re.compile(r"CREATE INDEX FOR \(n:(\w+)\) ON \(n\.(\w+)\)")
_CON = re.compile(r"GRAPH\.CONSTRAINT CREATE <key> UNIQUE NODE (\w+) PROPERTIES 1 (\w+)")


def parse_ddl(text: str | None = None) -> list[tuple]:
    """Statements in file order: ``("index", label, prop, cypher)`` / ``("constraint", label, prop)``.

    Raises ``ValueError`` on any statement that is neither shape (inline comments, lower-case keywords,
    multi-property constraints, trailing junk, a missing final ``;``): no silent skips."""
    text = DDL_V1.read_text() if text is None else text
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
    if body.strip() and not body.rstrip().endswith(";"):
        raise ValueError("final statement lacks its terminating ';'")
    out: list[tuple] = []
    for stmt in (s.strip() for s in body.split(";")):
        if not stmt:
            continue
        if m := _IDX.fullmatch(stmt):
            out.append(("index", m[1], m[2], stmt))
        elif m := _CON.fullmatch(stmt):
            out.append(("constraint", m[1], m[2]))
        else:
            raise ValueError(f"unrecognised DDL statement: {stmt!r}")
    return out
