"""The ``ccg`` command line (C-1001 P1-25a): ``ask``, ``resolve``, ``refresh``, ``raw``.

    cd cpg && .venv/bin/python -m clients.cli.main ask --request '{"useCase": "impact", "anchor": {"kind": "symbol", "repo": "r", "qname": "m.py::A"}}'

Output is canonical JSON on stdout (``ask --format text|markdown`` prints ``ccg.render.render_text``/``render_markdown`` instead). The rules live in ``clients/common.py``; this
module parses arguments, prints, and chooses the exit code (``docs/plans/connected-code-graph-p1-answers.md`` P1-25 (1), (8), (9)):

* ``3`` with ``--fail-on`` given, the document is ``ambiguous`` or ``unresolved`` (output printed; neither success nor tripped: a gate never reads "could not understand or locate" as clean; plan P1-25 (8b)). Without ``--fail-on`` such a document exits 0.
* ``0`` success; ``1`` ``--fail-on`` tripped (the output is still printed); ``2`` a typed error (``{"error": {"code", "message"}}`` on stderr, nothing on stdout) or a usage error.
* ``--fail-on possible`` trips when any item (a folded member included) is ``possible``; ``--fail-on any`` when the answer lists any item. A ``notices`` entry never changes the
  code (``-answer-spec2.md`` §5.1, §9 Q-1): the exit code keeps meaning "items of that status exist".
* The document is validated before it is printed or rendered; an invalid one prints nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from ccg import answer
from ccg.render.common import all_items
from clients import common
from clients.common import BadRequest, Client, ClientError

EXIT_OK, EXIT_FAIL_ON, EXIT_ERROR, EXIT_UNRESOLVED = 0, 1, 2, 3
FAIL_ON = ("possible", "any")


def fail_on_tripped(doc: dict, level: str | None) -> bool:
    """Whether ``doc`` makes ``--fail-on level`` exit 1. Only items count; ``notices`` and every other field are ignored."""
    if level is None:
        return False
    items = list(all_items(doc)) if doc.get("interpretation", {}).get("status") == "resolved" else []
    if level == "any":
        return bool(items)
    if level == "possible":
        return any(i["status"] == "possible" for i in items)
    raise BadRequest(f"--fail-on is one of {list(FAIL_ON)}")


def _json(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise BadRequest(f"{what} is not valid JSON") from None


def _text_arg(value: str) -> str:
    """``-`` reads standard input."""
    return sys.stdin.read() if value == "-" else value


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccg", description="Connected code graph client (P1: impact and depends-on over the local store).")
    p.add_argument("--system", help="system name (default: $CCG_SYSTEM, else the default manifest's)")
    sub = p.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer a structured request or a plain-language question")
    src = ask.add_mutually_exclusive_group(required=True)
    src.add_argument("--request", help="structured request as JSON ('-': stdin)")
    src.add_argument("--question", help="plain-language question; until the interpreter is built, name the anchor (path:line, path::symbol, 'VERB /route', A.b), "
                                        "optionally with 'impact' or 'depends-on'")
    ask.add_argument("--cache-bypass", action="store_true", help="plain-language ask: neither read nor write the interpretation cache (no effect on a structured request; the cache is in memory, so a one-process CLI ask never hits it, only the MCP server's long-lived process does)")
    ask.add_argument("--format", choices=common.FORMATS, default="json")
    ask.add_argument("--fail-on", choices=FAIL_ON, help="exit 1 when the answer lists a possible item / any item")
    res = sub.add_parser("resolve", help="list the candidates of an anchor, with their ids")
    res.add_argument("--anchor", required=True, help="anchor as JSON ('-': stdin)")
    ref = sub.add_parser("refresh", help="rebuild the system from its local clones")
    ref.add_argument("--scope", nargs="+", metavar="REPO_ID", help="repoIds to rebuild (default: all)")
    raw = sub.add_parser("raw", help="read-only Cypher as the ccg_raw_<system> credential (GRAPH.RO_QUERY)")
    raw.add_argument("--cypher", required=True, help="query text ('-': stdin)")
    raw.add_argument("--timeout-ms", type=int, help="time limit; the server refuses one above its TIMEOUT_MAX")
    return p


def _dump(value: Any) -> str:
    return common.dump_json(value) + "\n"


def run(args: argparse.Namespace, client: Client, out: TextIO) -> int:
    if args.command == "ask":
        doc = client.ask(_json(_text_arg(args.request), "the request") if args.request is not None else None, question=args.question, cache_bypass=args.cache_bypass)
        out.write(common.render_answer(doc, args.format))
        if args.fail_on and doc.get("interpretation", {}).get("status") in ("ambiguous", "unresolved"):
            return EXIT_UNRESOLVED
        return EXIT_FAIL_ON if fail_on_tripped(doc, args.fail_on) else EXIT_OK
    if args.command == "resolve":
        out.write(_dump(client.resolve(_json(_text_arg(args.anchor), "the anchor"))))
    elif args.command == "refresh":
        out.write(_dump(client.refresh(args.scope)))
    else:
        out.write(_dump(client.raw_query(_text_arg(args.cypher), timeout_ms=args.timeout_ms)))
    return EXIT_OK


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[str | None], Client] = lambda system: Client(system),
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    out, err = stdout or sys.stdout, stderr or sys.stderr
    args = _parser().parse_args(argv)
    try:
        return run(args, client_factory(args.system), out)
    except ClientError as exc:
        err.write(_dump(exc.payload()))
        return EXIT_ERROR
    except answer.AnswerError:
        err.write(_dump({"error": {"code": "invalid-answer", "message": "the answer does not validate; nothing was printed"}}))
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - the backstop: exit 1 means --fail-on and nothing else, so no failure may escape as a traceback
        err.write(_dump({"error": {"code": "internal-error", "message": f"unexpected failure ({type(exc).__name__})"}}))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
