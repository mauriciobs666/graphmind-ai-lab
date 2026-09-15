#!/usr/bin/env python3
"""A fake OpenAI-compatible endpoint for S15 Run A (docs/plans/salesperson-ui.md §6.4).

Purpose: isolate the storefront **server + graph** from real LLM variance. Every
`POST /v1/chat/completions` sleeps a fixed, configurable delay and then answers
with a plain-text, no-tool-call message — never a JSON/bare-call-syntax shape
`falkorchat/llm.py`'s `_parse_content_tool_calls` fallback would recognise as an
embedded tool call — so the `salesperson` workflow's `assistant` step always
completes the turn as an ordinary chat reply on the first round, at a
deterministic, near-instant latency. This is what lets Run A's numbers measure
queueing/graph/REST behaviour rather than model-serving variance.

Zero third-party dependencies — stdlib `http.server` only, so it needs nothing
beyond `python3` (this component's own toolchain is Node; this script does not
assume a Python virtualenv exists anywhere in `salesperson/`).

Usage:
    python3 stub_llm_server.py --port 8899 --delay-ms 300

Point a scratch `opencode.json`'s `provider.lmstudio.options.baseURL` at
`http://127.0.0.1:8899` and pass it via `FALKORCHAT_OPENCODE_CONFIG` when
starting the storefront server for Run A.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CANNED_REPLY = (
    "Thanks for reaching out. I can help you browse the catalog, add items to "
    "your cart, or check your order status."
)


def make_handler(delay_s: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A002 - quiet by default
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path.rstrip("/").endswith("/models"):
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "mistralai/ministral-3-3b", "object": "model"},
                            {
                                "id": "text-embedding-qwen3-embedding-0.6b",
                                "object": "model",
                            },
                        ],
                    },
                )
                return
            self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                req = {}

            if self.path.rstrip("/").endswith("/embeddings"):
                # Not called by the storefront post path (§4.4 measure 3 — no
                # `_safe_embed`), but kept for completeness / direct benchmarking.
                time.sleep(delay_s)
                inputs = req.get("input", [""])
                if isinstance(inputs, str):
                    inputs = [inputs]
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"object": "embedding", "index": i, "embedding": [0.0] * 8}
                            for i in range(len(inputs))
                        ],
                        "model": req.get("model", "stub-embedding"),
                    },
                )
                return

            if self.path.rstrip("/").endswith("/chat/completions"):
                time.sleep(delay_s)
                self._send_json(
                    200,
                    {
                        "id": "stub-chatcmpl",
                        "object": "chat.completion",
                        "model": req.get("model", "stub-model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": CANNED_REPLY},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                )
                return

            self._send_json(404, {"error": "not_found"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=300,
        help="fixed artificial latency per call, milliseconds (default 300)",
    )
    args = parser.parse_args()

    handler = make_handler(args.delay_ms / 1000.0)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"stub_llm_server: listening on http://{args.host}:{args.port} "
        f"(fixed delay {args.delay_ms} ms)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
