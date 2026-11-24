"""Test helpers: a real stdlib HTTP stand-in provider, a fake provider kind, config builder."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llmgateway.config import ConfigSource
from llmgateway.providers.base import (
    BuildContext,
    BuiltRequest,
    ParsedReply,
    Provider,
    encode_body,
    map_settings,
)
from llmgateway.types import Usage

ME = os.getuid()
OK_BODY = json.dumps(
    {
        "model": "reported-model",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
).encode()


class StandIn:
    """Serves scripted replies, one per request (the last repeats); records what it received."""

    def __init__(self, *replies):
        self.replies = list(replies) or [(200, "application/json", OK_BODY)]
        self.received: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                i = min(len(outer.received), len(outer.replies) - 1)
                outer.received.append(
                    {"path": self.path, "body": body, "headers": dict(self.headers.items())}
                )
                status, ctype, data = outer.replies[i]
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def refusal(code):
    return (
        499,
        "application/vnd.llmgateway.refusal+json",
        json.dumps({"holderRefusal": code}).encode(),
    )


def error(status, text="boom"):
    return (status, "application/json", json.dumps({"error": {"message": text}}).encode())


class FakeProvider(Provider):
    """OpenAI-shaped fake kind. Settings 1:1 except `seed`, which it cannot honour (None)."""

    ops: dict = {}

    def __init__(self, kind="openai_compat"):
        self.kind = kind

    TABLE = {
        "temperature": "temperature",
        "top_p": "top_p",
        "max_tokens": "max_tokens",
        "seed": "seed",
        "stop": "stop",
    }

    def build(self, ctx: BuildContext) -> BuiltRequest:
        payload = {"model": ctx.model, "messages": list(ctx.messages)}
        payload.update(map_settings(ctx.provider_id, ctx.settings, self.TABLE, ctx.optional))
        if ctx.response_format is not None:
            payload["response_format"] = ctx.response_format
        return BuiltRequest("/v1/chat/completions", encode_body(payload))

    def parse(self, body, ctx):
        data = json.loads(body)
        u = data["usage"]
        return ParsedReply(
            data["model"],
            data["choices"][0]["message"],
            Usage(u["prompt_tokens"], u["completion_tokens"]),
            raw=data,
        )


def write_config(root, text):
    (root / "etc").mkdir(parents=True, exist_ok=True)
    root.chmod(0o755)
    (root / "etc").chmod(0o755)
    f = root / "etc" / "gateway.toml"
    f.write_text(text)
    f.chmod(0o644)
    return ConfigSource(f, frozenset({ME}), anchor=root)
