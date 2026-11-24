"""Shared fixtures for the key-holder forward tests (egress-spec5 section 7): the committed wire vectors, a scripted
`AF_UNIX` holder double with a connection counter, and a path to the real holder's test harness in `llm-gateway/tests/`
(read only). A missing vectors file or an unimportable `llmgateway` makes the consumer FAIL, never skip (spec5 s7 item 11).
Expected values live in the tests, written as literals."""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path

GATEWAY_TESTS = Path(__file__).resolve().parents[2] / "llm-gateway" / "tests"
VECTORS_PATH = GATEWAY_TESTS / "holder_wire_vectors.json"
if str(GATEWAY_TESTS) not in sys.path:
    sys.path.insert(0, str(GATEWAY_TESTS))  # `holder_harness`, `holder_launcher`: the holder's own test pattern


def load_vectors() -> dict:
    """The shared wire literals. Raises (a test FAILURE) when the file is missing or unparsable."""
    vec = json.loads(VECTORS_PATH.read_text())
    assert vec["version"] == 1
    return vec


def secret_of(vec: dict) -> bytes:
    return bytes.fromhex(vec["secret_hex"])


class ScriptedHolder:
    """Listens on ``<directory>/holder.sock``; counts connections; answers each with ``script(payload) -> bytes | None``
    (the frame to send, length prefix included; None sends nothing and holds the connection until `close`)."""

    def __init__(self, directory: Path, script=None):
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.path = str(directory / "holder.sock")
        self.script = script or (lambda payload: None)
        self.connections = 0
        self.requests: list[bytes] = []
        self._held: list[socket.socket] = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(8)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn: socket.socket):
        try:
            conn.settimeout(10)
            head = b""
            while len(head) < 4:
                c = conn.recv(4 - len(head))
                if not c:
                    return
                head += c
            size = int.from_bytes(head, "big")
            payload = b""
            while len(payload) < size:
                c = conn.recv(size - len(payload))
                if not c:
                    return
                payload += c
            self.requests.append(payload)
            out = self.script(payload)
            if out is None:
                self._held.append(conn)
                return
            conn.sendall(out)
            conn.close()
        except OSError:
            return

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)  # wakes the blocked accept(): a bare close() leaves the listener alive
        except OSError:
            pass
        self.sock.close()
        self._t.join(timeout=2)
        for c in self._held:
            c.close()
