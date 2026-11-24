"""D9a — the broker's key-holder client (`ccg.broker.holderclient`), egress-spec5 section 3 items 3-5 and section 7 items
2, 10, 11, 13.

Wire parity is pinned by the shared literals in `llm-gateway/tests/holder_wire_vectors.json` (a missing file fails),
and a live round trip against the real `llmgateway.framing`; the closed reply table is driven against a scripted
`AF_UNIX` double (directory owned by the test uid, passed as the holder uid).
"""
from __future__ import annotations

import ast
import json
import os
import struct
import time
from pathlib import Path

import pytest

from broker_helpers import Harness, hosted_call, hosted_egress, load_system
from gateway_helpers import ScriptedHolder, load_vectors, secret_of
from ccg import manifest as mf
from ccg.broker import framing, holderclient
from ccg.broker.holderclient import HolderClient
from ccg.broker.structured import Exchange
from llmgateway import framing as gw  # the real holder wire: an unimportable llmgateway FAILS these tests

UID = os.getuid()
ENDPOINT = "https://api.hosted.example.com:443"
PATH = "/v1/chat/completions"
BODY = b'{"m":1}'


@pytest.fixture
def vec():
    return load_vectors()


def signed(secret, nonce, reason="completed", status=200, ctype="application/json", body=b"{}") -> bytes:
    return gw.build_response(secret, request_nonce=nonce, close_reason=reason, status=status, content_type=ctype,
                             body=body)


def holder_double(tmp_path, vec, make_reply):
    """A holder double: ``make_reply(secret, header) -> frame bytes | None`` for each request it receives."""
    secret = secret_of(vec)

    def script(payload):
        hdr, _ = gw.parse_request(payload)
        return make_reply(secret, hdr)

    return ScriptedHolder(tmp_path / "run", script), secret


def client_for(h: ScriptedHolder, secret: bytes, **kw) -> HolderClient:
    return HolderClient(h.path, UID, secret, **kw)


# ------------------------------------------------------------------ wire parity (item 11)


def test_a_missing_or_unparsable_vectors_file_fails_not_skips(monkeypatch, tmp_path):
    import gateway_helpers
    monkeypatch.setattr(gateway_helpers, "VECTORS_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        gateway_helpers.load_vectors()


def test_every_request_vector_is_reproduced_byte_for_byte(vec):
    assert len(vec["requests"]) >= 3
    for r in vec["requests"]:
        body = bytes.fromhex(r["body_hex"])
        mac = holderclient.request_mac(secret_of(vec), caller=r["caller"], ts=r["ts"], nonce=r["nonce"],
                                       method=r["method"], endpoint=r["endpoint"], path=r["path"],
                                       run_id=r["run_id"], body=body)
        assert mac == r["mac"], r["name"]
        frame = holderclient.build_request(secret_of(vec), caller=r["caller"], method=r["method"],
                                           endpoint=r["endpoint"], path=r["path"], run_id=r["run_id"], body=body,
                                           ts=r["ts"], nonce=r["nonce"])
        assert frame.hex() == r["frame_hex"], r["name"]


def test_a_none_run_id_is_the_empty_string_on_the_wire(vec):
    r = next(x for x in vec["requests"] if x["name"] == "empty-run-id")
    frame = holderclient.build_request(secret_of(vec), caller=r["caller"], method=r["method"],
                                       endpoint=r["endpoint"], path=r["path"], run_id=None,
                                       body=bytes.fromhex(r["body_hex"]), ts=r["ts"], nonce=r["nonce"])
    assert frame.hex() == r["frame_hex"]


def test_every_reply_vector_maps_to_its_exchange(vec):
    expected = {
        "signed-200": ("completed", 200, "application/json"),
        "signed-499-refusal": ("completed", 499, "application/vnd.llmgateway.refusal+json"),
        "unsigned-bad-credential": ("completed", 499, "application/vnd.llmgateway.refusal+json"),
    }
    assert {r["name"] for r in vec["replies"]} >= set(expected)
    for r in vec["replies"]:
        frame = bytes.fromhex(r["frame_hex"])
        assert int.from_bytes(frame[:4], "big") == len(frame) - 4
        if r["mac"]:
            assert holderclient.response_mac(
                secret_of(vec), request_nonce=r["request_nonce"], close_reason=r["close_reason"], status=r["status"],
                content_type=r["content_type"], body=bytes.fromhex(r["body_hex"])) == r["mac"], r["name"]
        ex = holderclient.parse_reply(frame[4:], secret_of(vec), r["request_nonce"])
        want = expected[r["name"]]
        assert (ex.close_reason, ex.status, ex.content_type) == want and ex.body == bytes.fromhex(r["body_hex"])


def test_a_signed_reply_vector_does_not_verify_against_another_nonce(vec):
    r = next(x for x in vec["replies"] if x["name"] == "signed-200")
    with pytest.raises(holderclient._Reject):
        holderclient.parse_reply(bytes.fromhex(r["frame_hex"])[4:], secret_of(vec), "another-nonce")


def test_the_canonical_origin_vectors_hold_for_the_broker_canonicaliser(vec):
    assert len(vec["origins"]) >= 4
    for o in vec["origins"]:
        if o["canonical"] is None:
            with pytest.raises(mf.ManifestError):
                mf.canonical_endpoint(o["input"])
        else:
            assert mf.canonical_endpoint(o["input"]) == o["canonical"], o["input"]


def test_the_real_gateway_wire_agrees_with_the_client_on_the_same_inputs(vec):
    """Live parity: the real `llmgateway.framing` produces the same request bytes and verifies the client's reply parse."""
    for r in vec["requests"]:
        body = bytes.fromhex(r["body_hex"])
        real, _ = gw.build_request(secret_of(vec), caller=r["caller"], method=r["method"], endpoint=r["endpoint"],
                                   path=r["path"], run_id=r["run_id"], body=body, ts=r["ts"], nonce=r["nonce"])
        mine = holderclient.build_request(secret_of(vec), caller=r["caller"], method=r["method"],
                                          endpoint=r["endpoint"], path=r["path"], run_id=r["run_id"], body=body,
                                          ts=r["ts"], nonce=r["nonce"])
        assert mine == real


def test_a_live_round_trip_against_the_real_framing(tmp_path, vec):
    """The double verifies the client's request with the real `parse_request` + `request_mac` and answers with the real
    `build_response`; the client's `Exchange` is the answer, and the real `parse_reply` accepts the same bytes."""
    seen = []

    def make(secret, hdr):
        seen.append(hdr)
        return signed(secret, hdr.nonce, body=b'{"ok":2}')

    h, secret = holder_double(tmp_path, vec, make)
    try:
        t = [1_900_000_123.4]
        ex = client_for(h, secret, now=lambda: t[0])(ENDPOINT, "POST", PATH, BODY, "eval-1")
    finally:
        h.close()
    [hdr] = seen
    assert ex == Exchange("completed", 200, "application/json", b'{"ok":2}', ex.bytes_out, ex.bytes_in)
    assert (hdr.caller, hdr.ts, hdr.method, hdr.endpoint, hdr.path, hdr.run_id) == (
        "broker", 1_900_000_123, "POST", ENDPOINT, PATH, "eval-1")
    assert gw.request_mac(secret, caller="broker", ts=hdr.ts, nonce=hdr.nonce, method="POST", endpoint=ENDPOINT,
                          path=PATH, run_id="eval-1", body=BODY) == hdr.mac
    assert len(hdr.nonce) == 32 and int(hdr.nonce, 16) >= 0
    frame = signed(secret, hdr.nonce, body=b'{"ok":2}')
    assert gw.parse_reply(frame[4:], secret, hdr.nonce).status == 200
    assert ex.bytes_out == 4 + len(h.requests[0]) and ex.bytes_in == len(frame)


def test_each_call_signs_a_fresh_nonce_and_none_run_id_goes_out_empty(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce))
    try:
        c = client_for(h, secret)
        c(ENDPOINT, "POST", PATH, BODY, None)
        c(ENDPOINT, "POST", PATH, BODY, None)
    finally:
        h.close()
    heads = [gw.parse_request(p)[0] for p in h.requests]
    assert heads[0].nonce != heads[1].nonce and {x.run_id for x in heads} == {""}


# ------------------------------------------------------------------ the closed reply table (item 2)


def _bad_mac(secret, hdr):
    frame = bytearray(signed(secret, hdr.nonce))
    h, body = gw.decode_payload(bytes(frame[4:]))
    h["mac"] = "0" * 64
    return gw.encode_frame(h, body)


def _raw_header(header: dict, body=b"") -> bytes:
    return gw.encode_frame(header, body)


def _signed_custom(secret, hdr, **over):
    f = {"reason": "completed", "status": 200, "ctype": "application/json", "body": b"{}", **over}
    mac = gw.response_mac(secret, request_nonce=hdr.nonce, close_reason=f["reason"], status=f["status"],
                          content_type=f["ctype"], body=f["body"])
    return _raw_header({"closeReason": f["reason"], "status": f["status"], "contentType": f["ctype"], "mac": mac},
                       f["body"])


REFUSAL = "application/vnd.llmgateway.refusal+json"


def _unsigned(code, ctype=REFUSAL):
    return lambda s, hdr: gw.build_response(None, request_nonce=hdr.nonce, close_reason="completed", status=499,
                                            content_type=ctype, body=json.dumps({"holderRefusal": code}).encode())


CASES = {
    "signed-200": (lambda s, h: signed(s, h.nonce, body=b'{"a":1}'),
                   ("completed", 200, "application/json", b'{"a":1}')),
    "empty-content-type-is-none": (lambda s, h: signed(s, h.nonce, ctype=""), ("completed", 200, None, b"{}")),
    "signed-499-refusal": (lambda s, h: signed(s, h.nonce, status=499, ctype=REFUSAL,
                                               body=b'{"holderRefusal": "run_unmatched"}'),
                           ("completed", 499, REFUSAL, b'{"holderRefusal": "run_unmatched"}')),
    "signed-status-100": (lambda s, h: signed(s, h.nonce, status=100), ("completed", 100, "application/json", b"{}")),
    "signed-status-599": (lambda s, h: signed(s, h.nonce, status=599), ("completed", 599, "application/json", b"{}")),
    "unsigned-bad-credential": (_unsigned("bad_credential"), ("completed", 499, REFUSAL,
                                                              b'{"holderRefusal": "bad_credential"}')),
    "unsigned-unknown-caller": (_unsigned("unknown_caller"), ("completed", 499, REFUSAL,
                                                              b'{"holderRefusal": "unknown_caller"}')),
    "signed-upstream-error": (lambda s, h: signed(s, h.nonce, reason="upstream-error", status=0, ctype="", body=b""),
                              ("upstream-error", None, None, None)),
    "signed-timeout": (lambda s, h: signed(s, h.nonce, reason="timeout", status=0, ctype="", body=b""),
                       ("timeout", None, None, None)),
    "signed-response-too-large": (lambda s, h: signed(s, h.nonce, reason="response-too-large", status=0, ctype="",
                                                      body=b""), ("response-too-large", None, None, None)),
    "unsigned-other-refusal": (_unsigned("run_unmatched"), ("upstream-error", None, None, None)),
    "unsigned-non-refusal": (lambda s, h: gw.build_response(None, request_nonce=h.nonce, close_reason="completed",
                                                            status=200, content_type="application/json", body=b"{}"),
                             ("upstream-error", None, None, None)),
    "unsigned-refusal-with-wrong-content-type": (_unsigned("bad_credential", "application/json"),
                                                 ("upstream-error", None, None, None)),
    "bad-mac": (_bad_mac, ("upstream-error", None, None, None)),
    "mac-over-another-nonce": (lambda s, h: signed(s, "someone-else"), ("upstream-error", None, None, None)),
    "close-reason-outside-the-set": (lambda s, h: _signed_custom(s, h, reason="weird"),
                                     ("upstream-error", None, None, None)),
    "completed-status-0": (lambda s, h: _signed_custom(s, h, status=0), ("upstream-error", None, None, None)),
    "completed-status-600": (lambda s, h: _signed_custom(s, h, status=600), ("upstream-error", None, None, None)),
    "status-is-a-string": (lambda s, h: _raw_header({"closeReason": "completed", "status": "200",
                                                     "contentType": "", "mac": ""}),
                           ("upstream-error", None, None, None)),
    "header-is-not-an-object": (lambda s, h: struct.pack(">II", 4 + 2 + 1, 2) + b"[]" + b"x",
                                ("upstream-error", None, None, None)),
    "duplicate-header-key": (lambda s, h: _frame_with_header(b'{"closeReason":"completed","closeReason":"timeout",'
                                                             b'"status":200,"contentType":"","mac":""}'),
                             ("upstream-error", None, None, None)),
    "header-length-over-frame": (lambda s, h: struct.pack(">II", 4, 900), ("upstream-error", None, None, None)),
    "short-frame": (lambda s, h: struct.pack(">I", 2) + b"ab", ("upstream-error", None, None, None)),
}


def _frame_with_header(head: bytes) -> bytes:
    payload = struct.pack(">I", len(head)) + head
    return struct.pack(">I", len(payload)) + payload


@pytest.mark.parametrize("case", list(CASES))
def test_each_row_of_the_reply_table_maps_to_its_exchange_and_close_record(tmp_path, vec, case):
    make, (reason, status, ctype, body) = CASES[case]
    h, secret = holder_double(tmp_path, vec, make)
    try:
        client = client_for(h, secret)
        ex = client(ENDPOINT, "POST", PATH, BODY, "eval-1")
        assert (ex.close_reason, ex.status, ex.content_type, ex.body) == (reason, status, ctype, body)
        # and through the broker: the `close` record is valid (read_records validates) and carries the same outcome
        (tmp_path / "b").mkdir()
        hh = Harness(tmp_path / "b", systems={"shop": load_system(tmp_path / "b", repos=(("a", "hosted"),))},
                     hosted_exchanger=client)
        g = hh.open("answer", ["a"])
        r = hh.ask(g["token"], hosted_call(codebases=["a"]), {"m": 1})
    finally:
        h.close()
    [close] = hh.records("close")
    assert close["closeReason"] == reason and close["httpStatus"] == status
    assert not hh.audit.latched
    assert (r["response"] is not None) == (reason == "completed")
    if reason == "completed":
        assert r["response"]["status"] == status


def test_an_oversized_reply_frame_is_response_too_large_before_the_body_is_read(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce, body=b"x" * 5000))
    try:
        ex = client_for(h, secret, max_reply_bytes=1000)(ENDPOINT, "POST", PATH, BODY, "eval-1")
    finally:
        h.close()
    assert ex.close_reason == "response-too-large" and (ex.status, ex.body) == (None, None)
    assert ex.bytes_in == 4 and ex.bytes_out > 0


def test_a_reply_exactly_at_the_cap_is_accepted(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce, body=b"y" * 100))
    try:
        size = len(signed(secret, "n" * 32, body=b"y" * 100)) - 4
        ok = client_for(h, secret, max_reply_bytes=size)(ENDPOINT, "POST", PATH, BODY, "eval-1")
        over = client_for(h, secret, max_reply_bytes=size - 1)(ENDPOINT, "POST", PATH, BODY, "eval-1")
    finally:
        h.close()
    assert (ok.close_reason, over.close_reason) == ("completed", "response-too-large")


def test_the_default_reply_cap_is_the_broker_response_cap_plus_slack(tmp_path):
    c = HolderClient(str(tmp_path / "x.sock"), UID, bytes(32))
    assert c.max_reply_bytes == 16 * 1024 * 1024 + 64 * 1024 and c.timeout == 300


def test_a_hang_past_the_deadline_is_a_timeout(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: None)
    try:
        t0 = time.monotonic()
        ex = client_for(h, secret, timeout=0.4)(ENDPOINT, "POST", PATH, BODY, "eval-1")
    finally:
        h.close()
    assert ex.close_reason == "timeout" and ex.bytes_out > 0 and ex.bytes_in == 0
    assert time.monotonic() - t0 < 5


def test_a_slow_trickle_is_cut_by_the_wall_clock_deadline_not_per_read(tmp_path, vec):
    secret = secret_of(vec)
    srv = ScriptedHolder(tmp_path / "run", None)  # connection is held open by the double; bytes are never sent
    try:
        ex = HolderClient(srv.path, UID, secret, timeout=0.3)(ENDPOINT, "POST", PATH, BODY, "eval-1")
    finally:
        srv.close()
    assert ex.close_reason == "timeout"


def test_a_closed_connection_with_no_reply_is_upstream_error(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: b"")
    try:
        ex = client_for(h, secret)(ENDPOINT, "POST", PATH, BODY, "eval-1")
    finally:
        h.close()
    assert ex.close_reason == "upstream-error" and ex.bytes_out > 0


def test_a_refused_connect_and_an_absent_socket_are_upstream_error_with_no_bytes(tmp_path, vec):
    secret = secret_of(vec)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    assert HolderClient(str(d / "none.sock"), UID, secret)(ENDPOINT, "POST", PATH, BODY, "x") == Exchange(
        "upstream-error", None, None, None, 0, 0)
    h = ScriptedHolder(tmp_path / "gone", None)
    h.close()  # the socket file stays, nothing listens: ECONNREFUSED
    assert HolderClient(h.path, UID, secret)(ENDPOINT, "POST", PATH, BODY, "x") == Exchange(
        "upstream-error", None, None, None, 0, 0)


def test_the_credential_never_appears_in_the_client_repr(vec):
    c = HolderClient("/x/y.sock", UID, secret_of(vec))
    assert vec["secret_hex"] not in repr(c) and "000102" not in repr(c)


# ------------------------------------------------------------------ the socket directory check (item 10)


def test_a_directory_owned_by_another_uid_never_connects(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce))
    try:
        ex = HolderClient(h.path, UID + 1, secret)(ENDPOINT, "POST", PATH, BODY, "eval-1")
        assert ex == Exchange("upstream-error", None, None, None, 0, 0) and h.connections == 0
        again = HolderClient(h.path, UID, secret)(ENDPOINT, "POST", PATH, BODY, "eval-1")  # the control: served
        assert again.close_reason == "completed" and h.connections == 1
    finally:
        h.close()


@pytest.mark.parametrize("mode", [0o720, 0o702, 0o770, 0o777])
def test_a_group_or_other_writable_directory_never_connects(tmp_path, vec, mode):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce))
    try:
        os.chmod(tmp_path / "run", mode)
        ex = client_for(h, secret)(ENDPOINT, "POST", PATH, BODY, "eval-1")
        assert ex.close_reason == "upstream-error" and h.connections == 0
    finally:
        os.chmod(tmp_path / "run", 0o700)
        h.close()


def test_a_symlinked_socket_directory_never_connects(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce))
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "run")
    try:
        ex = HolderClient(str(link / "holder.sock"), UID, secret)(ENDPOINT, "POST", PATH, BODY, "eval-1")
        assert ex.close_reason == "upstream-error" and h.connections == 0
    finally:
        h.close()


def test_the_directory_check_runs_before_every_connect(tmp_path, vec):
    h, secret = holder_double(tmp_path, vec, lambda s, hdr: signed(s, hdr.nonce))
    try:
        c = client_for(h, secret)
        assert c(ENDPOINT, "POST", PATH, BODY, "e").close_reason == "completed"
        os.chmod(tmp_path / "run", 0o777)
        assert c(ENDPOINT, "POST", PATH, BODY, "e").close_reason == "upstream-error"
        assert h.connections == 1
    finally:
        os.chmod(tmp_path / "run", 0o700)
        h.close()


# ------------------------------------------------------------------ import isolation (item 13)

BROKER_DIR = Path(holderclient.__file__).parent


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("__import__", "import_module"):
            names.add("<dynamic>")
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
            names.add("<dynamic>")
    return names


def test_no_module_under_ccg_broker_imports_llmgateway():
    files = sorted(BROKER_DIR.glob("*.py"))
    assert {f.name for f in files} >= {"holderclient.py", "server.py", "policy.py"}
    for f in files:
        bad = {n for n in _imports(f) if n.split(".")[0] == "llmgateway" or n == "<dynamic>"}
        assert not bad, (f.name, bad)
