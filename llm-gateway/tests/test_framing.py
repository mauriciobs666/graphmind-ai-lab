"""Framing and MAC (plan §3.4 Authentication; §5.2 FR-8 (5))."""

import hashlib
import socket
import threading
import time

import pytest

from llmgateway import framing as fr

SECRET = bytes(range(32))
BASE = dict(
    caller="broker",
    ts=1700000000,
    nonce="n1",
    method="POST",
    endpoint="https://api.example",
    path="/v1/messages",
    run_id="eval-1",
    body=b'{"a":1}',
)


def mac(**over):
    return fr.request_mac(SECRET, **{**BASE, **over})


def test_mac_is_stable_and_binds_every_field():
    assert mac() == mac()
    for field, other in [
        ("caller", "service"),
        ("ts", 1700000001),
        ("nonce", "n2"),
        ("method", "GET"),
        ("endpoint", "https://other.example"),
        ("path", "/v1/other"),
        ("run_id", "eval-2"),
        ("body", b'{"a":2}'),
    ]:
        assert mac(**{field: other}) != mac(), field


def test_length_prefixing_defeats_a_delimiter_shift():
    # With a `|` join, run_id="x|y" body-digest-less shifts would collide; length prefixes do not.
    assert mac(run_id="a|b", path="/p") != mac(run_id="a", path="b|/p")
    assert mac(run_id="a|", nonce="n") != mac(run_id="a", nonce="|n")
    assert mac(path="/v1/messages", run_id="") != mac(path="/v1/messag", run_id="es")


def test_mac_matches_an_independent_computation():
    import hmac

    def f(x):
        raw = x if isinstance(x, bytes) else x.encode()
        return len(raw).to_bytes(4, "big") + raw

    msg = b"".join(
        f(v)
        for v in (
            "broker",
            "1700000000",
            "n1",
            "POST",
            "https://api.example",
            "/v1/messages",
            "eval-1",
            hashlib.sha256(b'{"a":1}').digest(),
        )
    )
    assert mac() == hmac.new(SECRET, msg, hashlib.sha256).hexdigest()


def test_request_roundtrip_and_fresh_nonce_per_call():
    kw = dict(BASE)
    kw.pop("ts"), kw.pop("nonce")
    frame1, n1 = fr.build_request(SECRET, **kw)
    frame2, n2 = fr.build_request(SECRET, **kw)
    assert n1 != n2 and frame1 != frame2  # a retry signs a fresh nonce
    hdr, body = fr.parse_request(frame1[4:])
    assert body == BASE["body"] and hdr.caller == "broker" and hdr.nonce == n1
    assert hdr.mac == fr.request_mac(
        SECRET,
        caller=hdr.caller,
        ts=hdr.ts,
        nonce=hdr.nonce,
        method=hdr.method,
        endpoint=hdr.endpoint,
        path=hdr.path,
        run_id=hdr.run_id,
        body=body,
    )


@pytest.mark.parametrize(
    "payload",
    [b"", b"\x00\x00", b"\x00\x00\x00\x09ab", b"\x00\x00\x00\x02[]", b"\x00\x00\x00\x02{}"],
)
def test_malformed_payloads_are_frame_errors(payload):
    with pytest.raises(fr.FrameError):
        fr.parse_request(payload)


def test_request_header_types_are_checked():
    good = {
        "caller": "c",
        "ts": 1,
        "nonce": "n",
        "mac": "m",
        "method": "POST",
        "endpoint": "e",
        "path": "/",
        "run_id": "",
    }
    for bad in (
        {**good, "ts": "1"},
        {**good, "ts": True},
        {**good, "ts": 1.5},
        {**good, "nonce": ""},
        {**good, "nonce": "x" * 129},
        {**good, "caller": 3},
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "mac"},
    ):
        with pytest.raises(fr.FrameError):
            fr.parse_request(fr.encode_frame(bad, b"")[4:])
    fr.parse_request(fr.encode_frame(good, b"")[4:])
    assert fr.parse_request(fr.encode_frame({**good, "run_id": None}, b"")[4:])[0].run_id == ""


def test_duplicate_header_key_is_refused():
    head = b'{"caller":"a","caller":"b"}'
    payload = len(head).to_bytes(4, "big") + head
    with pytest.raises(fr.FrameError):
        fr.parse_request(payload)


@pytest.mark.parametrize(
    "text",
    ['{"a":1,"a":2}', '{"x":{"a":1,"a":2}}', '[{"a":1,"a":1}]', '{"a":NaN}', '{"a":Infinity}'],
)
def test_strict_loads_refuses(text):
    with pytest.raises(ValueError):
        fr.strict_loads(text)


def test_strict_loads_refuses_bad_utf8_and_deep_nesting():
    with pytest.raises(ValueError):
        fr.strict_loads(b'{"a":"\xff"}')
    with pytest.raises(ValueError):
        fr.strict_loads("[" * 100_000 + "]" * 100_000)
    assert fr.strict_loads('{"a":{"b":[1,2]}}') == {"a": {"b": [1, 2]}}


def _pair():
    return socket.socketpair()


def test_read_frame_over_the_cap_is_refused_before_reading_it():
    a, b = _pair()
    a.sendall((10_000_000).to_bytes(4, "big"))  # declares 10 MB, sends nothing more
    with pytest.raises(fr.FrameError, match="cap"):
        fr.read_frame(b, max_bytes=1000, timeout=2)


def test_read_frame_slow_peer_hits_the_total_deadline():
    a, b = _pair()
    a.sendall((100).to_bytes(4, "big") + b"x")

    def drip():
        for _ in range(20):
            time.sleep(0.1)
            try:
                a.sendall(b"y")
            except OSError:
                return

    threading.Thread(target=drip, daemon=True).start()
    start = time.monotonic()
    with pytest.raises(fr.FrameError, match="timeout"):
        fr.read_frame(b, max_bytes=1000, timeout=0.5)
    assert time.monotonic() - start < 1.5  # the deadline is total, not per recv


def test_read_frame_early_close_and_success():
    a, b = _pair()
    a.sendall((5).to_bytes(4, "big") + b"ab")
    a.close()
    with pytest.raises(fr.FrameError):
        fr.read_frame(b, max_bytes=100, timeout=1)
    c, d = _pair()
    c.sendall(fr.encode_frame({"k": 1}, b"body"))
    assert fr.decode_payload(fr.read_frame(d, max_bytes=1000, timeout=1)) == ({"k": 1}, b"body")


def test_response_mac_binds_the_request_nonce_and_body():
    frame = fr.build_response(
        SECRET,
        request_nonce="n1",
        close_reason="completed",
        status=200,
        content_type="application/json",
        body=b"{}",
    )
    reply = fr.parse_reply(frame[4:], SECRET, "n1")
    assert reply.authenticated and reply.status == 200 and reply.body == b"{}"
    with pytest.raises(fr.ReplyRejected):
        fr.parse_reply(frame[4:], SECRET, "other-nonce")  # replayed reply for another request
    with pytest.raises(fr.ReplyRejected):
        fr.parse_reply(frame[4:], bytes(32), "n1")  # a holder without the secret
    tampered = frame[:-1] + b"X"
    with pytest.raises(fr.ReplyRejected):
        fr.parse_reply(tampered[4:], SECRET, "n1")


def test_unsigned_reply_needs_explicit_allowance():
    frame = fr.build_response(
        None,
        request_nonce="n",
        close_reason="completed",
        status=499,
        content_type=fr.REFUSAL_CONTENT_TYPE,
        body=fr.refusal_body("bad_credential"),
    )
    with pytest.raises(fr.ReplyRejected):
        fr.parse_reply(frame[4:], SECRET, "n")
    reply = fr.parse_reply(frame[4:], SECRET, "n", require_mac=False)
    assert not reply.authenticated and reply.refusal_code == "bad_credential"


def test_socket_dir_must_be_owned_by_the_holder_and_not_writable(tmp_path):
    import os

    d = tmp_path / "run"
    d.mkdir()
    d.chmod(0o711)
    sock = str(d / "holder.sock")
    fr.check_socket_dir(sock, os.getuid())
    with pytest.raises(fr.HolderUnreachable):
        fr.check_socket_dir(sock, os.getuid() + 1)  # fake holder: wrong owner
    d.chmod(0o777)
    with pytest.raises(fr.HolderUnreachable):
        fr.check_socket_dir(sock, os.getuid())  # anyone could swap the socket
    with pytest.raises(fr.HolderUnreachable):
        fr.check_socket_dir(str(tmp_path / "missing" / "s"), os.getuid())


def test_load_secret_refuses_symlink_and_wrong_length(tmp_path):
    good = tmp_path / "c"
    good.write_text(SECRET.hex() + "\n")
    assert fr.load_secret(str(good)) == SECRET
    link = tmp_path / "l"
    link.symlink_to(good)
    with pytest.raises(OSError):
        fr.load_secret(str(link))
    short = tmp_path / "s"
    short.write_text("abcd")
    with pytest.raises(ValueError):
        fr.load_secret(str(short))
