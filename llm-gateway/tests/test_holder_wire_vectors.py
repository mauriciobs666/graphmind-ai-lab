"""The holder side of the shared wire vectors (egress-spec5 §7 item 11).

`holder_wire_vectors.json` holds literals generated once from `llmgateway.framing` and
`keyholder.canonical_origin`, then reviewed; it is never computed here. ccg's broker tests read the
same file. Every test fails (never skips) when the file is missing or unparsable.
`frame_hex` is the whole frame, the 4-byte length prefix included.
"""

import ast
import json
import pathlib

import pytest

from llmgateway import framing
from llmgateway.keyholder import canonical_origin

VECTORS = pathlib.Path(__file__).resolve().parent / "holder_wire_vectors.json"
CCG_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "cpg" / "ccg" / "manifest.py"


def load():
    data = json.loads(VECTORS.read_text())  # a missing or broken file raises: the test fails
    assert data["version"] == 1
    return data


DATA = load()
SECRET = bytes.fromhex(DATA["secret_hex"])


def test_the_vectors_cover_what_the_spec_requires():
    assert len(SECRET) == 32
    reqs = {r["name"]: r for r in DATA["requests"]}
    assert any("|" in r["run_id"] for r in reqs.values())
    assert any(r["run_id"] == "" for r in reqs.values())
    assert all(r["body_hex"] for r in reqs.values())
    reps = {r["name"]: r for r in DATA["replies"]}
    assert {r["status"] for r in reps.values()} == {200, 499}
    assert any(r["mac"] == "" for r in reps.values())
    assert any(r["canonical"] is None for r in DATA["origins"])
    assert any(r["input"].isupper() or r["input"].startswith("HTTPS") for r in DATA["origins"])
    assert any("[" in r["input"] for r in DATA["origins"])


@pytest.mark.parametrize("v", DATA["requests"], ids=lambda v: v["name"])
def test_request_mac_and_frame_match_the_literals(v):
    body = bytes.fromhex(v["body_hex"])
    mac = framing.request_mac(
        SECRET,
        caller=v["caller"],
        ts=v["ts"],
        nonce=v["nonce"],
        method=v["method"],
        endpoint=v["endpoint"],
        path=v["path"],
        run_id=v["run_id"],
        body=body,
    )
    assert mac == v["mac"]
    frame, nonce = framing.build_request(
        SECRET,
        caller=v["caller"],
        method=v["method"],
        endpoint=v["endpoint"],
        path=v["path"],
        run_id=v["run_id"],
        body=body,
        ts=v["ts"],
        nonce=v["nonce"],
    )
    assert nonce == v["nonce"] and frame.hex() == v["frame_hex"]
    hdr, parsed_body = framing.parse_request(bytes.fromhex(v["frame_hex"])[4:])
    assert (hdr.caller, hdr.ts, hdr.nonce, hdr.mac) == (v["caller"], v["ts"], v["nonce"], v["mac"])
    assert (hdr.method, hdr.endpoint, hdr.path, hdr.run_id) == (
        v["method"],
        v["endpoint"],
        v["path"],
        v["run_id"],
    )
    assert parsed_body == body


@pytest.mark.parametrize("v", DATA["replies"], ids=lambda v: v["name"])
def test_reply_frames_and_macs_match_the_literals(v):
    body = bytes.fromhex(v["body_hex"])
    signed = v["mac"] != ""
    frame = framing.build_response(
        SECRET if signed else None,
        request_nonce=v["request_nonce"],
        close_reason=v["close_reason"],
        status=v["status"],
        content_type=v["content_type"],
        body=body,
    )
    assert frame.hex() == v["frame_hex"]
    if signed:
        assert (
            framing.response_mac(
                SECRET,
                request_nonce=v["request_nonce"],
                close_reason=v["close_reason"],
                status=v["status"],
                content_type=v["content_type"],
                body=body,
            )
            == v["mac"]
        )
    reply = framing.parse_reply(
        bytes.fromhex(v["frame_hex"])[4:], SECRET, v["request_nonce"], require_mac=False
    )
    assert reply.authenticated is signed
    assert (reply.status, reply.content_type, reply.body) == (v["status"], v["content_type"], body)


@pytest.mark.parametrize("v", DATA["origins"], ids=lambda v: repr(v["input"]))
def test_holder_canonical_origin_matches_the_literals(v):
    if v["canonical"] is None:
        with pytest.raises(ValueError):
            canonical_origin(v["input"])
    else:
        assert canonical_origin(v["input"]) == v["canonical"]


@pytest.mark.parametrize("v", DATA["origins"], ids=lambda v: repr(v["input"]))
def test_broker_canonical_endpoint_matches_the_same_literals(v):
    """The broker's function, extracted by symbol from the read-only ccg source."""
    tree = ast.parse(CCG_MANIFEST.read_text())
    wanted = {"_bad_char", "canonical_endpoint", "ManifestError"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in wanted]
    nodes += [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_PORTS"
    ]
    ns: dict = {}
    exec(  # noqa: S102 - this repo's own source
        "import ipaddress\nfrom urllib.parse import urlsplit\n"
        + "\n".join(ast.unparse(n) for n in nodes),
        ns,
    )
    if v["canonical"] is None:
        with pytest.raises(ns["ManifestError"]):
            ns["canonical_endpoint"](v["input"])
    else:
        assert ns["canonical_endpoint"](v["input"]) == v["canonical"]
