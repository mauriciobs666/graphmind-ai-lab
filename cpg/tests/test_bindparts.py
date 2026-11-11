"""P1-14 — `bindParts` (`-bundle-spec5.md`, F-7): the URL parse, the keyed parts, the validator rules V1-V6, the writer rules
E1-E4/W6/W7, the key epoch and the linker's comparison primitives.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-14. Every expectation is parsed from the text of spec5 §9
(never copied here): the `entry_hmac` vectors (§9.1), the writer vectors BP-n (§9.2), the validator vectors BK-n (§9.3), the
key-epoch vectors BR-n (§9.4) and the linker vectors BL-n (§9.5). Both published test keys are constants that
`ccg.atrest.hmac_key` refuses, so they are handed to the pure functions directly; no test reads a key file.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import hmac
import json
import re
import unicodedata
from pathlib import Path

import pytest

from ccg import atrest, bundle, ids, valuehash
from ccg.valuehash import entry_hmac
from extractors.common import writer_rules as wr
from test_bundle import ordered, rich_bundle

CPG = Path(__file__).resolve().parent.parent
SPEC5 = (CPG.parent / "docs" / "plans" / "connected-code-graph-bundle-spec5.md").read_text(encoding="utf-8")

KEY1 = bytes.fromhex(re.search(r"32-byte key `([0-9a-f]{64})`", SPEC5).group(1))
KEY2 = bytes.fromhex(re.search(r"second key,\s+`([0-9a-f]{64})`", SPEC5).group(1))
DECL = frozenset({"falkordb", "server", "api"})          # §9.2 "Inputs"

HOST_TAG, PATH_TAG, KEYCHECK_TAG = "ccg.bindParts.host.v1", "ccg.bindParts.path.v1", "ccg.bindParts.keycheck.v1"

_EH_ROW = re.compile(r"^\| (EH-\d+) \| `([^`]+)` \| (?:`([^`]*)`|\(empty\)) \| `([0-9a-f]{64})` \|$", re.M)
EH = {m.group(1): (m.group(2), m.group(3) or "", m.group(4)) for m in _EH_ROW.finditer(SPEC5)}


def ref(key: bytes, domain: str, value: str) -> str:
    """§2's construction recomputed with the stdlib, independent of `ccg.valuehash`."""
    return hmac.new(key, domain.encode() + b"\x00" + unicodedata.normalize("NFC", value).encode(), hashlib.sha256).hexdigest()


# --- the tags and §9.1 -------------------------------------------------------------------------

def test_the_three_tags_are_spec5_section_2s():
    assert (valuehash.BIND_HOST_TAG, valuehash.BIND_PATH_TAG, valuehash.BIND_KEYCHECK_TAG) == (HOST_TAG, PATH_TAG, KEYCHECK_TAG)
    assert valuehash.BIND_KEYCHECK_TAG is atrest.BIND_KEYCHECK_TAG        # one definition of the key-check tag


def test_the_spec_rows_are_parsed():
    assert len(EH) == 32 and EH["EH-30"][1] == "" and KEY2 == bytes(range(32, 64))


# --- §9.2 writer vectors: parse_bind_url, bindValue, bindParts ------------------------------------

_BP_ROW = re.compile(r"^\| (BP-\d+) \| `([^`]*)` \| `([^`]*)` \| (bind|absent) \| `([^`]*)` \| (.*) \|$", re.M)
BP = [(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)) for m in _BP_ROW.finditer(SPEC5)]


def expected_parts(cell: str):
    if cell == "none":
        return None
    out: dict = {}
    for tok in (t.strip() for t in cell.split(";")):
        k, _, v = tok.partition("=")
        if k == "port":
            out[k] = int(v)
        elif k == "hostHmac":
            out[k] = EH[v][2]
        elif k == "pathHmacs":
            out[k] = "#32" if v == "#32" else [EH[x][2] for x in v.split(",")] if v else []
        else:
            out[k] = v
    return out


def _check_32(parts: dict, host: str) -> dict:
    """`pathHmacs=#32`: 32 items, each per §5.3 (the hash of the prefix under its host), computed here independently."""
    h = host if parts.get("host") is None else parts["host"]
    want = [ref(KEY1, PATH_TAG, h + "/" + "/".join(["a"] * i)) for i in range(1, 33)]
    assert parts["pathHmacs"] == want
    return {**parts, "pathHmacs": "#32"}


def test_nine_two_has_the_forty_seven_rows():
    assert [r[0] for r in BP] == [f"BP-{i}" for i in range(1, 48)]


@pytest.mark.parametrize("row", BP, ids=[r[0] for r in BP])
def test_writer_vectors_of_section_9_2(row):
    name, key_path, value, bv, cell, decided = row
    role = name != "BP-37"                                           # §9.2: every row but BP-37 is in a binding role
    got_bv = wr.bind_fields(key_path, value, binding_role=role, declared_services=DECL)
    assert got_bv == ({"bindValue": value} if bv == "bind" else {}), name
    got = wr.bind_parts_fields(KEY1, key_path, value, binding_role=role, declared_services=DECL)
    want = expected_parts(cell)
    if want is None:
        assert got == {}, name
        return
    assert set(got) == {"bindParts"}
    parts = got["bindParts"]
    if want["pathHmacs"] == "#32":
        parts = _check_32(parts, "localhost")
    assert parts == want, name
    assert value not in json.dumps(got)                                       # the URL text is never stored as such


@pytest.mark.parametrize("row", BP, ids=[r[0] for r in BP])
def test_the_url_parse_is_none_exactly_where_section_3_says(row):
    name, _kp, value, _bv, cell, decided = row
    got = bundle.parse_bind_url(value)
    if cell == "none" and decided.startswith("§3"):
        assert got is None, name
    else:
        assert got is not None, name            # parts rows, and the rows only a writer rule (E2-E4) refuses


def test_parse_bind_url_returns_the_named_tuple_of_section_8():
    got = bundle.parse_bind_url("http://localhost:8000/api/v1")
    assert tuple(got) == ("http", 8000, "localhost", "loopback", ("api", "v1"))
    assert (got.scheme, got.port, got.host, got.host_class, got.segments) == tuple(got)
    assert bundle.parse_bind_url("https://ntfy.sh/") == ("https", None, "ntfy.sh", "dnsname", ())
    assert bundle.parse_bind_url("redis://api") == ("redis", None, "api", "label", ())
    assert bundle.parse_bind_url("http://[2001:db8::5]:8080/v1/") == ("http", 8080, "[2001:db8::5]", "ip", ("v1",))
    assert bundle.parse_bind_url("http://127.0.0.2/") == ("http", None, "127.0.0.2", "loopback", ())
    assert bundle.parse_bind_url("http://10.0.0.5/") == ("http", None, "10.0.0.5", "ip", ())


@pytest.mark.parametrize("value", [
    "http://localhost:8000/" + "a" * 600, "http://[::1]:80x/", "http://[::1", "http://[]/", "http://[1:2:3]/",
    "http://:80/", "http://localhost:/", "http://localhost:65536/", "http:///x", "://x", "h ttp://x/", "http://x /",
    "http://a_b/", "http://" + "a" * 64 + "/", "http://" + ".".join(["a" * 60] * 5) + "/", "a" * 17 + "://x/",
    "x7k2q9w4m1p8z3v6://x/", "http://localhost:8000/a/./b", "http://localhost/é", "http://locé/", "http://[::]/",
    "http://[::1]x/", "http://[fe80::1]x:80/", "http://x/a b", "http://x:80:80/", "http://localhost:80/\x00",
    "ab\ud800://x/", "http://x\ud800/",
])
def test_the_url_parse_refuses_hostile_shapes_without_raising(value):
    assert bundle.parse_bind_url(value) is None


def test_the_url_parse_lowercases_a_bracketed_host_and_bind_parts_carries_the_lowercase_scheme():
    assert bundle.parse_bind_url("http://[2001:DB8::5]:80/") == ("http", 80, "[2001:db8::5]", "ip", ())
    assert valuehash.bind_parts(KEY1, "HTTPS://Api:8000/V1", DECL)["scheme"] == "https"


def test_the_length_bound_is_511_and_the_dns_name_bound_is_253():
    head = "http://localhost/"
    assert bundle.parse_bind_url(head + "a" * (511 - len(head))) is not None
    assert bundle.parse_bind_url(head + "a" * (512 - len(head))) is None
    name = lambda last: ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * last])             # noqa: E731
    assert len(name(61)) == 253 and bundle.parse_bind_url(f"http://{name(61)}/").host_class == "dnsname"
    assert bundle.parse_bind_url(f"http://{name(62)}/") is None


@pytest.mark.parametrize("field", ["scheme", "hostKind", "hostHmac", "pathHmacs"])
def test_v4b_each_required_field_is_required_and_the_failure_is_a_bundle_error(field):
    recs, _ = mutate(base(), f"bindParts.{field} = <absent>")
    with pytest.raises(bundle.BundleError, match="V4b"):
        bundle.validate(recs)


def test_the_url_parse_keeps_path_case_and_lowercases_scheme_and_host():
    assert bundle.parse_bind_url("HTTP://LocalHost:80/A/b") == ("http", 80, "localhost", "loopback", ("A", "b"))


# --- bind_parts: the keyed object (§5.3) -----------------------------------------------------------

def test_bind_parts_is_a_function_of_value_services_and_key_only():
    a = valuehash.bind_parts(KEY1, "http://api:8000/x", {"api"})
    assert a == valuehash.bind_parts(KEY1, "http://api:8000/x", ["API"])         # services compare lowercased
    assert a["hostKind"] == "service" and a["host"] == "api"
    assert valuehash.bind_parts(KEY1, "http://api:8000/x", set())["hostKind"] == "external"
    assert valuehash.bind_parts(KEY2, "http://api:8000/x", {"api"}) != a
    assert valuehash.bind_parts(KEY1, "http://localhost", set()) == valuehash.bind_parts(KEY1, "http://localhost/", set())


def test_bind_parts_returns_none_where_the_parse_does():
    assert valuehash.bind_parts(KEY1, "postgres://u:p@db/app", DECL) is None
    assert valuehash.bind_parts(KEY1, "ab\ud800", DECL) is None


def test_a_declared_label_without_a_letter_or_failing_the_screen_is_external():
    assert valuehash.bind_parts(KEY1, "http://1-2:80/", {"1-2"})["hostKind"] == "external"           # no letter (V5h)
    long = "x7k2q9w4m1p8z3v6a"
    assert valuehash.bind_parts(KEY1, f"http://{long}:80/", {long}) ["hostKind"] == "external"       # T1 (V5g)
    assert valuehash.bind_parts(KEY1, "http://api:80/", {"api"})["hostKind"] == "service"


def test_the_path_hash_is_host_scoped_and_the_host_hash_is_not_path_scoped():
    a = valuehash.bind_parts(KEY1, "http://localhost/mcp", set())
    b = valuehash.bind_parts(KEY1, "http://host.docker.internal/mcp", set())
    assert a["hostHmac"] != b["hostHmac"] and a["pathHmacs"] != b["pathHmacs"]
    assert a["hostHmac"] == ref(KEY1, HOST_TAG, "localhost") and a["pathHmacs"] == [ref(KEY1, PATH_TAG, "localhost/mcp")]
    assert a["hostHmac"] != ref(KEY1, PATH_TAG, "localhost")          # the two tags separate host from path


def test_the_hash_of_a_prefix_is_not_the_hash_of_one_segment():
    got = valuehash.bind_parts(KEY1, "http://localhost/a/b/c", set())["pathHmacs"]
    assert got == [ref(KEY1, PATH_TAG, p) for p in ("localhost/a", "localhost/a/b", "localhost/a/b/c")]
    assert ref(KEY1, PATH_TAG, "localhost/b") not in got


@pytest.mark.parametrize("value", [
    "http://169.254.169.254/latest", "http://169.254.169.254:80/", "http://[fe80::1]:8000/", "http://[fd00:ec2::254]/latest",
    "http://100.100.100.200/", "http://[fec0::1]/", "http://[100::1]/", "https://169.254.0.1/x",
    "http://168.63.129.16/", "http://192.0.0.192:80/x", "http://[2002::1]/", "http://[2002:a00:1::1]:8080/v1",
])
def test_a_link_local_or_metadata_host_is_external_with_no_clear_host(value):
    """The plan's P1-14 rule: hostKind `external`, hash only (and `bindValue` is withheld by `writer_rules`)."""
    parts = wr.bind_parts_fields(KEY1, "x.url", value, binding_role=True, declared_services=DECL)["bindParts"]
    assert parts["hostKind"] == "external" and "host" not in parts
    h = bundle.parse_bind_url(value).host
    assert parts["hostHmac"] == ref(KEY1, HOST_TAG, h)                  # still hashed: one comparison serves all four kinds
    assert wr.bind_fields("x.url", value, binding_role=True, declared_services=DECL) == {}
    rec = json.dumps(parts)
    assert h not in rec and h.strip("[]") not in rec


def test_a_link_local_host_is_external_even_if_the_caller_declares_it_a_service():
    parts = wr.bind_parts_fields(KEY1, "x.url", "http://0xa9fea9fe/", binding_role=True,
                                 declared_services={"0xa9fea9fe"})["bindParts"]
    assert parts["hostKind"] == "external" and "host" not in parts


@pytest.mark.parametrize("value", ["http://10.0.0.5:8080/api", "http://[2001:db8::5]/", "http://100.100.100.201/", "http://169.253.0.1/",
                                   "http://[fe7f::1]/", "http://100.64.0.1/", "http://168.63.129.17/",
                                   "http://192.0.0.193/", "http://[2003::1]/", "http://10.0.0.5/", "http://172.16.0.1/", "http://192.168.1.1/"])
def test_a_neighbouring_address_stays_a_clear_ip(value):
    parts = wr.bind_parts_fields(KEY1, "x.url", value, binding_role=True, declared_services=DECL)["bindParts"]
    assert parts["hostKind"] == "ip" and parts["host"] == bundle.parse_bind_url(value).host


# --- mount_prefix_hmac (§7) ---------------------------------------------------------------------------

@pytest.mark.parametrize("mount,want", [
    ("/", (0, None)), ("/mcp", (1, "localhost/mcp")), ("/mcp/", (1, "localhost/mcp")), ("/api/v1/", (2, "localhost/api/v1")),
    ("/MCP", (1, "localhost/MCP")),
])
def test_mount_prefix_hmac_normalises_like_section_3_step_4(mount, want):
    d, text = want
    assert valuehash.mount_prefix_hmac(KEY1, "localhost", mount) == (d, None if text is None else ref(KEY1, PATH_TAG, text))


@pytest.mark.parametrize("mount", ["mcp", "", "//", "/a//b", "/a/../b", "/a/./b", "/mcp;v=1", "/a b", "/é", "/a%2Fb", "/" + "/".join(["a"] * 33)])
def test_a_mount_outside_the_segment_class_matches_no_consumer(mount):
    assert valuehash.mount_prefix_hmac(KEY1, "localhost", mount) is None


def test_a_mount_of_thirty_two_segments_is_the_limit():
    assert valuehash.mount_prefix_hmac(KEY1, "localhost", "/" + "/".join(["a"] * 32))[0] == 32


# --- §9.4 key epoch --------------------------------------------------------------------------------------

_BR_ROW = re.compile(r"^\| (BR-\d+) \| (?:`)?([0-9a-f]{64}|EH-30)(?:`)? \| `?(K[12])`?[^|]* \| (true|false) \|", re.M)
BR = [(m.group(1), EH["EH-30"][2] if m.group(2) == "EH-30" else m.group(2), m.group(3), m.group(4) == "true")
      for m in _BR_ROW.finditer(SPEC5)]


def test_the_key_check_vectors_are_the_specs():
    assert [r[0] for r in BR] == ["BR-1", "BR-2", "BR-3"]
    assert atrest.bind_key_check(KEY1) == EH["EH-30"][2]
    assert atrest.bind_key_check(KEY2) == BR[1][1]


@pytest.mark.parametrize("name,check,which,usable", BR, ids=[r[0] for r in BR])
def test_bind_parts_usable_follows_the_key_epoch(name, check, which, usable):
    key = {"K1": KEY1, "K2": KEY2}[which]
    assert valuehash.bind_parts_usable({"bindKeyCheck": check}, key) is usable


def test_bind_parts_usable_is_false_without_a_check_and_never_raises_on_a_hostile_header():
    for h in ({}, {"bindKeyCheck": None}, {"bindKeyCheck": 5}, {"bindKeyCheck": "x"}, {"bindKeyCheck": EH["EH-30"][2].upper()}):
        assert valuehash.bind_parts_usable(h, KEY1) is False
    assert valuehash.bind_parts_usable({"bindKeyCheck": EH["EH-30"][2]}, KEY1) is True


def test_a_bundle_under_the_other_key_loads_but_its_parts_are_not_usable():
    recs = base()
    recs[0]["bindKeyCheck"] = BR[1][1]
    bundle.validate(recs)                                  # §5.4: not refused
    assert valuehash.bind_parts_usable(recs[0], KEY1) is False and valuehash.bind_parts_usable(base()[0], KEY1) is True


# --- §9.5 linker primitives ------------------------------------------------------------------------------------

_BL_ROW = re.compile(r"^\| (BL-\d+) \| (BP-\d+) \| `([^`]+)` \| (match|no match) \|", re.M)
BL = [(m.group(1), m.group(2), m.group(3), m.group(4) == "match") for m in _BL_ROW.finditer(SPEC5)]


def _consumer(bp: str) -> dict:
    row = next(r for r in BP if r[0] == bp)
    parts = expected_parts(row[4])
    return parts


def link_matches(consumer: dict, provider: str) -> bool:
    """§7 as a consumer of `valuehash` only: the decided host, then the prefix comparison."""
    mount, service = None, None
    for tok in (t.strip() for t in provider.split(";")):
        k, _, v = tok.partition("=")
        if k == "mount":
            mount = v
        else:
            service = v
    if service is not None:
        if consumer["hostHmac"] != entry_hmac(KEY1, HOST_TAG, service):
            return False
        host = service
    else:
        host = consumer["host"]
    if mount is None:
        return True
    got = valuehash.mount_prefix_hmac(KEY1, host, mount)
    if got is None:
        return False
    d, h = got
    return d == 0 or (len(consumer["pathHmacs"]) >= d and consumer["pathHmacs"][d - 1] == h)


def test_nine_five_has_the_seventeen_rows():
    assert [r[0] for r in BL] == [f"BL-{i}" for i in range(1, 18)]


@pytest.mark.parametrize("name,bp,provider,want", [r for r in BL if r[0] != "BL-16"], ids=[r[0] for r in BL if r[0] != "BL-16"])
def test_linker_vectors_of_section_9_5(name, bp, provider, want):
    assert link_matches(_consumer(bp), provider) is want


def test_bl_16_the_hash_would_equal_so_the_dotted_name_filter_is_the_linkers():
    """BL-16's outcome (no match) is P1-17's label filter, not a hash property: the primitives alone WOULD match."""
    assert link_matches(_consumer("BP-6"), "service=ntfy.sh") is True


# --- §9.3 validator vectors ---------------------------------------------------------------------------------------

BASE_KEYPATH = "mcpServers.falkor-chat-agent-team.url"


def base() -> list[dict]:
    """§9.3 'Base bundle': `rich_bundle()` with a 1.1 header, `bindKeyCheck` = EH-30 and the configEntry of BP-1."""
    recs = copy.deepcopy(rich_bundle())
    recs[0].update(bundleSchemaVersion="1.1", bindKeyCheck=EH["EH-30"][2])
    ce = next(r for r in recs if r["type"] == "configEntry")
    ce.update(keyPath=BASE_KEYPATH, bindValue="http://localhost:8200/mcp")
    ce["id"] = ids.config_entry_id(ce["repoId"], ce["path"], ce["keyPath"])
    ce["bindParts"] = expected_parts(next(r for r in BP if r[0] == "BP-1")[4])
    return ordered(recs)


_BK_ROW = re.compile(r"^\| (BK-\d+) \| (?:`(.*)`|\(none\)) \| (valid|reject) \| (.*) \|$", re.M)
BK = [(m.group(1), m.group(2) or "", m.group(3), m.group(4)) for m in _BK_ROW.finditer(SPEC5)]
_ABSENT = object()


def _value(text: str):
    if text == "<absent>":
        return _ABSENT
    if m := re.fullmatch(r"(\d+) × (EH-\d+)", text):
        return [EH[m.group(2)][2]] * int(m.group(1))
    if re.fullmatch(r"EH-\d+", text):
        return EH[text][2]
    return json.loads(text)


def mutate(recs: list[dict], mutation: str) -> tuple[list[dict], list[str]]:
    """Apply one §9.3 mutation cell; also return the string values it set (the §6.3 leak probe)."""
    recs = copy.deepcopy(recs)
    ce = next(r for r in recs if r["type"] == "configEntry")
    targets = {"header": recs[0], "configEntry": ce, "bindParts": ce.get("bindParts")}
    strings: list[str] = []
    for part in re.split(r";\s+(?=(?:header|configEntry|bindParts)\.)", mutation) if mutation else []:
        lhs, _, rhs = part.partition(" = ")
        target, _, field = lhs.partition(".")
        value = _value(rhs)
        if value is _ABSENT:
            targets[target].pop(field, None)
        else:
            targets[target][field] = value
            strings += [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
        if target == "configEntry" and field == "keyPath":
            ce["id"] = ids.config_entry_id(ce["repoId"], ce["path"], ce["keyPath"])
    return ordered(recs), strings


def test_nine_three_has_the_fifty_two_rows_and_every_validator_id_has_a_reject_row():
    assert [r[0] for r in BK] == [f"BK-{i}" for i in range(1, 53)]
    table = SPEC5[SPEC5.index("### 6.4 Validator checks"):SPEC5.index("## 7. The linker")]
    ids_in_spec = set(re.findall(r"^\| (V\d[a-z]?) \|", table, re.M))
    rejecting = {re.match(r"(V\d[a-z]?)", r[3]).group(1) for r in BK if r[2] == "reject"}
    assert len(ids_in_spec) == 30 and ids_in_spec == rejecting


@pytest.mark.parametrize("name,mutation,outcome,rule", BK, ids=[r[0] for r in BK])
def test_validator_vectors_of_section_9_3(name, mutation, outcome, rule):
    recs, strings = mutate(base(), mutation)
    if outcome == "valid":
        bundle.validate(recs)
        return
    with pytest.raises(bundle.BundleError) as e:
        bundle.validate(recs)
    msg = str(e.value)
    want = re.match(r"(V\d[a-z]?)", rule).group(1)
    assert want in msg, (name, msg)                                # the message names the rule id (§6.3) ...
    assert "configEntry" in msg or want.startswith("V1"), msg
    for s in strings:                                              # ... and never a value (§6.3)
        assert len(s) < 6 or s not in msg, (name, msg)
    assert not re.search(r"[0-9a-f]{64}", msg), msg


def test_the_base_bundle_and_the_unmodified_rich_bundle_are_valid():
    bundle.validate(base())
    bundle.validate(rich_bundle())                                   # positive control: a "1.0" bundle without the field
    assert rich_bundle()[0]["bundleSchemaVersion"] == "1.0"
    assert bundle.loads(b"".join(bundle.canonical_line(r) for r in base())) == base()


def test_a_message_for_a_bad_bind_value_never_carries_the_value():
    """BV-12's shape: a value that passes the schema and fails the screen, through `_Semantics._configEntry`'s message."""
    secret = "https://abcdefghij0123456789klmnopqrstuv.lambda-url.us-east-1.on.aws"
    recs, _ = mutate(base(), "")
    next(r for r in recs if r["type"] == "configEntry").update(bindValue=secret)
    with pytest.raises(bundle.BundleError) as e:
        bundle.validate(recs)
    assert "abcdefghij0123456789" not in str(e.value) and "bindValue" in str(e.value)


def test_a_schema_message_for_a_bad_bind_value_never_carries_the_value():
    secret = "HTTPS://Secret-Capability-Word.example.com/"
    recs, _ = mutate(base(), "")
    next(r for r in recs if r["type"] == "configEntry").update(bindValue=secret)
    with pytest.raises(bundle.BundleError) as e:
        bundle.validate(recs)
    assert "apability" not in str(e.value) and "bindValue" in str(e.value)


@pytest.mark.parametrize("bad", ["Zz" * 32, "00", 7, None])
def test_a_bad_key_check_is_rejected_without_echoing_it(bad):
    recs, _ = mutate(base(), "")
    recs[0]["bindKeyCheck"] = bad
    with pytest.raises(bundle.BundleError) as e:
        bundle.validate(recs)
    assert "bindKeyCheck" in str(e.value) and "ZzZz" not in str(e.value)


def test_an_unsupported_minor_is_rejected():
    recs, _ = mutate(base(), "")
    recs[0]["bundleSchemaVersion"] = "1.2"
    with pytest.raises(bundle.BundleError):
        bundle.validate(recs)


def test_a_bind_parts_on_a_launch_fact_or_other_record_is_rejected():
    recs, _ = mutate(base(), "")
    next(r for r in recs if r["type"] == "launchFact")["bindParts"] = copy.deepcopy(
        next(r for r in recs if r["type"] == "configEntry")["bindParts"])
    with pytest.raises(bundle.BundleError):
        bundle.validate(recs)


def test_v1_is_a_cross_record_rule_one_entry_with_parts_is_enough():
    recs = base()
    second = copy.deepcopy(next(r for r in recs if r["type"] == "configEntry"))
    second.update(keyPath="a.url", line=7)
    second["id"] = ids.config_entry_id(second["repoId"], second["path"], second["keyPath"])
    second.pop("bindParts")
    recs = ordered(recs + [second])
    recs[0]["counts"]["configEntry"] += 1
    bundle.validate(recs)                                          # an entry without parts beside one with: fine


# --- the writer rules: E1-E4, W6, W7 ----------------------------------------------------------------------------------

def fields(key_path="x.url", value="http://localhost:8000/mcp", **kw):
    kw.setdefault("binding_role", True)
    kw.setdefault("declared_services", DECL)
    return wr.bind_fields(key_path, value, **kw), wr.bind_parts_fields(KEY1, key_path, value, **kw)


def test_a_capability_url_gets_parts_and_no_bind_value():
    bv, parts = fields("notify.url", "https://ntfy.sh/x7k2-q9w4-m1p8-z3v6")
    assert bv == {} and parts["bindParts"]["hostKind"] == "external" and "host" not in parts["bindParts"]
    assert parts["bindParts"]["pathHmacs"] == [EH["EH-21"][2]]


def test_a_multi_segment_base_binds_by_path_hmacs_and_has_no_bind_value():
    bv, parts = fields("x.base_url", "http://localhost:8000/api/v1")
    assert bv == {} and parts["bindParts"]["pathHmacs"] == [EH["EH-19"][2], EH["EH-20"][2]]


def test_e3_no_binding_role_no_parts_and_withhold_is_honoured():
    assert wr.bind_parts_fields(KEY1, "x.url", "http://localhost:8000/", binding_role=False, declared_services=DECL) == {}
    assert wr.bind_parts_fields(KEY1, "x.url", "http://localhost:8000/", binding_role=True, declared_services=DECL,
                                withhold=True) == {}
    assert not wr.bind_parts_allowed("x.url", "http://localhost:8000/", binding_role=False, declared_services=DECL)
    assert wr.bind_parts_allowed("x.url", "http://localhost:8000/", binding_role=True, declared_services=DECL)


def test_e4_alias_half_both_ends_must_pass_the_name_rule():
    kw = dict(binding_role=True, declared_services=DECL, via_alias=True)
    assert wr.bind_parts_fields(KEY1, "x.url", "http://localhost:8000/", anchor_key_path="base.url", **kw)
    for anchor in ("base.auth_url", None, "a[", "ab\ud800"):
        assert wr.bind_parts_fields(KEY1, "x.url", "http://localhost:8000/", anchor_key_path=anchor, **kw) == {}, anchor
    assert wr.bind_parts_fields(KEY1, "x.auth_url", "http://localhost:8000/", anchor_key_path="base.url", **kw) == {}


@pytest.mark.parametrize("key_path", ["a[", "[01]", '["\\a"]', '["\\u00"]', "a[" + "x" * 300_000, "[01]" * 75_000, "ab\ud800"])
def test_a_hostile_key_path_withholds_and_never_raises(key_path):
    assert wr.bind_parts_fields(KEY1, key_path, "http://localhost:8000/", binding_role=True, declared_services=DECL) == {}
    assert not wr.bind_parts_allowed(key_path, "http://localhost:8000/", binding_role=True, declared_services=DECL)


@pytest.mark.parametrize("value", ["", "ab\ud800", "http://localhost:8000/" + "a" * 70_000, "http://\x00/", "\ud800://x/", None, 5, b"http://x/"])
def test_a_hostile_value_withholds_and_never_raises(value):
    assert wr.bind_parts_fields(KEY1, "x.url", value, binding_role=True, declared_services=DECL) == {}


def test_e2_w1_an_argv_element_is_never_a_parts_source():
    assert wr.bind_parts_fields(KEY1, "services.app.command[2]", "http://localhost:8000/mcp", binding_role=True,
                                declared_services=DECL) == {}
    assert wr.bind_parts_fields(KEY1, "a.entrypoint[0]", "http://localhost:8000/mcp", binding_role=True, declared_services=DECL) == {}


def test_w6_keeps_bind_value_only_for_loopback_and_service_hosts():
    kept = ["http://localhost:8200/mcp", "http://127.0.0.2:9000", "http://api:8000", "redis://falkordb:6379",
            "http://host.docker.internal:8200/mcp"]
    for v in kept:
        assert fields(value=v)[0] == {"bindValue": v}, v
    withheld = ["http://10.0.0.5:8080/api", "http://[2001:db8::5]:8080/v1/", "http://worker:8000", "https://ntfy.sh/deploy-alerts-team",
                "https://ntfy.sh/kxqzmwvptryhbnjdlc"]
    for v in withheld:
        bv, parts = fields(value=v)
        assert bv == {} and "bindParts" in parts, v


def test_w6_does_not_touch_a_non_url_bind_value():
    for v in ("8200", "ntfy.sh", "localhost:8000", "10.0.0.5", "[2001:db8::5]", "api", "10.0.0.5:80"):
        assert fields("x.addr", v)[0] == {"bindValue": v}, v
        assert fields("x.addr", v)[1] == {}, v


def test_w7_a_url_the_grammar_admits_and_the_parse_refuses_has_no_clear_copy():
    v = "http://[::ffff:7f00:1]:80/"
    assert bundle.value_grammar_ok(v) and bundle.parse_bind_url(v) is None
    assert fields(value=v) == ({}, {})
    for v in ("http://[::7f00:1]/", "http://[64:ff9b::7f00:1]/"):
        assert fields(value=v) == ({}, {}), v


def test_audit_records_checks_parts_against_the_text():
    recs = [{"type": "configEntry", "id": ids.config_entry_id("r", "a.json", "x.url"), "keyPath": "x.url", "valueHash": "0" * 64,
             **dict(wr.bind_parts_fields(KEY1, "x.url", "http://localhost:8000/mcp", binding_role=True, declared_services=DECL))}]
    rid = recs[0]["id"]
    recs[0]["valueHash"] = wr.value_hash(KEY1, rid, "http://localhost:8000/mcp")
    texts = {rid: "http://localhost:8000/mcp"}
    assert wr.audit_records(recs, key=KEY1, declared_services=DECL, texts=texts) == []
    for mutate_parts in (lambda p: p.update(hostHmac="0" * 64), lambda p: p.update(hostKind="external"),
                         lambda p: p.update(pathHmacs=[]), lambda p: p.update(port=1), lambda p: p.pop("host")):
        bad = copy.deepcopy(recs)
        mutate_parts(bad[0]["bindParts"])
        out = wr.audit_records(bad, key=KEY1, declared_services=DECL, texts=texts)
        assert out and "http" not in " ".join(out)
    nb = copy.deepcopy(recs)
    nb[0].update(bindValue="https://ntfy.sh/kxqzmwvptryhbnjdlc")      # W6/W7 hand-built violation
    nb[0].pop("bindParts")
    assert wr.audit_records(nb, key=KEY1, declared_services=DECL, texts={rid: "https://ntfy.sh/kxqzmwvptryhbnjdlc"})
    nul = copy.deepcopy(recs)
    nul[0]["bindParts"] = None
    assert wr.audit_records(nul, key=KEY1, declared_services=DECL)


def test_bind_header_fields_follow_the_version_rule():
    with_parts = [{"type": "configEntry", **wr.bind_parts_fields(KEY1, "x.url", "http://localhost:8000/", binding_role=True,
                                                                  declared_services=DECL)}]
    assert wr.bind_header_fields(KEY1, with_parts) == {"bundleSchemaVersion": "1.1", "bindKeyCheck": EH["EH-30"][2]}
    assert wr.bind_header_fields(KEY1, [{"type": "configEntry"}, {"type": "launchFact"}]) == {}
    assert wr.bind_header_fields(KEY1, []) == {}


# --- one definition of the key accessors ---------------------------------------------------------------------------------

def _defs(name: str) -> list[str]:
    out = []
    for root in ("ccg", "extractors"):
        for p in sorted((CPG / root).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(ast.parse(p.read_text("utf-8")))):
                out.append(str(p.relative_to(CPG)))
    return out


@pytest.mark.parametrize("name", ["hmac_key", "bind_key_check"])
def test_the_key_accessors_have_one_home(name):
    assert _defs(name) == ["ccg/atrest.py"]


def test_valuehash_defines_no_key_reading_function_and_imports_atrest_only_for_the_check():
    tree = ast.parse((CPG / "ccg" / "valuehash.py").read_text("utf-8"))
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not {f for f in funcs if "key" in f.lower() and f != "bind_parts_usable"}
    src = (CPG / "ccg" / "valuehash.py").read_text("utf-8")
    for needle in ("open(", "read_bytes", "read_text", "environ", "expanduser", "hmac_key("):
        assert needle not in src


# --- review follow-ups (P1-14 gates) ---------------------------------------------------------------------------------

def _rejects(mutation: str, rule: str, mod=bundle) -> str:
    recs, _ = mutate(base(), mutation)
    with pytest.raises(bundle.BundleError) as e:
        mod.validate(recs)
    assert rule in str(e.value), str(e.value)
    return str(e.value)


#: Rows beyond spec5 §9.3 (same syntax), each violating exactly one §6.4 conjunct; the in-suite mutation pass below runs them too.
EXTRA_BK = [
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[2001:0db8::5]"', "V5d"),      # non-canonical IPv6
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[2001:DB8::5]"', "V5d"),
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[zz]"', "V5d"),                 # not IPv6 at all
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[2001:db8::5"', "V5d"),
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "2001:db8::5"', "V5d"),         # unbracketed
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "10.0.0.05"', "V5d"),
    ('configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "127.1.2.3"', "V5d"),           # 127/8, not only 127.0.*
    ('configEntry.bindValue = <absent>; bindParts.host = "127.1.2.3"', None),                                      # valid loopback
    ('configEntry.bindValue = <absent>; bindParts.host = "127.255.255.254"', None),
    ('configEntry.bindValue = <absent>; bindParts.host = "128.0.0.1"', "V5c"),
    ("bindParts.scheme = <absent>", "V4b"),                       # BK-11 removes only hostHmac; one row per required field
    ("bindParts.hostKind = <absent>", "V4b"),
    ("bindParts.pathHmacs = <absent>", "V4b"),
    ('header.bundleSchemaVersion = "1.2"', "bundleSchemaVersion"),    # spec5 §1: only "1.0" and "1.1"
]


@pytest.mark.parametrize("mutation,rule", EXTRA_BK)
def test_extra_validator_rows(mutation, rule):
    if rule is None:
        bundle.validate(mutate(base(), mutation)[0])
    else:
        _rejects(mutation, rule)


# spec5 §8: "a mutation pass: disabling any one V-conjunct in bundle.py or the schema turns at least one BK row red", in the suite.
# Each mutant is a source edit of ccg/bundle.py (or the schema) executed as a private module; no file is written and the shared
# module is never swapped.
def _mutant_module(old: str | None, new: str | None, schema_edit=None, tmp=None):
    import types
    path = CPG / "ccg" / "bundle.py"
    src = path.read_text(encoding="utf-8")
    if old is not None:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    mod = types.ModuleType("bundle_mutant")
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    if schema_edit is not None:
        d = json.loads(bundle.SCHEMA_PATH.read_text(encoding="utf-8"))
        schema_edit(d["$defs"]["bindParts"], d["$defs"]["header"])
        f = tmp / "schema.json"
        f.write_text(json.dumps(d), encoding="utf-8")
        mod.SCHEMA_PATH = f
    return mod


def _corpus_outcomes(mod) -> list[bool]:
    """True where `mod` disagrees with the spec on a row (valid row rejected, reject row accepted or with the wrong rule id)."""
    out = []
    rows = [(m, o, re.match(r"(V\d[a-z]?)", r).group(1) if o == "reject" else None) for _n, m, o, r in BK]
    rows += [(m, "valid" if r is None else "reject", r) for m, r in EXTRA_BK]
    for mutation, outcome, rule in rows:
        recs = mutate(base(), mutation)[0]
        try:
            mod.validate(recs)
            out.append(outcome != "valid")
        except Exception as e:                               # the mutant module has its own BundleError class
            out.append(type(e).__name__ != "BundleError" or outcome == "valid" or rule not in str(e))     # a crash is not a rejection
    return out


_V_MUTANTS = [
    ("V1a", 'if v == "1.0" and (has_parts or has_check):', "if False:"),
    ("V1b", 'if v == "1.1" and not (has_parts or has_check):', "if False:"),
    ("V1c", "if has_parts and not has_check:", "if False:"),
    ("V1d", "if has_check and not has_parts:", "if False:"),
    ("V2", 'if _w1_fails(segs):\n            fail("V2")', 'if False:\n            fail("V2")'),
    ("V3", 'if _name_rule_fails(segs):\n            fail("V3")', 'if False:\n            fail("V3")'),
    ("V4d", 'if screen_fails(bp["scheme"]):', "if False:"),
    ("V5c", 'if kind == "loopback" and not (host in', 'if False and not (host in'),
    ("V5c-prefix", 'or (_ipv4(host) and host.startswith("127."))):', "or _ipv4(host)):"),
    ("V5d-v6-format", 'if not _ipv6(host[1:-1]) or host == "[::1]":', 'if host == "[::1]":'),
    ("V5d-v6-loopback", 'or host == "[::1]":', "or False:"),
    ("V5d-v4", 'elif not _ipv4(host) or host.startswith("127."):', "elif False:"),
    ("V5d-v4-127", 'elif not _ipv4(host) or host.startswith("127."):', "elif not _ipv4(host):"),
    ("V5d-v4-127-prefix", 'elif not _ipv4(host) or host.startswith("127."):', 'elif not _ipv4(host) or host.startswith("127.0"):'),
    ("V5e", 'if excluded:\n                    fail("V5e", "host")', 'if False:\n                    fail("V5e", "host")'),
    ("V5f", 'if not _label(host, False):\n                fail("V5f"', 'if False:\n                fail("V5f"'),
    ("V5g", 'if screen_fails(host):\n                fail("V5g"', 'if False:\n                fail("V5g"'),
    ("V5h", 'if not any(c.isalpha() for c in host):\n                fail("V5h"', 'if False:\n                fail("V5h"'),
    ("V5i", 'if host == "localhost":\n                fail("V5i"', 'if False:\n                fail("V5i"'),
    ("V6a", 'if u is None:\n                fail("V6a")', "if u is None:\n                return"),
    ("V6b", 'if bp["scheme"] != u.scheme:', "if False:"),
    ("V6c", 'if bp.get("port") != u.port:', "if False:"),
    ("V6c-absent", 'if bp.get("port") != u.port:', 'if "port" in bp and bp["port"] != u.port:'),
    ("V6d", 'if kind != {"loopback": "loopback"', 'if False and kind != {"loopback": "loopback"'),
    ("V6e", "if host is not None and host != u.host:", "if False:"),
    ("V6f", 'if len(bp["pathHmacs"]) != len(u.segments):', "if False:"),
    ("msg-echo", 'if path[:1] in (["bindParts"], ["bindValue"], ["bindKeyCheck"]):', 'if path[:1] in (["bindValue"], ["bindKeyCheck"]):'),
]
_S_MUTANTS = [
    ("V4a", lambda bp, hd: bp.update(additionalProperties=True)),
    ("V4b-scheme", lambda bp, hd: bp["required"].remove("scheme")),
    ("V4b-hostKind", lambda bp, hd: bp["required"].remove("hostKind")),
    ("V4b-hostHmac", lambda bp, hd: bp["required"].remove("hostHmac")),
    ("V4b-pathHmacs", lambda bp, hd: bp["required"].remove("pathHmacs")),
    ("V4c-length", lambda bp, hd: bp["properties"]["scheme"].update(pattern="^[a-z][a-z0-9+.-]{0,16}(?!\\n)$")),
    ("V4c-case", lambda bp, hd: bp["properties"]["scheme"].update(pattern="^[A-Za-z][a-zA-Z0-9+.-]{0,15}(?!\\n)$")),
    ("V4e-min", lambda bp, hd: bp["properties"]["port"].update(minimum=0)),
    ("V4e-max", lambda bp, hd: bp["properties"]["port"].update(maximum=65536)),
    ("V4e-type", lambda bp, hd: bp["properties"]["port"].update(type="string")),
    ("V4f", lambda bp, hd: bp["properties"]["hostKind"]["enum"].append("remote")),
    ("V4g", lambda bp, hd: bp["properties"].update(hostHmac={"type": "string"})),
    ("V4h", lambda bp, hd: bp["properties"]["pathHmacs"].update(items={"type": "string"})),
    ("V4i", lambda bp, hd: bp["properties"]["pathHmacs"].update(maxItems=33)),
    ("V5a", lambda bp, hd: bp.update(then={})),
    ("V5b", lambda bp, hd: bp.update(**{"else": {}})),
    ("version-enum", lambda bp, hd: hd["properties"]["bundleSchemaVersion"]["enum"].append("1.2")),
]


def test_the_unmutated_module_agrees_with_every_row():
    assert not any(_corpus_outcomes(_mutant_module(None, None)))


@pytest.mark.parametrize("name,old,new", _V_MUTANTS, ids=[m[0] for m in _V_MUTANTS])
def test_mutation_pass_disabling_one_validator_conjunct_turns_a_row_red(name, old, new):
    assert any(_corpus_outcomes(_mutant_module(old, new))), name


@pytest.mark.parametrize("name,edit", _S_MUTANTS, ids=[m[0] for m in _S_MUTANTS])
def test_mutation_pass_disabling_one_schema_conjunct_turns_a_row_red(name, edit, tmp_path):
    assert any(_corpus_outcomes(_mutant_module(None, None, edit, tmp_path))), name


# --- m1: no number is echoed by a bind-field failure path -------------------------------------------------------------------------

def test_a_float_in_a_bind_part_is_rejected_without_echoing_it_in_the_validator():
    recs, _ = mutate(base(), "")
    next(r for r in recs if r["type"] == "configEntry")["bindParts"]["port"] = 8200.5            # _check_values site
    with pytest.raises(bundle.BundleError) as e:
        bundle.validate(recs)
    assert "8200.5" not in str(e.value) and "integers only" in str(e.value)


def test_a_float_in_a_bind_part_is_rejected_without_echoing_it_in_the_parser():
    recs, _ = mutate(base(), "")
    next(r for r in recs if r["type"] == "configEntry")["bindParts"]["port"] = 31337.5
    data = b"".join(bundle.canonical_line(r) for r in recs)
    with pytest.raises(bundle.BundleError) as e:                                                    # _no_float site
        bundle.loads(data)
    assert "31337" not in str(e.value) and "integers only" in str(e.value)
    with pytest.raises(bundle.BundleError) as e2:
        bundle.loads(data.replace(b"31337.5", b"NaN"))
    assert "NaN" not in str(e2.value)


@pytest.mark.parametrize("where", ["header.bindKeyCheck", "configEntry.bindValue", "bindParts.scheme", "bindParts.port", "bindParts.hostKind",
                                   "bindParts.host", "bindParts.hostHmac", "bindParts.pathHmacs", "bindParts"])
@pytest.mark.parametrize("junk", [31337.5, 31337, True, [31337.5], {"k": 31337}, "31337-string"])
def test_no_number_or_value_set_on_a_bind_field_is_echoed_by_the_validator(where, junk):
    target, _, field = where.partition(".")
    recs, _ = mutate(base(), "")
    ce = next(r for r in recs if r["type"] == "configEntry")
    holder = {"header": recs[0], "configEntry": ce, "bindParts": ce["bindParts"]}.get(target if field else "bindParts")
    if field:
        holder[field] = junk
    else:
        ce["bindParts"] = junk
    try:
        bundle.validate(recs)
    except bundle.BundleError as e:
        assert "31337" not in str(e), (where, junk, str(e))
    else:
        assert where == "bindParts.port" and junk == 31337 or where == "bindParts.pathHmacs" and False, (where, junk)


# --- m2: the right rule id --------------------------------------------------------------------------------------------------------

def test_a_non_string_host_is_not_called_v5b():
    msg = _rejects('configEntry.bindValue = <absent>; bindParts.host = 123456', "bindParts.host")
    assert "V5b" not in msg and "123456" not in msg


@pytest.mark.parametrize("bad", ['"remote"', '"Secret-Word-Here"', "5", "null"])
def test_a_bad_host_kind_is_v4f_in_both_branches(bad):
    msg = _rejects(f"configEntry.bindValue = <absent>; bindParts.hostKind = {bad}", "V4f")
    assert "V5b" not in msg and "V5a" not in msg and "Secret" not in msg
    _rejects(f"configEntry.bindValue = <absent>; bindParts.hostKind = {bad}; bindParts.host = <absent>", "V4f")


def test_v5b_still_names_a_missing_host_for_a_valid_kind_and_v5a_a_stray_host():
    _rejects("bindParts.host = <absent>", "V5b")
    _rejects('configEntry.bindValue = <absent>; bindParts.hostKind = "external"', "V5a")


@pytest.mark.parametrize("bad", ['"Zz"', "7", "null"])
def test_a_malformed_key_check_borrows_no_v1_id(bad):
    recs, _ = mutate(base(), f"header.bindKeyCheck = {bad}")
    with pytest.raises(bundle.BundleError) as e:
        bundle.validate(recs)
    assert "bindKeyCheck" in str(e.value) and "V1" not in str(e.value)


# --- m3: 127/8 ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["127.1.2.3", "127.255.255.254", "127.0.0.1", "127.10.0.0"])
def test_every_127_slash_8_address_is_loopback_in_the_parse_and_the_parts(host):
    assert bundle.parse_bind_url(f"http://{host}:80/").host_class == "loopback"
    p = valuehash.bind_parts(KEY1, f"http://{host}/", set())
    assert p["hostKind"] == "loopback" and p["host"] == host
    assert bundle.parse_bind_url("http://128.0.0.1/").host_class == "ip" and bundle.parse_bind_url("http://12.7.0.1/").host_class == "ip"


# --- nit: mount_prefix_hmac validates the host ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["h/a", "/", "", "a/", None, 5, "ab\ud800"])
def test_mount_prefix_hmac_refuses_a_host_that_is_not_a_single_host_text(host):
    assert valuehash.mount_prefix_hmac(KEY1, host, "/b") is None


def test_a_host_with_a_slash_cannot_forge_a_path_prefix_of_another_host():
    second = valuehash.bind_parts(KEY1, "http://h/a/b", set())["pathHmacs"][1]
    assert valuehash.mount_prefix_hmac(KEY1, "h", "/a/b")[1] == second          # the honest call equals the parts' hash
    assert valuehash.mount_prefix_hmac(KEY1, "h/a", "/b") is None               # the spliced host is refused, not equal to it
