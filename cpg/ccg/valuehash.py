"""The keyed, domain-separated hash of `-bundle-spec5.md` §2 (the generalisation of `-bundle-spec3.md` §1.7.5).

``entry_hmac(key, domain, value) = lowercase-hex(HMAC-SHA-256(key, UTF-8(domain) || 0x00 || UTF-8(NFC(value))))``.

This module only computes. It never reads, stores or logs a key: the system key comes from ``ccg.atrest.hmac_key``
and is handed in. Conformance: spec5 §9.1 (``tests/test_valuehash.py`` parses the vectors from the spec text).
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import unicodedata
from collections.abc import Collection

from ccg import atrest, bundle

KEY_BYTES = 32            # `ccg.atrest.KEY_BYTES`; every spec5 §9 vector uses 32-byte keys

#: spec5 §2: the domain tags of ``bindParts``. A tag contains ``.``, so it never equals a 32-hex record id.
BIND_HOST_TAG = "ccg.bindParts.host.v1"
BIND_PATH_TAG = "ccg.bindParts.path.v1"
BIND_KEYCHECK_TAG = atrest.BIND_KEYCHECK_TAG        # one definition; `atrest.bind_key_check` is the one implementation of the check


class UnhashableValue(ValueError):
    """``value`` has no UTF-8 encoding (a lone surrogate). The message never carries the value."""


def entry_hmac(key: bytes, domain: str, value: str) -> str:
    """HMAC-SHA-256 of ``value`` under ``key``, separated by ``domain`` (non-empty ASCII without NUL)."""
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be exactly {KEY_BYTES} bytes (-bundle-spec5.md §2)")
    if not (isinstance(domain, str) and domain and domain.isascii() and "\x00" not in domain):
        raise ValueError("domain must be non-empty ASCII without 0x00")
    try:
        text = unicodedata.normalize("NFC", value).encode("utf-8")
    except UnicodeEncodeError:
        raise UnhashableValue("value has no UTF-8 encoding") from None
    msg = domain.encode("ascii") + b"\x00" + text
    return hmac.new(bytes(key), msg, hashlib.sha256).hexdigest()


# --- bindParts (spec5 §3-§5.3, §7) ---------------------------------------------------------------------------

def host_kind(url: bundle.BindUrl, declared_services: Collection[str], *, force_external: bool = False) -> str:
    """§4's table: ``loopback``, ``ip``, ``service`` or ``external``. A ``label`` host is a ``service`` iff it is in
    ``declared_services`` (compared lowercased), holds a letter and passes the token screen. ``force_external`` is the
    writer's link-local/metadata withholding (``writer_rules``): such a host is hashed and never in clear."""
    if force_external:
        return "external"
    if url.host_class in ("loopback", "ip"):
        return url.host_class
    if (url.host_class == "label" and url.host in {s.lower() for s in declared_services}
            and any(c.isalpha() for c in url.host) and not bundle.screen_fails(url.host)):
        return "service"
    return "external"


def bind_parts(key: bytes, value: str, declared_services: Collection[str], *, force_external: bool = False) -> dict | None:
    """The ``bindParts`` object of §5.3 for ``value``, or ``None`` if §3 refuses the text (E1). A pure function of the
    value, the declared-service set and the key; it never raises on the value. ``host`` is present only for ``loopback``,
    ``ip`` and ``service`` (§4); ``hostHmac`` always; ``pathHmacs`` are hashes of the path prefixes under the host."""
    url = bundle.parse_bind_url(value)
    if url is None:
        return None
    kind = host_kind(url, declared_services, force_external=force_external)
    out: dict = {"scheme": url.scheme}
    if url.port is not None:
        out["port"] = url.port
    out["hostKind"] = kind
    if kind != "external":
        out["host"] = url.host
    out["hostHmac"] = entry_hmac(key, BIND_HOST_TAG, url.host)
    out["pathHmacs"] = [entry_hmac(key, BIND_PATH_TAG, url.host + "/" + "/".join(url.segments[:i + 1]))
                        for i in range(len(url.segments))]
    return out


def bind_parts_usable(header: dict, key: bytes) -> bool:
    """§5.4: a bundle's ``bindParts`` are usable iff its ``header.bindKeyCheck`` equals ``atrest.bind_key_check(key)``.
    Absent or malformed: not usable (never raises)."""
    got = header.get("bindKeyCheck") if isinstance(header, dict) else None
    return isinstance(got, str) and hmac.compare_digest(got.encode("utf-8", "replace"),
                                                         atrest.bind_key_check(key).encode("ascii"))


def mount_prefix_hmac(key: bytes, host: str, mount_path: str) -> tuple[int, str | None] | None:
    """§7: a provider mount path ``M`` as ``(d, hash)``: ``d`` its segment count and ``hash`` the prefix hash under the
    decided ``host`` (``None`` when ``d == 0``, which matches every consumer). ``None`` when ``M`` is outside §3 step 4's
    class (no leading ``/``, an empty, dot or ``..`` segment, a character outside ``[A-Za-z0-9._~-]``, more than 32
    segments): such a mount matches no consumer by parts."""
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
        return None
    if not isinstance(host, str) or not host or "/" in host:        # ``host`` is H of §3, which has no ``/`` (§2): never splice a path into it
        return None
    if mount_path == "/":
        return 0, None
    body = mount_path[:-1] if mount_path.endswith("/") else mount_path
    segs = body[1:].split("/")
    if len(segs) > 32 or any(not bundle._PATH_SEGMENT.fullmatch(x) or x in (".", "..") for x in segs):
        return None
    try:
        return len(segs), entry_hmac(key, BIND_PATH_TAG, host + "/" + "/".join(segs))
    except UnhashableValue:                                          # a host with no UTF-8 encoding matches nothing
        return None


# --- link-local and cloud-metadata targets: the one rule (writer and linker) ------------------------------------------------
URL_AUTHORITY = re.compile(r"[a-z]+://([^/]*)", re.S)


def host_of(value: str) -> str | None:
    """The host of ``value`` (``host``, ``ipv6``, ``hostport`` or ``url``; brackets removed), ``None`` for a bare
    ``port``. Only sound on a value that already passed the §1.7.2 grammar."""
    if bundle._port(value):
        return None
    if bundle._ipv6(value):
        return value
    m = URL_AUTHORITY.match(value)
    host = m.group(1) if m else value
    if host.startswith("["):
        return host[1:host.index("]")] if "]" in host else host
    i = host.rfind(":")
    return host[:i] if i > 0 else host


# Never carried as a bindValue or a default, whatever the form of the URL: link-local and cloud-metadata targets.
# 169.254.0.0/16 and fe80::/10 (P1-11 security review M3), plus the list deferred in
# `docs/plans/connected-code-graph-egress-spec4.md` §6: `fec0::/10`, `100::/64`, `fd00:ec2::254`, `100.100.100.200`.
# Architect ruling (P1-14): also 168.63.129.16 (Azure wireserver), 192.0.0.192 and the whole 6to4 block 2002::/16 (which embeds
# any IPv4 address); RFC1918 stays clear. Used by the writer and the linker; the validator has no link-local rule.
V4_NETS = (ipaddress.ip_network("169.254.0.0/16"), ipaddress.ip_network("100.100.100.200/32"),
            ipaddress.ip_network("168.63.129.16/32"), ipaddress.ip_network("192.0.0.192/32"))
V6_NETS = (ipaddress.ip_network("fe80::/10"), ipaddress.ip_network("fec0::/10"), ipaddress.ip_network("100::/64"),
            ipaddress.ip_network("fd00:ec2::254/128"), ipaddress.ip_network("2002::/16"))
_INET_PART = re.compile(r"0[xX][0-9a-fA-F]+|[0-9]+")


def _v4_forbidden(ip: ipaddress.IPv4Address) -> bool:
    return any(ip in n for n in V4_NETS)


def inet_aton(host: str) -> ipaddress.IPv4Address | None:
    """``host`` read the way ``inet_aton`` does (1-4 parts, decimal, ``0x`` hex or leading-0 octal; the last part
    fills the remaining bytes), ``None`` if it is not such a form. Resolvers read these as addresses."""
    parts = host.split(".")
    if not (1 <= len(parts) <= 4) or not all(_INET_PART.fullmatch(p) for p in parts):
        return None
    try:
        nums = [int(p, 16) if p[:2] in ("0x", "0X") else int(p, 8) if len(p) > 1 and p[0] == "0" else int(p, 10)
                for p in parts]
    except ValueError:
        return None
    n = len(nums)
    if any(x > 255 for x in nums[:-1]) or nums[-1] >= 256 ** (5 - n):
        return None
    total = 0
    for x in nums[:-1]:
        total = total * 256 + x
    return ipaddress.IPv4Address(total * 256 ** (5 - n) + nums[-1])


def forbidden_target(value: str) -> bool:
    """True iff the host of ``value`` is a link-local or cloud-metadata address, in any form the parsers read as one
    (dotted, ``inet_aton`` decimal/hex/octal/short forms, bracketed or bare IPv6, IPv4-mapped/compatible/NAT64)."""
    host = host_of(value)
    if host is None:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = inet_aton(host)
        return ip is not None and _v4_forbidden(ip)
    if ip.version == 4:
        return _v4_forbidden(ip)
    if any(ip in n for n in V6_NETS):
        return True
    raw = ip.packed
    if ip.ipv4_mapped is not None:
        return _v4_forbidden(ip.ipv4_mapped)
    if raw[:12] == bytes(12) or raw[:12] == bytes.fromhex("0064ff9b0000000000000000"):     # ::/96, 64:ff9b::/96
        return _v4_forbidden(ipaddress.IPv4Address(raw[12:]))
    return False
