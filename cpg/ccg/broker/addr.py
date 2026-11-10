"""Address validation after an allow, and S6 (C-1001 P1-29). Spec: ``docs/plans/connected-code-graph-egress-spec3.md``
section 3.4 and section 4.3 (S6).

A denied request causes no DNS query: the broker calls `validate` only for a request a rule would allow. ``local``
entries are never resolved; other entries are resolved once, as the absolute name ``host + "."``, every returned
address must pass the forbidden set, and the connection goes to the first one. The forbidden set is explicit
networks, not a library's address-class predicates. Interface data comes from an injectable reader, by default
``/usr/bin/ip -j``, re-read at every validation.
"""
from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, NamedTuple
from urllib.parse import urlsplit

IP = "/usr/bin/ip"

_V4_ALWAYS = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "127.0.0.0/8", "169.254.0.0/16", "192.0.0.0/24", "198.18.0.0/15", "224.0.0.0/4", "240.0.0.0/4"))
_V6_ALWAYS = tuple(ipaddress.ip_network(n) for n in (
    "::/128", "::1/128", "fe80::/10", "ff00::/8", "::ffff:0:0/96", "::/96", "64:ff9b::/96", "64:ff9b:1::/48",
    "2002::/16", "2001::/32"))
_PRIVATE = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "fc00::/7"))
_LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))


class AddressDenied(Exception):
    """The allow becomes ``R0-INVALID`` with this reason: resolve-failed, forbidden-address or address-check-unavailable."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class IfaceUnavailable(Exception):
    """The interface data could not be read."""


@dataclass(frozen=True)
class IfaceData:
    addresses: tuple = ()       # every address of a host interface
    bridge_nets: tuple = ()     # the network of each bridge-kind interface address
    gateways: tuple = ()        # each default gateway


class Validated(NamedTuple):
    addr: str
    port: int
    private: bool  # the entry declared privateAddress: true


def _run_json(run: Callable[..., Any], argv: list[str]) -> Any:
    try:
        cp = run(argv, capture_output=True, timeout=10, check=True)
        return json.loads(cp.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise IfaceUnavailable(f"{' '.join(argv)}: {exc}") from None


def read_interfaces(run: Callable[..., Any] = subprocess.run, ip: str = IP) -> IfaceData:
    """``ip -j -d address show`` and ``ip [-6] -j route show default``, as the absolute ``/usr/bin/ip``."""
    addrs, nets, gws = [], [], []
    try:
        for ifc in _run_json(run, [ip, "-j", "-d", "address", "show"]):
            bridge = (ifc.get("linkinfo") or {}).get("info_kind") == "bridge"
            for a in ifc.get("addr_info", []):
                addr = ipaddress.ip_address(a["local"])
                addrs.append(addr)
                if bridge:
                    nets.append(ipaddress.ip_interface(f"{a['local']}/{a['prefixlen']}").network)
        for argv in ([ip, "-j", "route", "show", "default"], [ip, "-6", "-j", "route", "show", "default"]):
            for r in _run_json(run, argv):
                if r.get("gateway"):
                    gws.append(ipaddress.ip_address(r["gateway"]))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise IfaceUnavailable(f"unexpected `ip -j` output: {exc!r}") from None
    return IfaceData(tuple(addrs), tuple(nets), tuple(gws))


def default_resolver(name: str) -> list[str]:
    seen: list[str] = []
    for ai in socket.getaddrinfo(name, None, type=socket.SOCK_STREAM):
        if ai[4][0] not in seen:
            seen.append(ai[4][0])
    return seen


def _in(addr, nets: Iterable) -> bool:
    return any(addr.version == n.version and addr in n for n in nets)


def _literal(host: str):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def local_ok(addr, iface: Callable[[], IfaceData]) -> bool:
    """S6: a ``local`` IP literal is loopback or an address of a host interface, and never a default gateway. The
    interface data is read only for an address that is not loopback."""
    if _in(addr, _LOOPBACK):
        return True
    data = iface()
    return addr not in data.gateways and addr in data.addresses


def check_local_entries(egress: Mapping[str, Any], iface: Callable[[], IfaceData]) -> list[str]:
    """S6 over a manifest's ``local`` entries; one message per violation. Raises `IfaceUnavailable` if the data is
    needed and unreadable."""
    bad = []
    for e in egress["local"]:
        parts = urlsplit(e["endpoint"])
        ip = _literal(parts.hostname or "")
        if ip is not None and not local_ok(ip, iface):
            bad.append(f"S6: local endpoint {e['endpoint']} ({e['id']}) is not an address of this machine")
    return bad


def _forbidden(addr, data: IfaceData, local_addrs: set, private_ok: bool) -> bool:
    if "%" in str(addr):
        return True
    nets = _V4_ALWAYS if addr.version == 4 else _V6_ALWAYS
    if _in(addr, nets) or _in(addr, data.bridge_nets):
        return True
    if addr in data.addresses or addr in data.gateways or addr in local_addrs:
        return True
    return not private_ok and _in(addr, _PRIVATE)


def validate(egress: Mapping[str, Any], section: str, entry: Mapping[str, Any], *,
             resolver: Callable[[str], list[str]], iface: Callable[[], IfaceData]) -> Validated:
    """The address to connect to for an allowed entry, or `AddressDenied`."""
    parts = urlsplit(entry["endpoint"])
    host, port = parts.hostname or "", parts.port or 0
    private = entry.get("privateAddress") is True
    if section == "local":
        if host == "localhost":
            return Validated("127.0.0.1", port, False)
        ip = _literal(host)
        try:
            ok = ip is not None and local_ok(ip, iface)
        except IfaceUnavailable as exc:
            raise AddressDenied("address-check-unavailable", str(exc)) from None
        if not ok:
            raise AddressDenied("forbidden-address", f"{host} is no longer an address of this machine")
        return Validated(str(ip), port, False)
    ip = _literal(host)
    if ip is not None:
        names = [str(ip)]  # a literal needs no resolution
    else:
        try:
            names = list(resolver(host + "."))
        except Exception as exc:  # any resolver failure is a failed resolution
            raise AddressDenied("resolve-failed", f"{type(exc).__name__}") from None
        if not names:
            raise AddressDenied("resolve-failed", "no address")
    addrs = []
    for n in names:
        a = _literal(n) if isinstance(n, str) else None
        if a is None:
            raise AddressDenied("resolve-failed", "the resolver returned a non-address")
        addrs.append(a)
    try:
        data = iface()
    except IfaceUnavailable as exc:
        raise AddressDenied("address-check-unavailable", str(exc)) from None
    local_addrs = set()
    for e in egress["local"]:
        lip = _literal(urlsplit(e["endpoint"]).hostname or "")
        if lip is not None:
            local_addrs.add(lip)
    for a in addrs:
        if _forbidden(a, data, local_addrs, private):
            raise AddressDenied("forbidden-address", str(a))
    return Validated(str(addrs[0]), port, private)
