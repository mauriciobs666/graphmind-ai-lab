"""Writer rules of the P1 writer done-condition (F-10), shared by every extractor that emits a bundle.

An extractor decides these rules by importing this module; it does not reach ``ccg.bundle``'s gates, the key
accessors or ``ccg.valuehash`` itself, and does not write the fields below by hand. Spec:
`docs/plans/connected-code-graph-bundle-spec3.md` §1.6.1 (argv gate), §1.7.4 (W2, W3, W4), §1.7.5 (per-entry hash);
`-bundle-spec5.md` §2 (``entry_hmac``).

**Enforcement is a tripwire plus a per-obligation backstop, not a proof.** The tripwire is an AST test
(``tests/test_writer_rules.py``) that fails if another extractor module writes ``command``, ``args``, ``bindValue``,
``default``, ``valueHash``, ``defaultHash`` or ``bindParts`` (literal or constant-foldable keys) or reaches the rule-deciding names; it
cannot see ``dict(zip(..))``, ``r.update(var)``, a ``**`` splat of a built mapping or a data-computed key. The backstop is
``audit_records``, which a writer's own test runs over the records it produced: it re-derives W1-W3 and W2 for
``default``, the argv fixed point, the hash domain (the record's own id), ``defaultHash`` present whenever ``default`` is
withheld, and absent-not-null. The validator (``bundle.validate``) additionally re-checks grammar, name rule, W1 and
the argv fixed point; W2 and the hashes are writer-only, so for them the tripwire and ``audit_records`` are the only controls.

Obligations:

- **W2** a single-label host (bare, in a ``hostport`` or in a ``url``) other than ``localhost`` is emitted, as a
  ``bindValue`` or a ``launchFact.default``, only if it is a declared Compose service of the same repository.
- **W3** nothing is interpolated (the value grammar has no ``$``), and a YAML alias value binds only if the alias's
  ``keyPath`` and the anchor's both pass the name rule.
- **W4** withholding is always allowed, and hostile input is withheld, never raised: ``withhold=True``; a ``keyPath``
  that does not parse, or text with no UTF-8 encoding (``encodable``), withholds. A withheld field is *absent*, never
  ``null``. Text that cannot be encoded is never hashed (``valueHash`` is required, so the *entry* is dropped by the
  config extractor, P1-11); ``default_fields`` then returns ``{}``.
- **E1-E4, W6, W7** (`-bundle-spec5.md` §6.1, §6.2; P1-14) ``bind_parts_allowed`` decides whether a ``configEntry`` may carry
  ``bindParts``: the URL parse is not ``None``, W1, a binding role, the name rule on every ``keyPath`` segment and, for an alias,
  on the anchor's. ``bind_parts_fields`` is the one place the field is written. A link-local or metadata host is hostKind
  ``external`` (hash only, never in clear). A ``url``-form ``bindValue`` is kept only if ``bindParts`` is emitted (W7) and its
  hostKind is ``loopback`` or ``service`` (W6). ``bind_header_fields`` gives ``"1.1"`` and ``bindKeyCheck`` exactly when one
  entry carries parts. Hostile input is withheld, never raised, and no message carries it.
- **SA-1** ``launchFact.command``/``args`` exist only as the output of ``ccg.bundle.gate_argv``; an argv with no program
  is gated behind a ``<redacted>`` argv[0] and ``command`` never holds a flag.
- **Targets** a link-local or cloud-metadata host (169.254.0.0/16, fe80::/10, fec0::/10, 100::/64, fd00:ec2::254,
  100.100.100.200, 168.63.129.16, 192.0.0.192, 2002::/16, in any URL or numeric form) is never a ``bindValue`` or ``default``. W2's declared-service set is the
  caller's: P1-11 passes only services of primary-role Compose files.
- **§1.7.5** ``valueHash``/``defaultHash`` are ``entry_hmac`` over the record's own 32-hex id.

Whether an entry is in a *binding role* (SA-3, F-6) is P1-11's decision and arrives as ``binding_role``.
Every function returns data to write; none of them reads a key (``system_key`` delegates to ``ccg.atrest``).
"""
from __future__ import annotations

import ipaddress
import re
from collections.abc import Collection, Iterable, Mapping

from ccg import atrest, bundle, ids, valuehash
from ccg.valuehash import entry_hmac

__all__ = ["encodable", "bind_value", "bind_fields", "bind_parts_allowed", "bind_parts_fields", "bind_header_fields",
           "launch_default", "default_fields", "launch_argv_fields", "value_hash", "value_hash_fields", "default_hash",
           "config_entry_value_hash", "system_key", "audit_records", "mcp_exec_launch_path"]

_RECORD_ID = re.compile(r"[0-9a-f]{32}")
_HASH = re.compile(r"[0-9a-f]{64}")
_URL_AUTHORITY = re.compile(r"[a-z]+://([^/]*)", re.S)


def encodable(text: object) -> bool:
    """True iff ``text`` is a ``str`` with a UTF-8 encoding (no lone surrogate): the one test for hashing/emitting it."""
    if not isinstance(text, str):
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _host_of(value: str) -> str | None:
    """The host of ``value`` (``host``, ``ipv6``, ``hostport`` or ``url``; brackets removed), ``None`` for a bare
    ``port``. Only sound on a value that already passed the §1.7.2 grammar."""
    if bundle._port(value):
        return None
    if bundle._ipv6(value):
        return value
    m = _URL_AUTHORITY.match(value)
    host = m.group(1) if m else value
    if host.startswith("["):
        return host[1:host.index("]")] if "]" in host else host
    i = host.rfind(":")
    return host[:i] if i > 0 else host


def _single_label_host(value: str) -> str | None:
    """The host of ``value`` if it is a single label other than ``localhost``, else ``None``. Private: it is only
    sound on a value that already passed the §1.7.2 grammar (which admits lowercase ASCII hosts only)."""
    host = _host_of(value)
    if host is None or host == "localhost" or "." in host or ":" in host or not re.fullmatch(r"[a-z0-9-]+", host):
        return None
    return host


# Never carried as a bindValue or a default, whatever the form of the URL: link-local and cloud-metadata targets.
# 169.254.0.0/16 and fe80::/10 (P1-11 security review M3), plus the list deferred in
# `docs/plans/connected-code-graph-egress-spec4.md` §6: `fec0::/10`, `100::/64`, `fd00:ec2::254`, `100.100.100.200`.
# Architect ruling (P1-14): also 168.63.129.16 (Azure wireserver), 192.0.0.192 and the whole 6to4 block 2002::/16 (which embeds
# any IPv4 address); RFC1918 stays clear. Writer-only: the validator has no link-local rule.
_V4_NETS = (ipaddress.ip_network("169.254.0.0/16"), ipaddress.ip_network("100.100.100.200/32"),
            ipaddress.ip_network("168.63.129.16/32"), ipaddress.ip_network("192.0.0.192/32"))
_V6_NETS = (ipaddress.ip_network("fe80::/10"), ipaddress.ip_network("fec0::/10"), ipaddress.ip_network("100::/64"),
            ipaddress.ip_network("fd00:ec2::254/128"), ipaddress.ip_network("2002::/16"))
_INET_PART = re.compile(r"0[xX][0-9a-fA-F]+|[0-9]+")


def _v4_forbidden(ip: ipaddress.IPv4Address) -> bool:
    return any(ip in n for n in _V4_NETS)


def _inet_aton(host: str) -> ipaddress.IPv4Address | None:
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


def _forbidden_target(value: str) -> bool:
    """True iff the host of ``value`` is a link-local or cloud-metadata address, in any form the parsers read as one
    (dotted, ``inet_aton`` decimal/hex/octal/short forms, bracketed or bare IPv6, IPv4-mapped/compatible/NAT64)."""
    host = _host_of(value)
    if host is None:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = _inet_aton(host)
        return ip is not None and _v4_forbidden(ip)
    if ip.version == 4:
        return _v4_forbidden(ip)
    if any(ip in n for n in _V6_NETS):
        return True
    raw = ip.packed
    if ip.ipv4_mapped is not None:
        return _v4_forbidden(ip.ipv4_mapped)
    if raw[:12] == bytes(12) or raw[:12] == bytes.fromhex("0064ff9b0000000000000000"):     # ::/96, 64:ff9b::/96
        return _v4_forbidden(ipaddress.IPv4Address(raw[12:]))
    return False


def _w2_ok(value: str, declared_services: Collection[str]) -> bool:
    """W2, then the link-local/metadata withholding. ``declared_services`` is the CALLER's set (P1-11 restricts it to
    primary-role Compose files); this module never widens it, and an address literal is withheld even if listed."""
    if _forbidden_target(value):
        return False
    host = _single_label_host(value)
    return host is None or host in {s.lower() for s in declared_services}


def _name_rule_ok(key_path: str) -> bool:
    """False also when the key path does not parse: what cannot be judged is withheld (W4)."""
    try:
        return not any(k == "name" and bundle.name_fails(t) for k, t in bundle._keypath_segments(key_path))
    except ValueError:
        return False


def bind_value(key_path: str, value: str, *, binding_role: bool, declared_services: Collection[str],
               via_alias: bool = False, anchor_key_path: str | None = None, withhold: bool = False) -> str | None:
    """``value`` if a ``configEntry`` may carry it as ``bindValue``, else ``None`` (withheld). Never raises on input."""
    if withhold or not binding_role or not encodable(value) or not encodable(key_path):
        return None
    if via_alias and (not encodable(anchor_key_path) or not _name_rule_ok(anchor_key_path)):   # W3, both ends
        return None
    try:
        if not bundle.bind_value_allowed(key_path, value):                      # W1, §1.7.3, §1.7.2 (no `$`: W3)
            return None
    except ValueError:                                                          # a keyPath that does not parse
        return None
    if not _w2_ok(value, declared_services):                                    # W2
        return None
    if _URL_AUTHORITY.match(value):                                             # a `url`-form value (§6.2)
        if not bind_parts_allowed(key_path, value, binding_role=binding_role, declared_services=declared_services,
                                  via_alias=via_alias, anchor_key_path=anchor_key_path, withhold=withhold):
            return None                                                         # W7: no clear URL without parts
        url = bundle.parse_bind_url(value)
        if valuehash.host_kind(url, declared_services, force_external=_forbidden_target(url.host)) not in ("loopback", "service"):
            return None                                                         # W6: ip and external keep the parts only
    return value


def bind_fields(key_path: str, value: str, **kw) -> dict:
    """``{"bindValue": value}`` or ``{}``: a withheld field is absent, never ``null`` (W4)."""
    v = bind_value(key_path, value, **kw)
    return {} if v is None else {"bindValue": v}


def bind_parts_allowed(key_path: str, value: str, *, binding_role: bool, declared_services: Collection[str] = (),
                       via_alias: bool = False, anchor_key_path: str | None = None, withhold: bool = False) -> bool:
    """E1-E4 of `-bundle-spec5.md` §6.1: may a ``configEntry`` carry ``bindParts`` for ``value``? The URL parse must not be
    ``None`` (E1), W1 (E2), a binding role (E3), the name rule on every named ``keyPath`` segment and, for a YAML alias,
    on the anchor's too (E4, W3). What cannot be judged (a ``keyPath`` that does not parse, text with no UTF-8 encoding) is
    withheld, never raised. ``declared_services`` is accepted so one keyword set serves ``bind_value`` and
    ``bind_parts_fields``; the gate does not depend on it (§4's host classes are applied when the parts are built)."""
    if withhold or not binding_role or not encodable(value) or not encodable(key_path):
        return False
    if via_alias and (not encodable(anchor_key_path) or not _name_rule_ok(anchor_key_path)):       # E4, the alias half
        return False
    try:
        segs = bundle._keypath_segments(key_path)
    except ValueError:
        return False
    if bundle._w1_fails(segs) or bundle._name_rule_fails(segs):                                   # E2, E4
        return False
    return bundle.parse_bind_url(value) is not None                                               # E1


def _parts_for(key: bytes, value: str, declared_services: Collection[str]) -> dict | None:
    """The §5.3 object for a ``value`` the gate admitted: a link-local or metadata host is ``external`` (hash only)."""
    url = bundle.parse_bind_url(value)
    if url is None:
        return None
    return valuehash.bind_parts(key, value, declared_services, force_external=_forbidden_target(url.host))


def bind_parts_fields(key: bytes, key_path: str, value: str, *, binding_role: bool, declared_services: Collection[str],
                      via_alias: bool = False, anchor_key_path: str | None = None, withhold: bool = False) -> dict:
    """``{"bindParts": {...}}`` or ``{}``: the one way an extractor emits the field (a withheld field is absent, W4)."""
    if not bind_parts_allowed(key_path, value, binding_role=binding_role, declared_services=declared_services,
                              via_alias=via_alias, anchor_key_path=anchor_key_path, withhold=withhold):
        return {}
    parts = _parts_for(key, value, declared_services)
    return {} if parts is None else {"bindParts": parts}


def bind_header_fields(key: bytes, records: Iterable[dict]) -> dict:
    """The ``header`` fields of the version rule (§1): ``bundleSchemaVersion`` ``"1.1"`` and ``bindKeyCheck`` exactly when
    some ``configEntry`` in ``records`` carries ``bindParts``; ``{}`` otherwise (the header stays ``"1.0"``)."""
    if any(r.get("type") == "configEntry" and "bindParts" in r for r in records):
        return {"bundleSchemaVersion": "1.1", "bindKeyCheck": atrest.bind_key_check(key)}
    return {}


def launch_default(env_var: str, value: str, *, binding_role: bool, declared_services: Collection[str],
                   withhold: bool = False) -> str | None:
    """``value`` if a ``kind: env-default`` ``launchFact`` may carry it as ``default``, else ``None``."""
    if withhold or not binding_role or not encodable(value) or not encodable(env_var):
        return None
    if bundle.name_fails(env_var):                                              # §1.7.3 on envVar
        return None
    if not bundle.value_grammar_ok(value):                                      # §1.7.2 (no `$`: W3)
        return None
    return value if _w2_ok(value, declared_services) else None                  # W2


def default_fields(key: bytes, launch_fact_id: str, env_var: str, value: str, **kw) -> dict:
    """The ``default``/``defaultHash`` fields of an ``env-default`` ``launchFact``: the hash is always present, except
    for text with no UTF-8 encoding, which is never hashed nor emitted (``{}``; the caller drops the record)."""
    if not encodable(value):
        return {}
    out = {"defaultHash": default_hash(key, launch_fact_id, value)}
    d = launch_default(env_var, value, **kw)
    if d is not None:
        out["default"] = d
    return out


def launch_argv_fields(argv: list[str], *, program_known: bool = True) -> dict:
    """``{"command", "args"}`` of a ``command`` ``launchFact``, through the argv gate (§1.6.1, SA-1).

    An argv with no program (``program_known=False``: a Dockerfile ``CMD`` beside an ``ENTRYPOINT``; or detected,
    when the first element starts with ``-``) gives ``gate_argv`` no context for argv[0], so a short value after
    ``--password`` would pass. Such a list is gated as ``[REDACTED, *argv]``: ``command`` is ``"<redacted>"`` (§1.6's
    value for an argv[0] that fails) and every element is an argument. A ``command`` never holds a flag."""
    if not argv:
        raise ValueError("argv must hold at least argv[0]")
    no_program = not program_known or argv[0].startswith("-")
    gated = bundle.gate_argv([bundle.REDACTED, *argv] if no_program else list(argv))
    return {"command": gated[0], "args": gated[1:]}


# P1-11b: the ONE allow-listed exception to the redaction of a `bash -c` string (the first loosening of SA-1). A whole-string match
# of two literal shapes over one ASCII character class (no nested or overlapping quantifier, so linear), length-capped first.
_EXEC_SHELLS = ("bash", "sh")
_EXEC_PATH = r"([A-Za-z0-9._/@+-]{1,128})"
_EXEC_SHAPE = re.compile(r'exec (?:"\$CLAUDE_PROJECT_DIR/' + _EXEC_PATH + r'"|\$CLAUDE_PROJECT_DIR/' + _EXEC_PATH + r")")
_EXEC_MAX_CHARS = 600


_DENIED_DOT_SEGMENTS = frozenset((".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".netrc", ".npmrc", ".pypirc",
                                  ".git-credentials", ".git"))
_DENIED_STEMS = frozenset(("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials", "secrets", "secret", "token", "tokens",
                           "password", "passwd"))


def _secret_segment(seg: str) -> bool:
    """A path segment that names a secret-bearing file or directory (case-insensitive): ``.env``/``.env.*``, a denylisted dot
    entry, or a denylisted name as the whole segment or its dot-stem (``secrets.sh``). Other dot segments (``.claude``) pass."""
    low = seg.lower()
    return (low == ".env" or low.startswith(".env.") or low in _DENIED_DOT_SEGMENTS or low.split(".", 1)[0] in _DENIED_STEMS)


def mcp_exec_launch_path(command: object, args: object) -> str | None:
    """The repo-relative script path of an ``.mcp.json`` server launched as ``bash|sh -c 'exec "$CLAUDE_PROJECT_DIR/<path>"'``
    (quotes optional, balanced), else ``None``: every other ``bash -c`` string keeps the argv gate's redaction. ``command`` must be
    exactly ``bash`` or ``sh``, ``args`` exactly ``["-c", S]``; ``<path>`` is relative, has no empty, ``.`` or ``..`` segment,
    is at most 128 characters (the gate's ``argv[0]`` cap), does not start with ``-`` (it becomes ``argv[0]``), passes ``bundle.screen_fails`` whole, and has no segment that fails the name rule (``bundle.name_fails``) or names a secret-bearing file or directory (``_secret_segment``). The caller emits
    ``command = <path>`` with no args (through ``launch_argv_fields``). Never raises on input."""
    if command not in _EXEC_SHELLS or not isinstance(args, list) or len(args) != 2 or args[0] != "-c" \
            or not isinstance(args[1], str) or len(args[1]) > _EXEC_MAX_CHARS:
        return None
    m = _EXEC_SHAPE.fullmatch(args[1])
    if m is None:
        return None
    path = m.group(1) or m.group(2)
    segments = path.split("/")
    if path[0] == "-" or any(seg in ("", ".", "..") for seg in segments):
        return None
    if bundle.screen_fails(path) or any(_secret_segment(seg) or bundle.name_fails(seg) for seg in segments):
        return None
    return path


def _record_id(record_id: str) -> str:
    if not (isinstance(record_id, str) and _RECORD_ID.fullmatch(record_id)):
        raise ValueError(f"domain must be the record's own 32-hex id, got {bundle.cap_repr(record_id)}")
    return record_id


def value_hash(key: bytes, config_entry_id: str, text: str) -> str:
    """``valueHash = entry_hmac(key, r["id"], text)`` (§1.7.5)."""
    return entry_hmac(key, _record_id(config_entry_id), text)


def value_hash_fields(key: bytes, config_entry_id: str, text: str) -> dict:
    """``{"valueHash": ...}`` for the ``configEntry`` whose own 32-hex id is ``config_entry_id`` (the domain; anything
    else raises the same ``ValueError`` as ``value_hash``), the one way an extractor emits the field. Text with no
    UTF-8 encoding is never hashed: the result is ``{}``, and the caller must drop the whole entry (check
    ``encodable`` at the value-read step), because ``valueHash`` is required and cannot be withheld alone."""
    rid = _record_id(config_entry_id)
    return {"valueHash": value_hash(key, rid, text)} if encodable(text) else {}


def default_hash(key: bytes, launch_fact_id: str, text: str) -> str:
    """``defaultHash = entry_hmac(key, <the launchFact's id>, <default text>)`` (§1.7.5)."""
    return entry_hmac(key, _record_id(launch_fact_id), text)


def config_entry_value_hash(key: bytes, repo_id: str, path: str, key_path: str, text: str) -> str:
    """``valueHash`` of the ``configEntry`` at (``repo_id``, ``path``, ``key_path``)."""
    return value_hash(key, ids.config_entry_id(repo_id, path, key_path), text)


def system_key(system: str, keys_root=None) -> bytes:
    """The system's HMAC key, read only through ``ccg.atrest.hmac_key`` (it refuses the published test keys)."""
    return atrest.hmac_key(system, keys_root)


# --- the behavioural backstop ---------------------------------------------------------------

def audit_records(records: Iterable[dict], *, key: bytes, declared_services: Collection[str],
                  texts: Mapping[str, str] | None = None) -> list[str]:
    """Violations of the writer obligations in ``records`` (empty list: none found). Messages carry the record type,
    its (capped) id and the obligation, never a value. ``texts`` optionally maps a record id to the text the extractor
    hashed (a ``configEntry``'s value, a ``launchFact``'s default), to check hashes of withheld values.

    Without the entry's binding role and alias origin, ``bindValue`` is judged as if in a binding role and not an
    alias: what this proves is that no emitted ``bindValue``/``default`` breaks W1, W2 or the grammar/name rule."""
    texts = texts or {}
    out: list[str] = []
    for r in records:
        t, rid = r.get("type"), r.get("id")
        if t not in ("configEntry", "launchFact"):
            continue
        tag = f"{t} {bundle.cap_repr(rid, 40)}"
        if not (isinstance(rid, str) and _RECORD_ID.fullmatch(rid)):
            out.append(f"{tag}: id is not 32-hex")
            continue
        for f in ("bindValue", "default", "valueHash", "defaultHash"):
            if f in r and not isinstance(r[f], str):
                out.append(f"{tag}: {f} is not a string (a withheld field is absent, never null)")
        if t == "configEntry":
            _audit_config_entry(r, tag, rid, key, declared_services, texts, out)
        else:
            _audit_launch_fact(r, tag, rid, key, declared_services, texts, out)
    return out


def _hash_matches(domain: str, text: object, claimed: object, key: bytes) -> bool:
    if not (encodable(text) and isinstance(claimed, str)):
        return False
    return claimed == entry_hmac(key, domain, text)


def _audit_config_entry(r, tag, rid, key, declared, texts, out) -> None:
    bv, vh = r.get("bindValue"), r.get("valueHash")
    if isinstance(bv, str) and bind_value(str(r.get("keyPath", "")), bv, binding_role=True,
                                          declared_services=declared) != bv:
        out.append(f"{tag}: bindValue breaks W1/W2/W6/W7/§1.7.2/§1.7.3")
    if "bindParts" in r:
        bp = r["bindParts"]
        text = texts.get(rid, bv if isinstance(bv, str) else None)
        if not isinstance(bp, dict):
            out.append(f"{tag}: bindParts is not an object (a withheld field is absent, never null)")
        elif text is not None and (not encodable(text) or _parts_for(key, text, declared) != bp):
            out.append(f"{tag}: bindParts are not the §5.3 parts of the entry's text under the key (E1, §4, link-local)")
    if not (isinstance(vh, str) and _HASH.fullmatch(vh)):
        out.append(f"{tag}: valueHash missing or not 64-hex")
        return
    text = texts.get(rid, bv if isinstance(bv, str) else None)
    if text is not None and not _hash_matches(rid, text, vh, key):
        out.append(f"{tag}: valueHash is not entry_hmac(key, the record's own id, text)")


def _audit_launch_fact(r, tag, rid, key, declared, texts, out) -> None:
    kind = r.get("kind")
    if kind == "command":
        argv = [r.get("command"), *(r.get("args") if isinstance(r.get("args"), list) else [None])]
        if isinstance(argv[0], str) and argv[0].startswith("-"):
            out.append(f"{tag}: command is a flag (an args-only argv gets command \"<redacted>\")")
        elif not all(isinstance(a, str) for a in argv) or bundle.gate_argv(argv) != argv:
            out.append(f"{tag}: command/args are not a fixed point of gate_argv (SA-1)")
        return
    has = [f for f in ("default", "defaultHash") if f in r]
    if kind != "env-default":
        if has:
            out.append(f"{tag}: {has[0]} on a launchFact that is not env-default")
        return
    dh, d = r.get("defaultHash"), r.get("default")
    if not (isinstance(dh, str) and _HASH.fullmatch(dh)):
        out.append(f"{tag}: defaultHash missing or not 64-hex (it is present even when default is withheld)")
        return
    if isinstance(d, str) and launch_default(str(r.get("envVar", "")), d, binding_role=True,
                                             declared_services=declared) != d:
        out.append(f"{tag}: default breaks W2/§1.7.2/§1.7.3")
    text = texts.get(rid, d if isinstance(d, str) else None)
    if text is not None and not _hash_matches(rid, text, dh, key):
        out.append(f"{tag}: defaultHash is not entry_hmac(key, the record's own id, text)")
