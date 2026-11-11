"""Writer rules of the P1 writer done-condition (F-10), shared by every extractor that emits a bundle.

An extractor decides these rules by importing this module; it does not reach ``ccg.bundle``'s gates, the key
accessors or ``ccg.valuehash`` itself, and does not write the fields below by hand. Spec:
`docs/plans/connected-code-graph-bundle-spec3.md` §1.6.1 (argv gate), §1.7.4 (W2, W3, W4), §1.7.5 (per-entry hash);
`-bundle-spec5.md` §2 (``entry_hmac``).

**Enforcement is a tripwire plus a per-obligation backstop, not a proof.** The tripwire is an AST test
(``tests/test_writer_rules.py``) that fails if another extractor module writes ``command``, ``args``, ``bindValue``,
``default``, ``valueHash`` or ``defaultHash`` (literal or constant-foldable keys) or reaches the rule-deciding names; it
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
- **SA-1** ``launchFact.command``/``args`` exist only as the output of ``ccg.bundle.gate_argv``.
- **§1.7.5** ``valueHash``/``defaultHash`` are ``entry_hmac`` over the record's own 32-hex id.

Whether an entry is in a *binding role* (SA-3, F-6) is P1-11's decision and arrives as ``binding_role``.
Every function returns data to write; none of them reads a key (``system_key`` delegates to ``ccg.atrest``).
"""
from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping

from ccg import atrest, bundle, ids
from ccg.valuehash import entry_hmac

__all__ = ["encodable", "bind_value", "bind_fields", "launch_default", "default_fields", "launch_argv_fields",
           "value_hash", "default_hash", "config_entry_value_hash", "system_key", "audit_records"]

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


def _single_label_host(value: str) -> str | None:
    """The host of ``value`` if it is a single label other than ``localhost``, else ``None``. Private: it is only
    sound on a value that already passed the §1.7.2 grammar (which admits lowercase ASCII hosts only)."""
    if bundle._port(value) or bundle._ipv6(value):
        return None
    m = _URL_AUTHORITY.match(value)
    host = m.group(1) if m else value
    if not host.startswith("["):
        i = host.rfind(":")
        host = host[:i] if i > 0 else host
    if host == "localhost" or "." in host or host.startswith("[") or not re.fullmatch(r"[a-z0-9-]+", host):
        return None
    return host


def _w2_ok(value: str, declared_services: Collection[str]) -> bool:
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
    return value if _w2_ok(value, declared_services) else None                  # W2


def bind_fields(key_path: str, value: str, **kw) -> dict:
    """``{"bindValue": value}`` or ``{}``: a withheld field is absent, never ``null`` (W4)."""
    v = bind_value(key_path, value, **kw)
    return {} if v is None else {"bindValue": v}


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


def launch_argv_fields(argv: list[str]) -> dict:
    """``{"command", "args"}`` of a ``command`` ``launchFact``, through the argv gate (§1.6.1, SA-1)."""
    if not argv:
        raise ValueError("argv must hold at least argv[0]")
    gated = bundle.gate_argv(list(argv))
    return {"command": gated[0], "args": gated[1:]}


def _record_id(record_id: str) -> str:
    if not (isinstance(record_id, str) and _RECORD_ID.fullmatch(record_id)):
        raise ValueError(f"domain must be the record's own 32-hex id, got {bundle.cap_repr(record_id)}")
    return record_id


def value_hash(key: bytes, config_entry_id: str, text: str) -> str:
    """``valueHash = entry_hmac(key, r["id"], text)`` (§1.7.5)."""
    return entry_hmac(key, _record_id(config_entry_id), text)


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
        out.append(f"{tag}: bindValue breaks W1/W2/§1.7.2/§1.7.3")
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
        if not all(isinstance(a, str) for a in argv) or bundle.gate_argv(argv) != argv:
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
