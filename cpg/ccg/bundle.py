"""Fact-bundle JSONL: read, validate and write with deterministic order.

Spec: ``docs/plans/connected-code-graph-bundle-spec3.md`` (v1.1) read with ``-spec4`` (spec4 §0)
— §2 (envelope and header), §5
(records), §7 (reason codes), §8 (determinism). The record shapes live in
``ccg/schemas/bundle-v1.schema.json``, the contract; this module adds what JSON Schema cannot
say: line framing and canonical bytes (§8.1), record order and duplicate ids (§8.3), id
recomputation through ``ccg.ids`` (§3.2), in-bundle references (§5.0), and the cross-record
rules of §2.2 and §5.

Public API: ``loads``/``read_bundle`` (parse + validate), ``validate`` (records in file
order), ``dumps``/``write_bundle`` (sort + validate + serialize), ``canonical_line``.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import unicodedata
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import jsonschema

from ccg import ids

SCHEMA_PATH = Path(__file__).parent / "schemas" / "bundle-v1.schema.json"

#: §8.3: the header, then the 19 record types in file order.
RECORD_TYPES: tuple[str, ...] = (
    "header", "file", "symbol", "param", "ret", "cond", "callsite", "valueSet", "constant",
    "data", "controlledBy", "decorator", "fieldAccess", "import", "dependency", "testMarker",
    "configEntry", "launchFact", "schemaEntry", "secretSighting",
)
_ORDER = {t: i for i, t in enumerate(RECORD_TYPES)}

#: spec3 §3.3's file extensions of the ``javascript`` token (TS and JS are one token), the
#: ``<ext>`` of spec4 §1's ``*.test.<ext>`` / ``*.spec.<ext>`` test pattern.
JS_TEST_EXTENSIONS: tuple[str, ...] = ("ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs")


class BundleError(ValueError):
    """A bundle (or a record in it) violates the bundle-v1 spec."""


@cache
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@cache
def _validator() -> jsonschema.protocols.Validator:
    schema = load_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


# --- serialization (§8.1) -------------------------------------------------------------------

def _canon(value: object) -> bytes:
    """§3.1's canonical JSON: sorted keys, ``,``/``:`` with no spaces, non-ASCII literal."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def canonical_line(record: dict) -> bytes:
    """One record's line (§8.1): its canonical JSON plus ``\\n``."""
    return _canon(record) + b"\n"


def _sort_key(record: dict) -> tuple[int, str]:
    t = record.get("type")
    if t not in _ORDER:
        raise BundleError(f"unknown record type {t!r}")
    return (_ORDER[t], record.get("id", ""))


def dumps(records: Iterable[dict]) -> bytes:
    """Sort ``records`` into §8.3 order, validate them, and return the bundle's bytes."""
    recs = list(records)
    for r in recs:
        if isinstance(r, dict) and r.get("type") == "secretSighting":
            raise BundleError("v1 writers must not emit secretSighting (§5.19)")
    ordered = sorted(recs, key=_sort_key)
    validate(ordered)
    return b"".join(canonical_line(r) for r in ordered)


def write_bundle(path: str | os.PathLike, records: Iterable[dict]) -> None:
    """Write a bundle file atomically (an ``O_EXCL`` 0600 temp file + rename), deterministic bytes.

    Custody of the parent directory is ``ccg.atrest``'s job, not checked here. A crash between create and rename
    leaves a hidden 0600 ``.<name>.<hex>.tmp`` (complete, already-gated bytes; never a ``*.bundle.jsonl``); leftovers
    accumulate until the refresh sweeps them (owner: the P1 architect's call), and a later write never collides with
    one (64 random bits in the name, and ``O_EXCL``). The temp name is 22 characters longer than the target's (the old ``<name>.tmp``: 4)."""
    data = dumps(records)
    target = Path(path)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    # Defence in depth: a caller (or an umask) cannot widen a bundle, and a planted name is never written through.
    # (O_NOFOLLOW is redundant with O_EXCL, which already refuses a symlink at the name.)
    created = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        created = True                       # never unlink a name we did not create (O_EXCL refused it)
        try:
            f = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)
        os.replace(tmp, target)
    except BaseException:
        if created:
            tmp.unlink(missing_ok=True)
        raise


# --- parsing --------------------------------------------------------------------------------

def _no_dup_keys(pairs: list[tuple[str, object]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise BundleError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _no_float(text: str) -> object:
    raise BundleError(f"numbers are integers only (§8.1), got {text}")


def loads(data: bytes) -> list[dict]:
    """Parse and validate a bundle's bytes; return its records in file order."""
    if data.startswith(b"\xef\xbb\xbf"):
        raise BundleError("a bundle has no BOM (§8.1)")
    if b"\r" in data:
        raise BundleError("a bundle has no \\r (§8.1)")
    if not data:
        raise BundleError("empty bundle: line 1 must be the header (§2.1)")
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise BundleError("a bundle ends with exactly one \\n (§8.1)")
    records = []
    for n, raw in enumerate(data[:-1].split(b"\n"), start=1):
        try:
            rec = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_no_dup_keys,
                parse_float=_no_float,
                parse_constant=_no_float,
            )
        except BundleError as e:
            raise BundleError(f"line {n}: {e}") from None
        except ValueError as e:  # UnicodeDecodeError and JSONDecodeError
            raise BundleError(f"line {n}: not UTF-8 JSON: {e}") from None
        if not isinstance(rec, dict):
            raise BundleError(f"line {n}: a record is a JSON object")
        if canonical_line(rec) != raw + b"\n":
            raise BundleError(f"line {n}: not the canonical JSON of its record (§8.1)")
        records.append(rec)
    validate(records)
    return records


def read_bundle(path: str | os.PathLike) -> list[dict]:
    return loads(Path(path).read_bytes())


# --- validation -----------------------------------------------------------------------------

def validate(records: list[dict]) -> None:
    """Validate records given in file order. Raises ``BundleError`` on the first violation."""
    if not records or not isinstance(records[0], dict) or records[0].get("type") != "header":
        raise BundleError("line 1 must be the header record (§2.1)")
    for n, rec in enumerate(records, start=1):
        if n > 1 and isinstance(rec, dict) and rec.get("type") == "header":
            raise BundleError(f"line {n}: only line 1 is a header (§2.1)")
        _check_values(n, rec)
        _check_schema(n, rec)
    _check_order(records)
    _Semantics(records).check()


def _check_schema(n: int, rec: object) -> None:
    err = jsonschema.exceptions.best_match(_validator().iter_errors(rec))
    if err is not None:
        where = "/".join(str(p) for p in err.absolute_path) or "<record>"
        raise BundleError(f"line {n}: schema: {where}: {err.message}")


#: §8.1: identity fields are stripped as well as NFC (every ``*Id`` key counts as one).
_IDENTITY_KEYS = frozenset({"id", "qname", "path", "keyPath", "name", "calleeName"})


def _check_values(n: int, value: object, key: str = "") -> None:
    """§8.1: integers only (JSON Schema's ``integer`` admits ``1.0``), every string NFC,
    identity strings stripped."""
    if isinstance(value, float):
        raise BundleError(f"line {n}: {key}: numbers are integers only (§8.1), got {value!r}")
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise BundleError(f"line {n}: {key}: string is not NFC (§8.1)")
        if (key in _IDENTITY_KEYS or key.endswith("Id")) and value.strip() != value:
            raise BundleError(f"line {n}: {key}: identity field must be stripped (§8.1)")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_values(n, v, k)
    elif isinstance(value, list):
        for v in value:
            _check_values(n, v, key)


def _check_order(records: list[dict]) -> None:
    """§8.3: the fixed type order, each block by id; equal ids within a type are an error."""
    prev = None
    for n, rec in enumerate(records[1:], start=2):
        key = _sort_key(rec)
        if prev is not None:
            if key == prev:
                raise BundleError(f"line {n}: duplicate {rec['type']} id {rec['id']} (§8.3)")
            if key < prev:
                raise BundleError(f"line {n}: record out of §8.3 order")
        prev = key


# Access paths (BS2 §1.1 replaces spec §5.1's grammar), parsed for the cross-record rules.
#: ASCII only: the class on which Python's re and ECMA-262 agree (BS2 §1.1).
_IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
# Alternation order matters: `recv`/`ret` are tried before `r`.
_HEAD = re.compile(r"a(?:0|[1-9][0-9]*)|p(?:0|[1-9][0-9]*)|recv|ret|r|kw")
_FIELD = re.compile(r"\." + _IDENT + r'|\[(?P<q>"(?:[^"\\]|\\.)*")\]')


def _path_head(n: int, path: str, what: str) -> str:
    """The head token of an access path (``a0``, ``p1``, ``recv``, ``kw.x``, ``ret``, ``r``,
    or ``""``), after checking that every quoted field has §5.1's one encoding."""
    if path == "":
        return ""
    m = _HEAD.match(path)
    if not m:
        raise BundleError(f"line {n}: {what}: not an access path: {path!r}")
    head, pos = m.group(0), m.end()
    if head == "kw" and not _FIELD.match(path, pos):
        raise BundleError(f"line {n}: {what}: the kw head is kw plus one field (BS2 §1.1)")
    while pos < len(path):
        f = _FIELD.match(path, pos)
        if not f:
            raise BundleError(f"line {n}: {what}: bad field segment in {path!r}")
        if f.group("q") is not None:
            text = f.group("q")
            key = json.loads(text)
            if re.fullmatch(_IDENT, key) or json.dumps(key, ensure_ascii=False) != text:
                raise BundleError(
                    f"line {n}: {what}: field {text} must be quoted only when it is not an "
                    "ASCII identifier, with canonical JSON escaping (BS2 §1.1)")
        pos = f.end()
    return head


def _use_kind(head: str) -> str:
    """§5.10: the point kind a value lands on, from a use path's head."""
    return {"": "cond", "ret": "ret"}.get(head, "callsite")


def _def_kind(head: str) -> tuple[str, int | None]:
    """§5.10: the point kind (and param index) a value comes from, from a def path's head."""
    if head == "r":
        return "callsite", None
    if head == "recv":
        return "param", -1
    return "param", int(head[1:])


def _strictly_sorted(keys: list) -> bool:
    return all(a < b for a, b in zip(keys, keys[1:]))


def _pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# === BS2 gates (docs/plans/connected-code-graph-bundle-spec3.md, which consolidates BS2) ====
# One implementation each of the token screen (§1.7.1), the name rule (§1.7.3), the bindValue
# value grammar (§1.7.2) and the argv gate (§1.6.1). The writer and the validator share them:
# a record's argv is valid iff it is a fixed point of ``gate_argv`` (§1.6.1 "the emitted
# sequence decides").

# --- §1.7.1 the token screen ---------------------------------------------------------------
_RUN = re.compile(r"[A-Za-z0-9]+")
_HEXCH = frozenset("0123456789abcdefABCDEF")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_T5_PREFIXES = ("xoxa-", "xoxb-", "xoxp-", "xoxr-", "xoxs-", "ghp_", "gho_", "ghu_", "ghs_",
                "ghr_", "github_pat_", "glpat-", "sk-", "sk_live_", "sk_test_", "rk_live_",
                "pk_live_", "pk_test_", "AKIA", "ASIA", "AIza", "hvs.", "eyJ")
_T6_SPAN = re.compile(r"[A-Za-z0-9._:+,-]+")
_T6_SEPS = re.compile(r"[._:+,-]")


def _transitions(core: str) -> int:
    return sum(1 for a, b in zip(core, core[1:]) if a.isdigit() != b.isdigit())


def screen_fails(text: str) -> bool:
    """§1.7.1: True iff ``text`` fails the token screen (T1–T6), as one string."""
    for run in _RUN.findall(text):
        letters = any(c.isalpha() for c in run)
        digits = any(c.isdigit() for c in run)
        if len(run) >= 16 and letters and digits:                      # T1
            return True
        if len(run) >= 24:                                             # T2
            return True
        if len(run) >= 12 and all(c in _HEXCH for c in run):           # T3
            return True
    if _UUID.search(text):                                             # T4
        return True
    if len(text) >= 20:                                                # T5
        starts = [0] + [i + 1 for i, c in enumerate(text) if c in "=/:@,"]
        if any(text.startswith(p, i) for i in starts for p in _T5_PREFIXES):
            return True
    for span in _T6_SPAN.findall(text):                                # T6
        core = _T6_SEPS.sub("", span)
        if len(core) >= 16 and (all(c in _HEXCH for c in core) or _transitions(core) >= 6):
            return True
    return False


# --- §1.7.3 the name rule ------------------------------------------------------------------
#: 25 words matched as substrings of the folded segment: the spec's 15, then the 10 moved.
SUBSTRING_WORDS = ("pass", "pwd", "secret", "token", "key", "cred", "auth", "cookie",
                   "session", "private", "cert", "dsn", "conn", "hook", "sig", "pw", "psk",
                   "jwt", "hmac", "totp", "mnemonic", "bearer", "gpg", "pgp", "kms")
#: 11 words matched as whole folded tokens.
WHOLE_TOKEN_WORDS = ("salt", "pepper", "seed", "pin", "otp", "pat", "license", "nonce",
                     "encrypt", "ssh", "vault")


def fold(text: str) -> str:
    """NFKD, drop combining marks (Mn), NFKC, ``casefold`` (Python 3.12)."""
    d = unicodedata.normalize("NFKD", text)
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFKC", d).casefold()


def name_tokens(segment: str) -> list[str]:
    """Split a segment before folding: on ``_ - .``; at lower→upper; before the last capital
    of a capital run followed by a lowercase letter; at every letter↔digit boundary."""
    out = []
    for part in re.split(r"[_.-]", segment):
        cur = ""
        for i, c in enumerate(part):
            if cur:
                prev, nxt = part[i - 1], part[i + 1] if i + 1 < len(part) else ""
                if ((prev.islower() and c.isupper())
                        or (prev.isupper() and c.isupper() and nxt.islower())
                        or (prev.isalpha() and c.isdigit())
                        or (prev.isdigit() and c.isalpha())):
                    out.append(cur)
                    cur = ""
            cur += c
        if cur:
            out.append(cur)
    return out


def name_fails(segment: str) -> bool:
    """§1.7.3: True iff one segment fails the name rule."""
    folded = fold(segment)
    if any(w in folded for w in SUBSTRING_WORDS):
        return True
    return any(fold(t) in WHOLE_TOKEN_WORDS for t in name_tokens(segment))


def _has_context_word(value: str) -> bool:
    """§1.6.1 "Context": a bare value containing one of the 25 substring words."""
    folded = fold(value)
    return any(w in folded for w in SUBSTRING_WORDS)


# --- §1.7.2 the value grammar --------------------------------------------------------------
_SCHEMES = ("http", "https", "ws", "wss", "redis", "rediss", "postgres", "postgresql",
            "mysql", "mongodb", "amqp", "amqps", "nats", "grpc")


def _port(s: str) -> bool:
    return bool(re.fullmatch(r"[1-9][0-9]{0,4}", s)) and int(s) <= 65535


def _ipv4(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(
        re.fullmatch(r"0|[1-9][0-9]{0,2}", p) and int(p) <= 255 for p in parts)


def _ipv6(s: str) -> bool:
    """RFC 5952 canonical text, as Python 3.12 writes it: no '.', no '%' (§1.7.2)."""
    if not (2 <= len(s) <= 39 and re.fullmatch(r"[0-9a-f:]+", s)):
        return False
    try:
        return str(ipaddress.IPv6Address(s)) == s
    except ValueError:
        return False


def _label(s: str, screen: bool) -> bool:
    return (bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", s))
            and not (screen and screen_fails(s)))


def _dnsname(s: str, screen: bool) -> bool:
    labels = s.split(".")
    return (len(s) <= 253 and len(labels) >= 2 and all(_label(x, screen) for x in labels)
            and bool(re.fullmatch(r"[a-z]{2,63}", labels[-1])))


def _host(s: str, screen: bool) -> bool:
    if s in ("localhost", "host.docker.internal") or _ipv4(s):
        return True
    if s.startswith("[") and s.endswith("]"):
        return _ipv6(s[1:-1])
    if _label(s, screen):
        return any(c.isalpha() for c in s)  # a single-label host holds a letter
    return _dnsname(s, screen)


def _hostport(s: str, screen: bool) -> bool:
    i = s.rfind(":")
    return i > 0 and _host(s[:i], screen) and _port(s[i + 1:])


def _url(s: str, screen: bool) -> bool:
    m = re.fullmatch(r"([a-z]+)://([^/]*)(.*)", s, re.S)
    if not m or m.group(1) not in _SCHEMES:
        return False
    auth, path = m.group(2), m.group(3)
    if not (_host(auth, screen) or _hostport(auth, screen)):
        return False
    if path in ("", "/"):
        return True
    seg = re.fullmatch(r"/([a-z0-9_-]{1,32})/?", path)
    return bool(seg) and not (screen and screen_fails(seg.group(1)))


def _portmap(s: str) -> bool:
    """§1.6.1 (v6): ``[ipv4 ":"] port [":" port] ["/" ("tcp" / "udp")]``."""
    m = re.fullmatch(r"(?:([0-9.]+):)?([0-9]+)(?::([0-9]+))?(?:/(?:tcp|udp))?", s)
    return bool(m) and (m.group(1) is None or _ipv4(m.group(1))) and _port(m.group(2)) and (
        m.group(3) is None or _port(m.group(3)))


def value_grammar_ok(value: str) -> bool:
    """The public form of §1.7.2's value grammar, for the writer rules."""
    return _bind_grammar(value)


def _bind_grammar(value: str) -> bool:
    """§1.7.2 ``bindValue = port / host / ipv6 / hostport / url``, every label and segment
    screened on its own (§1.7.1's scope rule), fewer than 512 characters."""
    return len(value) < 512 and (_port(value) or _host(value, True) or _ipv6(value)
                                 or _hostport(value, True) or _url(value, True))


_KEYPATH_PART = re.compile(r'\[(?:(?P<i>0|[1-9][0-9]*)|(?P<q>"(?:[^"\\]|\\.)*"))\]'
                           r"|(?P<n>[^.\[]+)|\.")


def cap_repr(text: object, limit: int = 80) -> str:
    """``text`` as an escaped, ASCII-only repr of at most ``limit`` characters of it, for error messages: repo-derived
    text is never echoed whole (it may be the secret-bearing part of an entry)."""
    t = str(text)
    r = ascii(t[:limit])
    return r + ("..." if len(t) > limit else "")


def _keypath_segments(key_path: str) -> list[tuple[str, str]]:
    """§5.17's ``keyPath`` as (kind, text) pairs, kind ``name`` or ``index``."""
    out, pos = [], 0
    while pos < len(key_path):
        m = _KEYPATH_PART.match(key_path, pos)
        if not m:
            raise BundleError(f"keyPath {cap_repr(key_path)} does not parse (§5.17)")
        if m.group("i") is not None:
            out.append(("index", m.group("i")))
        elif m.group("q") is not None:
            try:
                out.append(("name", json.loads(m.group("q"))))
            except ValueError:
                raise BundleError(f"keyPath {cap_repr(key_path)} has a bad quoted segment (§5.17)") from None
        elif m.group("n") is not None:
            out.append(("name", m.group("n")))
        pos = m.end()
    return out


def bind_value_allowed(key_path: str, value: str) -> bool:
    """§1.7: the value grammar, the name rule on every named ``keyPath`` segment, and W1."""
    segs = _keypath_segments(key_path)
    if any(k == "name" and t in ("args", "command", "entrypoint") and i + 1 < len(segs)
           and segs[i + 1][0] == "index" for i, (k, t) in enumerate(segs)):
        return False                                                   # W1
    if any(k == "name" and name_fails(t) for k, t in segs):
        return False                                                   # §1.7.3
    return _bind_grammar(value)


# --- §1.6.1 the argv gate ------------------------------------------------------------------
REDACTED = "<redacted>"
_FORCING = ("-" + REDACTED, "--" + REDACTED)
_SENTINELS = frozenset((REDACTED, *_FORCING))
_VALUE_CHARS = re.compile(r"[A-Za-z0-9._/:@+=,\[\]-]+")
_ALLOW_EXACT = frozenset(("-jar", "-cp", "-classpath", "-port", "-addr", "-host", "-listen",
                          "-bind", "-ea", "-da", "-server", "-client"))
_USER_TOKENS = frozenset(("user", "login", "basic", "account", "principal", "identity"))


def _allow_listed_name(el: str) -> str | None:
    """Class 2's closed single-dash allow-list: the entry's name, or None."""
    if el in _ALLOW_EXACT or re.fullmatch(r"-X(?:mx|ms|ss)[0-9]+[kKmMgG]?", el):
        name = el[1:]
    elif m := re.fullmatch(r"-XX:[+-]([A-Z][A-Za-z0-9]{0,63})", el):
        name = m.group(1)
    elif re.fullmatch(r"-verbose:(?:gc|class|jni|module)", el):
        name = "verbose"
    else:
        return None
    if screen_fails(el) or any(name_fails(p) for p in re.split(r"[.:+-]", el)):
        return None
    return name


def _classify(el: str) -> tuple[int, str]:
    """§1.6.1's element classes 1–5, tested in order; with the flag's name for class 2."""
    if el in _SENTINELS:
        return 1, ""
    if m := re.fullmatch(r"--([A-Za-z][A-Za-z0-9._-]*)", el):
        return 2, m.group(1)
    if re.fullmatch(r"-[A-Za-z0-9]", el):
        return 2, el[1]
    if (name := _allow_listed_name(el)) is not None:
        return 2, name
    if "=" in el:
        return 3, ""
    if el.startswith("-") and not el.startswith("--") and len(el) >= 3:
        return 4, ""
    return 5, ""


def _address_shaped(v: str, screen: bool) -> bool:
    return (_ipv4(v) or _ipv6(v) or _hostport(v, screen) or _dnsname(v, screen)
            or _url(v, screen) or _portmap(v))


def _plain_ok(v: str) -> bool:
    """(v1) shape and (v2) screen: what ``command`` must meet, and every value's base."""
    if not (1 <= len(v) <= 128 and _VALUE_CHARS.fullmatch(v)):
        return False
    if ("[" in v or "]" in v) and not (_hostport(v, False) or _url(v, False)):
        return False
    if _address_shaped(v, False):
        return _address_shaped(v, True)
    return not screen_fails(v)


def _value_ok(v: str, context: list[str]) -> bool:
    """(v1)–(v8) for a value, an assignment's value or an attached value."""
    if not _plain_ok(v):
        return False
    if any(name_fails(seg) for n in context for seg in n.split(".")):   # (v3)
        return False
    if "@" in v and ":" in v[:v.rindex("@")]:                          # (v4)
        return False
    if "://" in v and not _url(v, True):                               # (v5)
        return False
    if any(len(n) == 1 for n in context) and not (                     # (v6)
            _port(v) or _ipv4(v) or _dnsname(v, True) or _hostport(v, True) or _portmap(v)):
        return False
    if ":" in v and any(fold(t) in _USER_TOKENS                        # (v7)
                        for n in context for t in name_tokens(n)):
        return False
    if ":" in v and "://" not in v and not _address_shaped(v, False):  # (v8)
        if any(name_fails(seg) for seg in v.split(":", 1)[0].split(".")):
            return False
    return True


def gate_argv(argv: list[str]) -> list[str]:
    """§1.6.1: write ``argv`` (argv[0] then ``args``) through the argv gate.

    A sentinel already in the input is kept as written, so the gate is idempotent on its own
    output; the validator accepts an emitted argv iff it is a fixed point. argv[0] is gated by
    (v1)/(v2) alone and supplies no context: context comes from the preceding ``args``
    element."""
    if not argv:
        return []
    out = [argv[0] if argv[0] in _SENTINELS or _plain_ok(argv[0]) else REDACTED]
    context: list[str] = []
    forced = False
    for el in argv[1:]:
        cls, name = _classify(el)
        if cls == 1:
            out.append(el)
            context, forced = [], el in _FORCING
            continue
        supplier = cls in (2, 4) or (cls == 5 and _has_context_word(el))
        sentinel = ("--" if el.startswith("--") else "-") + REDACTED if supplier else REDACTED
        if forced:
            ok = False
        elif cls == 2:
            ok = len(el) <= 128 and not screen_fails(el)
        elif cls == 3:
            n, v = el.split("=", 1)
            n = n.lstrip("-")
            ok = (bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", n)) and not screen_fails(n)
                  and _value_ok(v, [n, *context]))
        elif cls == 4:
            ok = (_value_ok(el[2:], [el[1]]) and not name_fails(el[1:])
                  and not screen_fails(el))
        else:
            ok = _value_ok(el, context)
        if ok:
            out.append(el)
            if cls == 2:
                context, forced = [name], name_fails(name)
            else:
                context = [el] if cls == 5 and supplier else []
                forced = False
        else:
            out.append(sentinel)
            context, forced = [], sentinel in _FORCING
    return out


_POINT_TYPES = ("param", "ret", "cond", "callsite")


class _Semantics:
    """The cross-record rules of §2.2, §3.2 and §5 over one ordered, schema-valid bundle."""

    def __init__(self, records: list[dict]):
        self.header = records[0]
        self.lines = list(enumerate(records, start=1))
        self.by_type: dict[str, dict[str, tuple[int, dict]]] = {t: {} for t in RECORD_TYPES}
        for n, r in self.lines[1:]:
            self.by_type[r["type"]][r["id"]] = (n, r)
        unit = self.header["unit"]
        self.repo, self.root = unit["repoId"], unit["rootPath"]
        self.vendored = set(self.header["vendoredDirs"])

    def check(self) -> None:
        self._header()
        # Pass 1: each record's own repoId and id, so a bad id is reported as itself and not as a
        # dangling reference from some later-checked record.
        for n, r in self.lines[1:]:
            if "repoId" in r:
                self._same_repo(n, r)
            fn = self._ID_FNS.get(r["type"])
            if fn:
                self._id(n, r, lambda: fn(self, r))
        # Pass 2: references and the cross-record rules.
        for n, r in self.lines[1:]:
            getattr(self, "_" + r["type"])(n, r)
            for field in ("path", "manifestPath", "resolvedPath"):
                if r["type"] != "secretSighting" and field in r and self._is_vendored(r[field]):
                    raise BundleError(
                        f"line {n}: {field} {r[field]!r} is vendored: a vendored file yields "
                        "no record (§5.0)")
        self._ordinals("import", lambda r: r["fileId"])
        self._ordinals("launchFact", lambda r: (r["path"], r["kind"]))

    #: §3.2/§5 tuples, through ``ccg.ids``. ``constant`` needs its file's lang (pass 2).
    #: ``import``/``launchFact`` carry their tuple's ``ordinal`` (BS2 §1.5, §1.6).
    _ID_FNS = {
        "file": lambda s, r: ids.file_id(r["repoId"], r["path"]),
        "symbol": lambda s, r: ids.symbol_id(r["lang"], r["repoId"], r["qname"],
                                             r.get("signature")),
        "param": lambda s, r: ids.point_id(r["symbolId"], "param", r["index"]),
        "ret": lambda s, r: ids.point_id(r["symbolId"], "ret", r["ordinal"]),
        "cond": lambda s, r: ids.point_id(r["symbolId"], "cond", r["ordinal"]),
        "callsite": lambda s, r: ids.callsite_id(r["symbolId"], r["calleeName"], r["ordinal"]),
        "valueSet": lambda s, r: ids.value_set_id(r["callSiteId"], r["arg"]),
        "data": lambda s, r: ids.data_id(r["use"], r["def"], r["toPath"], r["fromPath"]),
        "controlledBy": lambda s, r: ids.controlled_by_id(r["point"], r["cond"]),
        "decorator": lambda s, r: ids.decorator_id(r["targetId"], r["ordinal"]),
        "fieldAccess": lambda s, r: ids.field_access_id(r["pointId"], r["path"], r["mode"],
                                                        r["receiverType"], r["field"]),
        "dependency": lambda s, r: ids.dependency_record_id(r["ecosystem"], r["package"],
                                                            r["manifestPath"], r["line"]),
        "testMarker": lambda s, r: ids.test_marker_id(r["symbolId"], r["framework"]),
        "configEntry": lambda s, r: ids.config_entry_id(r["repoId"], r["path"], r["keyPath"]),
        "schemaEntry": lambda s, r: ids.schema_entry_id(s.repo, r["path"], r["objectKey"]),
        "import": lambda s, r: ids.import_id(r["fileId"], r["ordinal"]),
        "launchFact": lambda s, r: ids.launch_fact_id(s.repo, r["path"], r["kind"],
                                                      r["ordinal"]),
        "secretSighting": lambda s, r: ids.sighting_id(s.repo, r["path"], r["commit"],
                                                       r["fingerprint"]),
    }

    # -- helpers --
    def _id(self, n: int, r: dict, expected_fn) -> None:
        try:
            expected = expected_fn()
        except (TypeError, ValueError) as e:
            raise BundleError(f"line {n}: {r['type']} id cannot be computed: {e}") from None
        if r["id"] != expected:
            raise BundleError(
                f"line {n}: {r['type']} id {r['id']} does not match its fields "
                f"(expected {expected}, §3.2)")

    def _ref(self, n: int, r: dict, field: str, *types: str) -> dict:
        target = r[field]
        for t in types:
            if target in self.by_type[t]:
                return self.by_type[t][target][1]
        raise BundleError(
            f"line {n}: {field} {target} names no {'/'.join(types)} record in this bundle (§5.0)")

    def _same_repo(self, n: int, r: dict) -> None:
        if r["repoId"] != self.repo:
            raise BundleError(f"line {n}: repoId {r['repoId']!r} is not the unit's {self.repo!r}")

    def _point(self, n: int, r: dict, field: str, kind: str, index: int | None = None) -> dict:
        p = self._ref(n, r, field, *_POINT_TYPES)
        if p["type"] != kind or (index is not None and p["index"] != index):
            want = kind if index is None else f"{kind} index {index}"
            raise BundleError(
                f"line {n}: {field} is a {p['type']}, but its path names a {want} (§5.1, §5.10)")
        return p

    def _sorted_values(self, n: int, values: list, what: str) -> None:
        if not _strictly_sorted([_canon(v) for v in values]):
            raise BundleError(f"line {n}: {what} must be sorted by canonical JSON, no repeats "
                              "(§5.8)")

    def _is_vendored(self, path: str) -> bool:
        if not self.vendored:
            return False
        rel = path
        if self.root != "." and path.startswith(self.root + "/"):
            rel = path[len(self.root) + 1:]
        return any(seg in self.vendored for seg in rel.split("/")[:-1])

    def _ordinals(self, rtype: str, group_key) -> None:
        """BS2 §1.5/§1.6: each group's ``ordinal``s are exactly 0..k-1, and their lines do not
        decrease in ordinal order. (The id is recomputed from the field in pass 1.)"""
        groups: dict = {}
        for n, r in self.by_type[rtype].values():
            groups.setdefault(group_key(r), []).append((r["ordinal"], r["line"], n))
        for members in groups.values():
            members.sort()
            if [o for o, _, _ in members] != list(range(len(members))):
                raise BundleError(f"line {members[0][2]}: {rtype} ordinals of a group must be "
                                  f"exactly 0..{len(members) - 1} (BS2 §1.5)")
            for (_, a, _), (_, b, n) in zip(members, members[1:]):
                if b < a:
                    raise BundleError(f"line {n}: {rtype} ordinals must follow source order")

    # -- header (§2.2) --
    def _header(self) -> None:
        h = self.header
        u = h["unit"]
        expected = ids.unit_id(u["repoId"], u["rootPath"], u["lang"])
        if h["unitId"] != expected:
            raise BundleError(f"line 1: unitId does not match unit (expected {expected})")
        actual = {t: len(self.by_type[t]) for t in RECORD_TYPES[1:]}
        if h["counts"] != actual:
            diff = {t: (h["counts"][t], actual[t]) for t in actual if h["counts"][t] != actual[t]}
            raise BundleError(f"line 1: counts do not match the records (header, actual): {diff}")
        if h["vendoredDirs"] != sorted(h["vendoredDirs"]):
            raise BundleError("line 1: vendoredDirs must be sorted (§2.2)")
        if h["history"] != "available":
            raise BundleError("line 1: history must be 'available' in P0 (§2.2)")
        self._skipped_files(h.get("skippedFiles", []))

    def _skipped_files(self, entries: list[dict]) -> None:
        """spec4 §1: ``header.skippedFiles`` is sorted by path (byte order), unique, under
        ``rootPath``, outside ``vendoredDirs`` and has no ``file`` record. Completeness (every
        source file that has neither) needs the file system and is not checked here."""
        mapped = {r["path"] for _, r in self.by_type["file"].values()}
        prefix = "" if self.root == "." else self.root + "/"
        previous = None
        for e in entries:
            path = e["path"]
            if previous is not None:
                if path == previous:
                    raise BundleError(f"line 1: skippedFiles path {path!r} is listed twice: "
                                      "paths are unique (spec4 §1)")
                if path.encode("utf-8") < previous.encode("utf-8"):
                    raise BundleError("line 1: skippedFiles must be sorted by path in byte "
                                      f"order ({path!r} follows {previous!r}, spec4 §1)")
            previous = path
            if not path.startswith(prefix):
                raise BundleError(f"line 1: skippedFiles path {path!r} is not under the unit's "
                                  f"rootPath {self.root!r} (spec4 §1)")
            if self._is_vendored(path):
                raise BundleError(f"line 1: skippedFiles path {path!r} is vendored: "
                                  "vendoredDirs explains it (spec4 §1)")
            if path in mapped:
                raise BundleError(f"line 1: skippedFiles path {path!r} has a file record "
                                  "(spec4 §1)")
            is_test = self._is_test_path(path)
            if (e["reason"] == "test") != is_test:
                raise BundleError(
                    f"line 1: skippedFiles path {path!r} "
                    + ("matches the closed test pattern, so its reason is 'test' (spec4 §1 "
                       "precedence)" if is_test else "does not match the closed test pattern, "
                       "so its reason cannot be 'test' (spec4 §1)"))

    #: spec4 §1's closed ``test`` pattern: JS/TS units only (``lang`` ``javascript``, whose
    #: extensions are spec3 §3.3's); any other language has none in v1.
    def _is_test_path(self, path: str) -> bool:
        if self.header["unit"]["lang"] != "javascript":
            return False
        rel = path[len(self.root) + 1:] if self.root != "." else path
        *dirs, base = rel.split("/")
        if any(d in ("e2e", "__tests__") for d in dirs):
            return True
        return any(base.endswith(f".{kind}.{ext}")
                   for kind in ("test", "spec") for ext in JS_TEST_EXTENSIONS)

    # -- records (§5) --
    def _file(self, n, r):
        if r["unitId"] != self.header["unitId"]:
            raise BundleError(f"line {n}: unitId is not the header's")

    def _symbol(self, n, r):
        f = self._ref(n, r, "fileId", "file")
        if f["path"] != r["path"]:
            raise BundleError(f"line {n}: path {r['path']!r} is not its file's {f['path']!r}")
        if f["lang"] != r["lang"]:
            raise BundleError(f"line {n}: lang {r['lang']!r} is not its file's {f['lang']!r}")
        norm = unicodedata.normalize("NFC", unicodedata.normalize("NFC", r["name"]).casefold())
        if r["nameNorm"] != norm.strip():
            raise BundleError(f"line {n}: nameNorm must be NFC(casefold(NFC(name))).strip() "
                              f"= {norm.strip()!r} (BS2 §1.2)")
        if not r["qname"].startswith(r["path"] + "::"):
            raise BundleError(f"line {n}: qname must start with path + '::' (§4.1)")
        if "parentId" in r:
            self._ref(n, r, "parentId", "symbol")
        if r["lineStart"] > r["lineEnd"]:
            raise BundleError(f"line {n}: lineStart is after lineEnd")
        regions = r.get("ownedRegions", [])
        if not _strictly_sorted(regions) or any(a > b for a, b in regions):
            raise BundleError(f"line {n}: ownedRegions must be sorted [start, end] pairs (§5.3)")
        if "lastChangedAt" not in r:
            raise BundleError(f"line {n}: lastChangedAt is required when history is "
                              "available (§5.3)")

    def _param(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _ret(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _cond(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _callsite(self, n, r):
        self._ref(n, r, "symbolId", "symbol")
        targets = r["targets"]
        if not _strictly_sorted([t["symbolId"] for t in targets]):
            raise BundleError(f"line {n}: targets must be sorted by symbolId (§5.6)")
        # Not checked: a target's path against vendoredDirs (§5.6 "never vendored code").
        # §5.0's test is relative to the *defining* unit's rootPath (P3-4), which a targets
        # entry does not carry; one path can be vendored for unit "." and first-party for a
        # unit rooted at third_party/ourlib/. An in-bundle target is covered by its own
        # symbol/file record's path check; an out-of-bundle one is the assembler's to judge.
        for t in targets:
            hit = self.by_type["symbol"].get(t["symbolId"])
            if hit and (hit[1]["qname"], hit[1].get("signature")) != (t["qname"],
                                                                      t.get("signature")):
                raise BundleError(f"line {n}: targets entry disagrees with symbol "
                                  f"{t['symbolId']}")

    def _valueSet(self, n, r):
        self._ref(n, r, "callSiteId", "callsite")
        _path_head(n, r["arg"], "arg")
        self._sorted_values(n, r["values"], "values")

    def _constant(self, n, r):
        f = self._ref(n, r, "fileId", "file")
        self._sorted_values(n, r["values"], "values")
        self._id(n, r, lambda: ids.constant_id(f["lang"], self.repo, r["qname"]))

    def _data(self, n, r):
        use = self._point(n, r, "use", _use_kind(_path_head(n, r["toPath"], "toPath")))
        kind, index = _def_kind(_path_head(n, r["fromPath"], "fromPath"))
        def_ = self._point(n, r, "def", kind, index)
        if use["symbolId"] != def_["symbolId"]:
            raise BundleError(f"line {n}: use and def must be in the same symbol (§5.10)")

    def _controlledBy(self, n, r):
        point = self._ref(n, r, "point", *_POINT_TYPES)
        cond = self._point(n, r, "cond", "cond")
        if r["point"] == r["cond"]:
            raise BundleError(f"line {n}: point equals cond, a CDG self-edge (BS2 §1.4)")
        if point["symbolId"] != cond["symbolId"]:
            raise BundleError(f"line {n}: point and cond must be in the same symbol (§5.11)")

    def _decorator(self, n, r):
        self._ref(n, r, "targetId", "symbol")
        args = r["args"]
        if not _strictly_sorted([a["arg"] for a in args]):
            raise BundleError(f"line {n}: args must be sorted by arg, no repeats (§5.12)")
        for a in args:
            _path_head(n, a["arg"], "args.arg")
            self._sorted_values(n, a["values"], "args.values")

    def _fieldAccess(self, n, r):
        head = _path_head(n, r["path"], "path")
        if r["mode"] == "read":
            self._point(n, r, "pointId", _use_kind(head))
        else:
            self._point(n, r, "pointId", *_def_kind(head))

    def _import(self, n, r):
        self._ref(n, r, "fileId", "file")
        if "symbolId" in r:
            self._ref(n, r, "symbolId", "symbol")
        keys = [(x["name"], 0, "") if "alias" not in x else (x["name"], 1, x["alias"])
                for x in r["names"]]
        if not _strictly_sorted(keys):
            raise BundleError(f"line {n}: names must be sorted by (name, alias), an absent "
                              "alias first (§5.14)")

    def _dependency(self, n, r):
        eco, pkg = r["ecosystem"], r["package"]
        if (eco == "PyPI" and pkg != _pep503(pkg)) or (eco == "NuGet" and pkg != pkg.lower()):
            raise BundleError(f"line {n}: package {pkg!r} is not {eco}-normalized (§5.15)")
        expected = ids.dep_id(eco, pkg)
        if r["depId"] != expected:
            raise BundleError(f"line {n}: depId does not match (expected {expected})")

    def _testMarker(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _configEntry(self, n, r):
        if "bindValue" in r and not bind_value_allowed(r["keyPath"], r["bindValue"]):
            raise BundleError(f"line {n}: bindValue {r['bindValue']!r} under keyPath "
                              f"{r['keyPath']!r} fails BS2 §1.7 (grammar, screen, name rule or "
                              "W1)")
        f = self.by_type["file"].get(ids.file_id(self.repo, r["path"]))
        if f and r["example"] != (f[1]["role"] == "example"):
            raise BundleError(f"line {n}: example must be copied from its file's role (§5.17)")

    def _launchFact(self, n, r):
        if r["kind"] == "command":
            argv = [r["command"], *r["args"]]
            written = gate_argv(argv)
            for i, (got, want) in enumerate(zip(argv, written)):
                if got != want:
                    where = "command" if i == 0 else f"args[{i - 1}]"
                    raise BundleError(f"line {n}: launchFact argv: {where} {got!r} must be "
                                      f"written {want!r} (BS2 §1.6.1)")
        if "default" in r and not (_bind_grammar(r["default"]) and not name_fails(r["envVar"])):
            raise BundleError(f"line {n}: launchFact default is present only when it passes "
                              "the value grammar and envVar the name rule (BS2 §1.6)")

    def _schemaEntry(self, n, r):
        pass  # id only (pass 1)

    def _secretSighting(self, n, r):
        pass  # reserved (P5): shape and id only; P5 fixes its enclosing end's reference rule
