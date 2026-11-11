"""The keyed, domain-separated hash of `-bundle-spec5.md` §2 (the generalisation of `-bundle-spec3.md` §1.7.5).

``entry_hmac(key, domain, value) = lowercase-hex(HMAC-SHA-256(key, UTF-8(domain) || 0x00 || UTF-8(NFC(value))))``.

This module only computes. It never reads, stores or logs a key: the system key comes from ``ccg.atrest.hmac_key``
and is handed in. Conformance: spec5 §9.1 (``tests/test_valuehash.py`` parses the vectors from the spec text).
"""
from __future__ import annotations

import hashlib
import hmac
import unicodedata

KEY_BYTES = 32            # `ccg.atrest.KEY_BYTES`; every spec5 §9 vector uses 32-byte keys


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
