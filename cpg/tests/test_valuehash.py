"""P1-07 — `ccg.valuehash.entry_hmac` and the per-entry hashes of `-bundle-spec3.md` §1.7.5.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-07. The expected hex values are parsed from the text
of `docs/plans/connected-code-graph-bundle-spec5.md` §9.1 (never copied here), and the test key is the one §9
publishes; both keys are constants that `ccg.atrest.hmac_key` refuses, so tests hand them to `entry_hmac` directly.
No test touches the real key area.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from pathlib import Path

import pytest

from ccg import atrest, ids
from ccg.valuehash import entry_hmac
from extractors.common import writer_rules as wr

CPG = Path(__file__).resolve().parent.parent
SPEC5 = (CPG.parent / "docs" / "plans" / "connected-code-graph-bundle-spec5.md").read_text(encoding="utf-8")

_KEY1 = bytes.fromhex(re.search(r"32-byte key `([0-9a-f]{64})`", SPEC5).group(1))
_ROW = re.compile(r"^\| (EH-\d+) \| `([^`]+)` \| (?:`([^`]*)`|\(empty\)) \| `([0-9a-f]{64})` \|$", re.M)
VECTORS = [(m.group(1), m.group(2), m.group(3) or "", m.group(4)) for m in _ROW.finditer(SPEC5)]

KEY = bytes(range(64, 96))          # a private 32-byte test key, not either published one
REPO, PATH = "repo-1", "falkor-chat/compose.yaml"


def _ref(key: bytes, domain: str, value: str) -> str:
    """The construction as §1.7.5 writes it, recomputed with the stdlib."""
    msg = domain.encode() + b"\x00" + unicodedata.normalize("NFC", value).encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def test_spec_vectors_are_parsed_from_the_text():
    assert len(VECTORS) == 32 and VECTORS[30][0] == "EH-31"
    assert _KEY1 == bytes(range(32))


@pytest.mark.parametrize("name,domain,value,expected", VECTORS, ids=[v[0] for v in VECTORS])
def test_entry_hmac_reproduces_spec5_section_9_1(name, domain, value, expected):
    assert entry_hmac(_KEY1, domain, value) == expected


def test_eh31_is_the_id_domain_form_of_value_hash():
    eh31 = next(v for v in VECTORS if v[0] == "EH-31")
    assert eh31[1:3] == ("0123456789abcdef0123456789abcdef", "8200")
    assert wr.value_hash(_KEY1, eh31[1], eh31[2]) == eh31[3]


def test_bind_key_check_is_the_empty_value_under_the_keycheck_tag():
    eh30 = next(v for v in VECTORS if v[0] == "EH-30")
    assert atrest.bind_key_check(_KEY1) == eh30[3] == entry_hmac(_KEY1, atrest.BIND_KEYCHECK_TAG, "")


def test_value_is_nfc_normalised_before_hashing():
    composed, decomposed = "café", "café"
    assert composed != decomposed
    assert entry_hmac(KEY, "d", composed) == entry_hmac(KEY, "d", decomposed) == _ref(KEY, "d", composed)


def test_normalisation_is_nfc_not_a_compatibility_fold():
    # NFKC would fold the ligature and the fullwidth digits to "fi" and "8200"; the spec says NFC
    for text in ("\ufb01", "\uff18\uff12\uff10\uff10"):
        assert entry_hmac(KEY, "d", text) == _ref(KEY, "d", text)
        assert entry_hmac(KEY, "d", text) != entry_hmac(KEY, "d", unicodedata.normalize("NFKC", text))


def test_domain_separates_and_is_not_concatenated_with_the_value():
    # the 0x00 separator: ("ab", "c") and ("a", "bc") would collide without it
    assert entry_hmac(KEY, "ab", "c") != entry_hmac(KEY, "a", "bc")
    assert entry_hmac(KEY, "ab", "c") == _ref(KEY, "ab", "c")


def test_the_key_matters():
    assert entry_hmac(KEY, "d", "v") != entry_hmac(KEY[:-1] + b"\x01", "d", "v")


@pytest.mark.parametrize("domain", ["", "a\x00b", "café"])
def test_domain_must_be_non_empty_ascii_without_nul(domain):
    with pytest.raises(ValueError, match="domain"):
        entry_hmac(KEY, domain, "v")


def test_key_must_be_bytes_of_exactly_32():          # spec5 §2 / atrest.KEY_BYTES; every §9 vector uses 32 bytes
    with pytest.raises(TypeError, match="key"):
        entry_hmac("not-bytes", "d", "v")          # type: ignore[arg-type]
    for n in (0, 1, 31, 33, 64):
        with pytest.raises(ValueError, match="32 bytes"):
            entry_hmac(bytes(range(n)), "d", "v")
    entry_hmac(bytes(32), "d", "v")


def test_value_hash_hashes_the_text_exactly_unstripped():            # §1.7.5: the value as the format's unescaping yields it
    rid = ids.config_entry_id(REPO, PATH, "k")
    for text in (" 8200", "8200 ", "8200\n", "\t8200\r\n"):
        assert wr.value_hash(KEY, rid, text) == _ref(KEY, rid, text)
        assert wr.value_hash(KEY, rid, text) != wr.value_hash(KEY, rid, text.strip())
        assert wr.config_entry_value_hash(KEY, REPO, PATH, "k", text) == _ref(KEY, rid, text)


# --- §1.7.5 valueHash / defaultHash --------------------------------------------------------

def test_value_hash_uses_the_config_entrys_own_id_as_domain():
    rid = ids.config_entry_id(REPO, PATH, "services.falkordb.ports[0]")
    assert re.fullmatch(r"[0-9a-f]{32}", rid)
    assert wr.value_hash(KEY, rid, "6379") == _ref(KEY, rid, "6379")


def test_config_entry_value_hash_derives_the_id_from_the_triple():
    got = wr.config_entry_value_hash(KEY, REPO, PATH, "server.port", "8200")
    assert got == _ref(KEY, ids.config_entry_id(REPO, PATH, "server.port"), "8200")


def test_same_value_under_two_key_paths_gives_two_hashes():
    a = wr.config_entry_value_hash(KEY, REPO, PATH, "server.port", "8200")
    b = wr.config_entry_value_hash(KEY, REPO, PATH, "client.port", "8200")
    assert a != b
    # and the same entry hashed twice is stable (per-entry change detection)
    assert a == wr.config_entry_value_hash(KEY, REPO, PATH, "server.port", "8200")


def test_a_changed_value_changes_the_hash():
    assert (wr.config_entry_value_hash(KEY, REPO, PATH, "server.port", "8200")
            != wr.config_entry_value_hash(KEY, REPO, PATH, "server.port", "8201"))


def test_value_hash_refuses_a_domain_that_is_not_a_record_id():
    with pytest.raises(ValueError, match="32-hex"):
        wr.value_hash(KEY, "server.port", "8200")          # a keyPath is not an id
    with pytest.raises(ValueError, match="32-hex"):
        wr.default_hash(KEY, "ccg.bindParts.host.v1", "8200")    # nor is a tag


def test_default_hash_hashes_the_text_exactly_unstripped():
    lid = ids.launch_fact_id(REPO, PATH, "env-default", 0)
    for text in (" 8200", "8200\n"):
        assert wr.default_hash(KEY, lid, text) == _ref(KEY, lid, text) != wr.default_hash(KEY, lid, text.strip())


def test_default_hash_uses_the_launch_facts_own_id():
    lid = ids.launch_fact_id(REPO, PATH, "env-default", 0)
    assert wr.default_hash(KEY, lid, "8200") == _ref(KEY, lid, "8200")


def test_default_hash_is_per_launch_fact_not_per_value():
    l0, l1 = (ids.launch_fact_id(REPO, PATH, "env-default", n) for n in (0, 1))
    assert wr.default_hash(KEY, l0, "8200") != wr.default_hash(KEY, l1, "8200")


def test_a_config_entry_and_a_launch_fact_never_share_a_domain():
    # same text, one id from each family: no cross-record equality (SA-11, F-2)
    ce = ids.config_entry_id(REPO, PATH, "k")
    lf = ids.launch_fact_id(REPO, PATH, "env-default", 0)
    assert wr.value_hash(KEY, ce, "x") != wr.default_hash(KEY, lf, "x")


# --- the key is read only through ccg.atrest.hmac_key --------------------------------------

def test_system_key_delegates_to_atrest_hmac_key(tmp_path):
    keys = tmp_path / "keys"
    got = wr.system_key("demo", keys_root=keys)
    assert got == atrest.hmac_key("demo", keys)
    assert len(got) == atrest.KEY_BYTES and got not in (_KEY1, bytes(range(32, 64)))


@pytest.mark.parametrize("mod", ["ccg/valuehash.py", "extractors/common/writer_rules.py"])
def test_neither_module_reads_a_key_file_itself(mod):
    src = (CPG / mod).read_text(encoding="utf-8")
    for needle in ("open(", "read_bytes", "read_text", "environ", "expanduser", "read_private"):
        assert needle not in src, f"{mod} touches key storage ({needle}); only ccg.atrest.hmac_key may"
    assert not re.search(r"\.hmac(?!_key)", src), f"{mod} names a key file"
