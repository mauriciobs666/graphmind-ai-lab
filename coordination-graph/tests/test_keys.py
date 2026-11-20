import hashlib

import pytest

from coordgraph.keys import KeyHygieneError, norm_value, safe_key

# Note 4.1's crashing inputs: leading/trailing C isspace(), and over 4,096 bytes.
EDGE_INPUTS = ["set ", " a", "a\t", "a\n", "a\r", "a\v", "a\f", "\tkey\n", "  both  "]


@pytest.mark.parametrize("raw", EDGE_INPUTS)
def test_safe_key_strips_whitespace_edges(raw):
    out = safe_key(raw)
    assert out == raw.strip()
    assert not out[0].isspace() and not out[-1].isspace()


@pytest.mark.parametrize("raw", EDGE_INPUTS)
def test_norm_value_strips_whitespace_edges(raw):
    assert norm_value(raw) == raw.strip()


def test_safe_key_hashes_over_512_bytes_and_never_passes_a_4096_byte_value_through():
    long = "x" * 5000
    out = safe_key(long)
    assert out == "h:" + hashlib.sha256(long.encode()).hexdigest()
    assert len(out.encode()) <= 512
    # exactly at the limit stays as is, one byte over is hashed
    assert safe_key("y" * 512) == "y" * 512
    assert safe_key("y" * 513).startswith("h:")


def test_safe_key_counts_bytes_not_characters():
    two_byte = "é" * 300  # 300 chars, 600 bytes
    assert safe_key(two_byte).startswith("h:")


def test_safe_key_joins_stripped_parts():
    assert safe_key(" docs/a.md ", " U3~2\n") == "docs/a.md#U3~2"


def test_safe_key_strips_after_joining_so_no_whitespace_edge_survives():
    assert safe_key("a", " ") == "a#"
    assert not safe_key("a ", "b\t").endswith("\t")


@pytest.mark.parametrize("space", [" ", " "])
def test_nbsp_and_em_space_are_safe_controls_in_the_middle(space):
    value = f"a{space}b"
    assert safe_key(value) == value
    assert norm_value(value) == value


def test_norm_value_never_hashes_or_truncates():
    long = "z" * 5000
    assert norm_value(long) == long


def test_safe_key_requires_a_part():
    with pytest.raises(KeyHygieneError):
        safe_key()


def test_two_part_key_over_the_limit_hashes_the_whole_joined_string_not_the_path_alone():
    path, unit = "docs/plans/x-coordination.md", "U1 " + "é" * 300
    joined = f"{path}#{unit}"
    expected = "h:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()
    assert safe_key(path, unit) == expected
    assert safe_key(path, unit) != safe_key(path, unit + "b")
    assert safe_key(path, unit) != "h:" + hashlib.sha256(path.encode()).hexdigest()
