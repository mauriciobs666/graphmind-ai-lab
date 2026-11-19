"""U6f.4a: `decisionmodels.manifest` matches every row of the byte-copied `fit-manifest.jsonl`
(`docs/plans/typed-decision-models4.md` §3.11, §5 row U6f.4a). The vector's sha256 values were
computed by hand, never by the code under test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from decisionmodels.manifest import (
    ManifestRefused,
    declare_manifest,
    parse_manifest,
    render_manifest,
    verify_manifest,
)

_VECTOR = Path(__file__).parent / "fixtures" / "decision-contract" / "fit-manifest.jsonl"
_ROWS = [json.loads(line) for line in _VECTOR.read_text(encoding="utf-8").splitlines()[1:]]
_ACCEPTED = [r for r in _ROWS if r["accept"]]
_REFUSED = [r for r in _ROWS if not r["accept"]]
_IDS = lambda r: r["case"]  # noqa: E731


def test_vector_has_rows_of_both_verdicts():
    assert len(_ACCEPTED) == 3 and len(_REFUSED) == 8


@pytest.mark.parametrize("row", _ACCEPTED, ids=_IDS)
def test_accepted_row_verifies_and_parses(row):
    raw = row["text"].encode("utf-8")
    assert verify_manifest(raw, row["sha256"], row["size"]) == tuple(row["entries"])
    assert parse_manifest(raw) == tuple(row["entries"])


@pytest.mark.parametrize("row", _ACCEPTED, ids=_IDS)
def test_accepted_row_declaration_is_computed(row):
    raw = row["text"].encode("utf-8")
    assert declare_manifest(raw) == {"sha256": row["sha256"], "size": row["size"]}
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]


@pytest.mark.parametrize("row", _ACCEPTED, ids=_IDS)
def test_accepted_row_is_written_byte_for_byte(row):
    assert render_manifest(row["entries"]) == row["text"].encode("utf-8")
    assert render_manifest(reversed(row["entries"])) == row["text"].encode("utf-8")


@pytest.mark.parametrize("row", _REFUSED, ids=_IDS)
def test_refused_row_names_its_rule(row):
    with pytest.raises(ManifestRefused) as excinfo:
        verify_manifest(row["text"].encode("utf-8"), row["sha256"], row["size"])
    assert excinfo.value.rule == row["rule"]
    assert row["refusal"] in str(excinfo.value)


def test_size_is_entry_count_not_bytes():
    raw = render_manifest(["a" * 64, "b" * 64])
    assert len(raw) == 130
    assert declare_manifest(raw)["size"] == 2


def test_render_refuses_malformed_and_collapses_repeats():
    with pytest.raises(ManifestRefused) as excinfo:
        render_manifest(["A" * 64])
    assert excinfo.value.rule == "malformed"
    assert render_manifest(["a" * 64, "a" * 64]) == ("a" * 64 + "\n").encode()
    assert render_manifest([]) == b""


def test_bool_size_is_not_a_count():
    raw = render_manifest(["a" * 64])
    with pytest.raises(ManifestRefused) as excinfo:
        verify_manifest(raw, hashlib.sha256(raw).hexdigest(), True)
    assert excinfo.value.rule == "wrong-size"


def test_non_utf8_and_bom_are_refused():
    with pytest.raises(ManifestRefused) as excinfo:
        parse_manifest(b"\xff\n")
    assert excinfo.value.rule == "not-utf8"
    with pytest.raises(ManifestRefused):
        parse_manifest(b"\xef\xbb\xbf" + b"a" * 64 + b"\n")


_BAD_SHAPES = ["a" * 63, "a" * 65, "a" * 64 + "x", ""]


@pytest.mark.parametrize("bad", _BAD_SHAPES, ids=["63", "65", "trailing-x", "empty"])
def test_render_refuses_a_wrong_shape_as_malformed(bad):
    with pytest.raises(ManifestRefused) as excinfo:
        render_manifest(["b" * 64, bad])
    assert excinfo.value.rule == "malformed"


@pytest.mark.parametrize("bad", _BAD_SHAPES, ids=["63", "65", "trailing-x", "empty"])
def test_parse_refuses_a_wrong_shape_line_as_malformed(bad):
    with pytest.raises(ManifestRefused) as excinfo:
        parse_manifest(("a" * 64 + "\n" + bad + "\n").encode())
    assert excinfo.value.rule == "malformed"


@pytest.mark.parametrize("bad", [None, 7, b"a" * 64, ["a" * 64]], ids=repr)
def test_render_refuses_a_non_string_entry_as_malformed(bad):
    with pytest.raises(ManifestRefused) as excinfo:
        render_manifest(["a" * 64, bad])
    assert excinfo.value.rule == "malformed"
