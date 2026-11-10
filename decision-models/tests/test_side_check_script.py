"""`scripts/import_side_check_shingles.py` (typed-decision-models U5.10a; plan2 §3.5, note §Q1.4):
it writes hashes only, with the source's sha256 in a header line. Fixtures are synthetic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from decisionmodels import shingles

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_side_check_shingles.py"
ITEM = {"itemId": "zz-01",
        "condition": "CONDITIONSENTENCE is never shingled here though it has nine tokens",
        "understanding": {"request": "Zebra quokka marmot narwhal ocelot pangolin",
                          "known": ["quetzal rhea sable tapir uguisu vicuna"], "missing": []},
        "turns": [{"role": "user", "text": "wombat xerus yak zorilla aardvark bonobo"}]}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("import_side_check_shingles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def items_bytes(*items: dict) -> bytes:
    return "".join(json.dumps(item) + "\n" for item in items).encode("utf-8")


def test_the_output_is_header_lines_then_sorted_hex_digests_and_nothing_else() -> None:
    data = items_bytes(ITEM)
    lines = load_script().build(data).splitlines()
    assert lines[0] == f"# source-sha256: {hashlib.sha256(data).hexdigest()}"
    assert lines[1] == "# shingle-tokens: 8"
    body = lines[2:]
    assert body == sorted(set(body)) and body
    assert all(re.fullmatch(r"[0-9a-f]{64}", line) for line in body)


def test_no_token_item_id_or_text_appears_in_the_output() -> None:
    text = load_script().build(items_bytes(ITEM)).lower()
    for word in ("zebra", "quokka", "wombat", "zorilla", "zz-01", "conditionsentence", "request",
                 "understanding", "known"):
        assert word not in text


def test_the_hashes_are_those_a_generated_item_with_the_same_span_computes() -> None:
    out = load_script().build(items_bytes(ITEM))
    body = frozenset(line for line in out.splitlines() if not line.startswith("#"))
    paraphrase = "Please: QUETZAL rhea sable tapir uguisu vicuna zebra quokka marmot"
    assert shingles.hits_side_check(paraphrase, body)
    assert not shingles.hits_side_check(paraphrase, set())
    # the condition is never an evidence field, the turns are
    assert shingles.shingles(ITEM["condition"])             # it has shingles of its own
    assert not shingles.hits_side_check(ITEM["condition"], body)
    assert shingles.hits_side_check(
        "ocelot pangolin wombat xerus yak zorilla aardvark bonobo", body)


def test_main_writes_the_file_the_loader_reads_and_refuses_a_missing_source(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    script = load_script()
    source, out = tmp_path / "items.jsonl", tmp_path / "data" / "side.txt"
    source.write_bytes(items_bytes(ITEM, {**ITEM, "itemId": "zz-02"}))
    assert script.main(["--items", str(source), "--out", str(out)]) == 0
    assert "shingle hashes to" in capsys.readouterr().out
    loaded = shingles.load_side_check_hashes(out)
    assert loaded and len(loaded) == len(set(out.read_text().splitlines()[2:]))
    assert script.main(["--items", str(tmp_path / "none.jsonl"), "--out", str(out)]) == 1
    assert "is not a file" in capsys.readouterr().err


def test_the_committed_hash_file_is_exactly_what_the_script_builds_from_the_real_pack() -> None:
    """Whole-file equality, so a deleted or flipped hash, or a tokenizer, hash-form or evidence
    change the unit tests do not pin, reddens (Pass 30 P30-M1). Skips only without model-bench."""
    bench = Path(__file__).resolve().parents[2] / "model-bench"
    if not bench.is_dir():
        pytest.skip("model-bench/ is absent from this checkout")
    items = bench / "packs" / "guard-judge-understanding" / "items.jsonl"
    assert items.is_file(), f"model-bench/ is present but {items} is not"
    assert load_script().build(items.read_bytes()) == shingles.SIDE_CHECK_FILE.read_text("utf-8")
