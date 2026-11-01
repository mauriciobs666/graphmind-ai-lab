"""The answer ingester (typed-decision-models U5.4; plan `docs/plans/typed-decision-models4.md`
§3.2 "Answer", §3.3's answer rows, "Idempotence", §5 U5.4).

Answers are written here the way `points/common/*-answer.md` tells a subagent to write them: line 1
`{"bodySha256": …}`, then one `{"key", "answer"}` or `{"key", "item"}` line per key, in any order,
optionally split into `<opaque>.1.jsonl`, `<opaque>.2.jsonl`, … with the hash line in part 1 only.

Every void below is run-attributable (plan4 §3.3). The ledger is U5.6's; here a stand-in appends
one line per `Ingested` outcome, which is all the ingester asks of it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import exchange, ingest
from decisionmodels.points import load_point
from decisionmodels.seeds import draw_seeds

GUARD = load_point("guard-judge")


def guard_item(index: int) -> dict[str, Any]:
    return {
        "condition": "the user has stated a budget for the request",
        "path": "turns",
        "understanding": {},
        "turns": [{"role": "user", "displayName": "Ana", "text": f"Orçamento de R$ {index}00"}],
    }


@pytest.fixture
def label(tmp_path: Path) -> exchange.BuiltRequest:
    built = exchange.build_label_request(
        GUARD, [(f"it_{i}", guard_item(i)) for i in range(3)], exchange_dir=tmp_path, cap=50,
        held_keys=set(),
    )
    exchange.write_request(built)
    return built


@pytest.fixture
def generation(tmp_path: Path) -> exchange.BuiltRequest:
    draw = draw_seeds("guard-judge", 2, 11, design_stratum=None)
    built = exchange.build_generation_request(
        GUARD, draw.seeds, exchange_dir=tmp_path, cap=25, held_keys=set()
    )
    exchange.write_request(built)
    return built


def header_line(built: exchange.BuiltRequest) -> dict[str, Any]:
    return {"bodySha256": json.loads(built.text.split("\n", 1)[0])["bodySha256"]}


def answer_lines(built: exchange.BuiltRequest, answer: Any = True) -> list[dict[str, Any]]:
    return [header_line(built)] + [{"key": key, "answer": answer} for key in built.keys]


def encode(lines: list[Any]) -> bytes:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")


def ingest_bytes(built: exchange.BuiltRequest, *parts: bytes, prior: str | None = None):
    return ingest.ingest_answer(built.text, parts, ingested_answer_sha256=prior)


def item_for(seed) -> dict[str, Any]:
    """A schema-valid guard item obeying `seed` (`points/guard-judge/point.json`)."""
    axes = seed.axes
    if axes["evidenceKind"] == "understanding":
        missing = ["the budget"] if axes["missingForm"] in ("immaterial", "listed") else []
        understanding = {"request": "Um bolo", "known": ["sabor: chocolate"], "missing": missing}
        turns: list[dict[str, str]] = []
    else:
        understanding = {}
        turns = [{"role": "user", "displayName": "Ana", "text": f"Mensagem {n}"}
                 for n in range(axes["turnCount"])]
    return {"condition": axes["condition"], "path": axes["evidenceKind"],
            "understanding": understanding, "turns": turns}


# --- accepted answers ----------------------------------------------------------------------------


def test_a_complete_label_answer_ingests_every_key(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines[2]["answer"] = False
    outcome = ingest_bytes(label, encode(lines))
    assert isinstance(outcome, ingest.Ingested)
    keys = list(label.keys)
    assert outcome.answers == {keys[0]: True, keys[1]: False, keys[2]: True}
    assert outcome.answer_sha256 == hashlib.sha256(encode(lines)).hexdigest()


def test_answer_lines_may_come_in_any_order(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines[1:] = reversed(lines[1:])
    assert isinstance(ingest_bytes(label, encode(lines)), ingest.Ingested)


def test_a_two_part_answer_ingests_as_one(label: exchange.BuiltRequest, tmp_path: Path) -> None:
    lines = answer_lines(label)
    ans = tmp_path / "ans"
    ans.mkdir()
    (ans / f"{label.opaque}.1.jsonl").write_bytes(encode(lines[:2]))
    (ans / f"{label.opaque}.2.jsonl").write_bytes(encode(lines[2:]))
    parts, problem = ingest.read_answer_parts(ans, label.opaque)
    assert problem is None
    assert parts == (encode(lines[:2]), encode(lines[2:]))
    split = ingest_bytes(label, *parts)
    whole = ingest_bytes(label, encode(lines))
    assert isinstance(split, ingest.Ingested)
    assert split.answers == whole.answers


def test_a_part_without_a_final_newline_does_not_merge_lines(
    label: exchange.BuiltRequest,
) -> None:
    lines = answer_lines(label)
    outcome = ingest_bytes(label, encode(lines[:2]).rstrip(b"\n"), encode(lines[2:]))
    assert isinstance(outcome, ingest.Ingested)


def test_parts_are_read_in_numeric_order(label: exchange.BuiltRequest, tmp_path: Path) -> None:
    ans = tmp_path / "ans"
    ans.mkdir()
    for k in range(1, 12):
        (ans / f"{label.opaque}.{k}.jsonl").write_bytes(f"{k}\n".encode())
    parts, problem = ingest.read_answer_parts(ans, label.opaque)
    assert problem is None
    assert parts == tuple(f"{k}\n".encode() for k in range(1, 12))


def test_a_single_answer_file_is_read_whole(label: exchange.BuiltRequest, tmp_path: Path) -> None:
    ans = tmp_path / "ans"
    ans.mkdir()
    (ans / f"{label.opaque}.jsonl").write_bytes(b"x\n")
    (ans / "otheropaquename0.jsonl").write_bytes(b"y\n")
    assert ingest.read_answer_parts(ans, label.opaque) == ((b"x\n",), None)


@pytest.mark.parametrize(
    "names",
    [
        ["{o}.jsonl", "{o}.1.jsonl"],
        ["{o}.1.jsonl", "{o}.3.jsonl"],
        ["{o}.2.jsonl"],
        ["{o}.01.jsonl"],
        ["{o}.1.jsonl", "{o}.x.jsonl"],
    ],
)
def test_an_ambiguous_part_layout_is_a_problem(tmp_path: Path, names: list[str]) -> None:
    ans = tmp_path / "ans"
    ans.mkdir()
    for name in names:
        (ans / name.format(o="abcdefghijklmnop")).write_bytes(b"x\n")
    parts, problem = ingest.read_answer_parts(ans, "abcdefghijklmnop")
    assert problem is not None


def test_a_generation_answer_ingests_items_by_key(generation: exchange.BuiltRequest) -> None:
    lines = [header_line(generation)] + [
        {"key": key, "item": item_for(seed)} for key, seed in generation.keys.items()
    ]
    outcome = ingest_bytes(generation, encode(lines))
    assert isinstance(outcome, ingest.Ingested)
    assert outcome.answers == {key: item_for(seed) for key, seed in generation.keys.items()}


# --- run-attributable voids (plan4 §3.3) ---------------------------------------------------------


def assert_void(outcome: Any, reason: str) -> None:
    assert isinstance(outcome, ingest.Void), outcome
    assert outcome.void_class == "run-attributable"
    assert outcome.reason == reason


def test_no_answer_at_all_is_void(label: exchange.BuiltRequest, tmp_path: Path) -> None:
    (tmp_path / "ans").mkdir()
    assert ingest.read_answer_parts(tmp_path / "ans", label.opaque) == ((), None)
    assert_void(ingest_bytes(label), "answerMissing")


def test_a_missing_answer_directory_reads_as_no_answer(
    label: exchange.BuiltRequest, tmp_path: Path
) -> None:
    assert ingest.read_answer_parts(tmp_path / "ans", label.opaque) == ((), None)


def test_a_missing_first_line_is_void(label: exchange.BuiltRequest) -> None:
    assert_void(ingest_bytes(label, encode(answer_lines(label)[1:])), "bodyHashMissing")


def test_an_empty_answer_is_void_as_missing_its_first_line(label: exchange.BuiltRequest) -> None:
    assert_void(ingest_bytes(label, b""), "bodyHashMissing")


def test_a_wrong_hash_is_void(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines[0]["bodySha256"] = "f" * 64
    assert_void(ingest_bytes(label, encode(lines)), "bodyHashMismatch")


@pytest.mark.parametrize("which", [0, 1, 2])
def test_a_missing_key_is_void(label: exchange.BuiltRequest, which: int) -> None:
    lines = answer_lines(label)
    del lines[1 + which]
    assert_void(ingest_bytes(label, encode(lines)), "keyMissing")


def test_a_duplicate_key_is_void(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines.append(dict(lines[1]))
    assert_void(ingest_bytes(label, encode(lines)), "keyDuplicate")


def test_an_unknown_key_is_void(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label) + [{"key": "zzzzzzzzzz", "answer": True}]
    assert_void(ingest_bytes(label, encode(lines)), "keyUnknown")


@pytest.mark.parametrize("answer", ["true", 1, 0, None, "advance", [True]])
def test_a_label_answer_out_of_space_is_void(label: exchange.BuiltRequest, answer: Any) -> None:
    lines = answer_lines(label)
    lines[2]["answer"] = answer
    assert_void(ingest_bytes(label, encode(lines)), "outOfSpace")


def test_a_choice_answer_must_be_one_of_the_points_types() -> None:
    point = load_point("message-type-classifier")
    assert ingest.in_answer_space(point, "complaint")
    assert not ingest.in_answer_space(point, "Complaint")
    assert not ingest.in_answer_space(point, True)


def test_a_generated_item_out_of_schema_is_void(generation: exchange.BuiltRequest) -> None:
    lines = [header_line(generation)] + [
        {"key": key, "item": item_for(seed)} for key, seed in generation.keys.items()
    ]
    lines[1]["item"]["intent"] = "advance"
    assert_void(ingest_bytes(generation, encode(lines)), "outOfSpace")


@pytest.mark.parametrize(
    "raw",
    [
        b"{not json}\n",
        b"[1, 2]\n",
        b"\n",
        b"\xff\xfe\n",
        b'{"key": ["a"], "answer": true}\n',
        b'{"key": {"a": 1}, "answer": true}\n',
        b'{"key": 7, "answer": true}\n',
        b"[" * 100_000 + b"\n",
    ],
)
def test_a_malformed_line_is_void(label: exchange.BuiltRequest, raw: bytes) -> None:
    assert_void(ingest_bytes(label, encode(answer_lines(label)) + raw), "malformed")


def test_an_answer_line_with_another_field_is_void(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines[1]["why"] = "because"
    assert_void(ingest_bytes(label, encode(lines)), "malformed")


def test_a_label_line_in_a_generation_answer_is_void(generation: exchange.BuiltRequest) -> None:
    lines = [header_line(generation)] + [{"key": key, "answer": True} for key in generation.keys]
    assert_void(ingest_bytes(generation, encode(lines)), "malformed")


def test_a_hash_line_after_part_one_is_void(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    assert_void(ingest_bytes(label, encode(lines[:2]), encode([lines[0]] + lines[2:])), "malformed")


def test_a_hash_line_with_another_field_is_void(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines[0]["note"] = "x"
    assert_void(ingest_bytes(label, encode(lines)), "malformed")


def test_a_void_records_the_answer_hash_and_every_problem(label: exchange.BuiltRequest) -> None:
    lines = answer_lines(label)
    lines[1]["answer"] = "yes"
    del lines[2]
    outcome = ingest_bytes(label, encode(lines))
    assert_void(outcome, "outOfSpace")
    assert outcome.answer_sha256 == hashlib.sha256(encode(lines)).hexdigest()
    assert len(outcome.details) == 2


def test_an_invalid_request_is_refused_not_voided(label: exchange.BuiltRequest) -> None:
    broken = label.text.replace('"role":"label"', '"role":"label","split":"eval"', 1)
    with pytest.raises(exchange.RequestRefused):
        ingest.ingest_answer(broken, (encode(answer_lines(label)),), ingested_answer_sha256=None)


# --- idempotence and the second answer (plan4 §3.3) ----------------------------------------------


class StandInLedger:
    """U5.6's ledger reduced to what the ingester's caller does with an outcome."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.write_bytes(b"")

    def ingested_sha256(self, opaque: str) -> str | None:
        for line in self.path.read_text().splitlines():
            event = json.loads(line)
            if event["request"] == opaque:
                return event["answerSha256"]
        return None

    def ingest(self, built: exchange.BuiltRequest, *parts: bytes):
        outcome = ingest_bytes(built, *parts, prior=self.ingested_sha256(built.opaque))
        if isinstance(outcome, ingest.Ingested):
            with self.path.open("a") as handle:
                handle.write(json.dumps({"request": built.opaque,
                                         "answerSha256": outcome.answer_sha256}) + "\n")
        return outcome


def test_a_resent_identical_answer_leaves_the_ledger_unchanged(
    label: exchange.BuiltRequest, tmp_path: Path
) -> None:
    ledger = StandInLedger(tmp_path / "ledger.jsonl")
    answer = encode(answer_lines(label))
    assert isinstance(ledger.ingest(label, answer), ingest.Ingested)
    before = ledger.path.read_bytes()
    assert isinstance(ledger.ingest(label, answer), ingest.AlreadyIngested)
    assert ledger.path.read_bytes() == before


def test_the_answer_hash_covers_the_bytes_not_the_part_split(
    label: exchange.BuiltRequest,
) -> None:
    lines = answer_lines(label)
    first = ingest_bytes(label, encode(lines))
    resent = ingest_bytes(label, encode(lines[:2]), encode(lines[2:]),
                          prior=first.answer_sha256)
    assert isinstance(resent, ingest.AlreadyIngested)


def test_a_different_second_answer_is_refused_and_changes_nothing(
    label: exchange.BuiltRequest, tmp_path: Path
) -> None:
    ledger = StandInLedger(tmp_path / "ledger.jsonl")
    assert isinstance(ledger.ingest(label, encode(answer_lines(label, True))), ingest.Ingested)
    before = ledger.path.read_bytes()
    with pytest.raises(ingest.AnswerRefused, match="already ingested"):
        ledger.ingest(label, encode(answer_lines(label, False)))
    assert ledger.path.read_bytes() == before


def test_a_second_answer_is_refused_even_when_it_would_be_void(
    label: exchange.BuiltRequest,
) -> None:
    first = ingest_bytes(label, encode(answer_lines(label)))
    with pytest.raises(ingest.AnswerRefused):
        ingest_bytes(label, b"", prior=first.answer_sha256)


# --- from the answer directory -------------------------------------------------------------------


def test_ingest_from_the_answer_directory_reads_the_parts(
    label: exchange.BuiltRequest, tmp_path: Path
) -> None:
    lines = answer_lines(label)
    ans = tmp_path / "ans"
    ans.mkdir()
    (ans / f"{label.opaque}.1.jsonl").write_bytes(encode(lines[:3]))
    (ans / f"{label.opaque}.2.jsonl").write_bytes(encode(lines[3:]))
    outcome = ingest.ingest_answer_files(label.text, ans, label.opaque, ingested_answer_sha256=None)
    assert isinstance(outcome, ingest.Ingested)
    assert set(outcome.answers) == set(label.keys)


def test_an_unorderable_part_layout_is_a_malformed_void(
    label: exchange.BuiltRequest, tmp_path: Path
) -> None:
    lines = answer_lines(label)
    ans = tmp_path / "ans"
    ans.mkdir()
    (ans / f"{label.opaque}.1.jsonl").write_bytes(encode(lines[:3]))
    (ans / f"{label.opaque}.3.jsonl").write_bytes(encode(lines[3:]))
    outcome = ingest.ingest_answer_files(label.text, ans, label.opaque, ingested_answer_sha256=None)
    assert_void(outcome, "malformed")


# --- malformed lines that must void, never crash (review Pass 5 P5-F1) ---------------------------


@pytest.mark.parametrize("key", [["a"], {"a": 1}])
def test_an_unhashable_key_in_a_generation_answer_is_void(
    generation: exchange.BuiltRequest, key: Any
) -> None:
    lines = [header_line(generation)] + [
        {"key": k, "item": item_for(seed)} for k, seed in generation.keys.items()
    ]
    lines.append({"key": key, "item": {}})
    assert_void(ingest_bytes(generation, encode(lines)), "malformed")


def with_line_for_first_key(built: exchange.BuiltRequest, raw: str) -> bytes:
    """The complete answer, the first key's line replaced by `raw`."""
    lines = [json.dumps(line) for line in answer_lines(built)]
    lines[1] = raw
    return "".join(line + "\n" for line in lines).encode("utf-8")


def test_a_repeated_name_in_an_answer_line_is_void(label: exchange.BuiltRequest) -> None:
    key = next(iter(label.keys))
    raw = f'{{"key": "{key}", "answer": false, "answer": true}}'
    assert_void(ingest_bytes(label, with_line_for_first_key(label, raw)), "malformed")


def test_a_repeated_name_in_the_hash_line_is_void(label: exchange.BuiltRequest) -> None:
    sha = header_line(label)["bodySha256"]
    lines = [json.dumps(line) for line in answer_lines(label)]
    lines[0] = f'{{"bodySha256": "{"0" * 64}", "bodySha256": "{sha}"}}'
    answer = "".join(line + "\n" for line in lines).encode("utf-8")
    assert_void(ingest_bytes(label, answer), "malformed")


def test_reconciling_the_same_bad_batch_twice_voids_twice_without_crashing(
    label: exchange.BuiltRequest, tmp_path: Path
) -> None:
    """plan4 §3.5: `next` re-ingests an `issued` batch, so a crash would stall the campaign."""
    ans = tmp_path / "ans"
    ans.mkdir()
    (ans / f"{label.opaque}.jsonl").write_bytes(
        encode(answer_lines(label)) + b'{"key": ["a"], "answer": true}\n'
    )
    for _ in range(2):
        outcome = ingest.ingest_answer_files(
            label.text, ans, label.opaque, ingested_answer_sha256=None
        )
        assert_void(outcome, "malformed")


# --- answer normalization (P5-F3) ----------------------------------------------------------------


def test_a_part_without_its_final_newline_hashes_as_the_whole_answer(
    label: exchange.BuiltRequest,
) -> None:
    lines = answer_lines(label)
    whole = ingest_bytes(label, encode(lines))
    resent = ingest_bytes(label, encode(lines[:2]).rstrip(b"\n"), encode(lines[2:]),
                          prior=whole.answer_sha256)
    assert isinstance(resent, ingest.AlreadyIngested)


def test_a_crlf_answer_ingests(label: exchange.BuiltRequest) -> None:
    crlf = encode(answer_lines(label)).replace(b"\n", b"\r\n")
    outcome = ingest_bytes(label, crlf)
    assert isinstance(outcome, ingest.Ingested)
    assert set(outcome.answers) == set(label.keys)


@pytest.mark.parametrize("blank", [b"\n", b"\r\n"])
def test_a_blank_first_line_is_malformed_in_lf_and_crlf_alike(
    label: exchange.BuiltRequest, blank: bytes
) -> None:
    """A `\\r`-only line is a CRLF blank line: the same reason as an LF one (review Pass 6
    P6-F2), not a missing hash line."""
    assert_void(ingest_bytes(label, blank + encode(answer_lines(label))), "malformed")
