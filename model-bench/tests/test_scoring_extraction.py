"""`modelbench.scoring.extraction` — `nlq-structured-query`'s `ItemScorer` (S4 spec §5.2.4, §7
Step 3).

Order, red -> green, matches the spec's own step sequence (§7 Step 3):

Pass B (this file's first half, items 4-5): `_canon_str`/`_scalar_equal`/`score_pair` —
transcribed from `nlq_scoring.py`'s own `test_nlq_scoring.py` cases (reused directly, confirming
this port matches the source's tested behaviour), **explicitly including the `conflicting-facts`
subset-containment exception**; `_describe_dataset_schema`/`build_messages` against `schema.json`'s
real catalog/knowledge_base blocks.

Pass C (this file's second half, items 6-8): the three-gate `score_item` tests; `aggregate` over a
synthetic fixture spanning all seven shapes plus the unanswerable bucket; `validate --pack
packs/nlq-structured-query --strict` (via `packs.validate_pack` directly, the same pre-existing
`cli.py` `NotImplementedError` deferral the sibling pack's own test works around).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack, load_pack, validate_pack
from modelbench.results import ItemTiming
from modelbench.scoring import extraction

_REAL_PACK_ROOT = Path(__file__).resolve().parents[1] / "packs" / "nlq-structured-query"


def _row(*, shape: str, expected: dict) -> dict:
    return {"itemId": "nlq-test", "dataset": "catalog", "question": "q?", "shape": shape}


# ==================================================================================================
# Pass B, item 4: `_canon_str` / `_scalar_equal` / `score_pair` — transcribed from
# `nlq_scoring.py`'s own tested rules (`falkor-chat/server/tests/eval/test_nlq_scoring.py`)
# ==================================================================================================


class TestCanonStr:
    def test_case_folds_and_collapses_whitespace(self):
        assert extraction._canon_str("  Storage  Unit ") == "storage unit"

    def test_stringifies_non_string_values(self):
        assert extraction._canon_str(24.99) == "24.99"


class TestScalarEqual:
    def test_numbers_within_epsilon_are_equal(self):
        assert extraction._scalar_equal(24.99, 24.995) is True

    def test_numbers_at_epsilon_boundary_are_equal(self):
        assert extraction._scalar_equal(24.99, 25.00) is True  # exactly 0.01 off — inclusive

    def test_numbers_just_outside_epsilon_are_not_equal(self):
        assert extraction._scalar_equal(24.99, 25.01) is False

    def test_strings_are_compared_case_and_whitespace_folded(self):
        assert extraction._scalar_equal("Storage", "  storage  ") is True

    def test_a_numeric_expected_never_string_equals_a_formatted_price(self):
        """The ml note's own explicit caution: a numeric expected value compared against a
        string-typed actual value is a type mismatch, never stringified and matched."""
        assert extraction._scalar_equal(24.99, "$24.99") is False

    def test_bool_is_compared_by_identity_never_coerced_to_a_number(self):
        assert extraction._scalar_equal(True, True) is True
        assert extraction._scalar_equal(True, 1) is False


class TestScorePair:
    # -- scalar ------------------------------------------------------------------------------
    def test_scalar_exact_match_after_folding(self):
        result = {"items": [{"p.category": "  storage  "}]}
        correct, _ = extraction.score_pair(
            {"type": "scalar", "value": "Storage"}, "single-fact", result
        )
        assert correct is True

    def test_scalar_mismatch_is_incorrect(self):
        result = {"items": [{"p.category": "Audio"}]}
        correct, _ = extraction.score_pair(
            {"type": "scalar", "value": "Storage"}, "single-fact", result
        )
        assert correct is False

    def test_scalar_extraction_fails_on_zero_rows(self):
        correct, _ = extraction.score_pair(
            {"type": "scalar", "value": "Storage"}, "single-fact", {"items": []}
        )
        assert correct is False

    def test_scalar_extraction_fails_on_multiple_rows(self):
        correct, _ = extraction.score_pair(
            {"type": "scalar", "value": "Storage"},
            "single-fact",
            {"items": [{"p.category": "Storage"}, {"p.category": "Storage"}]},
        )
        assert correct is False

    def test_scalar_extraction_fails_on_multiple_columns(self):
        correct, _ = extraction.score_pair(
            {"type": "scalar", "value": "Storage"},
            "single-fact",
            {"items": [{"p.category": "Storage", "p.name": "x"}]},
        )
        assert correct is False

    # -- set (plain, exact match) --------------------------------------------------------------
    def test_set_unordered_match_regardless_of_result_order(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["A", "B"]},
            "filter-list",
            {"items": [{"p.name": "B"}, {"p.name": "A"}]},
        )
        assert correct is True

    def test_set_match_is_case_and_whitespace_folded(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["Widget A"]},
            "filter-list",
            {"items": [{"p.name": "  widget   a "}]},
        )
        assert correct is True

    def test_set_missing_a_member_is_incorrect(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["A", "B"]}, "filter-list", {"items": [{"p.name": "A"}]}
        )
        assert correct is False

    def test_set_with_an_extra_member_is_incorrect(self):
        """Plain set shapes are EXACT match — an extra, unrequested value is wrong, unlike
        conflicting-facts' own containment exception below."""
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["A"]},
            "filter-list",
            {"items": [{"p.name": "A"}, {"p.name": "B"}]},
        )
        assert correct is False

    def test_set_singleton_is_still_a_set_comparison(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["A"]}, "filter-list", {"items": [{"p.name": "A"}]}
        )
        assert correct is True

    # -- not_found -----------------------------------------------------------------------------
    def test_not_found_correct_when_result_is_genuinely_empty(self):
        correct, _ = extraction.score_pair({"type": "not_found"}, "not-found", {"items": []})
        assert correct is True

    def test_not_found_incorrect_when_mechanism_fabricates_a_value(self):
        correct, _ = extraction.score_pair(
            {"type": "not_found"}, "not-found", {"items": [{"p.name": "X"}]}
        )
        assert correct is False

    # -- conflicting-facts: THE named exception (subset-containment, never set-equality) -------
    def test_conflicting_facts_correct_when_all_expected_values_present(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["62", "140 employees"]},
            "conflicting-facts",
            {"items": [{"o.value": "62"}, {"o.value": "140 employees"}]},
        )
        assert correct is True

    def test_conflicting_facts_tolerates_extra_values_present(self):
        """THE mutation-test target: a scorer that applied set-EQUALITY uniformly (rather than
        conflicting-facts' own subset-containment) would score this incorrect — an extra value
        beyond the two required ones is tolerated ONLY for this shape (S4 spec §2.5/§3.8.3)."""
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["62", "140 employees"]},
            "conflicting-facts",
            {"items": [{"o.value": "62"}, {"o.value": "140 employees"}, {"o.value": "99"}]},
        )
        assert correct is True

    def test_conflicting_facts_incorrect_when_only_one_value_present(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["62", "140 employees"]},
            "conflicting-facts",
            {"items": [{"o.value": "62"}]},
        )
        assert correct is False

    def test_conflicting_facts_incorrect_when_result_is_empty(self):
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["62", "140 employees"]}, "conflicting-facts", {"items": []}
        )
        assert correct is False

    def test_a_plain_set_shape_with_the_same_extra_value_is_incorrect(self):
        """The direct contrast to the conflicting-facts test above: the SAME actual set
        (containing an extra member) against a PLAIN set shape (`filter-list`, not
        `conflicting-facts`) is scored by exact-match, so it is INCORRECT — proving the
        containment exception is shape-scoped, not a blanket relaxation."""
        correct, _ = extraction.score_pair(
            {"type": "set", "values": ["62", "140 employees"]},
            "filter-list",
            {"items": [{"o.value": "62"}, {"o.value": "140 employees"}, {"o.value": "99"}]},
        )
        assert correct is False

    # -- unknown type --------------------------------------------------------------------------
    def test_unknown_expected_type_raises(self):
        with pytest.raises(ValueError):
            extraction.score_pair({"type": "bogus"}, "single-fact", {"items": []})


# ==================================================================================================
# Pass B, item 5: `_describe_dataset_schema` / `build_messages`
# ==================================================================================================


class TestDescribeDatasetSchema:
    def test_one_line_per_label_sorted_properties_joined_by_semicolon(self):
        schema_block = {
            "labels": {
                "Product": {"price": "float", "name": "str", "category": "str"},
            }
        }
        assert (
            extraction._describe_dataset_schema(schema_block)
            == "Product (properties: category, name, price)"
        )

    def test_multiple_labels_joined_by_semicolon_space(self):
        schema_block = {
            "labels": {
                "Entity": {"name": "str", "type": "str"},
                "Document": {"title": "str"},
            }
        }
        assert (
            extraction._describe_dataset_schema(schema_block)
            == "Entity (properties: name, type); Document (properties: title)"
        )


@pytest.fixture(scope="module")
def real_pack() -> Pack:
    if not (_REAL_PACK_ROOT / "items.jsonl").exists():
        pytest.skip(
            f"{_REAL_PACK_ROOT / 'items.jsonl'} does not exist — run "
            "`scripts/refresh_golden.py --pack packs/nlq-structured-query` first"
        )
    return load_pack(_REAL_PACK_ROOT)


def _load_real_items() -> list[dict[str, Any]]:
    with (_REAL_PACK_ROOT / "items.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestBuildMessages:
    def test_system_message_fills_the_catalog_schema_for_a_catalog_item(self, real_pack: Pack):
        item = next(r for r in _load_real_items() if r["dataset"] == "catalog")
        messages = extraction.build_messages(item, pack=real_pack)
        template = (real_pack.root / "prompts" / "querygen.md").read_text(encoding="utf-8")
        schema = json.loads((real_pack.root / "schema.json").read_text(encoding="utf-8"))
        expected_schema_str = extraction._describe_dataset_schema(schema["catalog"])
        assert messages[0] == {
            "role": "system", "content": template.format(dataset_schema=expected_schema_str),
        }

    def test_system_message_fills_the_knowledge_base_schema_for_a_kb_item(self, real_pack: Pack):
        item = next(r for r in _load_real_items() if r["dataset"] == "knowledge_base")
        messages = extraction.build_messages(item, pack=real_pack)
        schema = json.loads((real_pack.root / "schema.json").read_text(encoding="utf-8"))
        expected_schema_str = extraction._describe_dataset_schema(schema["knowledge_base"])
        assert expected_schema_str in messages[0]["content"]
        assert "Entity (properties:" in messages[0]["content"]

    def test_user_message_is_the_items_own_question_verbatim(self, real_pack: Pack):
        item = next(r for r in _load_real_items() if r["itemId"] == "nlq-01")
        messages = extraction.build_messages(item, pack=real_pack)
        assert messages[1] == {"role": "user", "content": item["question"]}

    def test_never_reads_the_gold_label_fields(self, real_pack: Pack):
        """`build_messages` must never read `expected`/`rationale`/`answerable` — those would
        leak the gold answer into the prompt. Constructed by deleting the keys (where present)
        before calling: must not raise `KeyError`."""
        item = dict(next(r for r in _load_real_items() if r["itemId"] == "nlq-01"))
        item.pop("expected", None)
        item.pop("rationale", None)
        item.pop("answerable", None)
        messages = extraction.build_messages(item, pack=real_pack)  # must not raise
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


def test_prompts_querygen_md_matches_the_live_falkorchat_source_byte_for_byte():
    """Verified the same way `judge.md` was, not hand-typed and eyeballed: an AST
    `literal_eval` of `tools.py`'s own `_QUERY_REQUEST_INSTRUCTIONS` assignment against the live
    source, compared to the shipped prompt file's exact bytes (S4 spec §5.2/§2.5)."""
    import ast

    tools_py = (
        Path(__file__).resolve().parents[2]
        / "falkor-chat" / "server" / "falkorchat" / "tools.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(tools_py)
    live_value = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_QUERY_REQUEST_INSTRUCTIONS":
                live_value = ast.literal_eval(node.value)
                break
    assert live_value is not None
    shipped = (_REAL_PACK_ROOT / "prompts" / "querygen.md").read_text(encoding="utf-8")
    assert shipped == live_value


# ==================================================================================================
# Pass C, item 6: `score_item` — the three-gate structure, wiring `tools/exec.py`
# ==================================================================================================


@pytest.fixture()
def synthetic_pack(tmp_path: Path) -> Pack:
    """A minimal, self-contained pack: the REAL `tools/exec.py` (copied verbatim — this is the
    module under test's own execution seam, not a stand-in), a 2-row catalog fixture."""
    pack_root = tmp_path / "nlq-pack"
    (pack_root / "tools").mkdir(parents=True)
    (pack_root / "tools" / "exec.py").write_text(
        (_REAL_PACK_ROOT / "tools" / "exec.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (pack_root / "schema.json").write_text(
        json.dumps({"catalog": {"labels": {"Product": {"name": "str", "price": "float"}}}})
    )
    (pack_root / "tables.json").write_text(
        json.dumps(
            {
                "catalog": {
                    "Product": [
                        {"name": "Widget", "price": 10.0},
                        {"name": "Extra", "price": 20.0},
                    ]
                }
            }
        )
    )
    (pack_root / "items.jsonl").write_text("")
    (pack_root / "pack.json").write_text(
        json.dumps(
            {
                "packId": "nlq-structured-query-test", "packVersion": "1.0.0",
                "role": "nlq-generator", "scorer": "extraction",
                "environment": {"requires": ["lmstudio-chat"]},
                "data": {"items": "items.jsonl", "tables": "tables.json", "schema": "schema.json"},
                "tools": {"module": "tools/exec.py", "entrypoint": "compile_and_execute"},
                "sampling": {"seed": 1, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
                "metrics": {
                    "verdictMetrics": ["layer1ExactMatchRate"],
                    "headlineMetric": "layer1ExactMatchRate",
                },
            }
        )
    )
    return load_pack(pack_root)


def _chat_result(content: str) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": content},
        tool_calls=(), toolCallForm="prose", stats=None, model_info=None, runtime=None,
        usage=None, wallClockMs=5.0,
    )


def _timing() -> ItemTiming:
    return ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)


def _timeout_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")


def _no_response_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")


def _item_input(**overrides: Any) -> dict[str, Any]:
    base = {
        "itemId": "nlq-01", "dataset": "catalog", "question": "How much does Widget cost?",
        "shape": "single-fact", "expected": {"type": "scalar", "value": 10.0}, "answerable": True,
    }
    base.update(overrides)
    return base


_WIDGET_FILTER = [{"property": "name", "op": "=", "value": "Widget"}]
_SPEC_WIDGET_PRICE = json.dumps(
    {
        "matches": [{"var": "p", "label": "Product", "filters": _WIDGET_FILTER}],
        "returns": ["p.price"],
    }
)
_SPEC_WIDGET_NAME = json.dumps(
    {
        "matches": [{"var": "p", "label": "Product", "filters": _WIDGET_FILTER}],
        "returns": ["p.name"],
    }
)
_SPEC_NO_MATCH = json.dumps(
    {
        "matches": [
            {
                "var": "p", "label": "Product",
                "filters": [{"property": "name", "op": "=", "value": "Does Not Exist"}],
            }
        ],
        "returns": ["p.name"],
    }
)
_SPEC_ALL_ROWS = json.dumps(
    {"matches": [{"var": "p", "label": "Product", "filters": []}], "returns": ["p.name"]}
)


class TestScoreItem:
    def test_result_none_timeout_scores_fail_with_no_scoreable_metrics(self, synthetic_pack):
        result = extraction.score_item(_item_input(), None, _timeout_timing(), pack=synthetic_pack)
        assert result.outcome == "fail"
        assert result.scoreable == {}
        assert result.counts == {}

    def test_result_none_no_response_scores_unrunnable(self, synthetic_pack):
        timing = _no_response_timing()
        result = extraction.score_item(_item_input(), None, timing, pack=synthetic_pack)
        assert result.outcome == "unrunnable"
        assert result.scoreable == {}

    def test_gate_1_no_json_reply_is_a_parse_failure(self, synthetic_pack):
        reply = _chat_result("not json at all")
        result = extraction.score_item(_item_input(), reply, _timing(), pack=synthetic_pack)
        assert result.outcome == "parse_failure"
        assert result.detail == {"failureClass": "no_json"}
        assert result.scoreable == {}
        assert result.counts == {}

    def test_gate_1_own_line_object_missing_the_required_key_is_a_parse_failure(
        self, synthetic_pack
    ):
        """`require_key` only disambiguates the OWN-LINE-scan branch (per
        `extract_own_line_json_object`'s own docstring) — a reply that is a whole JSON object
        parses unconditionally even missing the key, and only THEN fails Layer A for missing
        `matches`. This case instead wraps the object in prose so it takes the own-line-scan
        path, where `require_key="matches"` genuinely gates it."""
        reply = _chat_result(
            'Here is my answer:\n' + json.dumps({"returns": ["p.name"]}) + '\nDone.'
        )
        result = extraction.score_item(_item_input(), reply, _timing(), pack=synthetic_pack)
        assert result.detail == {"failureClass": "no_json"}

    def test_gate_1_whole_reply_json_reaches_layer_a_as_malformed_without_matches(
        self, synthetic_pack
    ):
        """The other side of that same asymmetry: a reply that IS ENTIRELY one JSON object
        parses regardless of `require_key`, so a missing `matches` key surfaces one gate later,
        as a Layer A structural violation — not a parse failure."""
        reply = _chat_result(json.dumps({"returns": ["p.name"]}))
        result = extraction.score_item(_item_input(), reply, _timing(), pack=synthetic_pack)
        assert result.detail == {"failureClass": "malformed_spec"}

    def test_gate_2_layer_a_violation_fails_as_malformed_spec(self, synthetic_pack):
        reply = _chat_result(json.dumps({"matches": [], "returns": ["p.name"]}))
        result = extraction.score_item(_item_input(), reply, _timing(), pack=synthetic_pack)
        assert result.outcome == "fail"
        assert result.detail == {"failureClass": "malformed_spec"}
        assert result.scoreable == {}
        assert result.counts == {}

    def test_gate_3_layer_b_violation_fails_as_schema_violation(self, synthetic_pack):
        reply = _chat_result(
            json.dumps(
                {
                    "matches": [{"var": "p", "label": "Nonexistent", "filters": []}],
                    "returns": ["p.name"],
                }
            )
        )
        result = extraction.score_item(_item_input(), reply, _timing(), pack=synthetic_pack)
        assert result.outcome == "fail"
        assert result.detail == {"failureClass": "schema_violation"}
        assert result.scoreable == {}
        assert result.counts == {}

    def test_clean_correct_answerable_reply_scores_pass(self, synthetic_pack):
        reply = _chat_result(_SPEC_WIDGET_PRICE)
        item = _item_input(expected={"type": "scalar", "value": 10.0})
        result = extraction.score_item(item, reply, _timing(), pack=synthetic_pack)
        assert result.outcome == "pass"
        assert result.scoreable == {"layer1ExactMatchRate": True, "exactMatchBySingleFact": True}
        assert result.counts == {"layer1ExactMatchRate": 1, "exactMatchBySingleFact": 1}
        assert result.detail["shape"] == "single-fact"

    def test_clean_incorrect_answerable_reply_still_scores_outcome_pass_but_zero_counts(
        self, synthetic_pack
    ):
        """A wrong-but-well-formed answer still reaches execution — `outcome="pass"` either way
        (S4 spec §5.2.4's own "scoreable=True always, even on a wrong answer")."""
        reply = _chat_result(_SPEC_WIDGET_NAME)  # returns "Widget" (a string), not 10.0
        item = _item_input(expected={"type": "scalar", "value": 10.0})
        result = extraction.score_item(item, reply, _timing(), pack=synthetic_pack)
        assert result.outcome == "pass"
        assert result.counts == {"layer1ExactMatchRate": 0, "exactMatchBySingleFact": 0}

    def test_unanswerable_item_correctly_abstaining_scores_a_correct_abstain(self, synthetic_pack):
        reply = _chat_result(_SPEC_NO_MATCH)  # executes to an empty result
        item = _item_input(answerable=False, shape="relationship-traversal")
        result = extraction.score_item(item, reply, _timing(), pack=synthetic_pack)
        assert result.outcome == "pass"
        # Correction A3 (`docs/reviews/nlq-conflicting-facts-answerability-ml.md` Q2/F-1):
        # unanswerable ≠ unscored — the exploratory score_pair call also runs, but scores
        # incorrect here (a scalar shape can't extract from an empty result), so no luckyPass.
        assert result.scoreable == {
            "unanswerableAbstainRate": True, "exactMatchByRelationshipTraversal": True,
        }
        assert result.counts == {
            "unanswerableAbstainRate": 1, "exactMatchByRelationshipTraversal": 0,
        }
        assert "luckyPass" not in result.detail
        # Never scored toward layer1ExactMatchRate at all.
        assert "layer1ExactMatchRate" not in result.scoreable

    def test_unanswerable_item_fabricating_an_answer_scores_an_incorrect_abstain(
        self, synthetic_pack
    ):
        reply = _chat_result(_SPEC_ALL_ROWS)  # executes to a NON-empty result
        item = _item_input(answerable=False, shape="relationship-traversal")
        result = extraction.score_item(item, reply, _timing(), pack=synthetic_pack)
        assert result.counts == {
            "unanswerableAbstainRate": 0, "exactMatchByRelationshipTraversal": 0,
        }
        assert "luckyPass" not in result.detail

    def test_unanswerable_conflicting_facts_lucky_pass_is_flagged(self, synthetic_pack):
        """F-4's degenerate-spec class of outcome: an unfiltered/broad spec (`_SPEC_ALL_ROWS`
        returns every product name) happens to satisfy `conflicting-facts`'s subset-containment
        rule even though it is not a faithful answer. The unanswerable branch's exploratory
        `score_pair` call now runs against production-shaped data, so this is a real, reachable
        outcome — `detail["luckyPass"]` names it rather than letting it read as an ordinary win."""
        reply = _chat_result(_SPEC_ALL_ROWS)  # returns {"Widget", "Extra"} — a superset
        item = _item_input(
            answerable=False, shape="conflicting-facts",
            expected={"type": "set", "values": ["Widget"]},
        )
        result = extraction.score_item(item, reply, _timing(), pack=synthetic_pack)
        assert result.scoreable == {
            "unanswerableAbstainRate": True, "exactMatchByConflictingFacts": True,
        }
        # Fabricated (non-empty), so an incorrect abstain — independent of the lucky pass below.
        assert result.counts == {
            "unanswerableAbstainRate": 0, "exactMatchByConflictingFacts": 1,
        }
        assert result.detail["luckyPass"] is True


# ==================================================================================================
# Pass C, item 7: `aggregate` over a synthetic fixture spanning all seven shapes plus the
# unanswerable bucket
# ==================================================================================================


def _synthetic_extraction_items(pack: Pack) -> list[Any]:
    def scored(item_id: str, **overrides: Any) -> Any:
        reply_content = overrides.pop("_reply")
        item = _item_input(itemId=item_id, **overrides)
        return extraction.score_item(item, _chat_result(reply_content), _timing(), pack=pack)

    items = [
        # single-fact: one correct, one incorrect.
        scored(
            "sf-1", shape="single-fact", expected={"type": "scalar", "value": 10.0},
            _reply=_SPEC_WIDGET_PRICE,
        ),
        scored(
            "sf-2", shape="single-fact", expected={"type": "scalar", "value": 10.0},
            _reply=_SPEC_WIDGET_NAME,
        ),
        # filter-list: one correct, one incorrect.
        scored(
            "fl-1", shape="filter-list", expected={"type": "set", "values": ["Widget"]},
            _reply=_SPEC_WIDGET_NAME,
        ),
        scored(
            "fl-2", shape="filter-list", expected={"type": "set", "values": ["Widget"]},
            _reply=_SPEC_NO_MATCH,
        ),
        # compound-filter: one correct, one incorrect.
        scored(
            "cf-1", shape="compound-filter", expected={"type": "set", "values": ["Widget"]},
            _reply=_SPEC_WIDGET_NAME,
        ),
        scored(
            "cf-2", shape="compound-filter", expected={"type": "set", "values": ["Widget"]},
            _reply=_SPEC_NO_MATCH,
        ),
        # not-found: one correct (genuinely empty), one incorrect (fabricated).
        scored("nf-1", shape="not-found", expected={"type": "not_found"}, _reply=_SPEC_NO_MATCH),
        scored("nf-2", shape="not-found", expected={"type": "not_found"}, _reply=_SPEC_WIDGET_NAME),
        # aggregation: one correct, one incorrect.
        scored(
            "ag-1", shape="aggregation", expected={"type": "scalar", "value": 2},
            _reply=json.dumps(
                {
                    "matches": [{"var": "p", "label": "Product", "filters": []}],
                    "returns": ["count(p)"],
                }
            ),
        ),
        scored(
            "ag-2", shape="aggregation", expected={"type": "scalar", "value": 2},
            _reply=json.dumps(
                {
                    "matches": [
                        {
                            "var": "p", "label": "Product",
                            "filters": [{"property": "name", "op": "=", "value": "Widget"}],
                        }
                    ],
                    "returns": ["count(p)"],
                }
            ),  # count(p)=1 under this filter, not 2 — incorrect
        ),
        # conflicting-facts: one correct (superset ok), one incorrect (missing a value).
        scored(
            "cx-1", shape="conflicting-facts",
            expected={"type": "set", "values": ["Widget", "Extra"]}, _reply=_SPEC_ALL_ROWS,
        ),
        scored(
            "cx-2", shape="conflicting-facts",
            expected={"type": "set", "values": ["Widget", "Extra"]}, _reply=_SPEC_WIDGET_NAME,
        ),
        # unanswerable bucket (relationship-traversal): one correct abstain, one fabrication.
        scored("rt-1", answerable=False, shape="relationship-traversal", _reply=_SPEC_NO_MATCH),
        scored("rt-2", answerable=False, shape="relationship-traversal", _reply=_SPEC_ALL_ROWS),
        # the three failure gates — never contribute to layer1ExactMatchRate at all.
        scored("pf-1", _reply="not json"),
        scored("ms-1", _reply=json.dumps({"matches": [], "returns": ["p.name"]})),
        scored(
            "sv-1",
            _reply=json.dumps(
                {"matches": [{"var": "p", "label": "Nope", "filters": []}], "returns": ["p.name"]}
            ),
        ),
    ]
    return items


class TestAggregate:
    def test_exact_match_covers_only_the_twelve_answerable_items_reaching_execution(
        self, synthetic_pack
    ):
        items = _synthetic_extraction_items(synthetic_pack)
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        assert aggregates.exactMatch.n == 12
        assert aggregates.exactMatch.successes == 6  # one correct per shape, six shapes

    def test_by_shape_sums_back_to_exact_match_and_each_carries_its_own_n(self, synthetic_pack):
        items = _synthetic_extraction_items(synthetic_pack)
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        by_name = {m.name: m for m in aggregates.byShape}
        for name in (
            "exactMatchBySingleFact", "exactMatchByFilterList", "exactMatchByCompoundFilter",
            "exactMatchByNotFound", "exactMatchByAggregation", "exactMatchByConflictingFacts",
        ):
            assert by_name[name].n == 2, name
            assert by_name[name].successes == 1, name
        total_n = sum(
            by_name[n].n for n in (
                "exactMatchBySingleFact", "exactMatchByFilterList", "exactMatchByCompoundFilter",
                "exactMatchByNotFound", "exactMatchByAggregation", "exactMatchByConflictingFacts",
            )
        )
        assert total_n == aggregates.exactMatch.n

    def test_unanswerable_abstain_rate_is_its_own_separate_metric(self, synthetic_pack):
        items = _synthetic_extraction_items(synthetic_pack)
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        by_name = {m.name: m for m in aggregates.byShape}
        assert by_name["unanswerableAbstainRate"].n == 2
        assert by_name["unanswerableAbstainRate"].successes == 1

    def test_the_three_failure_counts_are_each_their_own_class_never_pooled(self, synthetic_pack):
        items = _synthetic_extraction_items(synthetic_pack)
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        assert aggregates.parseFailures == 1
        assert aggregates.malformedSpecCount == 1
        assert aggregates.schemaViolationCount == 1
        # And the three failure-gate items never touched layer1ExactMatchRate's own denominator.
        assert aggregates.exactMatch.n == 12

    def test_every_binary_metric_uses_the_item_unit(self, synthetic_pack):
        items = _synthetic_extraction_items(synthetic_pack)
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        assert aggregates.exactMatch.unit == "item"
        assert all(m.unit == "item" for m in aggregates.byShape)

    def test_by_shape_pools_answerable_and_unanswerable_scores_of_the_same_shape(
        self, synthetic_pack
    ):
        """Correction A3: `byShape`'s per-shape pooling is computed by filtering on each item's
        own `exactMatchBy{Shape}` scoreable flag, never by gating on `layer1ExactMatchRate`
        (the pre-correction approach). Today's real pack never mixes an answerable and an
        unanswerable item on the same shape, but the aggregation code must not special-case that
        away — this fixture proves it by mixing them directly on a hypothetical shape."""
        items = [
            extraction.score_item(
                _item_input(
                    itemId="rt-hyp", answerable=True, shape="relationship-traversal",
                    expected={"type": "scalar", "value": 10.0},
                ),
                _chat_result(_SPEC_WIDGET_PRICE), _timing(), pack=synthetic_pack,
            ),  # hypothetical answerable member of this shape — correct
            extraction.score_item(
                _item_input(
                    itemId="rt-1", answerable=False, shape="relationship-traversal",
                    expected={"type": "scalar", "value": 10.0},
                ),
                _chat_result(_SPEC_NO_MATCH), _timing(), pack=synthetic_pack,
            ),  # unanswerable-path member — exploratory score incorrect
        ]
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        by_name = {m.name: m for m in aggregates.byShape}
        pooled = by_name["exactMatchByRelationshipTraversal"]
        assert pooled.n == 2
        assert pooled.successes == 1
        # The unanswerable member never contributes to the headline denominator.
        assert aggregates.exactMatch.n == 1
        assert aggregates.exactMatch.successes == 1

    def test_lucky_pass_count_is_additive_with_its_shape_metric_not_an_alternative(
        self, synthetic_pack
    ):
        items = [
            extraction.score_item(
                _item_input(
                    itemId="cx-lucky", answerable=False, shape="conflicting-facts",
                    expected={"type": "set", "values": ["Widget"]},
                ),
                _chat_result(_SPEC_ALL_ROWS), _timing(), pack=synthetic_pack,
            ),  # F-4 degenerate spec — scores correct by accident
            extraction.score_item(
                _item_input(
                    itemId="cx-ordinary", answerable=False, shape="conflicting-facts",
                    expected={"type": "set", "values": ["Widget"]},
                ),
                _chat_result(_SPEC_NO_MATCH), _timing(), pack=synthetic_pack,
            ),  # ordinary incorrect case
        ]
        aggregates = extraction.aggregate(items, pack=synthetic_pack)
        assert aggregates.luckyPassCount == 1
        by_name = {m.name: m for m in aggregates.byShape}
        pooled = by_name["exactMatchByConflictingFacts"]
        assert pooled.n == 2
        # The lucky pass's success is counted here too — additive, not an alternative accounting.
        assert pooled.successes == 1


# ==================================================================================================
# Pass C, item 8: `validate --pack packs/nlq-structured-query --strict` — via `packs.validate_pack`
# directly, the same pre-existing `cli.py` `NotImplementedError` deferral the sibling pack's own
# test (`test_scoring_classification.py`) works around
# ==================================================================================================


def test_validate_pack_fails_a_fixture_missing_the_answerable_key(tmp_path: Path):
    pack_root = tmp_path / "unstamped"
    (pack_root / "tools").mkdir(parents=True)
    (pack_root / "tools" / "exec.py").write_text(
        (_REAL_PACK_ROOT / "tools" / "exec.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (pack_root / "schema.json").write_text(json.dumps({"catalog": {"labels": {}}}))
    (pack_root / "tables.json").write_text(json.dumps({"catalog": {}}))
    (pack_root / "pack.json").write_text(
        json.dumps(
            {
                "packId": "nlq-structured-query", "packVersion": "1.0.0", "role": "nlq-generator",
                "scorer": "extraction", "environment": {"requires": ["lmstudio-chat"]},
                "data": {"items": "items.jsonl", "tables": "tables.json", "schema": "schema.json"},
                "tools": {"module": "tools/exec.py", "entrypoint": "compile_and_execute"},
                "sampling": {"seed": 1, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
                "metrics": {
                    "verdictMetrics": ["layer1ExactMatchRate"],
                    "headlineMetric": "layer1ExactMatchRate",
                },
            }
        )
    )
    (pack_root / "items.jsonl").write_text(
        json.dumps({"itemId": "x", "dataset": "catalog", "answerable": True}) + "\n"
        + json.dumps({"itemId": "y", "dataset": "catalog"}) + "\n"  # missing "answerable"
    )
    pack = load_pack(pack_root)
    problems = validate_pack(pack)
    assert any("answerable" in p for p in problems)


def test_validate_pack_passes_once_every_row_is_stamped(tmp_path: Path):
    pack_root = tmp_path / "stamped"
    (pack_root / "tools").mkdir(parents=True)
    (pack_root / "tools" / "exec.py").write_text(
        (_REAL_PACK_ROOT / "tools" / "exec.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (pack_root / "schema.json").write_text(json.dumps({"catalog": {"labels": {}}}))
    (pack_root / "tables.json").write_text(json.dumps({"catalog": {}}))
    (pack_root / "pack.json").write_text(
        json.dumps(
            {
                "packId": "nlq-structured-query", "packVersion": "1.0.0", "role": "nlq-generator",
                "scorer": "extraction", "environment": {"requires": ["lmstudio-chat"]},
                "data": {"items": "items.jsonl", "tables": "tables.json", "schema": "schema.json"},
                "tools": {"module": "tools/exec.py", "entrypoint": "compile_and_execute"},
                "sampling": {"seed": 1, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
                "metrics": {
                    "verdictMetrics": ["layer1ExactMatchRate"],
                    "headlineMetric": "layer1ExactMatchRate",
                },
            }
        )
    )
    (pack_root / "items.jsonl").write_text(
        json.dumps({"itemId": "x", "dataset": "catalog", "answerable": True}) + "\n"
        + json.dumps({"itemId": "y", "dataset": "catalog", "answerable": False}) + "\n"
    )
    pack = load_pack(pack_root)
    problems = validate_pack(pack)
    assert not any("answerable" in p for p in problems)


def test_the_real_shipped_pack_validates_clean_except_for_the_still_pending_answerability_stamp():
    """S4 Step 3's own scope boundary (§7 Step 3's "Done when" / §6): the real shipped
    `items.jsonl` is NOT yet stamped with `answerable` — that is Step 4's `--stamp-answerability`
    run, which needs the real, human-snapshotted `tables.json` this step does not write. So the
    real pack is expected to show EXACTLY the missing-answerable-key problems and nothing else —
    proving the pack is otherwise fully valid."""
    pack = load_pack(_REAL_PACK_ROOT)
    problems = validate_pack(pack)
    assert problems, "expected the still-pending answerability stamp to be flagged"
    assert all("answerable" in p for p in problems), problems
