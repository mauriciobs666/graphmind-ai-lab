"""`modelbench.runner` — capture order, `LatencyBlock` accumulation, and the two driving loops
(spec §7 Step 1, `docs/plans/small-model-benchmarking-runner-spec.md`).

Offline throughout: hand-built stub `LMStudio`/`ToolEnvironment`/pack objects, matching
`tests/test_convo.py`'s and `tests/test_lmstudio.py`'s own "stub everything at the boundary"
convention. Test 15b's full case list (`docs/plans/small-model-benchmarking.md:6177-6274`) drives
`latency_block`/the two driving loops first, red→green; then v1.31's per-conversation
`ToolEnvironment` tests; then a light confirmation pass on `_drive_conversations`'s own reading of
`TurnTrace`/`ConversationTrace`; then the dispatch-failure note's E2-E4
(`docs/plans/small-model-benchmarking-ml-dispatch-failure.md` §5); then 12/12b's basis wiring.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelbench.convo import Conversation, ConversationTrace, Turn, TurnTrace
from modelbench.lmstudio import (
    ChatResult,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    ModelInfo,
    ResidentModel,
)
from modelbench.results import CallTiming, ClassificationAggregates, ItemResult, ItemTiming
from modelbench.runner import (
    UNEXPLAINED_MS_THRESHOLD,
    _drive_conversations,
    _drive_single_call_items,
    _gap_ms,
    _gap_withheld_for,
    _load_withheld_for,
    _turn_timings,
    latency_block,
)
from modelbench.tooling import DispatchRecord

# ==================================================================================================
# Shared fixtures — hand-built, offline (mirrors tests/test_convo.py's own convention)
# ==================================================================================================


def model_info(
    *, model_id: str = "qwen/qwen3-4b-2507", capabilities: tuple[str, ...] | None = ("tool_use",)
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        object="model",
        type="llm",
        publisher="qwen",
        arch="qwen3",
        compatibility_type="gguf",
        quantization="Q4_K_M",
        state="loaded",
        max_context_length=262144,
        capabilities=capabilities,
        loaded_context_length=8192,
    )


def resident(model_id: str = "qwen/qwen3-4b-2507") -> list[ResidentModel]:
    return [ResidentModel(id=model_id, state="loaded")]


def call_timing(
    *, wall_clock_ms=100.0, ttft_ms=10.0, generation_ms=80.0, prompt_tokens=50, tps=25.0
) -> CallTiming:
    return CallTiming(
        wallClockMs=wall_clock_ms,
        ttftMs=ttft_ms,
        generationMs=generation_ms,
        promptTokens=prompt_tokens,
        tokensPerSecond=tps,
    )


def item(
    item_id: str, timing: ItemTiming | None, *, outcome: str = "pass"
) -> ItemResult:
    return ItemResult(
        itemId=item_id,
        pairingKey=(item_id,),
        outcome=outcome,
        scoreable={},
        counts={},
        timing=timing,
    )


def timed_item(
    item_id: str, wall_clock_ms: float, *, calls: tuple[CallTiming, ...] = ()
) -> ItemResult:
    """A cleanly-timed single-call item: one call whose wall clock is the item's own."""
    if not calls:
        calls = (call_timing(wall_clock_ms=wall_clock_ms),)
    return item(item_id, ItemTiming(wallClockMs=wall_clock_ms, calls=calls, withheldFor=None))


class StubLMStudio:
    """A hand-built `LMStudio`-shaped stub: `.chat`/`.embed` pop responses in call order (an
    exception *instance* is raised instead of returned), `.residency` pops residency snapshots.
    Mirrors `tests/test_convo.py::stub_llm`'s own pattern one layer up."""

    def __init__(
        self,
        *,
        chat_responses: list[Any] | None = None,
        embed_responses: list[Any] | None = None,
        residency_sequence: list[list[ResidentModel]] | None = None,
        probe_result: str = "api-v0",
        catalog_result: list[Any] | None = None,
        warm_up_responses: list[Any] | None = None,
    ) -> None:
        self._chat = list(chat_responses or [])
        self._embed = list(embed_responses or [])
        self._residency = list(residency_sequence or [])
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.residency_calls = 0
        self._probe_result = probe_result
        self._catalog_result = list(catalog_result or [])
        self._warm_up = list(warm_up_responses or [])
        self.warm_up_calls: list[dict[str, Any]] = []

    def residency(self) -> list[ResidentModel]:
        self.residency_calls += 1
        return self._residency.pop(0) if self._residency else []

    def chat(self, messages, *, model, temperature=0.0, max_tokens=1, timeout_s, tools=None):
        self.chat_calls.append({"messages": messages, "model": model, "tools": tools})
        response = self._chat.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def embed(self, texts, *, model, timeout_s):
        self.embed_calls.append({"texts": texts, "model": model})
        response = self._embed.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def probe(self) -> str:
        return self._probe_result

    def catalog(self) -> list[Any]:
        return list(self._catalog_result)

    def warm_up(self, model_key, *, call_surface, system_prompt, was_resident_before, timeout_s):
        self.warm_up_calls.append({"model_key": model_key, "call_surface": call_surface})
        response = self._warm_up.pop(0) if self._warm_up else None
        if isinstance(response, BaseException):
            raise response
        if response is not None:
            return response
        from modelbench.lmstudio import LoadResult

        return LoadResult(
            wallClockMs=500.0,
            wasResidentBefore=was_resident_before,
            runtime={"name": "llama.cpp", "version": "1.52.0"} if call_surface == "chat" else None,
            stats=None,
        )


class StubToolEnvironment:
    """Mirrors `tests/test_convo.py::StubEnvironment` — its own `DispatchRecord`s, per the real
    plugin contract (`dispatch` recording is the environment's own job)."""

    def __init__(self, *, results: Mapping[str, Any] | None = None) -> None:
        self._results = dict(results or {})
        self._trace: list[DispatchRecord] = []
        self._state: dict[str, Any] = {}

    def schemas(self) -> list[dict[str, Any]]:
        return []

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        return_value = self._results.get(name, {"ok": True})
        self._trace.append(
            DispatchRecord(
                name=name,
                rawArguments=arguments,
                parsedArguments=arguments,
                returnValue=return_value,
                timestamp="2026-09-10T00:00:00Z",
            )
        )
        if name == "place_order":
            self._state["cart"] = []
            self._state["orders"] = self._state.get("orders", 0) + 1
        elif name == "add_to_cart":
            self._state.setdefault("cart", []).append(arguments)
        elif name == "view_cart":
            pass
        return return_value

    def trace(self) -> list[DispatchRecord]:
        return list(self._trace)

    def state(self) -> dict[str, Any]:
        return dict(self._state)


class EnvironmentFactory:
    """`pack.load_tool_module().build_environment` — records every `ToolEnvironment` it built, so
    v1.31's "one fresh instance per conversation" obligation is directly assertable."""

    def __init__(self) -> None:
        self.built: list[StubToolEnvironment] = []

    def build_environment(self) -> StubToolEnvironment:
        env = StubToolEnvironment(results=self._results_for_next())
        self.built.append(env)
        return env

    def _results_for_next(self) -> dict[str, Any]:
        return {"view_cart": {"cart": []}}


class FakePack:
    """A duck-typed stand-in for `packs.Pack` — every runner function reads `pack` structurally,
    never `isinstance`-checks it, so a hand-built object with the same surface is sufficient and
    matches this codebase's offline-stub convention (no on-disk pack tree needed)."""

    def __init__(
        self,
        *,
        role: str = "guard-judge",
        items: list[Mapping[str, Any]] | None = None,
        scripts: list[Conversation] | None = None,
        manifest: Mapping[str, Any] | None = None,
        tool_module: EnvironmentFactory | None = None,
        prompt_cfg: Any = None,
        pack_id: str = "fixture-pack",
    ) -> None:
        self.packId = pack_id
        self.packVersion = "1.0.0"
        self.role = role
        self.contentHash = "f" * 64
        self._items = list(items or [])
        self._scripts = list(scripts or [])
        self.manifest = dict(manifest or {})
        self._tool_module = tool_module
        self._prompt_cfg = prompt_cfg

    def prompt_config(self):
        return self._prompt_cfg

    def iter_items(self):
        return iter(self._items)

    def iter_scripts(self):
        return iter(self._scripts)

    def find_script(self, script_id: str) -> Conversation:
        for s in self._scripts:
            if s.scriptId == script_id:
                return s
        raise KeyError(script_id)

    def load_tool_module(self):
        return self._tool_module


def make_prompt_cfg(**overrides: Any):
    from modelbench.convo import PromptConfig

    base: dict[str, Any] = {
        "systemPrompt": "You are a helpful shop assistant.",
        "toolSchemas": (),
        "historyReplay": "structured",
        "representToolSchemasEachTurn": True,
        "historyTurns": 0,
        "maxIterationsPerTurn": 8,
        "temperature": 0.0,
        "maxTokens": 1024,
    }
    base.update(overrides)
    return PromptConfig(**base)


def script(
    script_id: str, n_turns: int = 1, *, shape: str = "A", replicate: int = 1
) -> Conversation:
    return Conversation(
        scriptId=script_id,
        shape=shape,
        replicate=replicate,
        turns=tuple(Turn(seq=i, user=f"turn {i}", expect={}) for i in range(n_turns)),
    )


def turn_trace(
    *,
    disposition: str = "replied",
    wall_clock_ms: float = 100.0,
    chat_results: tuple[ChatResult, ...] = (),
    iterations: int | None = None,
) -> TurnTrace:
    return TurnTrace(
        messagesSent=(),
        chatResults=chat_results,
        dispatches=(),
        envState={},
        iterations=iterations if iterations is not None else len(chat_results),
        turnDisposition=disposition,
        finalReplyText="ok" if disposition == "replied" else None,
        wallClockMs=wall_clock_ms,
    )


def chat_result_with_stats(
    *, wall_clock_ms=100.0, ttft_s=0.010, generation_s=0.080, prompt_tokens=50, tps=25.0
) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": "ok"},
        tool_calls=(),
        toolCallForm="prose",
        stats={
            "time_to_first_token": ttft_s,
            "generation_time": generation_s,
            "tokens_per_second": tps,
        },
        model_info=None,
        runtime=None,
        usage={"prompt_tokens": prompt_tokens},
        wallClockMs=wall_clock_ms,
    )


def chat_result_no_stats(*, wall_clock_ms=100.0) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": "ok"},
        tool_calls=(),
        toolCallForm="prose",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=wall_clock_ms,
    )


# ==================================================================================================
# `_gap_ms` / `_load_withheld_for` / `_gap_withheld_for` — the two load producers (spec §4)
# ==================================================================================================


def test_gap_ms_computes_wall_clock_minus_ttft_plus_generation():
    c = call_timing(wall_clock_ms=1200.0, ttft_ms=10.0, generation_ms=80.0)
    assert _gap_ms(c) == pytest.approx(1200.0 - (10.0 + 80.0))


@pytest.mark.parametrize(
    "field",
    ["wall_clock_ms", "ttft_ms", "generation_ms"],
)
def test_gap_ms_is_none_when_any_operand_is_none(field):
    kwargs = {"wall_clock_ms": 100.0, "ttft_ms": 10.0, "generation_ms": 80.0}
    kwargs[field] = None
    assert _gap_ms(call_timing(**kwargs)) is None


def test_load_withheld_for_none_when_model_was_resident():
    assert _load_withheld_for(model_info(), resident()) is None


def test_load_withheld_for_load_when_model_was_not_resident():
    assert _load_withheld_for(model_info(), []) == "load"


def test_gap_withheld_for_none_under_threshold():
    timing = ItemTiming(
        wallClockMs=1300.0,
        calls=(call_timing(wall_clock_ms=1300.0, ttft_ms=10.0, generation_ms=1280.0),),
        withheldFor=None,
    )
    assert timing.unexplainedMs < UNEXPLAINED_MS_THRESHOLD
    assert _gap_withheld_for(timing) is None


def test_gap_withheld_for_load_over_threshold():
    # -ml §11.5.1's own measured cold-call gap: 3485.6 ms, an over-threshold fixture value
    # (bounds the threshold from above; the real threshold, read directly, is 1000 ms).
    timing = ItemTiming(
        wallClockMs=5000.0,
        calls=(call_timing(wall_clock_ms=5000.0, ttft_ms=10.0, generation_ms=1504.4),),
        withheldFor=None,
    )
    assert timing.unexplainedMs == pytest.approx(3485.6)
    assert _gap_withheld_for(timing) == "load"


def test_gap_withheld_for_none_when_no_calls_have_a_reading():
    timing = ItemTiming(
        wallClockMs=100.0,
        calls=(call_timing(wall_clock_ms=100.0, ttft_ms=None, generation_ms=None),),
        withheldFor=None,
    )
    assert timing.unexplainedMs is None
    assert _gap_withheld_for(timing) is None


# ==================================================================================================
# `latency_block` — the nine invariants (spec §5), single-call fixtures
# ==================================================================================================


def test_latency_block_is_none_when_every_item_has_no_timing():
    items = [item("i1", None), item("i2", None)]
    assert latency_block(items, call_surface=None) is None


def test_latency_block_clean_cold_run_withholds_nothing():
    items = [timed_item(f"i{n}", 100.0 + n) for n in range(12)]
    block = latency_block(items, call_surface="chat")
    assert block.latencyTimedCount == block.latencyItemCount == 12
    assert block.latencyWithheldForLoad == 0
    assert block.latencyWithheldForNoResponse == 0
    assert block.latencyMsP50 is not None


def test_latency_block_load_withheld_item_keeps_its_call_siblings():
    """An item preceded by a not-resident snapshot has `latencyMs` withheld while `ttftMs`,
    prefill and `tokensPerSecond` are kept — -ml §11.4's measured ruling."""
    good = [timed_item(f"i{n}", 100.0) for n in range(11)]
    withheld_call = call_timing(wall_clock_ms=100.0, ttft_ms=10.0, generation_ms=80.0)
    withheld = item(
        "withheld", ItemTiming(wallClockMs=None, calls=(withheld_call,), withheldFor="load")
    )
    items = good + [withheld]
    block = latency_block(items, call_surface="chat")
    assert block.latencyItemCount == 12
    assert block.latencyTimedCount == 11
    assert block.latencyWithheldForLoad == 1
    assert block.latencyWithheldForNoResponse == 0
    # kept: the withheld item's call still contributes to statsCoveredCount / the medians
    assert block.statsCoveredCount == 12
    assert block.ttftMsMedian is not None


def test_latency_block_unexplained_ms_over_threshold_withholds_as_load():
    calls = (call_timing(wall_clock_ms=5000.0, ttft_ms=10.0, generation_ms=1504.4),)  # gap 3485.6
    timing = ItemTiming(wallClockMs=5000.0, calls=calls, withheldFor="load")
    items = [timed_item(f"i{n}", 100.0) for n in range(11)] + [item("gapped", timing)]
    block = latency_block(items, call_surface="chat")
    assert block.latencyWithheldForLoad == 1
    assert block.latencyTimedCount == 11
    assert block.unexplainedMsMax == pytest.approx(3485.6)


def test_latency_block_unexplained_ms_under_threshold_is_not_withheld():
    calls = (call_timing(wall_clock_ms=100.0, ttft_ms=10.0, generation_ms=88.0),)  # gap 2.0
    timing = ItemTiming(wallClockMs=100.0, calls=calls, withheldFor=None)
    items = [timed_item(f"i{n}", 100.0) for n in range(11)] + [item("clean", timing)]
    block = latency_block(items, call_surface="chat")
    assert block.latencyWithheldForLoad == 0
    assert block.latencyTimedCount == 12


def test_latency_block_timeout_scores_and_withholds_with_no_timing_figure():
    timing = ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")
    items = [timed_item(f"i{n}", 100.0) for n in range(11)] + [
        item("timedout", timing, outcome="fail")
    ]
    block = latency_block(items, call_surface="chat")
    assert block.latencyWithheldForNoResponse == 1
    assert block.latencyWithheldForLoad == 0
    assert block.latencyTimedCount == 11
    assert block.callCount == 11  # the timed-out item contributes zero calls


def test_latency_block_no_response_scores_and_withholds_with_no_timing_figure():
    timing = ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")
    items = [timed_item(f"i{n}", 100.0) for n in range(11)] + [
        item("noresponse", timing, outcome="fail")
    ]
    block = latency_block(items, call_surface="chat")
    # timeout and no_response share the SAME counter (-ml v1.14 §11.5.1, one counter over two
    # item states) — this is the assertion that pins that mapping.
    assert block.latencyWithheldForNoResponse == 1
    assert block.latencyWithheldForLoad == 0


def test_latency_block_co_presence_excludes_call_missing_prompt_tokens_from_all_three_medians():
    good = [timed_item(f"i{n}", 100.0) for n in range(11)]
    no_prompt_tokens = call_timing(wall_clock_ms=100.0, prompt_tokens=None)
    partial = timed_item("partial", 100.0, calls=(no_prompt_tokens,))
    items = good + [partial]
    block = latency_block(items, call_surface="chat")
    # excluded from statsCoveredCount...
    assert block.statsCoveredCount == 11
    # ...and it does not contaminate the medians (all computed from the 11 good calls' identical
    # inputs, so the median is unaffected by the excluded call's very different values)
    assert block.ttftMsMedian == pytest.approx(10.0)
    # the item's own latencyMs is unaffected — co-presence is a per-CALL exclusion, not per-item
    assert block.latencyTimedCount == 12


def test_latency_block_stats_less_chat_response_never_raises_and_counts_zero():
    calls = (
        call_timing(
            wall_clock_ms=100.0, ttft_ms=None, generation_ms=None, prompt_tokens=None, tps=None
        ),
    )
    items = [item("nostats", ItemTiming(wallClockMs=100.0, calls=calls, withheldFor=None))]
    block = latency_block(items, call_surface="chat")
    assert block.statsCoveredCount == 0
    assert block.ttftMsMedian is None


def test_latency_block_embeddings_surface_stats_covered_is_none_not_zero():
    items = [timed_item(f"i{n}", 100.0) for n in range(12)]
    block = latency_block(items, call_surface="embeddings")
    assert block.statsCoveredCount is None
    assert block.ttftMsMedian is None
    assert block.prefillMsPer1kMedian is None
    assert block.tokensPerSecondMedian is None
    assert block.unexplainedMsMax is None


def test_latency_block_identity_floor_renames_p95_to_max_at_small_x():
    items = [timed_item(f"i{n}", 100.0 + n) for n in range(12)]  # X = Y = 12 <= 19
    block = latency_block(items, call_surface="chat")
    assert block.latencyMsP95 is None
    assert block.latencyMsMax is not None
    assert block.latencyMsMax == max(i.latencyMs for i in items)


def _partial_coverage_block(x: int, y: int):
    good = [timed_item(f"i{n}", 100.0 + n) for n in range(x)]
    withheld = [
        item(f"w{n}", ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response"))
        for n in range(y - x)
    ]
    return latency_block(good + withheld, call_surface="chat")


def test_latency_block_coverage_gate_refuses_both_figures_below_both_floors():
    # -ml §11.6's own worked table: at Y = 38, neither figure prints at X <= 34.
    block = _partial_coverage_block(x=34, y=38)
    assert block.latencyMsP50 is None
    assert block.latencyMsP95 is None
    assert block.latencyMsMax is None


def test_latency_block_p50_survives_lower_coverage_than_the_tail_figure():
    # -ml §11.6's own worked table: at Y = 38, p50 alone prints at X = 35.
    block = _partial_coverage_block(x=35, y=38)
    assert block.latencyMsP50 is not None
    assert block.latencyMsP95 is None
    assert block.latencyMsMax is None


def test_latency_block_both_figures_print_once_coverage_clears_the_tail_floor():
    # -ml §11.6's own worked table: at Y = 38, both print at X >= 36.
    block = _partial_coverage_block(x=36, y=38)
    assert block.latencyMsP50 is not None
    assert block.latencyMsP95 is not None


def test_latency_block_call_attempted_count_is_call_count_plus_no_response():
    good = [timed_item(f"i{n}", 100.0) for n in range(3)]
    failed = item(
        "failed", ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")
    )
    block = latency_block(good + [failed], call_surface="chat")
    assert block.callCount == 3
    assert block.latencyWithheldForNoResponse == 1
    assert block.callAttemptedCount == 4


# ==================================================================================================
# `_turn_timings` — the multi-call cases (a)-(f) (spec §5, plan §5 test 15b)
# ==================================================================================================


def _warm_call(wall_clock_ms=430.0) -> ChatResult:
    # ttft + generation close to wallClockMs -- a small, warm per-call gap (well under 1000ms)
    return chat_result_with_stats(wall_clock_ms=wall_clock_ms, ttft_s=0.010, generation_s=0.400)


def test_turn_timings_case_a_warm_multi_call_turn_reports_no_load_contamination():
    """(a) A warm 3-iteration turn, and one at the cap, report NO model-load contamination and
    KEEP their latencyMs — the regression a per-call reading of the gap produces."""
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="replied",
                wall_clock_ms=1300.0,
                chat_results=(_warm_call(), _warm_call(), _warm_call()),
            ),
            turn_trace(
                disposition="cap-hit",
                wall_clock_ms=3400.0,
                chat_results=tuple(_warm_call() for _ in range(8)),
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace, resident_before=resident(), model_id="qwen/qwen3-4b-2507", lmstudio=StubLMStudio()
    )
    assert timings[0].withheldFor is None
    assert timings[0].wallClockMs == 1300.0
    assert timings[1].withheldFor is None
    assert timings[1].wallClockMs == 3400.0
    assert len(timings[1].calls) == 8


def test_turn_timings_case_b_one_call_with_a_load_gap_withholds_the_whole_turn():
    """(b) A 3-iteration turn one of whose calls carries a 3 485.6 ms gap IS withheld and counted
    under model load, while the other two calls' sibling figures are KEPT."""
    gapped = chat_result_with_stats(wall_clock_ms=5000.0, ttft_s=0.010, generation_s=1.5044)
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="replied",
                wall_clock_ms=6000.0,
                chat_results=(_warm_call(), gapped, _warm_call()),
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace, resident_before=resident(), model_id="qwen/qwen3-4b-2507", lmstudio=StubLMStudio()
    )
    assert timings[0].withheldFor == "load"
    assert len(timings[0].calls) == 3
    assert all(c.ttftMs is not None for c in timings[0].calls)


def test_turn_timings_case_c_one_call_with_no_stats_is_not_withheld_and_others_count():
    """(c) A 3-iteration turn one of whose calls has no `stats` has `unexplainedMs is None`, is
    NOT withheld, and contributes its two readable calls to `statsCoveredCount`."""
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="replied",
                wall_clock_ms=1300.0,
                chat_results=(
                    _warm_call(), chat_result_no_stats(wall_clock_ms=430.0), _warm_call()
                ),
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace, resident_before=resident(), model_id="qwen/qwen3-4b-2507", lmstudio=StubLMStudio()
    )
    assert timings[0].withheldFor is None
    assert timings[0].unexplainedMs is None
    readable = [c for c in timings[0].calls if c.ttftMs is not None]
    assert len(readable) == 2


def test_turn_timings_case_d_third_call_raises_leaves_two_completed_calls_no_response():
    """(d) A 3-iteration turn whose third call raises: `iterations == len(timing.calls) == 2`,
    `ItemTiming.wallClockMs is None`, `withheldFor == "no_response"`, and both completed calls
    contribute to `callCount`/eligible for `statsCoveredCount` (P15-2)."""
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="server-rejected",
                wall_clock_ms=900.0,
                chat_results=(_warm_call(), _warm_call()),
                iterations=2,
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace, resident_before=resident(), model_id="qwen/qwen3-4b-2507", lmstudio=StubLMStudio()
    )
    assert timings[0].wallClockMs is None
    assert timings[0].withheldFor == "no_response"
    assert timings[0].callCount == 2


def test_turn_timings_case_e_timed_out_turn_with_a_load_gap_lands_in_no_response_not_load():
    """(e) A 3-iteration timed-out turn, one of whose completed calls carries a 3 485.6 ms gap,
    lands in `latencyWithheldForNoResponse`, is ABSENT from `latencyWithheldForLoad` (the
    disposition-over-load precedence, §3.6) — with `unexplainedMs` still stored."""
    gapped = chat_result_with_stats(wall_clock_ms=5000.0, ttft_s=0.010, generation_s=1.5044)
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="timed-out",
                wall_clock_ms=8000.0,
                chat_results=(_warm_call(), gapped),
                iterations=2,
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace, resident_before=resident(), model_id="qwen/qwen3-4b-2507", lmstudio=StubLMStudio()
    )
    # the item STATE is "timeout" (the disposition's own record); the counter it lands in is the
    # shared latencyWithheldForNoResponse (-ml v1.14 §11.5.1's one counter over two states) —
    # never latencyWithheldForLoad, which is the precedence rule this case pins.
    assert timings[0].withheldFor == "timeout"
    assert timings[0].unexplainedMs > UNEXPLAINED_MS_THRESHOLD  # still stored, still eligible
    block = latency_block([item("i0", timings[0])], call_surface="chat")
    assert block.latencyWithheldForNoResponse == 1
    assert block.latencyWithheldForLoad == 0


def test_turn_timings_precedence_disposition_wins_over_the_residency_guard():
    """The same run's residency-guard half of case (e): an item whose PRECEDING residency snapshot
    showed the model not resident and whose turn then times out is counted under no_response and
    not under load — the mutation-target precedence rule (§3.6, P15-3/P16-1)."""
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="timed-out", wall_clock_ms=8000.0, chat_results=(), iterations=0
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace,
        resident_before=[],  # NOT resident — the load guard would fire here if evaluated first
        model_id="qwen/qwen3-4b-2507",
        lmstudio=StubLMStudio(),
    )
    assert timings[0].withheldFor == "timeout"


def test_turn_timings_cap_hit_is_never_folded_into_incomplete():
    trace = ConversationTrace(
        scriptId="s1",
        shape="A",
        replicate=1,
        turns=(
            turn_trace(
                disposition="cap-hit", wall_clock_ms=2000.0, chat_results=(_warm_call(),) * 8
            ),
        ),
    )
    timings, _ = _turn_timings(
        trace, resident_before=resident(), model_id="qwen/qwen3-4b-2507", lmstudio=StubLMStudio()
    )
    assert timings[0].wallClockMs == 2000.0
    assert timings[0].withheldFor is None


# ==================================================================================================
# `_drive_single_call_items` — the four item-level roles
# ==================================================================================================


class FakeItemScorer:
    """Records every call; `score_item` returns a minimal `ItemResult` carrying the runner's
    timing verbatim (§3.2: the scorer receives `timing` as an input, never invents it)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, ItemTiming]] = []

    def score_item(self, item_input, result, timing, *, pack):
        self.calls.append((item_input, result, timing))
        outcome = "fail" if timing.withheldFor == "timeout" else (
            "unrunnable" if timing.withheldFor == "no_response" else "pass"
        )
        return ItemResult(
            itemId=str(item_input.get("id", len(self.calls))),
            pairingKey=(str(item_input.get("id", len(self.calls))),),
            outcome=outcome,
            scoreable={},
            counts={},
            timing=timing,
        )

    def aggregate(self, items, *, pack):
        return ClassificationAggregates(perClass=(), n=len(items))


def _cfg(**overrides):
    from modelbench.runner import RunConfig

    base = dict(modelKey="qwen/qwen3-4b-2507", sessionId=None, referenceKey=None)
    base.update(overrides)
    return RunConfig(**base)


def test_drive_single_call_items_happy_path_scores_every_item(monkeypatch):
    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = FakePack(
        role="guard-judge",
        items=[{"id": "a"}, {"id": "b"}],
        prompt_cfg=make_prompt_cfg(maxIterationsPerTurn=None),
    )
    lms = StubLMStudio(
        chat_responses=[
            chat_result_with_stats(wall_clock_ms=100.0),
            chat_result_with_stats(wall_clock_ms=110.0),
        ],
        residency_sequence=[resident(), resident()],
    )
    items, design_effect, basis = _drive_single_call_items(
        pack,
        _cfg(),
        lmstudio=lms,
        model_info=model_info(),
        call_surface="chat",
        baseline_residency=resident(),
    )
    assert len(items) == 2
    assert design_effect == 1.0
    assert basis == "by-construction"
    assert all(i.timing is not None and i.timing.withheldFor is None for i in items)


def test_drive_single_call_items_timeout_scores_fail_with_no_calls(monkeypatch):
    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = FakePack(role="guard-judge", items=[{"id": "a"}], prompt_cfg=make_prompt_cfg())
    lms = StubLMStudio(
        chat_responses=[LMStudioCallTimeout("timed out")], residency_sequence=[resident()]
    )
    items, _, _ = _drive_single_call_items(
        pack,
        _cfg(),
        lmstudio=lms,
        model_info=model_info(),
        call_surface="chat",
        baseline_residency=resident(),
    )
    assert items[0].outcome == "fail"
    assert items[0].timing.withheldFor == "timeout"
    assert items[0].timing.calls == ()


def test_drive_single_call_items_no_response_scores_unrunnable_with_no_calls(monkeypatch):
    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = FakePack(role="guard-judge", items=[{"id": "a"}], prompt_cfg=make_prompt_cfg())
    lms = StubLMStudio(
        chat_responses=[LMStudioCallFailed("HTTP 500", status=500)],
        residency_sequence=[resident()],
    )
    items, _, _ = _drive_single_call_items(
        pack,
        _cfg(),
        lmstudio=lms,
        model_info=model_info(),
        call_surface="chat",
        baseline_residency=resident(),
    )
    assert items[0].outcome == "unrunnable"
    assert items[0].timing.withheldFor == "no_response"
    assert items[0].timing.calls == ()


def test_drive_single_call_items_residency_probe_withholds_load(monkeypatch):
    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = FakePack(role="guard-judge", items=[{"id": "a"}], prompt_cfg=make_prompt_cfg())
    lms = StubLMStudio(
        chat_responses=[chat_result_with_stats(wall_clock_ms=100.0)],
        residency_sequence=[[]],  # NOT resident before the item's call
    )
    items, _, _ = _drive_single_call_items(
        pack,
        _cfg(),
        lmstudio=lms,
        model_info=model_info(),
        call_surface="chat",
        baseline_residency=resident(),
    )
    assert items[0].timing.withheldFor == "load"
    assert items[0].latencyMs is None
    assert items[0].timing.wallClockMs is not None  # the raw wall clock survives (readable)
    assert items[0].timing.calls[0].ttftMs is not None  # siblings kept


def test_drive_single_call_items_case_f_failing_call_wins_over_the_residency_guard(monkeypatch):
    """(f), P16-1: a `guard-judge` item whose preceding residency snapshot showed the model NOT
    resident, and whose single call then returns an HTTP 500, is `withheldFor == "no_response"` —
    NEVER `"load"`. The mutation that makes this a test rather than a demonstration: evaluate the
    residency guard first (drop the `try/except` ordering) and this assertion is the one that
    reddens."""
    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = FakePack(role="guard-judge", items=[{"id": "a"}], prompt_cfg=make_prompt_cfg())
    lms = StubLMStudio(
        chat_responses=[LMStudioCallFailed("HTTP 500", status=500)],
        residency_sequence=[[]],  # NOT resident
    )
    items, _, _ = _drive_single_call_items(
        pack,
        _cfg(),
        lmstudio=lms,
        model_info=model_info(),
        call_surface="chat",
        baseline_residency=resident(),
    )
    assert items[0].timing.withheldFor == "no_response"
    block = latency_block(list(items), call_surface="chat")
    assert block.latencyWithheldForLoad == 0
    assert block.latencyWithheldForNoResponse == 1
    assert block.callAttemptedCount == block.latencyItemCount  # rule (ii)'s single-call branch


def test_drive_single_call_items_embeddings_surface_calls_embed_not_chat(monkeypatch):
    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = FakePack(role="embedder", items=[{"id": "a"}], prompt_cfg=make_prompt_cfg())
    from modelbench.lmstudio import EmbedResult

    embed_result = EmbedResult(
        vectors=((0.1, 0.2),), dimension=2, model="text-embedding", usage=None, wallClockMs=15.0
    )
    lms = StubLMStudio(
        embed_responses=[embed_result], residency_sequence=[resident("text-embedding")]
    )
    items, _, _ = _drive_single_call_items(
        pack,
        _cfg(),
        lmstudio=lms,
        model_info=model_info(model_id="text-embedding", capabilities=None),
        call_surface="embeddings",
        baseline_residency=resident(),
    )
    assert len(lms.embed_calls) == 1
    assert len(lms.chat_calls) == 0
    assert items[0].timing.withheldFor is None


# ==================================================================================================
# `_drive_conversations` — v1.31's per-conversation `ToolEnvironment`, censoring, basis wiring
# ==================================================================================================


class FakeConversationScorer:
    """Returns one `ItemResult` per turn (mirroring the real per-turn analysis unit) and a canned
    `determinismProbe`-carrying aggregates object — the spec's own §8 test-strategy guidance:
    "asserted against a stub ConversationScorer returning a canned determinismProbe dict"."""

    def __init__(self, determinism_probe: dict[str, Any] | None = None) -> None:
        self.determinism_probe = determinism_probe or {
            "scriptIds": [],
            "ran": False,
            "identical": False,
            "differingTurns": [],
        }
        self.scored_calls: list[Any] = []
        self.probes_calls: list[Any] = []

    def score_conversations(self, scored, probes, *, pack):
        self.scored_calls = list(scored)
        self.probes_calls = list(probes)
        items = []
        for script, trace, timings in scored:
            for i, (turn, timing) in enumerate(zip(trace.turns, timings, strict=True)):
                items.append(
                    ItemResult(
                        itemId=f"{script.scriptId}#{i}",
                        pairingKey=(script.scriptId, script.replicate, i),
                        outcome="pass" if turn.turnDisposition == "replied" else "unrunnable",
                        scoreable={},
                        counts={},
                        timing=timing,
                    )
                )

        class _Aggregates:
            determinismProbe = self.determinism_probe

        return tuple(items), _Aggregates()


def _tool_caller_pack(
    *,
    scripts: list[Conversation],
    determinism_probe_scripts: list[str] | None = None,
    replicates_per_script: int = 1,
    tool_module: EnvironmentFactory | None = None,
) -> FakePack:
    return FakePack(
        role="tool-caller",
        scripts=scripts,
        manifest={
            "sampling": {
                "determinismProbeScripts": determinism_probe_scripts or [],
                "replicatesPerScript": replicates_per_script,
            }
        },
        tool_module=tool_module or EnvironmentFactory(),
        prompt_cfg=make_prompt_cfg(maxIterationsPerTurn=8),
    )


def _place_order_reply() -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": "ordered"},
        tool_calls=(),
        toolCallForm="prose",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=50.0,
    )


def test_drive_conversations_builds_one_fresh_environment_per_conversation(monkeypatch):
    """v1.31: twelve distinct `ToolEnvironment` instances across a twelve-conversation fixture
    pack, none shared across conversations."""
    scorer = FakeConversationScorer()
    monkeypatch.setattr("modelbench.runner._load_conversation_scorer", lambda pack: scorer)
    scripts = [script(f"script-{n:02d}", n_turns=1) for n in range(12)]
    factory = EnvironmentFactory()
    pack = _tool_caller_pack(scripts=scripts, tool_module=factory)
    lms = StubLMStudio(
        chat_responses=[chat_result_with_stats(wall_clock_ms=50.0) for _ in range(12)],
        residency_sequence=[resident() for _ in range(13)],
    )
    _drive_conversations(
        pack, _cfg(), lmstudio=lms, model_info=model_info(), baseline_residency=resident()
    )
    assert len(factory.built) == 12
    assert len(set(id(e) for e in factory.built)) == 12


def test_drive_conversations_conversation_k_plus_1_opens_with_an_empty_cart(monkeypatch):
    """v1.31's second done-condition: conversation k+1 opens with an empty cart after conversation
    k's script places an order — a `place_order` call in script k, a `view_cart` first turn in
    script k+1, asserting an empty result."""
    scorer = FakeConversationScorer()
    monkeypatch.setattr("modelbench.runner._load_conversation_scorer", lambda pack: scorer)

    scripts = [script("order-script", n_turns=1), script("check-script", n_turns=1)]
    factory = EnvironmentFactory()
    pack = _tool_caller_pack(scripts=scripts, tool_module=factory)

    place_order_call = ChatResult(
        message={"role": "assistant", "content": None},
        tool_calls=(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "place_order", "arguments": "{}"},
            },
        ),
        toolCallForm="native",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=50.0,
    )
    lms = StubLMStudio(
        chat_responses=[place_order_call, _place_order_reply(), _place_order_reply()],
        residency_sequence=[resident(), resident(), resident()],
    )
    _drive_conversations(
        pack, _cfg(), lmstudio=lms, model_info=model_info(), baseline_residency=resident()
    )
    # conversation 2's environment is a FRESH one -- an empty cart by construction, since nothing
    # was ever dispatched against it (the place_order call landed on conversation 1's own env).
    second_env = factory.built[1]
    assert second_env.state().get("cart", []) == []
    assert "orders" not in second_env.state()
    # and conversation 1's own environment really did record the order (the mechanism, not just
    # the absence, is what the test pins).
    assert factory.built[0].state().get("orders") == 1


def test_drive_conversations_reads_turn_trace_and_conversation_trace_correctly(monkeypatch):
    """A light confirmation pass: `_drive_conversations` threads `ConversationTrace`/`TurnTrace`
    from `drive()` into the scorer unchanged (10/10b/10c's own behavior is `convo.drive`'s tested
    surface, not re-tested here)."""
    scorer = FakeConversationScorer()
    monkeypatch.setattr("modelbench.runner._load_conversation_scorer", lambda pack: scorer)
    scripts = [script("s1", n_turns=2)]
    pack = _tool_caller_pack(scripts=scripts)
    lms = StubLMStudio(
        chat_responses=[
            chat_result_with_stats(wall_clock_ms=50.0),
            chat_result_with_stats(wall_clock_ms=60.0),
        ],
        residency_sequence=[resident(), resident()],
    )
    _drive_conversations(
        pack, _cfg(), lmstudio=lms, model_info=model_info(), baseline_residency=resident()
    )
    assert len(scorer.scored_calls) == 1
    driven_script, trace, timings = scorer.scored_calls[0]
    assert driven_script.scriptId == "s1"
    assert isinstance(trace, ConversationTrace)
    assert len(trace.turns) == 2
    assert len(timings) == 2
    assert all(t.turnDisposition == "replied" for t in trace.turns)


# ==================================================================================================
# Dispatch-failure note E2-E4 (`docs/plans/small-model-benchmarking-ml-dispatch-failure.md` §5)
# E1 (sim totality) is S5's; E5 (disclosure survives storage) is `compare`'s already-shipped read
# side — neither is this step's to rebuild.
# ==================================================================================================


class RaisingToolEnvironment(StubToolEnvironment):
    """A `ToolEnvironment` whose `dispatch` raises on a named tool call — the pack defect
    `ToolDispatchFailed` exists to name (dispatch-failure note §4(b))."""

    def __init__(self, *, raise_on: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._raise_on = raise_on

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name == self._raise_on:
            raise RuntimeError(f"{name} is broken")
        return super().dispatch(name, arguments)


def _tool_call_chat_result(tool_name: str) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": None},
        tool_calls=(
            {"id": "c1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}},
        ),
        toolCallForm="native",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=50.0,
    )


def test_drive_conversations_e2_censoring_is_wired(monkeypatch):
    """E2: a 9-turn script whose sim raises at turn 4 (0-indexed turn 3) is stored censored — the
    runner catches `ToolDispatchFailed`, stores the 3 completed turns, records one disclosure, and
    proceeds to the next script."""
    scorer = FakeConversationScorer()
    monkeypatch.setattr("modelbench.runner._load_conversation_scorer", lambda pack: scorer)

    censored_script = script("dispatch-fails", n_turns=9)
    clean_script = script("clean", n_turns=1)

    class Factory:
        built: list[Any] = []

        def build_environment(self):
            env = (
                RaisingToolEnvironment(raise_on="lookup_product_fact")
                if not self.built
                else StubToolEnvironment()
            )
            self.built.append(env)
            return env

    tool_module = Factory()
    pack = _tool_caller_pack(scripts=[censored_script, clean_script], tool_module=tool_module)

    # turns 1-3 reply cleanly, turn 4 issues the tool call that raises.
    responses = [chat_result_with_stats(wall_clock_ms=50.0) for _ in range(3)]
    responses.append(_tool_call_chat_result("lookup_product_fact"))
    responses.append(chat_result_with_stats(wall_clock_ms=50.0))  # the clean script's one turn
    lms = StubLMStudio(chat_responses=responses, residency_sequence=[resident()] * 5)

    items, disclosures, basis, design_effect, aggregates = _drive_conversations(
        pack, _cfg(), lmstudio=lms, model_info=model_info(), baseline_residency=resident()
    )
    assert len(disclosures) == 1
    disclosure = disclosures[0]
    assert disclosure.scriptId == "dispatch-fails"
    assert disclosure.turn == 3  # 0-indexed turnIndex
    assert disclosure.tool == "lookup_product_fact"
    assert "RuntimeError" in disclosure.reason

    censored_trace = scorer.scored_calls[0][1]
    assert censored_trace.scriptId == "dispatch-fails"
    assert len(censored_trace.turns) == 3  # nothing at or after the failing turn
    # the runner proceeded to the next script with a fresh environment
    assert scorer.scored_calls[1][0].scriptId == "clean"
    assert len(tool_module.built) == 2


def test_drive_conversations_e4_censoring_differs_from_a_normal_scored_failure(monkeypatch):
    """E4, the negative control: a dispatch-censored conversation and one with a NORMAL scored
    failure at the same turn must render differently. If a censored trace and a scored-failure
    trace had the same shape, E2 would have passed on a tautology (censoring not actually wired
    would look identical to a mundane no-response turn)."""
    scorer = FakeConversationScorer()
    monkeypatch.setattr("modelbench.runner._load_conversation_scorer", lambda pack: scorer)

    # --- run A: dispatch raises at turn 4 of a 9-turn script -----------------------------------
    class RaisingFactory:
        def build_environment(self):
            return RaisingToolEnvironment(raise_on="lookup_product_fact")

    pack_a = _tool_caller_pack(scripts=[script("a", n_turns=9)], tool_module=RaisingFactory())
    responses_a = [chat_result_with_stats(wall_clock_ms=50.0) for _ in range(3)]
    responses_a.append(_tool_call_chat_result("lookup_product_fact"))
    lms_a = StubLMStudio(chat_responses=responses_a, residency_sequence=[resident()] * 4)
    _, disclosures_a, _, _, _ = _drive_conversations(
        pack_a, _cfg(), lmstudio=lms_a, model_info=model_info(), baseline_residency=resident()
    )
    trace_a = scorer.scored_calls[0][1]

    # --- run B: turn 4 is a normal scored failure (server 500), never a dispatch raise ---------
    scorer_b = FakeConversationScorer()
    monkeypatch.setattr("modelbench.runner._load_conversation_scorer", lambda pack: scorer_b)
    pack_b = _tool_caller_pack(scripts=[script("b", n_turns=9)], tool_module=EnvironmentFactory())
    responses_b = [chat_result_with_stats(wall_clock_ms=50.0) for _ in range(3)]
    responses_b.append(LMStudioCallFailed("HTTP 500", status=500))  # turn 4: fails, not censored
    responses_b.extend(chat_result_with_stats(wall_clock_ms=50.0) for _ in range(5))  # turns 5-9
    lms_b = StubLMStudio(chat_responses=responses_b, residency_sequence=[resident()] * 9)
    _, disclosures_b, _, _, _ = _drive_conversations(
        pack_b, _cfg(), lmstudio=lms_b, model_info=model_info(), baseline_residency=resident()
    )
    trace_b = scorer_b.scored_calls[0][1]

    # The discrimination: censoring truncates the stored conversation; a scored failure does not
    # (§4.1's never-skip-a-turn rule keeps driving turns 5-9 past a mere no-response/timeout).
    assert len(trace_a.turns) == 3
    assert len(trace_b.turns) == 9
    assert len(disclosures_a) == 1
    assert len(disclosures_b) == 0
    assert trace_b.turns[3].turnDisposition == "server-rejected"


# ==================================================================================================
# Test 12/12b — the determinism probe's `basis` wiring (four cases, stub ConversationScorer)
# ==================================================================================================


def _basis_pack(*, determinism_probe_scripts, replicates_per_script=1):
    scripts = [script("a", n_turns=1), script("b", n_turns=1)]
    return _tool_caller_pack(
        scripts=scripts,
        determinism_probe_scripts=determinism_probe_scripts,
        replicates_per_script=replicates_per_script,
    )


def _run_basis(pack, *, determinism_probe, n_chat_calls):
    scorer = FakeConversationScorer(determinism_probe=determinism_probe)
    lms = StubLMStudio(
        chat_responses=[chat_result_with_stats(wall_clock_ms=50.0) for _ in range(n_chat_calls)],
        residency_sequence=[resident() for _ in range(n_chat_calls + 1)],
    )
    import modelbench.runner as runner_module

    original = runner_module._load_conversation_scorer
    runner_module._load_conversation_scorer = lambda pack: scorer
    try:
        return _drive_conversations(
            pack, _cfg(), lmstudio=lms, model_info=model_info(), baseline_residency=resident()
        )
    finally:
        runner_module._load_conversation_scorer = original


def test_basis_probe_ran_and_identical_is_by_construction():
    pack = _basis_pack(determinism_probe_scripts=["a"])
    _, _, basis, _, _ = _run_basis(
        pack,
        determinism_probe=dict(scriptIds=["a"], ran=True, identical=True, differingTurns=[]),
        n_chat_calls=3,  # 2 scored scripts + 1 probe script
    )
    assert basis == "by-construction"


def test_basis_probe_ran_and_differed_is_assumed():
    pack = _basis_pack(determinism_probe_scripts=["a"])
    _, _, basis, _, _ = _run_basis(
        pack,
        determinism_probe=dict(scriptIds=["a"], ran=True, identical=False, differingTurns=[0]),
        n_chat_calls=3,
    )
    assert basis == "assumed"


def test_basis_probe_did_not_run_is_assumed_the_fail_safe():
    # no probe scripts declared -> the probe never runs
    pack = _basis_pack(determinism_probe_scripts=[])
    _, _, basis, _, _ = _run_basis(
        pack,
        determinism_probe=dict(scriptIds=[], ran=False, identical=False, differingTurns=[]),
        n_chat_calls=2,  # 2 scored scripts, no probe
    )
    assert basis == "assumed"


def test_basis_replicates_per_script_above_one_is_assumed_regardless():
    pack = _basis_pack(determinism_probe_scripts=["a"], replicates_per_script=2)
    _, _, basis, _, _ = _run_basis(
        pack,
        # even a "perfect" probe result must not buy by-construction under replicatesPerScript > 1
        determinism_probe=dict(scriptIds=["a"], ran=True, identical=True, differingTurns=[]),
        n_chat_calls=3,
    )
    assert basis == "assumed"


def test_basis_design_effect_is_always_1_by_construction_for_the_tool_caller_pack():
    """Module docstring gap 1: `-ml` §4.5.1/R1's own "DEFF 1.00 by construction" — the 12×1
    design's sampling unit is the script, matching the analysis unit, so no clustering exists at
    the run's own `designEffect` regardless of the determinism probe's outcome."""
    pack = _basis_pack(determinism_probe_scripts=["a"])
    _, _, _, design_effect, _ = _run_basis(
        pack,
        determinism_probe=dict(scriptIds=["a"], ran=True, identical=False, differingTurns=[0]),
        n_chat_calls=3,
    )
    assert design_effect == 1.0


# ==================================================================================================
# `run_pack` — capture order: refusal paths, and one full offline happy path
# ==================================================================================================


def _host_json(root, *, runtime_name="llama.cpp", runtime_version="1.52.0"):
    from modelbench import hostinfo

    hostinfo.write_host_info(
        root,
        {
            "schemaVersion": 1,
            "apiBaseUrl": "http://localhost:1234",
            "attested": {
                "lmStudioAppVersion": "0.3.31",
                "kvCacheSetting": "f16",
                "hostRamGb": 16,
                "otherResidentWorkloads": [],
            },
            "attestedAt": "2026-09-10T00:00:00Z",
            "observedAtAttestation": {
                "residencySource": "lmstudio-api-v0",
                "runtimeName": runtime_name,
                "runtimeVersion": runtime_version,
                "runtimeObservedAt": "2026-09-10T00:00:00Z",
            },
        },
    )


def _item_pack(**kwargs):
    return FakePack(
        role="guard-judge",
        items=[{"id": "a"}],
        manifest={"environment": {"requires": ["lmstudio-chat"]}},
        prompt_cfg=make_prompt_cfg(maxIterationsPerTurn=None),
        **kwargs,
    )


def test_run_pack_refuses_exit_5_when_host_json_is_absent(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=StubLMStudio(), root=tmp_path)
    assert exc_info.value.exitCode == 5


def test_run_pack_refuses_exit_3_when_lm_studio_is_unreachable(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    _host_json(tmp_path)
    lms = StubLMStudio(probe_result="unreachable")
    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert exc_info.value.exitCode == 3


def test_run_pack_refuses_exit_3_when_lm_studio_is_v1_only(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    _host_json(tmp_path)
    lms = StubLMStudio(probe_result="v1-only")
    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert exc_info.value.exitCode == 3


def test_run_pack_refuses_exit_4_when_model_is_not_in_the_catalog(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    _host_json(tmp_path)
    lms = StubLMStudio(catalog_result=[])
    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert exc_info.value.exitCode == 4


def test_run_pack_refuses_exit_4_on_call_surface_cross_check_mismatch(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    _host_json(tmp_path)
    embeddings_model = model_info(model_id="qwen/qwen3-4b-2507")
    object.__setattr__(embeddings_model, "type", "embeddings")  # frozen dataclass
    lms = StubLMStudio(catalog_result=[embeddings_model])
    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert exc_info.value.exitCode == 4


def test_run_pack_refuses_exit_3_when_warm_up_times_out(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    _host_json(tmp_path)
    lms = StubLMStudio(
        catalog_result=[model_info()], warm_up_responses=[LMStudioCallTimeout("timed out")]
    )
    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert exc_info.value.exitCode == 3


def test_run_pack_refuses_exit_5_when_attestation_is_stale(tmp_path):
    from modelbench.runner import RunRefused, run_pack

    _host_json(tmp_path, runtime_name="llama.cpp", runtime_version="1.51.0")  # attested: 1.51.0
    lms = StubLMStudio(catalog_result=[model_info()])  # warm-up reports runtime 1.52.0 (mismatch)
    with pytest.raises(RunRefused) as exc_info:
        run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert exc_info.value.exitCode == 5


def test_run_pack_happy_path_stores_a_complete_run(tmp_path, monkeypatch):
    """Full offline capture order, item-level role: no refusal fires, the fingerprint and
    `LatencyBlock` are both assembled, and `attestationTripWire` is populated."""
    from modelbench.runner import run_pack

    scorer = FakeItemScorer()
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    _host_json(tmp_path)
    lms = StubLMStudio(
        catalog_result=[model_info()],
        chat_responses=[chat_result_with_stats(wall_clock_ms=100.0)],
        residency_sequence=[resident(), resident(), resident(), resident()],
    )
    run, disclosures = run_pack(_item_pack(), _cfg(), lmstudio=lms, root=tmp_path)
    assert disclosures == ()
    assert run.fingerprint.validate() == []
    assert run.latency is not None
    assert run.latency.latencyItemCount == 1
    assert run.attestationTripWire == "compared"
    assert run.designEffect == 1.0
    assert run.basis == "by-construction"
