"""Unit tests for the OpenAI-compatible chat client (K-013; K-042 Landing 1 renamed it
from `LMStudioLLM`), offline via injected transport.

Mirrors `test_embedding.py`: the real `OpenAICompatibleLLM` is exercised with an injected
transport so its request payload + `choices[0].message.content` parsing are pinned
without a live LM Studio server. Unit tests never touch the network.
"""

from __future__ import annotations

import json

import pytest

from falkorchat.llm import OpenAICompatibleLLM


def test_lmstudio_llm_parses_content_and_posts_expected_payload():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "hi there"}}]}

    llm = OpenAICompatibleLLM(
        base_url="http://localhost:1234/v1", model="qwen/qwen3-4b-2507",
        transport=fake_transport,
    )

    out = llm.complete([{"role": "user", "content": "hello"}])

    assert out == "hi there"
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["payload"] == {
        "model": "qwen/qwen3-4b-2507",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_lmstudio_llm_strips_trailing_slash_on_base_url():
    def fake_transport(url: str, payload: dict) -> dict:
        assert url == "http://localhost:1234/v1/chat/completions"
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = OpenAICompatibleLLM(
        base_url="http://localhost:1234/v1/", model="m", transport=fake_transport
    )
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"


def test_llm_requires_explicit_base_url_and_model():
    # K-042 FR-20: model choice is no longer an env var — base_url/model have no
    # config-derived defaults, so both are required positional args.
    def fake_transport(url: str, payload: dict) -> dict:
        assert url == "http://localhost:1234/v1/chat/completions"
        assert payload["model"] == "qwen/qwen3-4b-2507"
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = OpenAICompatibleLLM(
        "http://localhost:1234/v1", "qwen/qwen3-4b-2507", transport=fake_transport
    )
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"


def test_llm_merges_params_into_the_chat_payload():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = OpenAICompatibleLLM(
        "http://x/v1", "m", transport=fake_transport, params={"top_p": 0.9}
    )
    llm.complete([{"role": "user", "content": "hi"}])

    assert captured["payload"]["top_p"] == 0.9


def test_llm_omits_params_key_entirely_when_none_given():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = OpenAICompatibleLLM("http://x/v1", "m", transport=fake_transport)
    llm.complete([{"role": "user", "content": "hi"}])

    assert "top_p" not in captured["payload"]


def test_chat_result_model_carries_the_answering_ref():
    def fake_transport(url: str, payload: dict) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    llm = OpenAICompatibleLLM(
        "http://x/v1", "qwen3", transport=fake_transport, ref="lmstudio/qwen3"
    )
    result = llm.chat([{"role": "user", "content": "hi"}], [])

    assert result.model == "lmstudio/qwen3"


def test_chat_result_model_defaults_to_bare_model_when_no_ref_given():
    def fake_transport(url: str, payload: dict) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    llm = OpenAICompatibleLLM("http://x/v1", "qwen3", transport=fake_transport)
    result = llm.chat([{"role": "user", "content": "hi"}], [])

    assert result.model == "qwen3"


def test_complete_raises_provider_call_error_on_missing_choices():
    from falkorchat.transport import ProviderCallError

    def fake_transport(url: str, payload: dict) -> dict:
        return {"unexpected": "shape"}

    llm = OpenAICompatibleLLM("http://x/v1", "m", transport=fake_transport, ref="p/m")
    with pytest.raises(ProviderCallError) as excinfo:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "p/m" in str(excinfo.value)


# --- chat(): tool-calling seam (U6) ------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "graphrag_retrieve",
            "description": "Retrieve context.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


def _chat_transport(message: dict):
    """Build a fake transport returning `message` as the sole choice, capturing the payload."""
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": message}]}

    return fake_transport, captured


def test_chat_sends_tools_field_and_hits_chat_completions():
    fake_transport, captured = _chat_transport(
        {"role": "assistant", "content": "hello"}
    )
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    llm.chat([{"role": "user", "content": "hi"}], _TOOLS)

    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["tools"] == _TOOLS
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["payload"]["model"] == "m"


def test_chat_parses_native_tool_calls_shape():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "graphrag_retrieve",
                    "arguments": json.dumps({"query": "reset password"}),
                },
            }
        ],
    }
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "reset password"}


def test_chat_parses_content_embedded_json_fallback():
    # Qwen3 failure mode: the tool call arrives as JSON text in `content`,
    # with an empty/absent structured `tool_calls` field.
    message = {
        "role": "assistant",
        "content": json.dumps(
            {"name": "graphrag_retrieve", "arguments": {"query": "billing"}}
        ),
    }
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "billing"}


def test_chat_parses_structured_output_action_shape_in_fenced_content():
    # Structured-output prompting shape `{"action", "args"}`, wrapped in a
    # markdown code fence and surrounding prose.
    blob = json.dumps({"action": "graphrag_retrieve", "args": {"query": "vpn"}})
    message = {
        "role": "assistant",
        "content": f"Sure, let me look that up.\n```json\n{blob}\n```",
    }
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "vpn"}


def test_chat_returns_plain_text_when_no_tool_call():
    message = {"role": "assistant", "content": "The office opens at 9am."}
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "hours?"}], _TOOLS)

    assert not result.is_tool_call
    assert result.tool_calls == []
    assert result.text == "The office opens at 9am."


def test_chat_treats_non_tool_json_content_as_text():
    # A JSON object with no recognizable tool-call keys is a plain answer,
    # not a tool call — it must stay text.
    payload = json.dumps({"answer": 42, "confidence": "high"})
    message = {"role": "assistant", "content": payload}
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "q"}], _TOOLS)

    assert not result.is_tool_call
    assert result.text == payload


def test_chat_tolerates_string_arguments_that_fail_to_parse():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "function": {"name": "graphrag_retrieve", "arguments": "not json"}}
        ],
    }
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "x"}], _TOOLS)

    assert result.is_tool_call
    assert result.tool_calls[0].arguments == {}


# --- chat(): bare function-call syntax in content (K-027 slice A, DEF-K027-B) -----
#
# Small local models emit the call as an *expression* rather than JSON. The JSON
# probe recovers nothing from it (`{"text": ...}` has no name/action/tool key), so
# the call used to be silently lost as prose. Recovery rule pinned below: the call
# must occupy whole lines on its own and its argument must be a JSON **object**.

# **Reconstructed (de-wrapped)** from the K-025 QA pass (DEF-K027-B / report §3.9
# AC-2a): the `intake` node's step output on a live run. Not verbatim — the report
# line-wraps it and elides the tail with "…", and the run's real `StepRun.output` no
# longer exists in any live graph. The recorded prefix is joined onto a single line
# here; the wrapped form is *not* recovered (a raw newline inside the JSON string
# fails `json.loads`), so this fixture does not establish that the observed run would
# have converted — see gates m-4 / N-4.
OBSERVED_BARE_CALL = (
    'post_message({"text": "I\'m sorry to hear that you\'re experiencing a broken '
    "deploy on the billing service. Could you please provide more details about "
    'the issue?"})'
)


def _chat_content(content):
    """Drive `chat` with `content` as the sole assistant message content."""
    fake_transport, _ = _chat_transport({"role": "assistant", "content": content})
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)
    return llm.chat([{"role": "user", "content": "help"}], _TOOLS)


def test_chat_recovers_the_observed_bare_function_call():
    result = _chat_content(OBSERVED_BARE_CALL)

    assert result.is_tool_call
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "post_message"
    assert call.arguments["text"].startswith("I'm sorry to hear")
    assert "provide more details" in call.arguments["text"]


def test_chat_recovers_a_bare_call_inside_a_json_code_fence():
    result = _chat_content(f'```json\n{OBSERVED_BARE_CALL}\n```')

    assert result.is_tool_call
    assert result.tool_calls[0].name == "post_message"


def test_chat_recovers_a_bare_call_inside_an_unlabelled_code_fence():
    result = _chat_content('```\npost_message({"text": "hi"})\n```')

    assert result.is_tool_call
    assert result.tool_calls[0].arguments == {"text": "hi"}


def test_chat_recovers_a_bare_call_on_its_own_line_after_prose():
    result = _chat_content(
        'Let me ask the user for more detail.\npost_message({"text": "Which service?"})'
    )

    assert result.is_tool_call
    assert result.tool_calls[0].arguments == {"text": "Which service?"}
    # consistent with the JSON-embedded fallback: a recovered call carries no text
    assert result.text is None


def test_chat_recovers_a_bare_call_with_multiline_json_arguments():
    result = _chat_content('post_message({\n  "text": "Which service?"\n})')

    assert result.is_tool_call
    assert result.tool_calls[0].arguments == {"text": "Which service?"}


def test_chat_recovers_a_bare_call_with_no_arguments():
    result = _chat_content("human_handoff()")

    assert result.is_tool_call
    assert result.tool_calls[0].name == "human_handoff"
    assert result.tool_calls[0].arguments == {}


def test_chat_recovers_a_bare_call_with_an_empty_json_object():
    result = _chat_content("human_handoff({})")

    assert result.is_tool_call
    assert result.tool_calls[0].arguments == {}


def test_chat_recovers_a_bare_call_with_a_space_before_the_paren():
    # gate N-6: `^[ \t]*name[ \t]*\(` also tolerates whitespace *between* the
    # identifier and the paren. Pinned so the tolerance is a stated rule, not an
    # accident of the regex.
    result = _chat_content('post_message ({"text": "hi"})')

    assert result.is_tool_call
    assert result.tool_calls[0].name == "post_message"
    assert result.tool_calls[0].arguments == {"text": "hi"}


def test_chat_recovers_two_bare_calls_on_separate_lines():
    result = _chat_content(
        'graphrag_retrieve({"query": "billing"})\npost_message({"text": "ok"})'
    )

    assert [c.name for c in result.tool_calls] == ["graphrag_retrieve", "post_message"]
    assert [c.id for c in result.tool_calls] == ["call_0", "call_1"]


# --- the negative direction: prose must stay prose --------------------------------
#
# A false-positive tool call is worse than the miss this slice fixes: it dispatches
# something the model never asked for (or, with an ungranted name, burns a re-prompt
# iteration). Each case below must remain plain text.


def test_chat_keeps_prose_mentioning_a_tool_name_as_text():
    prose = (
        "I could use post_message to reply, but the request is still unclear, so "
        "I will wait. graphrag_retrieve would not help either."
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_an_inline_mid_sentence_call_as_text():
    # Narration *about* a call, not a call: the expression does not start its line.
    prose = 'I will call post_message({"text": "hi"}) once the user answers.'
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_a_call_with_trailing_prose_on_the_same_line_as_text():
    prose = 'post_message({"text": "hi"}) — but I am not sure yet.'
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_a_bare_call_with_non_json_arguments_as_text():
    # Keyword/positional argument syntax carries no reliable mapping to the tool
    # schema; guessing would dispatch wrong arguments, so it stays text.
    for prose in (
        'post_message(text="hi")',
        "post_message('hi')",
        'post_message("hi")',
        "post_message(...)",
    ):
        result = _chat_content(prose)
        assert not result.is_tool_call, prose
        assert result.text == prose


def test_chat_keeps_a_bare_call_with_an_unbalanced_paren_as_text():
    prose = 'post_message({"text": "hi"}'
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


# --- the negative direction, multi-line (gate M-1) --------------------------------
#
# Every negative pin above is a *single-line* case, which is what let the following
# three shapes through: a call the model is **quoting** or **illustrating** fired
# recovery, dispatched a real thread write, and (because a recovered call carries
# `text=None`) threw away the model's actual answer. The rule that closes them: from
# the first recovered call onward, the message holds nothing but calls and whitespace.
# (Gate M-1 first anchored only the *last* call; gate N-1 — pinned in the next block —
# found that half-rule re-opened all of these.)


def test_chat_keeps_a_quoted_call_in_a_fenced_block_mid_message_as_text():
    prose = (
        "You should not do this:\n```python\n"
        'post_message({"text": "hi"})\n'
        "```\nInstead, ask first."
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_an_illustrative_call_in_an_unlabelled_fence_mid_message_as_text():
    prose = (
        "Here is the syntax:\n```\n"
        'post_message({"text": "ex"})\n'
        "```\nDoes that help?"
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_a_call_followed_by_prose_on_a_later_line_as_text():
    prose = (
        "The API is simple.\n"
        'post_message({"text": "x"})\n'
        "That is all you need to know."
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


# --- the negative direction, prose *between* calls (gate N-1) ---------------------
#
# Every pin above ends with prose, which is exactly what hid this family: the
# "final non-whitespace content" rule anchors only the **last** accepted call, so
# each earlier own-line call-shaped expression rode along — and `executor.py:595`
# dispatches every element of `result.tool_calls`. "Here is what I considered /
# here is what I actually do" ends with a *genuine* call, the common shape for a
# well-behaved turn, so all of the quoted/illustrated shapes above re-opened
# through it. The rule that closes them: from the first accepted call onward, only
# calls and whitespace may appear.


def test_chat_keeps_a_quoted_call_before_a_genuine_call_as_text():
    prose = (
        "I considered handing off:\n```\nhuman_handoff()\n```\n"
        'But instead I will ask:\npost_message({"text": "Which service?"})'
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_a_disclaimed_call_before_a_genuine_call_as_text():
    prose = (
        "I will NOT do this:\nhuman_handoff()\n"
        'Instead:\npost_message({"text": "Which service?"})'
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_keeps_an_echoed_call_before_a_genuine_call_as_text():
    # The worst shape of the family: the echoed snippet is itself a `post_message`,
    # so recovering both writes the *user's own quoted text* to the thread.
    prose = (
        "You wrote:\n\n"
        '    post_message({"text": "hello"})\n\n'
        'I will do it:\npost_message({"text": "done"})'
    )
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_still_recovers_a_contiguous_catalogue_of_calls():
    # Characterisation, not endorsement: contiguous own-line calls with no prose
    # between them are structurally identical to an intended multi-call, so
    # position alone cannot separate them. This is the accepted residual — closing
    # it needs the granted-tool-name recognition filter (K-027 open question 4).
    result = _chat_content(
        "Available tools:\n"
        'post_message({"text": "hi"})\n'
        'graphrag_retrieve({"query": "billing"})\n'
        "human_handoff()"
    )

    assert [c.name for c in result.tool_calls] == [
        "post_message",
        "graphrag_retrieve",
        "human_handoff",
    ]


def test_chat_does_not_recover_an_identifier_nested_in_a_rejected_expression():
    # gate m-2: the scan cursor must advance past an expression it *rejected*, so a
    # nested identifier inside it is never lifted out as a call of its own.
    prose = 'outer(text="\nfoo({"a": 1})\n")'
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_recovers_identical_repeated_bare_calls_only_once():
    # gate m-5: self-repetition is a known small-model failure mode; dispatching it
    # twice posts the same message to the thread twice.
    result = _chat_content('post_message({"text": "a"})\npost_message({"text": "a"})')

    assert [c.name for c in result.tool_calls] == ["post_message"]
    assert result.tool_calls[0].arguments == {"text": "a"}


# --- the JSON probe's `tool_calls` envelope (gate m-1) ----------------------------


def test_chat_falls_through_an_empty_tool_calls_envelope_to_the_sibling_call():
    # An empty `tool_calls` list no longer short-circuits: the same object's
    # name/arguments keys are still a call.
    result = _chat_content(
        '{"tool_calls": [], "name": "graphrag_retrieve", "arguments": {"query": "q"}}'
    )

    assert [c.name for c in result.tool_calls] == ["graphrag_retrieve"]
    assert result.tool_calls[0].arguments == {"query": "q"}


def test_chat_keeps_a_bare_empty_tool_calls_envelope_as_text():
    prose = '{"tool_calls": []}'
    result = _chat_content(prose)

    assert not result.is_tool_call
    assert result.text == prose


def test_chat_prefers_native_tool_calls_over_a_bare_call_in_content():
    # Precedence is unchanged: the structured field stays authoritative and the
    # content is returned as text alongside it.
    message = {
        "role": "assistant",
        "content": OBSERVED_BARE_CALL,
        "tool_calls": [
            {"id": "call_native", "function": {"name": "graphrag_retrieve",
                                               "arguments": '{"query": "billing"}'}}
        ],
    }
    fake_transport, _ = _chat_transport(message)
    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert [c.name for c in result.tool_calls] == ["graphrag_retrieve"]
    assert result.text == OBSERVED_BARE_CALL


def test_complete_still_omits_tools_field():
    # Regression: the completion path the responder uses must not gain a `tools`
    # field or change shape.
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "unchanged"}}]}

    llm = OpenAICompatibleLLM(base_url="http://x/v1", model="m", transport=fake_transport)
    out = llm.complete([{"role": "user", "content": "hello"}])

    assert out == "unchanged"
    assert "tools" not in captured["payload"]
    assert captured["payload"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
    }
