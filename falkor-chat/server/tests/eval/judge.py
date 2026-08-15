"""LLM-as-judge for GraphRAG generation quality (K-026, `docs/plans/graphrag-eval.md`
§5 Unit 3).

Judges a `(question, context, answer)` triple on two binary axes — **faithfulness**
(is the answer grounded in / not contradicted by the retrieved context) and
**relevance** (does the answer address the question asked) — per the method note's
Layer 2 design and plan §3 D5.

Mirrors `guards.GuardVerdict` / `app._LlmGuardJudge`'s shape and parsing precedent: the
prompt asks for a single JSON verdict object, and `llm.extract_own_line_json_object`
(reused, never reimplemented — the conservative sibling of `extract_json_object`, K-027
gate B-1) is the only parser used. An unparseable, malformed, or ambiguously-typed
reply resolves to the **conservative** reading (`faithfulness=None`, `relevance=False`,
`parse_failed=True`) rather than a fabricated favorable verdict — this harness's
numbers are directional and explicitly non-gating (plan §3 D1), so a judge that
silently guessed `True` on a bad parse would be worse than one that visibly abstains.

`faithfulness: bool | None` carries two distinct "can't score this true/false" cases
(plan §3 D5 / §6 edge cases):

  * the answer legitimately **abstained** — no retrieved context was available, so the
    responder answered from general knowledge (the method note's explicit "don't
    penalize a correct 'context didn't help'"). `judge_triple` enforces this
    **unconditionally in code**, not just via the prompt: whenever `context` is empty,
    the returned `faithfulness` is forced to `None` regardless of what the judge model
    said, because scoring faithfulness against nothing is structurally meaningless, not
    a judgment call the model should be trusted to make correctly on its own.
  * the judge's reply **failed to parse** into a usable verdict at all.

`JudgeVerdict.parse_failed` is the field that tells those two apart for the report
generator's parse-failure count (plan §6: "visible in the report as a parse-failure
count, never silently dropped from the denominator") — a legitimate abstain is not a
parse failure, and must not be counted as one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from falkorchat.llm import LLM, ChatMessage, extract_own_line_json_object

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of a retrieval-augmented answer. You are given a "
    "QUESTION, the RETRIEVED CONTEXT that was available to the answering model, and "
    "its ANSWER. Score two axes:\n"
    "  faithfulness: does the answer avoid contradicting the retrieved context, and "
    "stay grounded in it when it draws on it? If RETRIEVED CONTEXT is empty and the "
    "answer correctly says so / answers from general knowledge instead, faithfulness "
    "is inapplicable — answer null, not true or false.\n"
    "  relevance: does the answer actually address the question asked (regardless of "
    "whether it used the context)?\n"
    "Reply with a single JSON object and nothing else: "
    '{"faithfulness": <true|false|null>, "relevance": <true|false>, "rationale": '
    '"<one short sentence>"}. When in doubt, prefer the conservative reading: '
    "faithfulness=null over a guessed true, relevance=false over a guessed true."
)


def build_judge_prompt(question: str, context: list[str], answer: str) -> list[ChatMessage]:
    """Assemble the judge's chat messages for one `(question, context, answer)` triple.

    Empty `context` renders as an explicit "(none — ...)" line rather than an empty
    bulleted block, so the judge can tell "genuinely no retrieved context" apart from
    "the context section was accidentally omitted" — only the former licenses the
    abstained/null faithfulness reading (D5).
    """
    if context:
        rendered = "\n".join(f"- {c}" for c in context)
    else:
        rendered = "(none — the answer had no retrieved context)"
    user = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{rendered}\n\n"
        f"ANSWER:\n{answer}"
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True)
class JudgeVerdict:
    """A judge's verdict on one `(question, context, answer)` triple (plan §3 D5).

    `faithfulness`: `True`/`False` when scored against non-empty context; `None` when
    the answer abstained (empty context) or the reply failed to parse — see
    `parse_failed` to tell those two apart.
    `relevance`: always `True`/`False` — never abstained (D5: relevance has no
    "context didn't help" exemption; even a parse failure resolves it to the
    conservative `False`, never left unset).
    `parse_failed`: `True` iff the judge's raw reply did not parse into a usable
    verdict at all. Never set merely because `faithfulness` is a legitimate `None`
    abstain.
    """

    faithfulness: bool | None
    relevance: bool
    rationale: str = ""
    parse_failed: bool = False


def _coerce_verdict(raw: Any) -> JudgeVerdict:
    """Normalize a parsed judge reply into a `JudgeVerdict`, biasing conservative.

    Mirrors `guards._coerce_verdict`'s posture: only a clean, well-typed reply is
    trusted at face value.

    `relevance` accepts only the real bool `True` — anything else (missing, `"true"`,
    `1`, `False`) resolves to `False`.
    `faithfulness` accepts the real bool `True`/`False` or JSON `null` (Python `None`)
    — any other value (missing key, a string, a number) also resolves to `None`, since
    it is not a clean assertion either way.
    """
    rationale = raw.get("rationale")
    rationale = rationale if isinstance(rationale, str) else ""

    faithfulness_raw = raw.get("faithfulness", "__missing__")
    faithfulness: bool | None
    if faithfulness_raw is True or faithfulness_raw is False:
        faithfulness = faithfulness_raw
    else:
        faithfulness = None  # None, missing key, or any other junk — conservative

    relevance = raw.get("relevance") is True

    return JudgeVerdict(faithfulness=faithfulness, relevance=relevance, rationale=rationale)


def judge_triple(
    judge_llm: LLM, *, question: str, context: list[str], answer: str
) -> JudgeVerdict:
    """Call `judge_llm` on one triple and return a conservative `JudgeVerdict`.

    Parsing uses `llm.extract_own_line_json_object` (K-027's conservative extractor,
    reused verbatim — not reimplemented): a reply that is not entirely one JSON
    object, or is an ambiguously-quoted one, resolves to `parse_failed=True` rather
    than being coerced out of prose. `require_key="relevance"` because `relevance` is
    the one axis that is never legitimately absent from a clean verdict (unlike
    `faithfulness`, which may be a genuine `null`).

    Empty `context` (§6 edge case) forces `faithfulness=None` unconditionally,
    regardless of what the judge said — faithfulness is never scored against nothing.
    """
    prompt = build_judge_prompt(question, context, answer)
    text = judge_llm.complete(prompt)
    parsed = extract_own_line_json_object(text, require_key="relevance")
    if parsed is None:
        return JudgeVerdict(
            faithfulness=None,
            relevance=False,
            rationale="unparseable judge output",
            parse_failed=True,
        )
    verdict = _coerce_verdict(parsed)
    if not context and verdict.faithfulness is not None:
        verdict = replace(verdict, faithfulness=None)
    return verdict
