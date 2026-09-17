"""`chat-responder`'s `ItemScorer` — the deterministic layer only (FR-21a): grounded-reply
containment against retrieved context, plus format compliance. No judge, no reply-quality score
(S7 spec §1/§3.4). One chat call per item; `ItemScorer`-Protocol-shaped, structural, no base class
(`classification.py`'s/`extraction.py`'s own precedent, S7 spec §2.2).

Design: `docs/plans/small-model-benchmarking-s7-spec.md` §3.4 (the exact functions this module
builds), §2.7/§2.8 (why `_canon_str` is imported from `extraction.py` rather than reimplemented,
and why `_ABSTENTION_MARKERS` is transcribed as DATA + CODE — D1, `classification.py`'s own module
docstring precedent — rather than imported from `falkor-chat/server/tests/eval/nlq_scoring.py`),
§5 Step 0 (this module's own red-then-green test order, `tests/test_scoring_grounding.py`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack
from modelbench.results import BinaryMetric, GroundingAggregates, ItemResult, ItemTiming
from modelbench.runner import ChatMessage
from modelbench.scoring.extraction import _canon_str

# Transcribed from falkor-chat/server/tests/eval/nlq_scoring.py:68-84 (D1: data + code, never an
# import, per classification.py's own module docstring precedent) — Layer 2's abstention
# phrasings, adapted to this pack's own mustContain/mustNotContain/mustAbstain checklist shape
# rather than nlq_scoring's own scalar/set/not_found shape (S7 spec §2.8).
_ABSTENTION_MARKERS: tuple[str, ...] = (
    "not found", "no matching", "couldn't find", "could not find", "don't have", "do not have",
    "no data", "unable to find", "no information", "not available", "no record",
    "i'm not sure", "i don't know", "cannot find", "can't find",
)

#: `qwen/qwen3-4b-2507`'s own dominant, live-confirmed abstention idiom ("The passages don't
#: mention X" / "...doesn't mention X") was missing from `_ABSTENTION_MARKERS` above (S7 live-run
#: defect, `docs/test-reports/small-model-benchmarking-s7-report.md` "Defect —
#: `_ABSTENTION_MARKERS` does not recognize..."; fix scoped by `data-scientist` consult). Checked
#: separately from the fixed-marker list, not folded into it, because a plain substring match on
#: "don't mention" also matches a reply that *hedges but still answers* ("The passages don't
#: mention this directly, but based on the numbers given, the answer is 42,000") — a real,
#: grounded, non-abstaining reply. So this phrase counts as abstention only when it is not followed
#: later in the reply by a contrastive continuation ("but"/"however") that signals the reply goes
#: on to actually answer.
_MENTION_ABSTENTION_RE = re.compile(r"\b(?:don't|doesn't) mention\b")
_CONTRASTIVE_CONTINUATION_RE = re.compile(r"\b(?:but|however)\b")


def looks_like_abstention(reply: str) -> bool:
    """`layer2_contains`'s own `not_found` branch, adapted (S7 spec §2.8): true iff any of the
    fixed abstention phrasings appears, as a canonicalized substring, in `reply` — or the reply
    uses the "don't/doesn't mention" idiom without a later hedge-then-answer continuation."""
    canon = _canon_str(reply)
    if any(marker in canon for marker in _ABSTENTION_MARKERS):
        return True
    match = _MENTION_ABSTENTION_RE.search(canon)
    if match is None:
        return False
    return _CONTRASTIVE_CONTINUATION_RE.search(canon, match.end()) is None


#: The three format keys every pack/item `format` block may declare, and their trivially-passing
#: defaults (S7 spec §3.4) — `maxWords: None`/`forbiddenPatterns: ()` never refuse anything on
#: their own; `resolve_format`'s three-way merge (pack default first, item override last) is what
#: turns this into a real constraint.
_DEFAULT_FORMAT: Mapping[str, Any] = {
    "maxWords": None, "mustBeSingleParagraph": False, "forbiddenPatterns": (),
}


def resolve_format(pack_format: Mapping[str, Any], item_format: Mapping[str, Any] | None) -> dict:
    """The three-way merge S7 spec §3.1/§3.2 both describe: the trivially-passing default first,
    the pack-level `format` manifest block next, the item's own `format` override (if any) last —
    an item that declares only one key overrides only that key, the other two falling back to the
    pack default."""
    return {**_DEFAULT_FORMAT, **pack_format, **(item_format or {})}


def checklist_pass(
    reply: str, *, must_contain: Sequence[str], must_not_contain: Sequence[str], must_abstain: bool
) -> bool:
    """The verdict metric's own predicate — the plan's own three-clause definition verbatim
    (`-ml` §6.2): all `mustContain` present, no `mustNotContain` present, and abstention matches
    `mustAbstain`. All three clauses are independent; a reply that gets the containment right but
    abstains when it should not (or vice versa) still fails the checklist."""
    canon_reply = _canon_str(reply)
    return (
        all(_canon_str(c) in canon_reply for c in must_contain)
        and not any(_canon_str(c) in canon_reply for c in must_not_contain)
        and looks_like_abstention(reply) == must_abstain
    )


def format_checks(reply: str, fmt: Mapping[str, Any]) -> dict[str, bool]:
    """The three independent, never-pooled format constraints (S7 spec §3.4). A pack/item that
    declares no `maxWords` (`None`) trivially passes, never refuses. `mustBeSingleParagraph` uses
    a blank line as its own minimal single-paragraph test — a reply with no blank line is one
    paragraph by construction. `forbiddenPatterns` is checked with `re.MULTILINE`, because the
    plan's own example pattern (`"^\\s*[-*]\\s"`) anchors per line, not per string."""
    return {
        "maxWords": fmt["maxWords"] is None or len(reply.split()) <= fmt["maxWords"],
        "mustBeSingleParagraph": (
            not fmt["mustBeSingleParagraph"] or "\n\n" not in reply.strip()
        ),
        "forbiddenPatterns": not any(
            re.search(p, reply, re.MULTILINE) for p in fmt["forbiddenPatterns"]
        ),
    }


def _format_directive(fmt: Mapping[str, Any]) -> str:
    """One deterministic sentence naming the resolved per-item format constraints, appended to the
    system prompt by `build_messages` (S7 spec §3.4) — synthesis, not a transcription, and cheap
    to revise (S7 spec §6/§7). Names only the constraints actually in force; an item/pack that
    declares none of the three still gets a (trivial) sentence, never a silent no-op."""
    clauses = []
    if fmt["maxWords"] is not None:
        clauses.append(f"stay under {fmt['maxWords']} words")
    if fmt["mustBeSingleParagraph"]:
        clauses.append("write a single paragraph")
    if fmt["forbiddenPatterns"]:
        clauses.append("do not use bullet points, numbered lists, or code fences")
    if not clauses:
        return "For this reply: no additional format constraints apply."
    return "For this reply: " + ", ".join(clauses) + "."


def build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]:
    """Never reads `item_input["mustContain"]`/`["mustNotContain"]`/`["mustAbstain"]`/
    `["provenance"]` — those would leak the answer key into the prompt (S7 spec §2.3's own
    load-bearing finding). Resolves this item's own format constraints and appends them to the
    pack's system prompt as one deterministic sentence, then renders the context passages
    labelled and numbered, followed by the question."""
    system_prompt = pack.prompt_config().systemPrompt
    fmt = resolve_format(pack.manifest.get("format") or {}, item_input.get("format"))
    system_prompt += "\n\n" + _format_directive(fmt)
    passages = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(item_input.get("context") or []))
    user = (
        f"CONTEXT:\n{passages}\n\nQUESTION: {item_input['question']}"
        if passages
        else f"CONTEXT: (none provided)\n\nQUESTION: {item_input['question']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def score_item(
    item_input: Mapping[str, Any], result: ChatResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """`result is None` -> outcome `"fail"` (timeout) / `"unrunnable"` (no response), per
    `timing.withheldFor`, mirroring `classification.py`'s/`extraction.py`'s own precedent —
    `scoreable={}`, no contribution to any metric's denominator.

    Otherwise `outcome` is always `"pass"` — this role's `outcome` states "the call happened",
    never "the checklist passed" (`classification.py`'s own established convention, S7 spec §2.2);
    correctness lives entirely in `counts`. Every item declares all four metrics scoreable
    unconditionally — grounding and format are independent axes over the same reply, never
    mutually exclusive."""
    item_id = item_input["itemId"]
    if result is None:
        outcome = "fail" if timing.withheldFor == "timeout" else "unrunnable"
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome=outcome,
            scoreable={}, counts={}, timing=timing,
        )

    reply = result.message.get("content")
    reply = reply if isinstance(reply, str) else ""

    checklist_ok = checklist_pass(
        reply,
        must_contain=item_input["mustContain"],
        must_not_contain=item_input["mustNotContain"],
        must_abstain=item_input["mustAbstain"],
    )
    fmt = resolve_format(pack.manifest.get("format") or {}, item_input.get("format"))
    checks = format_checks(reply, fmt)

    return ItemResult(
        itemId=item_id, pairingKey=(item_id,), outcome="pass",
        scoreable={
            "groundingRate": True, "formatMaxWords": True,
            "formatSingleParagraph": True, "formatNoForbiddenPatterns": True,
        },
        counts={
            "groundingRate": int(checklist_ok), "formatMaxWords": int(checks["maxWords"]),
            "formatSingleParagraph": int(checks["mustBeSingleParagraph"]),
            "formatNoForbiddenPatterns": int(checks["forbiddenPatterns"]),
        },
        timing=timing,
        detail={
            "checklistPass": checklist_ok, "abstained": looks_like_abstention(reply),
            "wordCount": len(reply.split()),
        },
    )


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> GroundingAggregates:
    """One pass, mirroring `classification.aggregate`'s own `declaring(name)` helper: four
    independent `BinaryMetric`s, each over only the items that declared it scoreable (trivially
    all of them, since `score_item` always declares all four — still computed generically rather
    than assumed, since a future edit could accidentally special-case one metric).

    `parseFailures` stays `0` always for this scorer (S7 spec §3.4) — this role scores free text
    by containment, never by parsing a structured reply, so the "no JSON found" failure class the
    field exists for on `guard-judge`/`nlq-generator` has no analogue here. The field stays on
    `GroundingAggregates` because it is shared union shape, not because this scorer populates it.
    """

    def declaring(name: str) -> list[ItemResult]:
        return [it for it in items if it.scoreable.get(name)]

    grounding_items = declaring("groundingRate")
    checklist_pass_metric = BinaryMetric(
        name="groundingRate",
        successes=sum(it.counts.get("groundingRate", 0) for it in grounding_items),
        n=len(grounding_items), unit="item",
    )
    per_check = tuple(
        BinaryMetric(
            name=name, successes=sum(it.counts.get(name, 0) for it in declaring(name)),
            n=len(declaring(name)), unit="item",
        )
        for name in ("formatMaxWords", "formatSingleParagraph", "formatNoForbiddenPatterns")
    )
    return GroundingAggregates(
        checklistPass=checklist_pass_metric, perCheck=per_check, parseFailures=0
    )
