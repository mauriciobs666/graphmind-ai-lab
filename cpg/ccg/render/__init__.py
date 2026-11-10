"""Renderers of a canonical answer document (P1-24; answer spec §7.1-§7.3, spec2 §3-§4).

``render_text(doc)`` is the grouped plain-text answer and ``render_markdown(doc)`` the saved impact report; both are
deterministic templates, and both open with ``answer.FENCE_SENTENCE``. ``summary_input(doc)`` is the only text a
plain-language summary prompt may be built from: counts and engine-minted codes, never a data field. It validates
its input and raises ``AnswerError`` on a document that fails, so a hostile minted string cannot reach the model.
"""
from __future__ import annotations

from ccg.answer import validate
from ccg.render.common import all_items
from ccg.render.markdown import render_markdown
from ccg.render.text import render_text

__all__ = ["render_markdown", "render_text", "summary_input"]


def summary_input(doc: dict) -> str:
    """Aggregate facts for the optional plain-language summary (F-3): numbers and registry codes only.

    No ``DATA_FIELDS`` value can appear here: a count, an enum, a registry code or a distance is engine-minted, and a
    path, name, key or unit label is data and is left out. The minted values are echoed only after ``answer.validate``
    has held them to the schema's patterns and enums; an invalid document raises ``AnswerError`` (the summary is
    optional, so the caller omits it, rather than summarising an answer that failed its contract).
    """
    from ccg.render import common as c

    validate(doc)
    items = list(all_items(doc))
    lines = [f"interpretation: {doc['interpretation']['status']}"]
    if doc["interpretation"]["status"] == "resolved":
        lines.append(f"direction: {doc['interpretation']['request']['direction']}")
    conf = sum(1 for i in items if i["status"] == "confirmed")
    lines.append(f"items: {len(items)} (confirmed {conf}, possible {len(items) - conf})")
    lines.append(f"folded groups: {len(doc['groups'])}")
    dists: dict[int, int] = {}
    reasons: dict[str, int] = {}
    for it in items:
        dists[it["distance"]] = dists.get(it["distance"], 0) + 1
        if "reason" in it:
            reasons[it["reason"]] = reasons.get(it["reason"], 0) + 1
    lines += [f"distance {d}: {n}" for d, n in sorted(dists.items())]
    lines += [f"reason {r}: {n}" for r, n in sorted(reasons.items())]
    states: dict[str, int] = {}
    for u in doc["staleness"]["units"]:
        states[u["buildStatus"]] = states.get(u["buildStatus"], 0) + 1
    lines += [f"units {s}: {n}" for s, n in sorted(states.items())]
    note = c.reach_note(doc)
    if note:
        lines.append(note)
    return "\n".join(lines) + "\n"
