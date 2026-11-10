"""The grouped plain-text answer (FR-10, AC-11): repository, component, distance, confirmed before possible.

Sentences come from :mod:`ccg.render.common`; this module only lays them out. See that module for F-3/F-4.
"""
from __future__ import annotations

from ccg.answer import FENCE_SENTENCE
from ccg.render import common as c

_IND = "  "


def _item(item: dict, units: dict[str, dict], level: int) -> list[str]:
    pad = _IND * level
    out = [f"{pad}- {c.subject_text(item, units)} [{c.status_text(item)}; {c.distance_text(item)}]"]
    if c.has_data_hop(item):
        out.append(f"{pad}{_IND}via data flow")
    for n, hop in enumerate(item["chain"], 1):
        line, details = c.hop_lines(hop, units)
        out.append(f"{pad}{_IND}{n}. {line}")
        out.extend(f"{pad}{_IND}{_IND}{d}" for d in details)
    return out


def render_text(doc: dict) -> str:
    units = c.units_of(doc)
    out = [FENCE_SENTENCE, "", f"Snapshot {doc['stamp']['snapshotId']}, verdict log offset {doc['stamp']['verdictLogOffset']}."]
    out += c.interpretation_lines(doc)
    if doc["staleness"]["units"]:
        out += ["", "Code views used:"] + [f"{_IND}{line}" for line in c.staleness_lines(doc)]
    if doc["interpretation"]["status"] == "resolved":
        out += ["", c.counts_line(doc["items"], doc["groups"])]
        note = c.reach_note(doc)
        if note:
            out.append(note)
        for repo, comps in c.component_groups(doc, doc["items"]):
            out += ["", f"Repository {repo}"]
            for unit_id, dists in comps:
                out.append(f"{_IND}Component {c.unit_label(units, unit_id)}")
                for dist, stats in dists:
                    for status, items in stats:
                        out.append(f"{_IND * 2}Distance {dist}, {status}")
                        for item in items:
                            out += _item(item, units, 3)
        for group in doc["groups"]:
            out += ["", f"Folded group: {c.GROUP_TITLE[group['kind']]}, {group['count']} members; "
                        f"reason {c.reason_text(group['reason'])}"]
            for item in group["members"]:
                out += _item(item, units, 1)
    return "\n".join(out) + "\n"
