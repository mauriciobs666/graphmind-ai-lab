"""The Markdown impact report (FR-10, AC-11): the same grouped answer as :mod:`ccg.render.text`, saved as a document.

Sentences come from :mod:`ccg.render.common`; this module only lays them out. See that module for F-3/F-4.
"""
from __future__ import annotations

from ccg.answer import FENCE_SENTENCE
from ccg.render import common as c


def _item(item: dict, units: dict[str, dict], head: str) -> list[str]:
    out = [f"{head} {c.subject_text(item, units)}", "", f"Status: {c.status_text(item)}. {c.distance_text(item).capitalize()}."]
    if c.has_data_hop(item):
        out.append("Reached through data flow.")
    out.append("")
    for n, hop in enumerate(item["chain"], 1):
        line, details = c.hop_lines(hop, units)
        out.append(f"{n}. {line}")
        out.extend(f"    - {d}" for d in details)
    out.append("")
    return out


def render_markdown(doc: dict) -> str:
    units = c.units_of(doc)
    out = ["# Connected code graph: impact report", "", f"> {FENCE_SENTENCE}", "",
           f"Snapshot {doc['stamp']['snapshotId']}, verdict log offset {doc['stamp']['verdictLogOffset']}.", ""]
    out += [f"{line}\n" for line in c.interpretation_lines(doc)]
    if doc["staleness"]["units"]:
        out += ["## Code views used", ""] + [f"- {line}" for line in c.staleness_lines(doc)] + [""]
    if doc["interpretation"]["status"] == "resolved":
        out += ["## Summary", "", c.counts_line(doc["items"], doc["groups"]), ""]
        note = c.reach_note(doc)
        if note:
            out += [note, ""]
        for repo, comps in c.component_groups(doc, doc["items"]):
            out += [f"## Repository {repo}", ""]
            for unit_id, dists in comps:
                out += [f"### Component {c.unit_label(units, unit_id)}", ""]
                for dist, stats in dists:
                    for status, items in stats:
                        out += [f"#### Distance {dist}, {status}", ""]
                        for item in items:
                            out += _item(item, units, "#####")
        for group in doc["groups"]:
            out += [f"## Folded group: {c.GROUP_TITLE[group['kind']]}", "",
                    f"{group['count']} members; reason {c.reason_text(group['reason'])}.", ""]
            for item in group["members"]:
                out += _item(item, units, "###")
    return "\n".join(out).rstrip("\n") + "\n"
