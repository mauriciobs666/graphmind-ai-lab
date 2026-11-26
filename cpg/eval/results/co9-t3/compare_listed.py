#!/usr/bin/env python3
"""Carry-forward comparator of the ``ac5-gate-1-r4`` record (``noiseFix.probeRehearsal``, ``noiseFix.t3ListedSet``): is the rehearsal's LISTED answer set the one
the t3 labels and the carried Q were made on?

The compared object (defined here once and in the record): everything the carried labels and Q read, i.e. every listed item of every dev answer, group members
included, as ``[anchorId, itemId, chainSha256, itemSha256]`` (``itemId`` = ``noise.item_id``; ``chainSha256`` = sha256 of the item's ``chain``; ``itemSha256`` = sha256 of the
item's full JSON, every field: status, distance, reason, band, split, subject lines, ...), plus one ``["group", anchorId, headSha256, memberItemIds]`` per group (``headSha256`` =
sha256 of the group without its ``members``; member ids sorted). JSON is canonical (sorted keys, separators ``,`` and ``:``, ensure_ascii). The set hash is the sha256 of the
entries sorted by their canonical line, one line each. No per-build value is excluded because none differs: t2 and t3 (two builds, two HMAC keys) hold 491 items equal in
(anchor, item, chain) whose full JSON is equal. A duplicate ``(anchorId, itemId)`` is a difference.

Usage (from ``cpg/``): ``python eval/results/co9-t3/compare_listed.py (<answers.jsonl> | --run <run.json>) [--baseline listed-triples.jsonl] [--pinned-sha <sha>]``; ``--run`` builds
``{anchorId, items, groups}`` from the rehearsal's ``run.json`` ``answers`` (groups default to ``[]``) and fails when any entry has no ``answer`` (a harness error is not a listed set).
Prints counts and booleans only (never an id); exit 0 iff identical, 1 on a difference, 2 on an unreadable run. The baseline holds dev ids and hashes of the t3 run;
``--pinned-sha`` defaults to the record's ``noiseFix.t3ListedSet.sha256``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CPG = _HERE.parents[2]
sys.path.insert(0, str(_CPG / "eval" / "tools"))
import noise as N  # noqa: E402

BASELINE = _HERE / "listed-triples.jsonl"
PREREG = _CPG / "eval" / "preregistration.jsonl"
GATE_ID = "ac5-gate-1-r4"


def _cj(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(obj) -> str:
    return hashlib.sha256(_cj(obj).encode()).hexdigest()


def triples(answers: list[dict]) -> list[list]:
    """Every compared entry (items, then group lines), sorted by canonical line."""
    out = []
    for ans in answers:
        for it in N._listed(ans):
            out.append([ans["anchorId"], N.item_id(it), _sha(it.get("chain")), _sha(it)])
        for g in ans["groups"]:
            out.append(["group", ans["anchorId"], _sha({k: v for k, v in g.items() if k != "members"}), sorted(N.item_id(m) for m in g["members"])])
    return sorted(out, key=_cj)


def listed_set_sha256(trs: list[list]) -> str:
    return hashlib.sha256("".join(_cj(t) + "\n" for t in sorted(trs, key=_cj)).encode()).hexdigest()


def write_baseline(path: Path, trs: list[list]) -> None:
    Path(path).write_text("".join(_cj(t) + "\n" for t in sorted(trs, key=_cj)))


def load_baseline(path: Path) -> list[list]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def load_answers(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def answers_from_run(path: Path) -> list[dict]:
    """``{anchorId, items, groups}`` per ``run.json`` ``answers`` entry; an entry without ``answer`` (a harness error) is a ValueError, never dropped."""
    entries = json.loads(Path(path).read_text())["answers"]
    bad = sum(1 for e in entries.values() if not isinstance(e, dict) or "answer" not in e)
    if bad:
        raise ValueError(f"{bad} entr{'y' if bad == 1 else 'ies'} of run.json answers hold no answer: not a listed set")
    return [{"anchorId": a, "items": e["answer"]["items"], "groups": e["answer"].get("groups", [])} for a, e in sorted(entries.items())]


def pinned_sha(prereg: Path = PREREG, gate_id: str = GATE_ID) -> str:
    for ln in Path(prereg).read_text().splitlines():
        d = json.loads(ln) if ln.strip() else {}
        if isinstance(d, dict) and d.get("stratum") == gate_id:
            return d["noiseFix"]["t3ListedSet"]["sha256"]
    raise SystemExit(f"no {gate_id} record in {prereg}")


def compare(answers_path: Path, baseline_path: Path, pinned: str, run: bool = False) -> dict:
    base = load_baseline(baseline_path)
    now = triples(answers_from_run(answers_path) if run else load_answers(answers_path))
    items = lambda trs: [t for t in trs if t[0] != "group"]
    groups = lambda trs: Counter(_cj(t) for t in trs if t[0] == "group")
    b_items, n_items = items(base), items(now)
    b_keys, n_keys = Counter((t[0], t[1]) for t in b_items), Counter((t[0], t[1]) for t in n_items)
    b_chain, n_chain = {(t[0], t[1]): t[2:] for t in b_items}, {(t[0], t[1]): t[2:] for t in n_items}
    added, removed = sum(1 for k in n_keys if k not in b_keys), sum(1 for k in b_keys if k not in n_keys)
    both = [k for k in n_keys if k in b_keys]
    chain_changed = sum(1 for k in both if n_chain[k][0] != b_chain[k][0])
    fields_changed = sum(1 for k in both if n_chain[k][0] == b_chain[k][0] and n_chain[k][1] != b_chain[k][1])
    dup = sum(1 for c in n_keys.values() if c > 1)
    gdiff = sum(((groups(now) - groups(base)) + (groups(base) - groups(now))).values())
    equal, base_ok = listed_set_sha256(now) == pinned, listed_set_sha256(base) == pinned
    return {"items": len(n_items), "baselineItems": len(b_items), "groups": sum(groups(now).values()), "added": added, "removed": removed,
            "chainChanged": chain_changed, "fieldsChanged": fields_changed, "groupDiffs": gdiff, "duplicatePairs": dup,
            "sha256Equal": equal, "baselineMatchesPinned": base_ok,
            "identical": equal and base_ok and not (added or removed or chain_changed or fields_changed or gdiff or dup)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("answers", type=Path, nargs="?")
    ap.add_argument("--run", type=Path, default=None, help="the rehearsal's run.json (instead of an answers file)")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--pinned-sha", default=None)
    a = ap.parse_args(argv)
    if (a.answers is None) == (a.run is None):
        ap.error("give an answers file or --run run.json")
    try:
        res = compare(a.run or a.answers, a.baseline, a.pinned_sha or pinned_sha(), run=a.run is not None)
    except ValueError as e:
        print(_cj({"identical": False, "error": str(e)}))
        return 2
    print(_cj(res))
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
