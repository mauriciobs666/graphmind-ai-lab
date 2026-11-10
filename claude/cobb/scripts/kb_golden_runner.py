#!/usr/bin/env python3
"""kb_golden_runner.py -- run the 45-pair golden set against a falkor-chat workspace, read-only (W7a).

Runs each pair the way an agent does (`skills/agent-kb-retrieval/SKILL.md`): the skill's literal instruction prefix
around the situation text (built with `pwq_kb.extract_prefix` / `pwq_kb.build_query`), `limit=5`, and a score floor that
rejects any chunk scoring strictly above it. Reports per pair a document-level top-5 hit and a floor-applied hit, plus
aggregates at every floor, as text or (`--json`) as a record to compare across runs by pair id. Search only: the only
request ever made is `GET /documents/search`.

    python3 claude/cobb/scripts/kb_golden_runner.py --base-url http://localhost:8210 --ws pwq-test --floor 0.42 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pwq_kb as kb  # noqa: E402  (reused by import, never modified)

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "fixtures" / "kb-golden-45.json"
TOP_K = 5
DEFAULT_FLOORS = (0.42, 0.43)
DOC_ID = re.compile(r"[0-9a-f]{32}")


def validate_fixture(fixture: dict) -> None:
    pairs = fixture.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise kb.KBError("fixture: no pairs")
    if not DOC_ID.fullmatch(str(fixture.get("r6_document", ""))):
        raise kb.KBError("fixture: r6_document must be a 32-hex document id")
    seen = set()
    for p in pairs:
        pid = p.get("id")
        if not pid or pid in seen:
            raise kb.KBError(f"fixture: missing or duplicate pair id {pid!r}")
        seen.add(pid)
        if not (isinstance(p.get("situation"), str) and p["situation"].strip()):
            raise kb.KBError(f"fixture {pid}: empty situation")
        if p.get("answer_class") not in ("single", "family", "negative"):
            raise kb.KBError(f"fixture {pid}: bad answer_class")
        for d in list(p.get("expected", [])) + list(p.get("distractors", [])):
            if not DOC_ID.fullmatch(str(d)):
                raise kb.KBError(f"fixture {pid}: {d!r} is not a 32-hex document id")
        if p["answer_class"] == "negative" and (p.get("expected") or p.get("applies")):
            raise kb.KBError(f"fixture {pid}: a negative has no expected document and does not apply to recall")
        if p["answer_class"] != "negative" and not p.get("expected"):
            raise kb.KBError(f"fixture {pid}: a positive needs expected documents")
        if not p.get("applies") and not p.get("na_reason"):
            raise kb.KBError(f"fixture {pid}: a pair that does not apply needs na_reason")


def load_fixture(path: Path) -> dict:
    try:
        fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise kb.KBError(f"cannot read fixture {path}: {e}") from e
    validate_fixture(fixture)
    return fixture


def _key(floor: float) -> str:
    return str(floor)


def document_view(fetched: list[dict]) -> list[dict]:
    """Distinct documents of the returned chunks, in first-appearance order, each with its best (lowest) chunk score."""
    best: dict[str, float] = {}
    for c in fetched:
        d = c["documentId"]
        best[d] = min(best.get(d, c["score"]), c["score"])
    return [{"documentId": d, "score": s, "rank": i + 1} for i, (d, s) in enumerate(best.items())]


def accepted(score, floor: float) -> bool:
    """The skill's rule: reject a hit with score strictly above the floor."""
    return score is not None and score <= floor


def run_pair(p: dict, searcher, template: str, ws: str, primary_floor: float, floors) -> dict:
    fetched = searcher.search(ws, kb.build_query(template, p["situation"]), TOP_K)
    docs = document_view(fetched)
    by = {d["documentId"]: d for d in docs}
    out = {"id": p["id"], "answer_class": p["answer_class"], "applies": bool(p["applies"]), "na_reason": p.get("na_reason"),
           "docs": docs, "top1": docs[0] if docs else None}
    if p["answer_class"] == "negative":
        closest = min((c["score"] for c in fetched), default=None)
        out.update(closest_score=closest, false_accept={_key(f): accepted(closest, f) for f in floors},
                   hit_top5=None, hit_floor=None, hit_floor_primary=None)
        return out
    hits = []
    for d in p["expected"]:
        got = by.get(d)
        hits.append({"documentId": d, "found": got is not None, "rank": got["rank"] if got else None,
                     "score": got["score"] if got else None,
                     "under_floor": {_key(f): bool(got) and accepted(got["score"], f) for f in floors}})
    distractors = []
    best_expected = min((h["score"] for h in hits if h["found"]), default=None)
    for d in p.get("distractors", []):
        got = by.get(d)
        distractors.append({"documentId": d, "in_top5": got is not None, "score": got["score"] if got else None,
                            "outranks": bool(got) and (best_expected is None or got["score"] < best_expected)})
    out.update(expected_hits=hits, distractors=distractors, total=len(hits), found=sum(h["found"] for h in hits),
               floor_found={_key(f): sum(h["under_floor"][_key(f)] for h in hits) for f in floors})
    out["hit_floor"] = {k: n == out["total"] for k, n in out["floor_found"].items()}
    out["hit_top5"] = out["found"] == out["total"]
    out["hit_floor_primary"] = out["hit_floor"][_key(primary_floor)]
    if not p["applies"]:
        out["hit_top5"] = out["hit_floor_primary"] = None
        out["hit_floor"] = None
    return out


def aggregate(results: list[dict], floor: float) -> dict:
    k = _key(floor)
    recall = [r for r in results if r["applies"] and r["answer_class"] != "negative"]
    singles = [r for r in recall if r["answer_class"] == "single"]
    fams = [r for r in recall if r["answer_class"] == "family"]
    negs = [r for r in results if r["answer_class"] == "negative"]
    closest = [r["closest_score"] for r in negs if r["closest_score"] is not None]
    return {
        "floor": floor,
        "pairs_total": len(results), "pairs_not_applicable": sum(not r["applies"] for r in results),
        "pairs_applicable": len(recall),
        "pairs_hit_top5": sum(r["hit_top5"] for r in recall), "pairs_hit_floor": sum(r["hit_floor"][k] for r in recall),
        "singles": len(singles), "singles_hit_top5": sum(r["hit_top5"] for r in singles),
        "singles_hit_floor": sum(r["hit_floor"][k] for r in singles),
        "family_pairs": len(fams), "family_pairs_hit_top5": sum(r["hit_top5"] for r in fams),
        "family_pairs_hit_floor": sum(r["hit_floor"][k] for r in fams),
        "siblings_total": sum(r["total"] for r in fams), "siblings_found_top5": sum(r["found"] for r in fams),
        "siblings_found_floor": sum(r["floor_found"][k] for r in fams),
        "distractors_outranking": sum(d["outranks"] for r in recall for d in r["distractors"]),
        "negatives": len(negs), "negative_false_accepts": sum(r["false_accept"][k] for r in negs),
        "closest_negative_score": min(closest) if closest else None,
    }


def r6_record(fixture: dict, results: list[dict], floors) -> dict:
    doc = fixture["r6_document"]
    for r in results:
        for h in r.get("expected_hits", []):
            if h["documentId"] == doc:
                return {"documentId": doc, "pair": r["id"], "in_top5": h["found"], "rank": h["rank"], "score": h["score"],
                        "under_floor": dict(h["under_floor"])}
    return {"documentId": doc, "pair": None, "in_top5": False, "rank": None, "score": None,
            "under_floor": {_key(f): False for f in floors}}


def run_all(fixture: dict, searcher, template: str, *, ws: str, primary_floor: float, floors) -> dict:
    floors = sorted({*floors, primary_floor})
    results = [run_pair(p, searcher, template, ws, primary_floor, floors) for p in fixture["pairs"]]
    return {"pairs": results, "floors": floors, "primary_floor": primary_floor,
            "aggregates": {_key(f): aggregate(results, f) for f in floors}, "r6_document": r6_record(fixture, results, floors)}


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def render_text(rep: dict) -> str:
    pf = _key(rep["primary_floor"])
    lines = [f"ws={rep['ws']} base_url={rep['base_url']} primary floor {pf}; floors {[f for f in rep['floors']]}"]
    for r in rep["pairs"]:
        if r["answer_class"] == "negative":
            lines.append(f"{r['id']:4} NEG  closest={_fmt(r['closest_score'])} false-accept " +
                         " ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["false_accept"].items()))
        elif not r["applies"]:
            lines.append(f"{r['id']:4} N/A  {r['na_reason']}")
        else:
            best = min((h["score"] for h in r["expected_hits"] if h["found"]), default=None)
            lines.append(f"{r['id']:4} {'HIT ' if r['hit_top5'] else 'MISS'} top5 {r['found']}/{r['total']} best={_fmt(best)} "
                         f"floor {pf}: {'HIT' if r['hit_floor_primary'] else 'MISS'}")
    for k, a in rep["aggregates"].items():
        lines.append(f"floor {k}: top5 pairs {a['pairs_hit_top5']}/{a['pairs_applicable']} (singles {a['singles_hit_top5']}/{a['singles']}, "
                     f"siblings {a['siblings_found_top5']}/{a['siblings_total']}); floor-applied pairs {a['pairs_hit_floor']}/{a['pairs_applicable']} "
                     f"(singles {a['singles_hit_floor']}/{a['singles']}, siblings {a['siblings_found_floor']}/{a['siblings_total']}); "
                     f"negatives false-accepted {a['negative_false_accepts']}/{a['negatives']}, closest {_fmt(a['closest_negative_score'])}; "
                     f"distractors outranking {a['distractors_outranking']}")
    r6 = rep["r6_document"]
    lines.append(f"R6 document {r6['documentId']}: in_top5={r6['in_top5']} rank={r6['rank']} score={_fmt(r6['score'])}")
    return "\n".join(lines)


def main(argv=None, *, searcher=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--ws", required=True)
    ap.add_argument("--floor", type=float, required=True, help="the floor behind the floor-applied hit (strictly above is rejected)")
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--skill", default=str(kb.DEFAULT_SKILL))
    ap.add_argument("--json", action="store_true", help="machine-readable record on stdout")
    args = ap.parse_args(argv)
    try:
        # read-only: the production confirmation guards writes, and this runner never writes
        target = kb.check_target(args.ws, args.base_url, confirm_production=True)
        fixture = load_fixture(Path(args.fixture))
        template = kb.extract_prefix(Path(args.skill).read_text(encoding="utf-8"))
        rep = run_all(fixture, searcher or kb.RestSearcher(target), template, ws=args.ws, primary_floor=args.floor,
                      floors=DEFAULT_FLOORS)
    except (kb.KBError, OSError) as e:
        print(f"kb_golden_runner: {e}", file=sys.stderr)
        return 2
    rep = {"ws": args.ws, "base_url": target.base_url, "fixture": str(args.fixture), **rep}
    print(json.dumps(rep, indent=1) if args.json else render_text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
