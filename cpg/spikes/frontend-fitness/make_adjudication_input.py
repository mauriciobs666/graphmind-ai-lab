#!/usr/bin/env python3
"""make_adjudication_input.py -- S25 deliverable (P0-18a item 5), run by `data-scientist` at the start of S57
(ML §11 N4), never by `graph-dba`.

    make_adjudication_input.py <lang> [--root cpg/spikes/frontend-fitness] [--out adjudication/<lang>.input.jsonl]

Compares the blind hand labels with the typed reference answer, site by site, and writes one line per
labelled site to adjudication/<lang>.input.jsonl for the pre-Joern adjudication (ML §5.1 step 3).

FIREWALL (checked statically by `analyst`): this script opens **only** `labels/<lang>.jsonl` and
`truth/<lang>.ref.jsonl` for reading, and writes **only** `adjudication/<lang>.input.jsonl`. It imports
nothing from this directory, reads no corpus source, no `sample/`, no `results/`, no Joern artifact, and
no other language's file. The site coordinates it needs come from `truth/<lang>.ref.jsonl`'s `site`
field (the same four public fields the labeller was shown). Standard library only.

Output line (no Joern columns):
  {ordinal, siteId, site{path,line,column,callText}, status, reasons[], label{...}, reference{... + relaxed / ambiguous /
   byConstruction when the reference row carries them: the strata of ML §5.1 step 4},
   inCorpusLabel[], inCorpusReference[], declaredLabel[]}
  status  "agreed"             same in-corpus target set (or both without one): the proposed truth
          "needs-adjudication" at least one reason below
  reasons "disagree-targets"   in-corpus sets differ (label vs reference static + impls)
          "reference-silent"   the label lists an in-corpus target and the reference found none
          "label-external-reference-internal" / "label-internal-reference-external"
          "unsure"             label.confidence == "unsure" (or dynamic with no listed target)
          A label line that violates ML §6.5's schema rejects the whole file (stderr "REJECTED", exit 2).
          "dynamic"            label.dynamic is true
`declared` labels are carried as `declaredLabel` unmapped: ML §6.5 assigns the harness the mapping to role
`decl`/`impl`/`N(s)` from the declaration form at `defLine`, which needs source this script must not open;
the adjudicator applies bundle spec rule 8 there.
A summary line goes to stderr: labelled sites, agreed, per-reason counts, hand-vs-reference agreement rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABEL_KEYS = {"ordinal", "siteId", "callees", "notACall", "dynamic", "confidence", "evidence"}
CALLEE_KINDS = {"internal", "declared", "external"}
LANGS = {"python", "javascript", "java", "kotlin", "csharp", "go", "cpp"}


class LabelSchemaError(ValueError):
    pass


def validate_label(o: dict, line_no: int) -> None:
    """ML §6.5's output schema; the first invalid line rejects the file (ML §6.3)."""
    def bad(msg):
        raise LabelSchemaError(f"labels line {line_no}: {msg}")
    if not isinstance(o, dict):
        bad("not an object")
    if set(o) != LABEL_KEYS:
        bad(f"keys {sorted(set(o) ^ LABEL_KEYS)} differ from the schema")
    if not isinstance(o["ordinal"], int) or o["ordinal"] < 1:
        bad("ordinal")
    if not isinstance(o["siteId"], str) or len(o["siteId"]) != 16:
        bad("siteId")
    if not isinstance(o["notACall"], bool) or not isinstance(o["dynamic"], bool):
        bad("notACall/dynamic must be booleans")
    if o["confidence"] not in ("sure", "unsure"):
        bad("confidence")
    if not isinstance(o["evidence"], str) or len(o["evidence"]) > 300:
        bad("evidence must be a string of at most 300 characters")
    if not isinstance(o["callees"], list):
        bad("callees")
    for c in o["callees"]:
        k = c.get("kind") if isinstance(c, dict) else None
        if k not in CALLEE_KINDS:
            bad(f"callee kind {k!r}")
        if k == "external":
            if set(c) != {"kind", "package"} or not isinstance(c["package"], str) or not c["package"]:
                bad("external callee needs exactly kind+package")
        else:
            if set(c) != {"kind", "path", "qname", "defLine"} or not isinstance(c["path"], str) \
                    or not (c["qname"] is None or isinstance(c["qname"], str)) \
                    or not isinstance(c["defLine"], int) or c["defLine"] < 1:
                bad(f"{k} callee needs exactly kind+path+qname+defLine")


def read_jsonl(path: Path, firewall_dir: str) -> list[dict]:
    if path.parent.name != firewall_dir:
        raise SystemExit(f"refusing to open {path}: only {firewall_dir}/ is allowed here")
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError as e:
                    raise LabelSchemaError(f"{path.name} line {i}: {e}")
    return rows


def label_targets(label: dict) -> tuple[set, set, set]:
    inc, dec, ext = set(), set(), set()
    for c in label["callees"]:
        if c["kind"] == "internal":
            inc.add((c["path"], c["defLine"]))
        elif c["kind"] == "declared":
            dec.add((c["path"], c["defLine"]))
        else:
            ext.add(c["package"])
    return inc, dec, ext


def ref_targets(ref: dict) -> tuple[set, set]:
    inc, ext = set(), set()
    st = ref["static"]
    if st["kind"] in ("internal", "declared"):
        inc.add((st["path"], st["defLine"]))
    elif st["kind"] == "external":
        ext.add(st.get("package", "?"))
    for r in ref.get("impls", []):
        inc.add((r["path"], r["defLine"]))
    return inc, ext


def compare(label: dict, ref: dict) -> list[str]:
    reasons = []
    l_inc, l_dec, l_ext = label_targets(label)
    l_all = l_inc | l_dec
    r_inc, _ = ref_targets(ref)
    kind = ref["static"]["kind"]
    if label["confidence"] == "unsure":
        reasons.append("unsure")
    if label["dynamic"]:
        reasons.append("dynamic")
    if label["notACall"]:
        # a site the labeller calls "not a call" agrees with a silent reference; anything else is a disagreement
        if kind not in ("silent",) and r_inc:
            reasons.append("disagree-targets")
        return reasons
    if l_all != r_inc:
        if l_all and kind == "silent":
            reasons.append("reference-silent")
        elif l_all and not r_inc:
            reasons.append("label-internal-reference-external")
        elif r_inc and not l_all:
            reasons.append("label-external-reference-internal")
        else:
            reasons.append("disagree-targets")
    return reasons


def build(lang: str, root: Path) -> tuple[list[dict], dict]:
    labels = read_jsonl(root / "labels" / f"{lang}.jsonl", "labels")
    for i, o in enumerate(labels, 1):
        validate_label(o, i)
    refs = {r["ordinal"]: r for r in read_jsonl(root / "truth" / f"{lang}.ref.jsonl", "truth")}
    seen = set()
    out, counts = [], {}
    for i, lab in enumerate(labels, 1):
        if lab["ordinal"] in seen:
            raise LabelSchemaError(f"labels line {i}: ordinal {lab['ordinal']} repeated")
        seen.add(lab["ordinal"])
        if lab["ordinal"] != i:
            raise LabelSchemaError(f"labels line {i}: ordinals must run 1,2,3... without skipping (got {lab['ordinal']})")
        ref = refs.get(lab["ordinal"])
        if ref is None:
            raise LabelSchemaError(f"labels line {i}: ordinal {lab['ordinal']} has no reference line")
        if ref["siteId"] != lab["siteId"]:
            raise LabelSchemaError(f"labels line {i}: siteId differs from the reference's for ordinal {lab['ordinal']}")
        reasons = compare(lab, ref)
        l_inc, l_dec, _ = label_targets(lab)
        r_inc, _ = ref_targets(ref)
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
        out.append({
            "ordinal": lab["ordinal"], "siteId": lab["siteId"], "site": ref.get("site"),
            "status": "needs-adjudication" if reasons else "agreed", "reasons": reasons,
            "label": {k: lab[k] for k in ("callees", "notACall", "dynamic", "confidence", "evidence")},
            "reference": {"static": ref["static"], "impls": ref.get("impls", []), "tool": ref.get("tool"), "toolVersion": ref.get("toolVersion"),
                          **{k: ref[k] for k in ("relaxed", "ambiguous", "byConstruction") if k in ref}},   # strata of ML §5.1 step 4
            "inCorpusLabel": sorted([list(x) for x in l_inc]), "declaredLabel": sorted([list(x) for x in l_dec]),
            "inCorpusReference": sorted([list(x) for x in r_inc]),
        })
    n = len(out)
    agreed = sum(1 for o in out if o["status"] == "agreed")
    summary = {"lang": lang, "labelled": n, "agreed": agreed, "needsAdjudication": n - agreed,
               "agreementRate": round(agreed / n, 4) if n else None, "reasonCounts": dict(sorted(counts.items()))}
    return out, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lang", choices=sorted(LANGS))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out_path = a.out or (a.root / "adjudication" / f"{a.lang}.input.jsonl")
    if out_path.parent.name != "adjudication":
        raise SystemExit("refusing to write outside adjudication/")
    try:
        rows, summary = build(a.lang, a.root)
    except LabelSchemaError as e:
        print(f"REJECTED: {e}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(out_path)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
