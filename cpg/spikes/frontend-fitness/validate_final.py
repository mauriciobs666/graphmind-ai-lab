"""Validator for the S57 truth files: `truth/<lang>.final.jsonl` and `adjudication/<lang>.noise.json`.

Closes analyst gate blocker B1 (`docs/reviews/connected-code-graph-frontend-fitness.md` Pass 3;
procedure `docs/plans/connected-code-graph-p0-ml.md` §5.4 steps 6-7, stop-list item 7). The seven
languages were written to different shapes; this module defines the ONE canonical shape (below) and
`normalize_final.py` converts the existing files to it. Run:

    cd cpg/spikes/frontend-fitness && python validate_final.py [--truth DIR] [--adjudication DIR] [--input DIR] [lang ...]

Exit 0 only when every line of every named language (default: all seven) validates; each violation
prints as `<file>: ordinal <n>: [rule] message`.

CANONICAL final.jsonl record (exactly these 14 keys, no others; one line per labelled site, in
`input.jsonl` order, `ordinal` 1..N):
  ordinal int>=1 | siteId str | site "path:line" str (never a dict: callText/column live in input.jsonl)
  T  list of {path str, qname str|null, defLine int>=1, role in ROLES} plus optional bool
     `qnameDerived` / `macroGenerated` (cpp's marks; kept, not dropped) | N list[str] (the N(s) notes)
  external list[str] of package/header names (never a bare str, None or {kind,package} dict)
  notACall / dynamic / scored bools | notScoredReason null iff scored, else one of NOT_SCORED_REASONS
  basis in BASIS | strata list of distinct plain tags (no "audit-U:AI" compounds); may be empty (go's 6 adjudicated
  sites carry no tag; none is invented)
  evidence str containing at least one `path:line` (several citations are joined with "; ")
Cross-field rules: unscored when notACall or (dynamic and T empty); a scored line decides something
(T, N or external non-empty); `label-only` implies an `AS` tag.
(Choice of shape: majority/spec. `site`, `external` list[str], `evidence` str are the majority forms
and §5.4 step 6 names `evidence`; `citation`/`citations` are the minority spellings.)

CANONICAL noise.json: a fixed envelope of machine-checked scalars (NOISE_KEYS) plus `detail`, the
original file verbatim, so no prose/finding is lost. The envelope is cross-checked against the final
file (N, basisCounts, unscored). `handError.noiseLimited` must agree with `e_res.point > 0.05` (§5.4
step 7: the flag uses e_res). `flagByVersion` (v1.13) is required: exactly v1.10/v1.11/v1.12 each {figure, flagged}, flagged == figure > 0.05,
v1.10 == top-level e_all.point, v1.12 == e_res.point, v1.12 <= v1.11 <= v1.10. `openPointsS27` (v1.12 §5.4 step 7) is a required top-level key: a non-empty list, or a dict holding
`classLineTargets` and two more category entries, each possibly a 'none found' statement. `census.targetsRead` may be null only when no closed class is claimed
(condition (ii) binds only a claimed closed class).

Vocabulary lives in the constants right below; change them there (the AS-basis ruling amends BASIS).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent

LANGS = ("python", "javascript", "go", "java", "csharp", "cpp", "kotlin")
# §5.4 step 6 (v1.11). The data-scientist's AS-basis ruling amends this tuple only.
BASIS = ("agreed", "audit-confirmed", "adjudicated", "audit-overturned", "sweep-overturned", "label-only")
ROLES = ("impl", "decl")
NOT_SCORED_REASONS = ("notACall", "dynamic", "macro-underdetermined", "underivable")
ERROR_KINDS = ("target", "kind", "role", "coordinate", "shape", "none")
CLOSED_KINDS = ("role", "coordinate", "shape")  # §5.4 step 7 (i)
WRONG = ("label", "reference", "both", "neither")
AUDIT_STRATA = ("AI", "AE", "AS")
CLOSED_SHARE_CAP = 0.10  # §5.4 step 7 (iv)
FLAG_VERSIONS = ("v1.10", "v1.11", "v1.12")  # §5.4 step 7 flagByVersion (v1.13)
FLAG = 0.05  # §5.4 step 7: e_res above this is noise-limited

FINAL_KEYS = ("ordinal", "siteId", "site", "T", "N", "external", "notACall", "dynamic", "scored",
              "notScoredReason", "basis", "strata", "evidence")
T_KEYS = ("path", "qname", "defLine", "role")
T_OPTIONAL = ("qnameDerived", "macroGenerated")
NOISE_KEYS = ("lang", "N", "comparable", "referenceSilent", "agreement", "adjudicated", "referenceError",
              "audit", "handError", "census", "closedClasses", "unscored", "basisCounts", "flagByVersion", "openPointsS27", "detail")

Violation = namedtuple("Violation", "ordinal rule message")
_PATH_LINE = re.compile(r"\S+:\d+")
_SITE = re.compile(r"^.+:\d+$")


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_record(rec) -> list[tuple[str, str]]:
    """Violations (rule, message) of one final.jsonl line."""
    if not isinstance(rec, dict):
        return [("keys", "line is not a JSON object")]
    out: list[tuple[str, str]] = []

    def bad(rule, msg):
        out.append((rule, msg))

    missing, extra = set(FINAL_KEYS) - set(rec), set(rec) - set(FINAL_KEYS)
    if missing or extra:
        bad("keys", f"missing {sorted(missing)} unknown {sorted(extra)}")
    g = rec.get
    if "ordinal" in rec and not (_int(g("ordinal")) and g("ordinal") >= 1):
        bad("ordinal", f"ordinal must be an int >= 1, got {g('ordinal')!r}")
    if "siteId" in rec and not (isinstance(g("siteId"), str) and g("siteId")):
        bad("siteId", "siteId must be a non-empty str")
    if "site" in rec and not (isinstance(g("site"), str) and _SITE.match(g("site"))):
        bad("site", f"site must be a 'path:line' str, got {g('site')!r}")

    T = g("T")
    if "T" in rec:
        if not isinstance(T, list):
            bad("T", "T must be a list")
            T = None
        else:
            for i, t in enumerate(T):
                if not isinstance(t, dict):
                    bad("T", f"T[{i}] is not an object")
                    continue
                miss, ext = set(T_KEYS) - set(t), set(t) - set(T_KEYS) - set(T_OPTIONAL)
                if miss or ext:
                    bad("T", f"T[{i}] missing {sorted(miss)} unknown {sorted(ext)}")
                if "path" in t and not (isinstance(t["path"], str) and t["path"]):
                    bad("T", f"T[{i}].path must be a non-empty str")
                if "qname" in t and not (t["qname"] is None or isinstance(t["qname"], str)):
                    bad("T", f"T[{i}].qname must be str or null")
                if "defLine" in t and not (_int(t["defLine"]) and t["defLine"] >= 1):
                    bad("T", f"T[{i}].defLine must be an int >= 1")
                if "role" in t and t["role"] not in ROLES:
                    bad("T", f"T[{i}].role {t['role']!r} not in {ROLES}")
                for k in T_OPTIONAL:
                    if k in t and not isinstance(t[k], bool):
                        bad("T", f"T[{i}].{k} must be a bool")
    N = g("N")
    if "N" in rec and not (isinstance(N, list) and all(isinstance(x, str) for x in N)):
        bad("N", "N must be a list of str")
        N = None
    ext = g("external")
    if "external" in rec and not (isinstance(ext, list) and all(isinstance(x, str) and x for x in ext)):
        bad("external", f"external must be a list of non-empty str, got {type(ext).__name__}")
        ext = None
    bools_ok = True
    for k in ("notACall", "dynamic", "scored"):
        if k in rec and not isinstance(g(k), bool):
            bad("bools", f"{k} must be a bool, got {g(k)!r}")
            bools_ok = False
    if "basis" in rec and g("basis") not in BASIS:
        bad("basis", f"basis {g('basis')!r} not in {BASIS}")
    st = g("strata")
    if "strata" in rec:
        if not (isinstance(st, list) and all(isinstance(x, str) and x for x in st)):
            bad("strata", "strata must be a list of non-empty str")
            st = None
        elif len(set(st)) != len(st):
            bad("strata", f"duplicate tags in {st}")
        elif any(":" in x for x in st):
            bad("strata", f"compound tag (use separate tags): {[x for x in st if ':' in x]}")
    ev = g("evidence")
    if "evidence" in rec and not (isinstance(ev, str) and _PATH_LINE.search(ev)):
        bad("evidence", "evidence must be a str with at least one path:line")

    reason = g("notScoredReason")
    if "notScoredReason" in rec and "scored" in rec and bools_ok:
        if g("scored") and reason is not None:
            bad("notScoredReason", f"scored line carries notScoredReason {reason!r}")
        elif not g("scored") and reason not in NOT_SCORED_REASONS:
            bad("notScoredReason", f"unscored line needs a reason in {NOT_SCORED_REASONS}, got {reason!r}")
    if bools_ok and all(k in rec for k in ("scored", "notACall", "dynamic")) and T is not None:
        if g("scored") and g("notACall"):
            bad("scored-consistency", "notACall line must be unscored")
        if g("scored") and g("dynamic") and not T:
            bad("scored-consistency", "dynamic line with empty T must be unscored")
        if g("scored") and T is not None and N is not None and ext is not None and not (T or N or ext):
            bad("nothing-decided", "scored line with empty T, N and external")
    if g("basis") == "label-only" and st is not None and "AS" not in st:
        bad("label-only-AS", "basis label-only requires the AS stratum tag")
    return out


def validate_rows(rows, expected=None) -> list[Violation]:
    """Violations of a whole final file. `expected`: [(ordinal, siteId)] from input.jsonl, in order."""
    out: list[Violation] = []
    seen_ids: dict = {}
    for i, rec in enumerate(rows, 1):
        ordv = rec.get("ordinal") if isinstance(rec, dict) else None
        for rule, msg in validate_record(rec):
            out.append(Violation(ordv, rule, msg))
        if ordv != i:
            out.append(Violation(ordv, "ordinal-sequence", f"line {i} has ordinal {ordv!r}; expected {i}"))
        sid = rec.get("siteId") if isinstance(rec, dict) else None
        if sid is not None:
            if sid in seen_ids:
                out.append(Violation(ordv, "siteId-duplicate", f"siteId {sid} also at ordinal {seen_ids[sid]}"))
            seen_ids.setdefault(sid, ordv)
    if expected is not None:
        got = [(r.get("ordinal"), r.get("siteId")) if isinstance(r, dict) else (None, None) for r in rows]
        if len(got) != len(expected):
            out.append(Violation(None, "input-mismatch", f"{len(got)} lines but input.jsonl has {len(expected)}"))
        for i, (a, b) in enumerate(zip(got, expected), 1):
            if tuple(a) != tuple(b):
                out.append(Violation(i, "input-mismatch", f"final has {a}, input.jsonl has {tuple(b)}"))
    return out


def _wilson_ok(w):
    return isinstance(w, list) and len(w) == 2 and all(_num(x) for x in w) and 0 <= w[0] <= w[1] <= 1


def validate_noise(noise, rows=None) -> list[tuple[str, str]]:
    """Violations (rule, message) of a noise.json against the canonical envelope; `rows` = the final file."""
    if not isinstance(noise, dict):
        return [("noise-keys", "noise.json is not an object")]
    out: list[tuple[str, str]] = []

    def bad(rule, msg):
        out.append((rule, msg))

    miss, extra = set(NOISE_KEYS) - set(noise), set(noise) - set(NOISE_KEYS)
    if miss or extra:
        bad("noise-keys", f"missing {sorted(miss)} unknown {sorted(extra)}")
        if miss:
            return out
    N = noise["N"]
    if not _int(N) or N < 1 or (rows is not None and N != len(rows)):
        bad("noise-N", f"N={N!r}" + (f" but final has {len(rows)} lines" if rows is not None else ""))
        N = None
    comp, sil = noise["comparable"], noise["referenceSilent"]
    if not (_int(comp) and _int(sil)) or (N is not None and comp + sil != N):
        bad("noise-comparable", f"comparable {comp!r} + referenceSilent {sil!r} != N {N!r}")
    ag = noise["agreement"]
    if not (isinstance(ag, dict) and _int(ag.get("agreed")) and _int(ag.get("comparable")) and _num(ag.get("rate"))
            and _wilson_ok(ag.get("wilson95"))):
        bad("noise-agreement", "agreement needs int agreed/comparable, numeric rate, wilson95 pair")
    else:
        if ag["comparable"] != comp or ag["agreed"] > ag["comparable"]:
            bad("noise-agreement", "agreement.comparable must equal comparable and agreed <= comparable")
        elif ag["comparable"] and abs(ag["rate"] - ag["agreed"] / ag["comparable"]) > 0.001:
            bad("noise-agreement", f"rate {ag['rate']} != agreed/comparable")
    ad = noise["adjudicated"]
    if not (isinstance(ad, dict) and _int(ad.get("n")) and _int(ad.get("W")) and isinstance(ad.get("byWrong"), dict)
            and isinstance(ad.get("byErrorKind"), dict)):
        bad("noise-adjudicated", "adjudicated needs int n, W and dicts byWrong, byErrorKind")
    else:
        bw, bk = ad["byWrong"], ad["byErrorKind"]
        if set(bw) - set(WRONG) or set(bk) - set(ERROR_KINDS):
            bad("noise-adjudicated", f"vocabulary: byWrong {sorted(set(bw) - set(WRONG))} byErrorKind {sorted(set(bk) - set(ERROR_KINDS))}")
        if sum(bw.values()) != ad["n"] or sum(bk.values()) != ad["n"]:
            bad("noise-adjudicated", f"byWrong/byErrorKind must each sum to n={ad['n']}")
        if ad["W"] != bw.get("label", 0) + bw.get("both", 0):
            bad("noise-adjudicated", f"W {ad['W']} != byWrong label+both {bw.get('label', 0) + bw.get('both', 0)}")
    re_ = noise["referenceError"]
    if not (isinstance(re_, dict) and _int(re_.get("wrong")) and _int(re_.get("comparable")) and _num(re_.get("rate"))
            and (re_.get("wilson95") is None or _wilson_ok(re_.get("wilson95")))):
        bad("noise-referenceError", "referenceError needs int wrong/comparable, numeric rate, wilson95 pair|null")
    au = noise["audit"]
    if not isinstance(au, dict) or set(au) != set(AUDIT_STRATA):
        bad("noise-audit", f"audit needs exactly the strata {AUDIT_STRATA}")
    else:
        for s in AUDIT_STRATA:
            a = au[s]
            if not (isinstance(a, dict) and all(_int(a.get(k)) for k in ("N_s", "n", "k"))
                    and (a.get("wilson95") is None or _wilson_ok(a.get("wilson95")))):
                bad("noise-audit", f"{s}: needs int N_s, n, k and wilson95 pair|null")
            elif not 0 <= a["k"] <= a["n"] <= a["N_s"]:
                bad("noise-audit", f"{s}: need 0 <= k <= n <= N_s, got k={a['k']} n={a['n']} N_s={a['N_s']}")
    he = noise["handError"]
    pts = {}
    if not isinstance(he, dict) or not isinstance(he.get("noiseLimited"), bool):
        bad("noise-handError", "handError needs bool noiseLimited and e_all/e_res {point, conservative}")
    else:
        for k in ("e_all", "e_res"):
            v = he.get(k)
            if not (isinstance(v, dict) and _num(v.get("point")) and _num(v.get("conservative"))):
                bad("noise-handError", f"{k} needs numeric point and conservative")
            else:
                pts[k] = v["point"]
        if len(pts) == 2:
            if he["noiseLimited"] != (pts["e_res"] > FLAG):
                bad("noise-handError", f"noiseLimited={he['noiseLimited']} contradicts e_res.point={pts['e_res']} (flag at > {FLAG})")
            if pts["e_res"] > pts["e_all"] + 1e-9:
                bad("noise-handError", "e_res cannot exceed e_all")
    fv = noise["flagByVersion"]
    fig = {}
    if not (isinstance(fv, dict) and set(fv) == set(FLAG_VERSIONS)):
        bad("noise-flagByVersion", f"flagByVersion needs exactly the keys {FLAG_VERSIONS}")
    else:
        for v in FLAG_VERSIONS:
            e = fv[v]
            if not (isinstance(e, dict) and set(e) == {"figure", "flagged"}):
                bad("noise-flagByVersion", f"{v} needs exactly figure and flagged")
            elif not (_num(e["figure"]) and 0 <= e["figure"] <= 1):
                bad("noise-flagByVersion", f"{v}.figure must be a number in [0,1], got {e['figure']!r}")
            elif not isinstance(e["flagged"], bool) or e["flagged"] != (e["figure"] > FLAG):
                bad("noise-flagByVersion", f"{v}.flagged {e['flagged']!r} must be a bool equal to figure > {FLAG} (figure {e['figure']})")
            else:
                fig[v] = e["figure"]
        if "v1.10" in fig and "e_all" in pts and abs(fig["v1.10"] - pts["e_all"]) > 1e-9:
            bad("noise-flagByVersion", f"v1.10.figure {fig['v1.10']} must equal the top-level e_all.point {pts['e_all']}")
        if "v1.12" in fig and "e_res" in pts and abs(fig["v1.12"] - pts["e_res"]) > 1e-9:
            bad("noise-flagByVersion", f"v1.12.figure {fig['v1.12']} must equal the top-level e_res.point {pts['e_res']}")
        if len(fig) == 3 and not fig["v1.12"] <= fig["v1.11"] <= fig["v1.10"]:
            bad("noise-flagByVersion", f"v1.11.figure {fig['v1.11']} must lie in [v1.12 {fig['v1.12']}, v1.10 {fig['v1.10']}]")
    cc = noise["closedClasses"]
    cc_count = None
    if not (isinstance(cc, dict) and _int(cc.get("count")) and isinstance(cc.get("byKind"), dict)):
        bad("noise-closedClasses", "closedClasses needs int count and dict byKind")
    else:
        bk = cc["byKind"]
        if set(bk) != set(CLOSED_KINDS) or not all(isinstance(v, list) and all(_int(x) for x in v) for v in bk.values()):
            bad("noise-closedClasses", f"byKind needs exactly {CLOSED_KINDS}, each a list of ordinals")
        else:
            union = {x for v in bk.values() for x in v}
            if cc["count"] != len(union):
                bad("noise-closedClasses", f"count {cc['count']} != distinct ordinals {len(union)}")
            if N and cc["count"] / N > CLOSED_SHARE_CAP:
                bad("noise-closedClasses", f"closed-class share {cc['count']}/{N} above {CLOSED_SHARE_CAP} (return to coordinator)")
            if N and any(x < 1 or x > N for x in union):
                bad("noise-closedClasses", "ordinal outside 1..N")
            cc_count = cc["count"]
    ce = noise["census"]
    if not isinstance(ce, dict) or "targetsRead" not in ce:
        bad("noise-census", "census needs targetsRead")
    else:
        tr = ce["targetsRead"]
        if tr is None:
            if cc_count != 0:
                bad("noise-census", "targetsRead null while a closed class is claimed (condition (ii))")
        elif not (_int(tr) and tr >= 0):
            bad("noise-census", f"targetsRead must be int >= 0 or null, got {tr!r}")
    un = noise["unscored"]
    if not (isinstance(un, dict) and isinstance(un.get("byReason"), dict) and _int(un.get("underivable"))
            and set(un["byReason"]) <= set(NOT_SCORED_REASONS) and all(_int(v) and v > 0 for v in un["byReason"].values())):
        bad("noise-unscored", "unscored needs byReason {reason: count>0} and int underivable")
    elif rows is not None:
        c = Counter(r.get("notScoredReason") for r in rows if isinstance(r, dict) and r.get("notScoredReason"))
        if dict(c) != un["byReason"] or c.get("underivable", 0) != un["underivable"]:
            bad("noise-unscored", f"unscored {un} differs from final {dict(c)}")
    bc = noise["basisCounts"]
    if not (isinstance(bc, dict) and set(bc) <= set(BASIS) and all(_int(v) for v in bc.values())):
        bad("noise-basisCounts", f"basisCounts keys must be in {BASIS}")
    else:
        if N and sum(bc.values()) != N:
            bad("noise-basisCounts", f"basisCounts sum {sum(bc.values())} != N {N}")
        if rows is not None:
            c = Counter(r.get("basis") for r in rows if isinstance(r, dict))
            if {k: v for k, v in c.items() if v} != {k: v for k, v in bc.items() if v}:
                bad("noise-basisCounts", f"basisCounts {bc} differs from final {dict(c)}")
    op = noise["openPointsS27"]
    if isinstance(op, dict) and op:
        if "classLineTargets" not in op or len(op) < 3:
            bad("noise-openPointsS27", "dict needs classLineTargets plus the other two §5.4 categories (anonymous-lambda indices, externals absent from the tree); an empty category is an explicit 'none found' entry")
    elif not (isinstance(op, list) and op):
        bad("noise-openPointsS27", "openPointsS27 must be a non-empty dict or list (one entry per category, 'none found' written out)")
    if not isinstance(noise["detail"], dict):
        bad("noise-keys", "detail must be the original noise.json object")
    return out


def validate_lang(lang, truth_dir, adj_dir, input_dir=None) -> list[str]:
    """All violations for one language as printable lines."""
    out: list[str] = []
    fpath, npath = Path(truth_dir) / f"{lang}.final.jsonl", Path(adj_dir) / f"{lang}.noise.json"
    ipath = Path(input_dir or adj_dir) / f"{lang}.input.jsonl"
    rows = None
    if not fpath.exists():
        out.append(f"{fpath.name}: missing file")
    else:
        rows, parse_bad = [], False
        for n, line in enumerate(fpath.read_text().splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                out.append(f"{fpath.name}: line {n}: [json] {e}")
                parse_bad = True
        if not parse_bad:
            expected = None
            if ipath.exists():
                expected = [(r["ordinal"], r["siteId"]) for r in map(json.loads, ipath.read_text().splitlines())]
            for v in validate_rows(rows, expected):
                where = f"ordinal {v.ordinal}" if v.ordinal is not None else "file"
                out.append(f"{fpath.name}: {where}: [{v.rule}] {v.message}")
        else:
            rows = None
    if not npath.exists():
        out.append(f"{npath.name}: missing file")
    else:
        try:
            noise = json.loads(npath.read_text())
        except ValueError as e:
            return out + [f"{npath.name}: [json] {e}"]
        for rule, msg in validate_noise(noise, rows):
            out.append(f"{npath.name}: [{rule}] {msg}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("langs", nargs="*", help="default: all seven")
    ap.add_argument("--truth", default=str(HERE / "truth"))
    ap.add_argument("--adjudication", default=str(HERE / "adjudication"), help="dir of <lang>.noise.json")
    ap.add_argument("--input", default=None, help="dir of <lang>.input.jsonl (default: --adjudication)")
    a = ap.parse_args(argv)
    total = 0
    for lang in a.langs or LANGS:
        v = validate_lang(lang, a.truth, a.adjudication, a.input)
        total += len(v)
        print(f"{lang}: {'OK' if not v else f'{len(v)} violation(s)'}")
        for line in v:
            print("  " + line)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
