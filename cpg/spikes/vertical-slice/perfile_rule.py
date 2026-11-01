#!/usr/bin/env python3
"""perfile_rule.py -- S11 (P0-10): measure the stricter per-file content rule on a slice unit.

Rule under test (plan §2 D15 / S11 row): *every non-empty source file has >= 1 non-external METHOD*.
Input: the staged unit and cpg-content-check.sc's output (``FILE_METHODS <cpg>\\t<file>=<n>`` lines).
Prints one JSON object; the S11 notes record it.  "Source file" = a file the frontend's language owns
by extension; "non-empty" = at least one non-whitespace character.

  perfile_rule.py --lang python|javascript --stage <dir> --check <contentcheck.txt>
"""
import argparse
import json
import re
from pathlib import Path

# jssrc2cpg 4.0.579 default exclusions, read from the constant pool of
# io/joern/jssrc2cpg/utils/AstGenRunner$.class in the sandbox image (S11 gate fix) and confirmed on a synthetic
# tree through joern-parse and through astgen.sc (S11-notes.md, finding 3).  Rule F matches the file name's
# tail (``xtest.ts`` and ``conf.ts`` are excluded too); rule D matches any directory segment of the path
# (``i18n`` yes, ``locales``/``intl`` no).  Only i18n, tests, e2e, docs, examples, www, build, vendor were
# individually confirmed on the synthetic tree; the other listed folders are from the constant pool only.
JS_EXCLUDE_FILE = re.compile(r"(conf|test|spec|[.-]min|\.d)\.(js|jsx|cjs|mjs|xsjs|xsjslib|ts|tsx)$")
JS_EXCLUDE_DIRS = ("node_modules", "venv", "docs", "test", "tests", "e2e", "e2e-beta", "examples", "cypress",
                   "jest-cache", "eslint-rules", "codemods", "flow-typed", "i18n", "vendor", "www", "dist", "build")


def js_exclusion_rules(rel):
    """Names of the default jssrc2cpg exclusion rules that match ``rel`` (a staged relative path)."""
    parts = rel.split("/")
    rules = []
    if JS_EXCLUDE_FILE.search(parts[-1]):
        rules.append("F:file-suffix")
    rules += [f"D:dir={d}" for d in parts[:-1] if d in JS_EXCLUDE_DIRS]
    return rules


EXT = {"python": (".py",), "javascript": (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--check", required=True)
    a = ap.parse_args()
    stage = Path(a.stage)
    counts = {}
    for line in Path(a.check).read_text(encoding="utf-8").splitlines():
        m = re.match(r"FILE_METHODS [^\t]+\t(.*)=(\d+)$", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
    src = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file() and p.suffix in EXT[a.lang])
    nonempty = [f for f in src if (stage / f).read_text(encoding="utf-8", errors="replace").strip()]
    empty = [f for f in src if f not in nonempty]
    in_cpg = [f for f in nonempty if f in counts]
    not_in_cpg = [f for f in nonempty if f not in counts]
    zero_in_cpg = [f for f in in_cpg if counts[f] == 0]
    placeholder = sorted(f for f in counts if f not in src)
    placeholder_zero = [f for f in placeholder if counts[f] == 0]
    res = {
        "lang": a.lang, "sourceFiles": len(src), "nonEmptySourceFiles": len(nonempty), "emptySourceFiles": len(empty),
        "fileNodesInCpg": len(counts), "nonEmptyInCpgWithMethods": len(in_cpg) - len(zero_in_cpg),
        "nonEmptyInCpgWithZeroMethods": zero_in_cpg,
        "nonEmptyNotInCpg": not_in_cpg,
        "emptyFilesInCpgWithZeroMethods": [f for f in empty if counts.get(f, -1) == 0],
        "emptyFilesNotInCpg": [f for f in empty if f not in counts],
        "fileNodesNotASourceFile": {f: counts[f] for f in placeholder},
        # the rule applied naively to every FILE node (what a literal per-file check would see)
        "naiveFalseFailures": sorted([f for f, n in counts.items() if n == 0]),
        # the rule applied to non-empty source files, counting a file the frontend never parsed as a failure
        "ruleOverNonEmptySource_failures": sorted(zero_in_cpg + not_in_cpg),
        "ruleOverCpgFilesOnly_failures": sorted(zero_in_cpg),
    }
    if a.lang == "javascript":
        rules = {f: js_exclusion_rules(f) for f in nonempty}
        res["jsExclusionRules"] = {f: r for f, r in sorted(rules.items()) if r}
        # prediction check: a file is absent from the CPG iff some rule matches it
        res["jsRulePredictionMismatches"] = sorted(f for f in nonempty if bool(rules[f]) != (f not in counts))
    res["holdsOverNonEmptySource"] = not res["ruleOverNonEmptySource_failures"]
    res["holdsOverCpgFilesOnly"] = not res["ruleOverCpgFilesOnly_failures"]
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
