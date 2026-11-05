"""Fill the required top-level `flagByVersion` of `adjudication/<lang>.noise.json` (ML note v1.13, §5.4 step 7).

    python populate_flag_by_version.py [--adjudication DIR] [lang ...]

  v1.10.figure = the file's top-level handError.e_all.point; v1.12.figure = handError.e_res.point;
  v1.11.figure = the e_res the custodian stored under the v1.11 rules, read from the path in V111_SOURCE
  (inside `detail`). The v1.11 figure is never computed: if the stored value is absent or not a number the
  script stops (NoStoredFigure) and the file is left unedited. `flagged` is `figure > 0.05`.
  Also removes the superseded `detail.v112.verdictByVersion` from cpp.noise.json (DROP).

Deterministic and idempotent (a second run is a no-op; an existing flagByVersion that differs is refused);
the custodians' format (indent=1, ASCII-escaped, no trailing newline) is preserved and anything else refused.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = "flagByVersion"
FLAG = 0.05
LANGS = ("python", "javascript", "go", "java", "csharp", "cpp", "kotlin")
# where each custodian kept the v1.11-rules e_res (the "without role errors" point, for the pre-v1.11 shapes)
V111_SOURCE = {
    "python": ("detail", "handError", "e_res"),
    "javascript": ("detail", "handErrorRate", "withoutRoleErrors"),
    "go": ("detail", "original_v1_11", "handErrorRate", "pointWithoutRoleErrors"),
    "java": ("detail", "original_v111", "handErrorRate", "withoutRoleErrors", "point"),
    "csharp": ("detail", "handErrorRate", "withoutRoleErrors", "point"),
    "cpp": ("detail", "v111Original", "handErrorRate", "e_res", "point"),
    "kotlin": ("detail", "handError", "v1.11_e_res_for_comparison"),
}
DROP = {"cpp": ("detail", "v112", "verdictByVersion")}


class NoStoredFigure(ValueError):
    pass


def _get(doc, path):
    node = doc
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _entry(figure):
    return {"figure": figure, "flagged": figure > FLAG}


def populate(noise: dict, source: tuple, drop: tuple | None = None):
    """Return (new_noise, changed); `source` is the path of the stored v1.11 figure, `drop` a key path to remove."""
    v111 = _get(noise, source)
    if not isinstance(v111, (int, float)) or isinstance(v111, bool):
        raise NoStoredFigure(f"no stored numeric v1.11 figure at {'.'.join(source)} (found {v111!r}); not computing one")
    he = noise["handError"]
    want = {"v1.10": _entry(he["e_all"]["point"]), "v1.11": _entry(v111), "v1.12": _entry(he["e_res"]["point"])}
    changed = False
    if KEY in noise:
        if noise[KEY] != want:
            raise ValueError(f"existing {KEY} {noise[KEY]} differs from the stored figures {want}")
    else:
        out = {}
        for k, v in noise.items():
            if k == "openPointsS27":
                out[KEY] = want
            out[k] = v
        if KEY not in out:
            out[KEY] = want
        noise, changed = out, True
    if drop is not None and _get(noise, drop) is not None:
        noise = copy.deepcopy(noise)
        del _get(noise, drop[:-1])[drop[-1]]
        changed = True
    return noise, changed


def _format(doc) -> str:
    return json.dumps(doc, indent=1)


def populate_file(path, source, drop=None) -> bool:
    path = Path(path)
    text = path.read_text()
    before = json.loads(text)
    if text != _format(before):
        raise ValueError(f"{path.name}: unrecognised formatting (expected indent=1, ASCII, no trailing newline)")
    after, changed = populate(copy.deepcopy(before), source, drop)
    if not changed:
        return False
    check = copy.deepcopy(after)
    check.pop(KEY, None)
    if drop is not None:
        base = copy.deepcopy(before)
        parent = _get(base, drop[:-1])
        if parent is not None:
            parent.pop(drop[-1], None)
        before_cmp = base
    else:
        before_cmp = before
    if check != before_cmp:
        raise AssertionError(f"{path.name}: changed more than {KEY}" + (f" and {'.'.join(drop)}" if drop else ""))
    path.write_text(_format(after))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("langs", nargs="*")
    ap.add_argument("--adjudication", default=str(HERE / "adjudication"))
    a = ap.parse_args(argv)
    langs = a.langs or LANGS
    plan = []
    for lang in langs:  # all-or-nothing: resolve every language before editing any file
        p = Path(a.adjudication) / f"{lang}.noise.json"
        try:
            populate(json.loads(p.read_text()), V111_SOURCE[lang], DROP.get(lang))
        except NoStoredFigure as e:
            print(f"{lang}: STOP: {e}", file=sys.stderr)
            return 2
        plan.append((lang, p))
    for lang, p in plan:
        print(f"{lang}: {'populated' if populate_file(p, V111_SOURCE[lang], DROP.get(lang)) else 'already populated, untouched'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
