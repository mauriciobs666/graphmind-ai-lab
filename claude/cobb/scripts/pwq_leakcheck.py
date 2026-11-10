#!/usr/bin/env python3
"""pwq_leakcheck.py -- the mechanical leak check of ML §5 rule 3 / plan S5.

A positive query must share no 4-gram with its own section (title or body) of the pinned
python-web-quirks source. Tokens are lowercase `[a-z0-9]+` runs, so case, punctuation and
code-span backticks never hide a shared phrase. Only the query's own section is compared.

    pwq_leakcheck.py --set fixtures/pwq-golden-set-a.json --pin <sha>   # source via `git show`
    pwq_leakcheck.py --set SET --source SKILL.md                         # source from a file

Exit 0 = no shared 4-gram; 1 = hits (printed); 2 = unreadable input. Queries are never edited.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pwq_kb import DEFAULT_SET_A, KBError, load_set, parse_source, read_pinned  # noqa: E402

N = 4


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(toks):
    return {" ".join(toks[i : i + N]) for i in range(len(toks) - N + 1)}


def check(data, sections):
    """Hits as dicts {id, section, where, gram}, one per shared 4-gram, for the positives of `data`."""
    secs = list(sections)
    hits = []
    for p in data["positives"]:
        if not 1 <= p["section"] <= len(secs):
            raise KBError(f"{p['id']}: section {p['section']} not in 1..{len(secs)}")
        sec = secs[p["section"] - 1]
        grams = ngrams(tokens(p["query"]))
        for where, text in (("title", sec.heading), ("body", sec.body)):
            for g in sorted(grams & ngrams(tokens(text))):
                hits.append({"id": p["id"], "section": p["section"], "where": where, "gram": g})
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=str(DEFAULT_SET_A), help="golden set JSON (default: Set A)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pin", help="commit sha; reads the source with `git show`")
    src.add_argument("--source", help="path to the source markdown")
    a = ap.parse_args(argv)
    try:
        text = Path(a.source).read_text(encoding="utf-8") if a.source else read_pinned(a.pin)
        data = load_set(Path(a.set))
        hits = check(data, parse_source(text).sections)
    except (KBError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for h in hits:
        print(f"{h['id']} section {h['section']} {h['where']}: {h['gram']!r}")
    print(f"{len(data['positives'])} positives checked, {len(hits)} shared {N}-grams")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
