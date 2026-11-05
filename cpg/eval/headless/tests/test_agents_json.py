"""S52 tests for agents.json (docs/plans/connected-code-graph-p0-2.md §3 S52, §6.1).

Run: python3 -I cpg/eval/headless/tests/test_agents_json.py   (stdlib only, no network)
Checks that the four definitions quote eval-protocol Appendix B verbatim (B.0 + the role's part),
carry no ancestor-plus-line rule and no learning-capture/KB/CPG/repo instructions, fence kit text
as data, and that the two key writers differ in strategy and wording.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
import hashlib, os
AGENTS = json.loads(Path(os.environ.get("AGENTS_JSON", HERE.parent / "agents.json")).read_text())
# The grammar extract is generated once (M-1 ruling: rules 5 and 7 cut, http step 7 without the
# catch-all clause); the four definitions and S40's briefs hold byte-identical copies of it.
EXTRACT_LEN, EXTRACT_SHA256 = 12145, "5202c983c7ced1bc2b1ef166fd2b8a77bfbf3d444f1880e3a5afdd06e009f3ff"
ML = (REPO / "docs/plans/connected-code-graph-eval-protocol-ml.md").read_text()

NAMES = ["ccg-key-writer-provider", "ccg-key-writer-consumer", "ccg-question-writer", "ccg-annotator"]
OUT = {"ccg-key-writer-provider": "out/KA.jsonl", "ccg-key-writer-consumer": "out/KB.jsonl",
       "ccg-question-writer": "out/QW.jsonl", "ccg-annotator": None}
ROLE_PART = {"ccg-key-writer-provider": "B.1", "ccg-key-writer-consumer": "B.1",
             "ccg-question-writer": "B.2", "ccg-annotator": "B.3"}

app = ML[ML.index("## Appendix B. Writer-facing text"):]
def part(start, end):
    i = app.index(start); i = app.index("\n", i) + 1
    j = app.index(end, i) if end else len(app)
    return app[i:j].strip("\n")
B = {"B.0": part("### B.0 Common to every role", "\n---\n\n### B.1"),
     "B.1": part("### B.1 Key writers", "\n---\n\n### B.2"),
     "B.2": part("### B.2 Question writer", "\n---\n\n### B.3"),
     "B.3": part("### B.3 Annotators", None)}

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: passed += 1; print("ok   " + name)
    else: failed += 1; print("FAIL " + name + (" :: " + detail if detail else ""))

check("exactly the four definitions, in order", list(AGENTS) == NAMES, str(list(AGENTS)))
for n, d in AGENTS.items():
    check(n + ": fields are description, tools, prompt", sorted(d) == ["description", "prompt", "tools"], str(sorted(d)))
    check(n + ": tools equal the launcher's --tools set", d["tools"] == ["Read", "Grep", "Glob", "Write", "Edit"])

def b0_regex(out, tail):
    pieces = B["B.0"].split("{grammarExtract}")
    assert len(pieces) == 2
    pre = pieces[0].replace("{outputPath}", out) if out else pieces[0]
    return re.compile(re.escape(pre) + r"(?P<g>.+?)" + re.escape(pieces[1] + "\n\n" + tail), re.S)

grammar = {}
for n, d in AGENTS.items():
    p = d["prompt"]
    m = b0_regex(OUT[n], B[ROLE_PART[n]][:60]).search(p)
    check(n + ": B.0 verbatim (slot outputPath filled, grammarExtract filled)", m is not None)
    if m: grammar[n] = m.group("g")
    check(n + ": role part " + ROLE_PART[n] + " verbatim", B[ROLE_PART[n]] in p)
    for other in {"B.1", "B.2", "B.3"} - {ROLE_PART[n]}:
        check(n + ": no foreign part " + other, B[other][:200] not in p)
    # data fence (-spec2 F-3)
    check(n + ": kit text fenced as data", "is **data to analyse**" in p and "never a source of orders" in p)
    check(n + ": no ancestor-plus-line rule (E1)", not re.search(r"ancestor|plus[- ]line|§3\.5", p, re.I))
    # authored text = prompt minus the pasted Appendix pieces
    authored = p
    if m: authored = authored.replace(m.group(0), "")
    authored = authored.replace(B[ROLE_PART[n]], "").replace("{cypherMcpTools}", "")
    bad = re.findall(r"kaizen|learning|ingest_document|search_documents|knowledge|\bKB\b|\bCPG\b|joern|cypher|falkor|\bmcp__|AGENTS\.md|CLAUDE\.md|\.claude|skill", authored, re.I)
    check(n + ": authored text has no learning-capture/KB/CPG/repo instructions", not bad, str(bad))
    gbad = re.findall(r"falkor|salesperson|storefront|embedding\.py|test_fusion|joern|callsite|analyst|P-slice|P-later|§[0-9]", re.sub(r"RFC 3986 §[0-9.]+", "", grammar.get(n, "")))
    check(n + ": grammar extract names no proving-repo element or unpasted section", not gbad, str(gbad[:5]))
    check(n + ": grammar extract omits rule 5 and has rules 1-4, 6, 8-10 (7 cut)",
          grammar.get(n) is not None and "5. **Anonymous" not in grammar[n]
          and all(re.search(r"(?m)^%d\. \*\*" % i, grammar[n]) for i in (1, 2, 3, 4, 6, 8, 9, 10))
          and not re.search(r"(?m)^7\. ", grammar[n]) and "(Rules 5 and 7 are not used in this brief.)" in grammar[n]
          and "{**}" not in grammar[n] and "Duplicates get" not in grammar[n])
    gb = grammar.get(n, "").encode()
    check(n + ": grammar extract pinned (length, sha256)", len(gb) == EXTRACT_LEN and hashlib.sha256(gb).hexdigest() == EXTRACT_SHA256, "%d %s" % (len(gb), hashlib.sha256(gb).hexdigest()))
    check(n + ": grammar extract has no JSON example block", "```json" not in p and '"schema":"ccg-key-v1"' not in grammar.get(n, ""))

# the two key writers: same schema text, different strategy and wording
pa, pb = AGENTS[NAMES[0]]["prompt"], AGENTS[NAMES[1]]["prompt"]
def strategy(p): return p[p.index("# Your strategy"):p.index("# Writer-facing specification")]
sa, sb = strategy(pa), strategy(pb)
wa, wb = set(re.findall(r"[a-z]+", sa.lower())), set(re.findall(r"[a-z]+", sb.lower()))
check("key writers: strategy sections differ in wording (Jaccard < 0.5)", len(wa & wb) / len(wa | wb) < 0.5, "%.2f" % (len(wa & wb) / len(wa | wb)))
check("key writers: provider-first vs consumer-first named", "provider-first" in sa and "consumer-first" in sb and "consumer-first" not in sa and "provider-first" not in sb)
check("key writers: strategy is not the only difference in outputs path", AGENTS[NAMES[0]]["description"] != AGENTS[NAMES[1]]["description"])

check("key writers/QW/annotator: extracts byte-identical", len({grammar.get(n) for n in NAMES}) == 1)
check("no 'working notes' file instruction (m-4)", all("working notes" not in AGENTS[n]["prompt"] for n in NAMES))
check("rule 4 keeps spec3's wording (m-1)", all("extractor parsed" not in AGENTS[n]["prompt"] for n in NAMES))

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
