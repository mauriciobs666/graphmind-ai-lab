"""Drift guard for SKILL.md / references/glossary.md (AC-15, FR-1, FR-25)."""
import re
from pathlib import Path

import pytest

import wilib.rules as rules
from wilib.cli import build_parser

SKILL_DIR = Path(__file__).resolve().parents[1]
SKILL = SKILL_DIR / "SKILL.md"
GLOSSARY = SKILL_DIR / "references" / "glossary.md"

# FR-25, written out literally (docs/requirements/work-item-numbering.md): child -> allowed parents.
# "project" = no parent; "I>=03" = an initiative numbered 3 or more.
FR25 = {
    ("I",): {"project"},
    ("F",): {"project", "I>=03"},
    ("S",): {"F"},
    ("T",): {"S", "F", "D", "K"},
    ("D",): {"I01"},
    ("K",): {"I02"},
    ("FR", "AC"): {"F"},
}
CANDIDATES = [  # (token-matching key, parent_type, parent_num)
    ("project", None, None), ("I01", "I", 1), ("I02", "I", 2), ("I>=03", "I", 3),
    ("I>=03", "I", 10), ("F", "F", 1), ("S", "S", 1), ("T", "T", 1), ("D", "D", 1),
    ("K", "K", 1), ("FR", "FR", 1), ("AC", "AC", 1),
]


def _frontmatter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, "no frontmatter"
    fm, key = {}, None
    for line in m.group(1).splitlines():
        top = re.match(r"([a-z-]+):\s*(.*)", line)
        if top and not line.startswith(" "):
            key = top.group(1)
            fm[key] = top.group(2).strip()
            if fm[key] in (">-", ">", "|"):
                fm[key] = ""
        elif key:
            fm[key] = (fm[key] + " " + line.strip()).strip()
    return fm


def _skill_table(text):
    sec = text.split("## Allowed parents", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 2 and not set(cells[0]) <= set("-: ") \
                and cells[0] != "Child":
            rows[tuple(x.strip() for x in cells[0].split(","))] = \
                {x.strip() for x in cells[1].split(",")}
    return rows


def test_frontmatter_valid():
    fm = _frontmatter(SKILL.read_text())
    assert fm["name"] == SKILL_DIR.name == "work-items"
    assert 0 < len(fm["description"]) <= 1024


def test_glossary_defines_each_term_once():
    text = GLOSSARY.read_text()
    for term in ["initiative", "feature", "story", "task", "requirement",
                 "acceptance criterion", "defect", "kaizen"]:
        defs = re.findall(rf"^- \*\*{re.escape(term)}\*\*", text, re.M)
        assert len(defs) == 1, f"{term}: {len(defs)} definitions"


def test_glossary_epic_is_alias_of_feature():
    defs = re.findall(r"^- \*\*epic\*\*.*$", GLOSSARY.read_text(), re.M)
    assert len(defs) == 1
    assert re.search(r"alias of \*\*feature\*\*", defs[0])


def test_skill_table_equals_literal_fr25():
    assert _skill_table(SKILL.read_text()) == FR25


@pytest.mark.parametrize("children", list(FR25))
def test_fr25_literal_agrees_with_rules(children):
    for child in children:
        for token, ptype, pnum in CANDIDATES:
            allowed = token in FR25[children]
            try:
                rules.validate_parent(child, ptype, pnum)
                got = True
            except rules.IllegalParent:
                got = False
            assert got == allowed, f"{child} under {token}/{ptype}{pnum}: rules={got}"


def test_every_subcommand_named_in_skill_exists():
    sub = next(a for a in build_parser()._actions if a.dest == "cmd").choices
    named = set(re.findall(r"`wi ([a-z][a-z-]*)", SKILL.read_text()))
    assert named, "no subcommands found in SKILL.md"
    missing = named - set(sub)
    assert not missing, f"SKILL.md names unknown subcommands: {missing}"
    assert {"new", "resolve", "move", "promote", "flush", "check", "key"} <= named
