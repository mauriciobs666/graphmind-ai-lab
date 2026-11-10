"""ID grammar (parse/format, widths, FR/AC suffixes) and FR-25 parent legality."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wilib import rules  # noqa: E402
from wilib.rules import IllegalParent, MalformedId  # noqa: E402


def roundtrip(text):
    p = rules.parse(text)
    return rules.format_id(p.project, p.segments)


@pytest.mark.parametrize("text", [
    "GAL-I03", "GAL-I03-F0042", "GAL-F0042", "GAL-I03-F0042-S01", "GAL-F0042-S100",
    "GAL-I03-F0042-S01-T01", "GAL-F0042-T01", "GAL-I01-D0015", "GAL-I02-K0015",
    "GAL-I01-D0015-T02", "GAL-F10000", "GAL-F0042-S100-T100",
    "GAL-F0042-FR3", "GAL-I03-F0042-AC21a", "GAL-F0042-FR2a", "GAL-F0042-AC12",
])
def test_valid_ids_roundtrip_unchanged(text):
    assert roundtrip(text) == text


def test_parse_exposes_project_and_segments():
    p = rules.parse("GAL-I03-F0042-S01")
    assert p.project == "GAL"
    assert [(s.type, s.num, s.suffix) for s in p.segments] == [
        ("I", 3, ""), ("F", 42, ""), ("S", 1, "")]


def test_parse_keeps_letter_suffix_on_fr_and_ac():
    p = rules.parse("GAL-F0042-AC21a")
    last = p.segments[-1]
    assert (last.type, last.num, last.suffix) == ("AC", 21, "a")


def test_widths_grow_on_overflow_without_renaming():
    assert rules.format_id("GAL", [rules.Segment("S", 99), rules.Segment("T", 1)]) == "GAL-S99-T01"
    assert rules.format_id("GAL", [rules.Segment("F", 1)]) == "GAL-F0001"
    assert rules.format_id("GAL", [rules.Segment("F", 10000)]) == "GAL-F10000"
    assert rules.format_id("GAL", [rules.Segment("S", 100)]) == "GAL-S100"
    assert rules.format_id("GAL", [rules.Segment("I", 3)]) == "GAL-I03"
    assert rules.format_id("GAL", [rules.Segment("FR", 2, "a")]) == "GAL-FR2a"
    assert rules.format_id("GAL", [rules.Segment("AC", 7)]) == "GAL-AC7"


@pytest.mark.parametrize("text", [
    "", "GAL", "GAL-", "gal-F0042", "G-F0042", "GALLERYX-F0042",
    "GAL-F042",          # below minimum width
    "GAL-S1",            # below minimum width
    "GAL-I3",
    "GAL-F00042",        # non-canonical leading zero
    "GAL-F0000",         # zero is never issued
    "GAL-X0042", "GAL-F0042-", "GAL-F0042--S01", "GAL-F0042-s01",
    "GAL-F0042-FR02",    # FR/AC are unpadded
    "GAL-F0042-FR0",
    "GAL-F0042-FRa",     # number required
    "GAL-F0042-FR2ab",   # single-letter suffix only
    "GAL-F0042-FR2A",    # lowercase suffix only
    "GAL-F0042-S01a",    # suffix only on FR/AC
    "GAL-F0042 ", " GAL-F0042", "GAL-F0042-S01\n",
])
def test_malformed_ids_refused(text):
    with pytest.raises(MalformedId):
        rules.parse(text)


# --- FR-25, written out literally from the requirement's text -------------
# "an initiative has the project; a feature has an initiative from I03 up, or
#  none, and never I01 or I02; a story has a feature only; a task has a story,
#  a feature, a defect or a kaizen item; a defect has I01 only; a kaizen item
#  has I02 only; FR and AC have a feature. Any other parent is refused."
# parent spec: None = the project itself, else (type, num).
ACCEPTED = [
    ("I", None),
    ("F", None), ("F", ("I", 3)), ("F", ("I", 4)), ("F", ("I", 99)),
    ("S", ("F", 42)),
    ("T", ("S", 1)), ("T", ("F", 42)), ("T", ("D", 15)), ("T", ("K", 15)),
    ("D", ("I", 1)),
    ("K", ("I", 2)),
    ("FR", ("F", 42)), ("AC", ("F", 42)),
]
REFUSED = [
    ("I", ("I", 3)), ("I", ("F", 42)), ("I", ("S", 1)), ("I", ("T", 1)),
    ("I", ("D", 15)), ("I", ("K", 15)), ("I", ("FR", 1)), ("I", ("AC", 1)),
    ("F", ("FR", 1)), ("F", ("AC", 1)), ("S", ("FR", 1)), ("S", ("AC", 1)),
    ("T", ("FR", 1)), ("T", ("AC", 1)),
    ("D", ("T", 1)), ("D", ("K", 15)), ("D", ("FR", 1)), ("D", ("AC", 1)),
    ("K", ("S", 1)), ("K", ("T", 1)), ("K", ("D", 15)), ("K", ("FR", 1)),
    ("K", ("AC", 1)), ("FR", ("AC", 1)), ("AC", ("FR", 1)),
    ("F", ("I", 1)), ("F", ("I", 2)), ("F", ("F", 42)), ("F", ("S", 1)),
    ("F", ("T", 1)), ("F", ("D", 15)), ("F", ("K", 15)),
    ("S", None), ("S", ("I", 3)), ("S", ("S", 1)), ("S", ("T", 1)),
    ("S", ("D", 15)), ("S", ("K", 15)),
    ("T", None), ("T", ("I", 3)), ("T", ("T", 1)),
    ("D", None), ("D", ("I", 2)), ("D", ("I", 3)), ("D", ("F", 42)),
    ("D", ("S", 1)), ("D", ("D", 15)),
    ("K", None), ("K", ("I", 1)), ("K", ("I", 3)), ("K", ("F", 42)),
    ("K", ("K", 15)),
    ("FR", None), ("FR", ("I", 3)), ("FR", ("S", 1)), ("FR", ("T", 1)),
    ("FR", ("D", 15)), ("FR", ("K", 15)), ("FR", ("FR", 1)),
    ("AC", None), ("AC", ("I", 3)), ("AC", ("S", 1)), ("AC", ("T", 1)),
    ("AC", ("D", 15)), ("AC", ("K", 15)), ("AC", ("AC", 1)),
]


def _call(child, parent):
    ptype, pnum = (None, None) if parent is None else parent
    rules.validate_parent(child, ptype, pnum)


@pytest.mark.parametrize("child,parent", ACCEPTED)
def test_allowed_parent_pairs_accepted(child, parent):
    _call(child, parent)


@pytest.mark.parametrize("child,parent", REFUSED)
def test_other_parent_pairs_refused(child, parent):
    with pytest.raises(IllegalParent):
        _call(child, parent)


def test_matrix_covers_every_type_pair_exactly_once():
    types = ["I", "F", "S", "T", "D", "K", "FR", "AC"]
    parents = [None] + [(t, n) for t in types for n in (1, 2, 3, 15, 42)]
    seen = set(ACCEPTED) | set(REFUSED)
    assert not set(ACCEPTED) & set(REFUSED)
    # every (child, parent type) pair is decided by the literal tables
    decided = {(c, p[0] if p else None) for c, p in seen}
    for c in types:
        for p in [None] + types:
            assert (c, p) in decided, (c, p)


def test_validate_path_accepts_legal_ids():
    for text in ["GAL-I03-F0042-S01-T01", "GAL-F0042-T01", "GAL-I01-D0015-T02",
                 "GAL-I02-K0015", "GAL-F0042-AC21a", "GAL-I03-F0042-FR2"]:
        rules.validate_path(rules.parse(text))


@pytest.mark.parametrize("text", [
    "GAL-I01-F0042", "GAL-I02-F0042", "GAL-S01", "GAL-T01", "GAL-I03-D0015",
    "GAL-I02-D0015", "GAL-I01-K0015", "GAL-F0042-D0015", "GAL-I01-D0015-S01",
    "GAL-I02-K0015-S01", "GAL-I03-FR1", "GAL-F0042-FR2-AC1", "GAL-F0042-S01-FR1",
])
def test_validate_path_refuses_illegal_ids(text):
    with pytest.raises(IllegalParent):
        rules.validate_path(rules.parse(text))
