"""Grammar and legality of work-item IDs (FR-2, FR-3, FR-10, FR-21, FR-25).

An ID is ``<KEY>-<segment>(-<segment>)*``.  Segment letters are
``I F S T D K`` (initiative, feature, story, task, defect, kaizen) and
``FR`` / ``AC`` (requirement / acceptance criterion, path descendants of a
feature).  Numbers are zero-padded to a minimum width (I/S/T 2, F/D/K 4) and
grow on overflow; FR/AC are unpadded and may carry one lowercase letter
suffix (``FR2a``).  Parsing is canonical: ``format_id(*parse(x)) == x``.
"""
import re
from collections import namedtuple

PROJECT_RE = re.compile(r"[A-Z]{2,6}")
_PLAIN_RE = re.compile(r"([IFSTDK])(\d+)")
_REQ_RE = re.compile(r"(FR|AC)(\d+)([a-z]?)")

WIDTH = {"I": 2, "S": 2, "T": 2, "F": 4, "D": 4, "K": 4, "FR": 1, "AC": 1}
LETTER_SUFFIX_TYPES = frozenset({"FR", "AC"})
LOCAL_TYPES_IDS = frozenset({"S", "T"})
# I01 is the Defects initiative, I02 the Kaizen initiative (FR-14).
DEFECTS_INITIATIVE = 1
KAIZEN_INITIATIVE = 2


class MalformedId(ValueError):
    """The string does not satisfy the ID grammar."""


class IllegalParent(ValueError):
    """The parent/child pair is not allowed by FR-25."""


Segment = namedtuple("Segment", "type num suffix", defaults=("",))
ParsedId = namedtuple("ParsedId", "project segments")


def width(type_):
    return WIDTH[type_]


def format_segment(seg):
    if seg.type not in WIDTH:
        raise MalformedId(f"unknown segment type {seg.type!r}")
    if seg.num < 1:
        raise MalformedId(f"segment number must be >= 1: {seg!r}")
    if seg.suffix and seg.type not in LETTER_SUFFIX_TYPES:
        raise MalformedId(f"only FR/AC take a letter suffix: {seg!r}")
    return f"{seg.type}{str(seg.num).zfill(WIDTH[seg.type])}{seg.suffix}"


def format_id(project, segments):
    if not PROJECT_RE.fullmatch(project):
        raise MalformedId(f"bad project key {project!r}")
    return "-".join([project, *(format_segment(s) for s in segments)])


def _parse_segment(token, whole):
    m = _REQ_RE.fullmatch(token)
    if m:
        seg = Segment(m.group(1), int(m.group(2)), m.group(3))
    else:
        m = _PLAIN_RE.fullmatch(token)
        if not m:
            raise MalformedId(f"bad segment {token!r} in {whole!r}")
        seg = Segment(m.group(1), int(m.group(2)), "")
    if seg.num < 1 or format_segment(seg) != token:  # width / leading zeros
        raise MalformedId(f"non-canonical segment {token!r} in {whole!r}")
    return seg


def parse(text):
    """Parse an ID into ``ParsedId(project, segments)``; grammar only."""
    if not isinstance(text, str):
        raise MalformedId(f"not a string: {text!r}")
    tokens = text.split("-")
    if len(tokens) < 2 or not PROJECT_RE.fullmatch(tokens[0]):
        raise MalformedId(f"bad ID {text!r}")
    return ParsedId(tokens[0], tuple(_parse_segment(t, text) for t in tokens[1:]))


def validate_parent(child_type, parent_type=None, parent_num=None):
    """Raise IllegalParent unless FR-25 allows ``child_type`` under the parent.

    ``parent_type=None`` means the project itself.  ``parent_num`` matters for
    initiatives (I01 = Defects, I02 = Kaizen, I>=03 = ordinary).
    """
    if child_type == "I":
        ok = parent_type is None
    elif child_type == "F":
        ok = parent_type is None or (
            parent_type == "I" and parent_num is not None and parent_num >= 3)
    elif child_type == "S":
        ok = parent_type == "F"
    elif child_type == "T":
        ok = parent_type in ("S", "F", "D", "K")
    elif child_type == "D":
        ok = parent_type == "I" and parent_num == DEFECTS_INITIATIVE
    elif child_type == "K":
        ok = parent_type == "I" and parent_num == KAIZEN_INITIATIVE
    elif child_type in ("FR", "AC"):
        ok = parent_type == "F"
    else:
        raise IllegalParent(f"unknown type {child_type!r}")
    if not ok:
        where = "the project" if parent_type is None else f"{parent_type}{parent_num or ''}"
        raise IllegalParent(f"{child_type} may not have {where} as parent")


def validate_path(parsed):
    """Check every segment of a parsed ID against the segment before it."""
    prev = None
    for seg in parsed.segments:
        if prev is None:
            validate_parent(seg.type)
        else:
            validate_parent(seg.type, prev.type, prev.num)
        prev = seg
