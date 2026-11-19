"""The draft of the plain-language front (C-1001 P1-26; ``docs/plans/connected-code-graph-p1-ml.md`` §4.2's ``InterpretDraft``): what a question names before any id is known.

The deterministic parsers (:mod:`ccg.front.parsers`) and the interpreter (dispatch 2) both produce one; :mod:`ccg.front.resolve` alone turns it into ids. It carries **spans**
(verbatim question text with a ``kindHint``), never an id: ``use_case`` (``impact`` or an unbuilt use-case name), ``direction`` (``impact``, ``depends-on``, or ``None`` when the
question states none), ``anchors`` (1-3 span groups; one group names one anchor, its spans qualify each other), ``scope`` (a span or ``None``), ``readings`` (alternative
``(use_case, direction)`` pairs when the question names two use cases). Pure data: no import beyond the standard library, so the interpreter and its cache can share it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BUILT_USE_CASE = "impact"
#: FR-12 to FR-15 (``docs/requirements/connected-code-graph.md``): understood, answered "not built in this phase"
UNBUILT_USE_CASES = ("root-cause", "code-review", "test-coverage", "security")
DIRECTIONS = ("impact", "depends-on")
KIND_HINTS = ("symbol", "file", "line", "route", "tool", "server", "other")
MAX_GROUPS = 3
MAX_SPANS = 3


@dataclass(frozen=True)
class Span:
    text: str
    kind_hint: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("a span's text is a non-empty string")
        if self.kind_hint not in KIND_HINTS:
            raise ValueError(f"kindHint is one of {list(KIND_HINTS)}")


@dataclass(frozen=True)
class Group:
    spans: tuple[Span, ...]

    def __post_init__(self):
        if not 1 <= len(self.spans) <= MAX_SPANS:
            raise ValueError(f"a group holds 1 to {MAX_SPANS} spans")


@dataclass(frozen=True)
class Reading:
    use_case: str
    direction: str | None


@dataclass(frozen=True)
class Draft:
    use_case: str
    direction: str | None
    anchors: tuple[Group, ...]
    scope: str | None = None
    readings: tuple[Reading, ...] = field(default=())

    def __post_init__(self):
        if self.use_case != BUILT_USE_CASE and self.use_case not in UNBUILT_USE_CASES:
            raise ValueError("useCase is impact or an unbuilt use-case name")
        if self.direction is not None and self.direction not in DIRECTIONS:
            raise ValueError(f"direction is one of {list(DIRECTIONS)} or null")
        if len(self.anchors) > MAX_GROUPS:
            raise ValueError(f"a draft names at most {MAX_GROUPS} anchor groups")
        for r in self.readings:
            if (r.use_case != BUILT_USE_CASE and r.use_case not in UNBUILT_USE_CASES) or (r.direction is not None and r.direction not in DIRECTIONS):
                raise ValueError("a reading is a use case and a direction (or null)")
