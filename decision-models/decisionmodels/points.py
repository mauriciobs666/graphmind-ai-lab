"""The three decision points' definitions, read from `points/<role>/` (plan
`docs/plans/typed-decision-models4.md` §4.1 U5.3, §3.2; method note
`docs/plans/typed-decision-models-ml.md` §Q1.1–§Q1.3).

Each point directory holds data, not code, so a reviewer can read and change it without touching
this package:

- `point.json` — the answer kind and `expected` vocabulary, the input-field schema, the seed
  bindings (which item field a seed axis fixes), the seed design that `decisionmodels.seeds`
  draws from, the table cells and sizes of note §Q1.3, and the generated-text screen's token
  lists (note §Q1.2 "Hits");
- `rubric.md` — the labeller rubric, the body of a label request's line 2 (plan4 §3.2);
- `generator.md` — the generator request template, the body of a generation request's line 2.

`points/common/` holds the answer-file section appended to every rubric and template, so the
request line 2 a subagent reads is self-contained (plan4 §3.4: "follow the instructions on line
2"). `Point.rubric` and `Point.generator_instructions` are those composed texts.

The input-field schema is a small closed language: `string` (`minLength`), `enum` (`values`),
`array` (`items`, `maxItems`) and `object` (`fields`, every field required, no other key;
`allowEmpty` admits `{}`, `nullable` admits `null`). `input_field_errors` checks an item against
it; `seed_binding_errors` checks an item against the seed it was generated from.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The `points/` data directory, beside this package in the component.
POINTS_DIR = Path(__file__).resolve().parents[1] / "points"

#: The three points, by model-bench role, in note §Q1.3's spend order (guard, message type,
#: join-in).
ROLES: tuple[str, ...] = ("guard-judge", "message-type-classifier", "join-in-judge")

_SECTION_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Point:
    """One decision point's definition (module docstring). `definition` is `point.json` as read."""

    role: str
    definition: Mapping[str, Any]
    rubric_body: str
    generator_body: str
    label_answer_section: str
    generate_answer_section: str

    @property
    def kind(self) -> str:
        return self.definition["kind"]

    @property
    def expected(self) -> tuple[Any, ...]:
        """The `expected` vocabulary: the whole answer space, in definition order."""
        return tuple(self.definition["expected"])

    @property
    def input_fields(self) -> tuple[str, ...]:
        """The item's input field names, in definition order."""
        return tuple(self.definition["inputFields"])

    @property
    def design(self) -> Mapping[str, Any]:
        return self.definition["design"]

    @property
    def rubric(self) -> str:
        """A label request's line-2 text: the rubric, then the answer-file section."""
        return self.rubric_body.rstrip() + _SECTION_SEPARATOR + self.label_answer_section

    @property
    def generator_instructions(self) -> str:
        """A generation request's line-2 text: the template, then the answer-file section."""
        return self.generator_body.rstrip() + _SECTION_SEPARATOR + self.generate_answer_section


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@functools.cache
def load_point(role: str) -> Point:
    """The point for a model-bench `role`. An unknown role is a `ValueError`."""
    if role not in ROLES:
        raise ValueError(f"no decision point is defined for role {role!r}")
    directory = POINTS_DIR / role
    definition = json.loads(_read_text(directory / "point.json"))
    if definition.get("role") != role:
        raise ValueError(f"{directory / 'point.json'} declares role {definition.get('role')!r}")
    return Point(
        role=role,
        definition=definition,
        rubric_body=_read_text(directory / "rubric.md"),
        generator_body=_read_text(directory / "generator.md"),
        label_answer_section=_read_text(POINTS_DIR / "common" / "label-answer.md"),
        generate_answer_section=_read_text(POINTS_DIR / "common" / "generate-answer.md"),
    )


# --- the input-field schema ----------------------------------------------------------------------


def _value_errors(schema: Mapping[str, Any], value: Any, where: str) -> list[str]:
    kind = schema["type"]
    if value is None:
        return [] if schema.get("nullable") else [f"{where}: null is not allowed"]
    if kind == "string":
        if not isinstance(value, str):
            return [f"{where}: expected a string"]
        if len(value) < schema.get("minLength", 0):
            return [f"{where}: shorter than {schema['minLength']} characters"]
        return []
    if kind == "enum":
        allowed = schema["values"]
        if isinstance(value, bool) or value not in allowed:
            return [f"{where}: {value!r} is not one of {allowed!r}"]
        return []
    if kind == "array":
        if not isinstance(value, list):
            return [f"{where}: expected a list"]
        errors: list[str] = []
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{where}: more than {schema['maxItems']} entries")
        for index, entry in enumerate(value):
            errors += _value_errors(schema["items"], entry, f"{where}[{index}]")
        return errors
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{where}: expected an object"]
        if not value and schema.get("allowEmpty"):
            return []
        return _object_errors(schema["fields"], value, where)
    raise ValueError(f"unknown schema type {kind!r} at {where}")


def _object_errors(fields: Mapping[str, Any], value: Mapping[str, Any], where: str) -> list[str]:
    errors = [f"{where}: unexpected field {name!r}" for name in value if name not in fields]
    for name, schema in fields.items():
        if name not in value:
            errors.append(f"{where}: missing field {name!r}")
        else:
            errors += _value_errors(schema, value[name], f"{where}.{name}")
    return errors


def input_field_errors(point: Point, item: Any) -> list[str]:
    """Every way `item` departs from the point's input-field schema: exactly the input fields,
    each of its declared shape. Empty means the item conforms."""
    if not isinstance(item, dict):
        return ["item: expected an object"]
    return _object_errors(point.definition["inputFields"], item, "item")


# --- seed bindings -------------------------------------------------------------------------------


def _is_present(value: Any) -> bool:
    return value is not None and value != {} and value != []


@dataclass(frozen=True)
class BindingError:
    """One seed-binding failure: the `point.json` binding that failed, and what was wrong. U5.8's
    ingester classifies on `binding` (plan5 §3.5: a `missingForm` binding is `formMismatch`, any
    other `bindingMismatch`), never on the message text."""

    binding: Mapping[str, Any]
    message: str

    @property
    def axis(self) -> str:
        """The seed axis the failed binding reads."""
        binding = self.binding
        for key in ("equalsAxis", "lengthAxis", "emptinessAxis"):
            if key in binding:
                return binding[key]
        return binding["presentWhen"]["axis"]


def seed_binding_errors(
    point: Point, item: Mapping[str, Any], axes: Mapping[str, Any]
) -> list[BindingError]:
    """Every way `item` departs from what its seed's `axes` fix (`point.json` `seedBindings`),
    each with the binding that failed: `equalsAxis` (the field equals the axis value),
    `lengthAxis` (the list has that many entries), `presentWhen` (the field is non-empty exactly
    when the named axis has the named value) and `emptinessAxis` (a member list is empty or not as
    the axis value requires)."""
    errors: list[BindingError] = []
    for binding in point.definition["seedBindings"]:
        messages = _binding_messages(binding, item, axes)
        errors += [BindingError(binding, message) for message in messages]
    return errors


def _binding_messages(
    binding: Mapping[str, Any], item: Mapping[str, Any], axes: Mapping[str, Any]
) -> list[str]:
    name = binding["field"]
    value = item.get(name)
    if "equalsAxis" in binding:
        if value != axes[binding["equalsAxis"]]:
            return [f"item.{name}: does not equal the seed's {binding['equalsAxis']}"]
        return []
    if "lengthAxis" in binding:
        wanted = axes[binding["lengthAxis"]]
        if wanted is not None and (not isinstance(value, list) or len(value) != wanted):
            return [f"item.{name}: does not hold the seed's {binding['lengthAxis']}"]
        return []
    if "emptinessAxis" in binding:
        return _emptiness_errors(binding, value, axes)
    if "presentWhen" in binding:
        rule = binding["presentWhen"]
        should = axes[rule["axis"]] == rule["value"]
        if _is_present(value) != should:
            state = "empty" if should else "present"
            return [f"item.{name}: {state} although the seed's {rule['axis']} is "
                    f"{axes[rule['axis']]!r}"]
        return []
    raise ValueError(f"unknown seed binding {binding!r}")


def _emptiness_errors(
    binding: Mapping[str, Any], value: Any, axes: Mapping[str, Any]
) -> list[str]:
    """`emptinessAxis`: the list at `field.member` is empty when the axis value is in `emptyFor`
    and non-empty when it is in `nonEmptyFor` (note §Q1.1 "Obedience is checked per item"); a
    seed whose axis is `None` is not bound. A `field` that is not a non-empty mapping is silent
    here: the field's `presentWhen` binding owns that case, so a wrong-kind item is never also a
    form error (Pass 4 P4-F1)."""
    form = axes[binding["emptinessAxis"]]
    if form is None:
        return []
    where = f"item.{binding['field']}.{binding['member']}"
    if form not in binding["emptyFor"] and form not in binding["nonEmptyFor"]:
        raise ValueError(f"{where}: no emptiness rule for {form!r}")
    if not isinstance(value, Mapping) or not value:
        return []
    member = value.get(binding["member"])
    if not isinstance(member, list):
        return [f"{where}: expected a list for the seed's {binding['emptinessAxis']} {form!r}"]
    if form in binding["emptyFor"] and member:
        return [f"{where}: not empty although the seed's {binding['emptinessAxis']} is {form!r}"]
    if form in binding["nonEmptyFor"] and not member:
        return [f"{where}: empty although the seed's {binding['emptinessAxis']} is {form!r}"]
    return []


def screen_tokens(point: Point) -> Mapping[str, Sequence[str]]:
    """The generated-text screen's token lists (note §Q1.2 "Hits"): `anywhere` and
    `metaPosition`. The minted-id form is `decisionmodels.seeds.ID_PATTERN`."""
    return point.definition["screen"]
