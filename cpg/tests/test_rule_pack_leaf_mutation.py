"""P3-1: every scalar leaf of the three real packs, mutated in the spec text, must turn the spec and validator tests red.

The spec gate found that a pack value a walkthrough sentence depends on (``include_router``, ``["fastapi"]``,
``{"mcp.server.fastmcp", "fastmcp"}``) could change in the YAML block while every spec test stayed green, because the tests
restate the same literal or never read it. This file closes that class mechanically instead of by a longer hand list:

* ``leaves()`` enumerates every scalar leaf of ``fastapi``, ``fetch`` and ``fastmcp`` with its offset in the spec text.
* ``mutate()`` rewrites one leaf in a COPY of the spec (strings gain a ``Z``, integers gain 1, booleans flip), in a scratch
  repository whose ``docs/`` and ``cpg/`` link back to this one (the test file is a real copy, so ``ROOT`` resolves there),
  and a subprocess runs ``GATE`` (the spec tests and the validator tests) against it. Nothing in the repository is edited.
* ``KILLED`` and ``UNEXERCISED_UNTIL_P1`` classify every leaf. The fast test pins ``leaves == KILLED | UNEXERCISED``, so a new
  leaf fails closed until the slow run (``CCG_LEAF_MUTATION=1 pytest tests/test_rule_pack_leaf_mutation.py``, about a minute on
  8 cores) classifies it; the slow test asserts each leaf behaves as classified.

The ``UNEXERCISED_UNTIL_P1`` leaves are the agreed list left to the P1 pack-runtime conformance test (spec 12.1: run ``fastapi``
and ``fetch`` over ``s14a`` and compare to ``assemble.py``, plus the fixtures it names): they are values that only a runtime
reading the pack can test, and the spec test has no evaluator for them.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

import test_rule_pack_spec as spec

CPG = Path(__file__).resolve().parents[1]
REPO = CPG.parent
REAL = ("fastapi", "fetch", "fastmcp")


def leaf_nodes(node, path=()):
    """(path, ScalarNode) for every scalar value of a composed YAML tree; keys are structure, not leaves."""
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            yield from leaf_nodes(v, path + (k.value,))
    elif isinstance(node, yaml.SequenceNode):
        for i, v in enumerate(node.value):
            yield from leaf_nodes(v, path + (i,))
    else:
        yield path, node


def fmt(path) -> str:
    return "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path).lstrip(".")


def leaves() -> dict[str, tuple[int, int, str, str]]:
    """leaf path -> (start, end, tag, raw) with offsets into the spec text, for the three real packs."""
    out = {}
    for _line, body in spec.yaml_blocks():
        data = yaml.safe_load(body)
        if not (isinstance(data, dict) and data.get("id") in REAL and "packFormat" in data):
            continue
        base = spec.TEXT.index(body)
        for path, node in leaf_nodes(yaml.compose(body)):
            out[f"{data['id']}:{fmt(path)}"] = (base + node.start_mark.index, base + node.end_mark.index, node.tag, node.style)
    return out


def mutate(text: str, start: int, end: int, tag: str, style) -> str:
    raw = text[start:end]
    if style in ("'", '"'):
        new = raw[:-1] + "Z" + raw[-1]
    elif tag.endswith(":int"):
        new = str(int(raw) + 1)
    elif tag.endswith(":bool"):
        new = "false" if raw.lower() == "true" else "true"
    else:
        new = raw + "Z"
    return text[:start] + new + text[end:]


def scratch_root(parent: Path, name: str) -> Path:
    """A repository whose docs/ and cpg/ entries are links, except the spec file and tests/ (real copies)."""
    root = parent / name
    for sub in ("docs", "cpg"):
        (root / sub).mkdir(parents=True)
        for entry in (REPO / sub).iterdir():
            if (sub, entry.name) in (("docs", "plans"), ("cpg", "tests")):
                continue
            (root / sub / entry.name).symlink_to(entry)
    (root / "docs/plans").mkdir()
    for entry in (REPO / "docs/plans").iterdir():
        if entry != spec.SPEC:
            (root / "docs/plans" / entry.name).symlink_to(entry)
    shutil.copytree(CPG / "tests", root / "cpg/tests", ignore=shutil.ignore_patterns("__pycache__"))
    return root


# The tests a mutated spec must turn red: the spec's own (block against prose) and the shipped validator's (kinds, enums,
# references, the registry). Not run: the pack-file equality tests (a mutated block differs from the pack file by construction)
# and the generated probes (slow, and about the schema's closure, which a one-leaf edit cannot change).
GATE = ["tests/test_rule_pack_spec.py", "tests/test_rule_pack_schema.py", "-k",
        "not pack_file and not planted and not scalar_leaf and not every_object and not differential"]


def survives(root: Path, mutated: str) -> bool:
    """Write the mutated spec into the scratch repository and run the spec test there; True when it still passes."""
    (root / "docs/plans" / spec.SPEC.name).write_text(mutated)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *GATE],
                       cwd=root / "cpg", capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 1, 2), r.stdout[-400:] + r.stderr[-400:]
    return r.returncode == 0


def run_all(workers: int = 6) -> dict[str, bool]:
    found = leaves()
    result: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        roots = [scratch_root(Path(tmp), f"w{i}") for i in range(workers)]
        jobs = list(found.items())

        def work(i):
            for key, (s, e, tag, style) in jobs[i::workers]:
                result[key] = survives(roots[i], mutate(spec.TEXT, s, e, tag, style))

        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, range(workers)))
    return result


# Classification from the slow run. Regenerate by running the slow test and moving leaves.
# UNEXERCISED_UNTIL_P1: values only a runtime reading the pack can test. The spec 12.1 P1 pack-runtime conformance test covers
# them: the fastapi: and fetch: ones by the fastapi/fetch rows over s14a against assemble.py (plus its named fixtures), the
# fastmcp: ones (import gates, callee names, the tool-call key, launch defaults and flags, URL-source patterns) by the
# `fastmcp` row, whose launch/config extractor is first exercised there. Each must flip a result when mutated.
UNEXERCISED_UNTIL_P1: set[str] = {
    "fastapi:mounts[0].call.callee",
    "fastapi:mounts[0].prefix.arg",
    "fastapi:mounts[0].prefix.default",
    "fastapi:mounts[0].recordConditions",
    "fastapi:mounts[0].scope.imports[0].module.prefix",
    "fastapi:rules[0].scope.imports[0].module.prefix",
    "fastapi:rules[1].scope.imports[0].module.prefix",
    "fastmcp:launch.frameworkDefaults[0].command.basename",
    "fastmcp:launch.frameworkDefaults[0].port",
    "fastmcp:launch.frameworkDefaults[0].protocol",
    "fastmcp:launch.mountPaths[0].call.callee",
    "fastmcp:launch.portFlags[0].command.basename",
    "fastmcp:launch.portFlags[0].flags[0]",
    "fastmcp:launch.targetArg[0].index",
    "fastmcp:rules[0].scope.imports[0].module.prefix",
    "fastmcp:rules[0].scope.imports[1].module.prefix",
    "fastmcp:rules[1].emit[0].consumes.key",
    "fastmcp:rules[1].match.callsite.callee",
    "fastmcp:rules[1].scope.imports[0].module.prefix",
    "fastmcp:urlBindings[0].urlSources[0].config.keyPath.regex",
    "fastmcp:urlBindings[0].urlSources[1].callsite.callee",
    "fetch:liftings[0].companions[0].arg",
    "fetch:liftings[0].seeds[0].call.callee",
    "fetch:rules[0].emit[0].consumes.key",
}

KILLED: set[str] = {
    "fastapi:id",
    "fastapi:kind",
    "fastapi:link[0].kind",
    "fastapi:link[0].matcher",
    "fastapi:mounts[0].child.arg",
    "fastapi:mounts[0].child.resolves",
    "fastapi:mounts[0].constructorPrefix[0].arg",
    "fastapi:mounts[0].constructorPrefix[0].call.callee",
    "fastapi:mounts[0].id",
    "fastapi:mounts[0].roots[0].call.callee",
    "fastapi:mounts[0].roots[0].how",
    "fastapi:mounts[0].roots[0].notWith[0].callee",
    "fastapi:mounts[0].unmounted.reason",
    "fastapi:mounts[0].unmounted.status",
    "fastapi:packFormat",
    "fastapi:rules[0].emit[0].provides.key",
    "fastapi:rules[0].emit[0].provides.kind",
    "fastapi:rules[0].extract.route.firstOf[0].arg",
    "fastapi:rules[0].extract.route.firstOf[1].arg",
    "fastapi:rules[0].extract.verb.group",
    "fastapi:rules[0].extract.verb.upper",
    "fastapi:rules[0].id",
    "fastapi:rules[0].match.decorator.form",
    "fastapi:rules[0].match.decorator.name.regex",
    "fastapi:rules[1].emit[0].provides.key",
    "fastapi:rules[1].emit[0].provides.kind",
    "fastapi:rules[1].extract.route.firstOf[0].arg",
    "fastapi:rules[1].extract.route.firstOf[1].arg",
    "fastapi:rules[1].extract.verb.literal",
    "fastapi:rules[1].id",
    "fastapi:rules[1].match.decorator.form",
    "fastapi:rules[1].match.decorator.name.regex",
    "fastapi:version",
    "fastapi:vocabVersion",
    "fastmcp:id",
    "fastmcp:kind",
    "fastmcp:launch.envPorts[0].adoptedBy",
    "fastmcp:launch.envPorts[0].envVar.regex",
    "fastmcp:launch.envPorts[0].excludeVars[0].regex",
    "fastmcp:launch.mountPaths[0].app.arg",
    "fastmcp:launch.mountPaths[0].app.fromCall.externalTarget.glob",
    "fastmcp:launch.mountPaths[0].path.arg",
    "fastmcp:launch.mountPaths[0].pathMatch",
    "fastmcp:launch.targetArg[0].form",
    "fastmcp:link[0].kind",
    "fastmcp:link[0].matcher",
    "fastmcp:packFormat",
    "fastmcp:rules[0].emit[0].provides.key",
    "fastmcp:rules[0].emit[0].provides.kind",
    "fastmcp:rules[0].extract.tool.firstOf[0].arg",
    "fastmcp:rules[0].extract.tool.firstOf[1].arg",
    "fastmcp:rules[0].extract.tool.firstOf[2].target",
    "fastmcp:rules[0].id",
    "fastmcp:rules[0].match.decorator.form",
    "fastmcp:rules[0].match.decorator.name.regex",
    "fastmcp:rules[1].emit[0].consumes.kind",
    "fastmcp:rules[1].extract.tool.arg",
    "fastmcp:rules[1].id",
    "fastmcp:urlBindings[0].id",
    "fastmcp:urlBindings[0].kinds[0]",
    "fastmcp:urlBindings[0].urlSources[1].arg",
    "fastmcp:urlBindings[0].urlSources[1].scope",
    "fastmcp:version",
    "fastmcp:vocabVersion",
    "fetch:id",
    "fetch:kind",
    "fetch:liftings[0].companions[0].bareDefault",
    "fetch:liftings[0].companions[0].role",
    "fetch:liftings[0].companions[0].unknown",
    "fetch:liftings[0].id",
    "fetch:liftings[0].role.kind",
    "fetch:liftings[0].role.slot",
    "fetch:liftings[0].seeds[0].arg",
    "fetch:liftings[0].seeds[0].call.resolution[0]",
    "fetch:liftings[0].seeds[0].call.resolution[1]",
    "fetch:link[0].kind",
    "fetch:link[0].matcher",
    "fetch:packFormat",
    "fetch:rules[0].emit[0].consumes.kind",
    "fetch:rules[0].extract.url.lifted",
    "fetch:rules[0].extract.verb.lifted",
    "fetch:rules[0].id",
    "fetch:rules[0].match.lifted.lifting",
    "fetch:version",
    "fetch:vocabVersion",
}


def test_the_mutation_harness_can_tell_a_killed_leaf_from_a_survivor(tmp_path):
    """The harness is not vacuous: an edit the spec test depends on dies, an edit to prose-free text survives."""
    root = scratch_root(tmp_path, "w")
    found = leaves()
    s, e, tag, style = found["fastapi:id"]
    assert not survives(root, mutate(spec.TEXT, s, e, tag, style))
    assert survives(root, spec.TEXT)  # the unmutated copy passes: the failure above is the edit's, not the scratch repo's


def test_every_leaf_of_the_three_real_packs_is_classified():
    found = set(leaves())
    assert len(found) > 100
    assert found == KILLED | UNEXERCISED_UNTIL_P1, sorted(found ^ (KILLED | UNEXERCISED_UNTIL_P1))
    assert not KILLED & UNEXERCISED_UNTIL_P1


@pytest.mark.skipif(not os.environ.get("CCG_LEAF_MUTATION"), reason="slow: set CCG_LEAF_MUTATION=1 (about a minute on 8 cores)")
def test_each_leaf_behaves_as_classified():
    result = run_all()
    survivors = {k for k, alive in result.items() if alive}
    assert survivors == UNEXERCISED_UNTIL_P1, (sorted(survivors - UNEXERCISED_UNTIL_P1), sorted(UNEXERCISED_UNTIL_P1 - survivors))
