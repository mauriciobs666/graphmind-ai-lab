"""P3-1 / S32: every scalar leaf of the seven packs, mutated in the spec text, must turn the spec and validator tests red.

The spec gate found that a pack value a walkthrough sentence depends on (``include_router``, ``["fastapi"]``,
``{"mcp.server.fastmcp", "fastmcp"}``) could change in the YAML block while every spec test stayed green, because the tests
restate the same literal or never read it. This file closes that class mechanically instead of by a longer hand list:

* ``leaves()`` enumerates every scalar leaf of the seven packs with its offset in the spec text: the pack blocks of sections 2 to 6, 8.1
  and 9.1 and, for ``infra-signatures``, the section 8.2 ``signatures`` list (leaves ``infra-signatures:signatures[i]...``). The
  section 8 negatives block is test data, not pack content, and is outside the harness (a mutated negative line cannot fail anything).
* ``mutate()`` rewrites one leaf in a COPY of the spec (strings gain a ``Z``, integers gain 1, booleans flip), in a scratch
  repository whose ``docs/`` and ``cpg/`` link back to this one (the test file is a real copy, so ``ROOT`` resolves there),
  and a subprocess runs ``GATE`` (the spec tests and the validator tests) against it. Nothing in the repository is edited.
* ``KILLED`` and ``UNEXERCISED_UNTIL_P1`` classify every leaf. The fast test pins ``leaves == KILLED | UNEXERCISED``, so a new
  leaf fails closed until the slow run (``CCG_LEAF_MUTATION=1 pytest tests/test_rule_pack_leaf_mutation.py``, about five and a half minutes on
  6 workers) classifies it; the slow test asserts each leaf behaves as classified.

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
REAL = ("fastapi", "fetch", "fastmcp", "kafka", "paper-inhouse-bus", "infra-signatures", "severity-rules")


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
    """leaf path -> (start, end, tag, raw) with offsets into the spec text, for the seven packs."""
    out = {}
    for _line, body in spec.yaml_blocks():
        data = yaml.safe_load(body)
        if isinstance(data, dict) and data.get("id") in REAL and "packFormat" in data:
            prefix = f"{data['id']}:"
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "class" in data[0]:
            prefix = "infra-signatures:signatures"  # the section 8.2 list is the pack's `signatures` body
        else:
            continue
        base = spec.TEXT.index(body)
        for path, node in leaf_nodes(yaml.compose(body)):
            key = prefix + fmt(path)
            out[key] = (base + node.start_mark.index, base + node.end_mark.index, node.tag, node.style)
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
GATE = ["tests/test_rule_pack_spec.py", "tests/test_rule_pack_schema.py", "tests/test_rule_pack_topic_facts.py",
        "tests/test_rule_pack_signature_pins.py", "-k",
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
# S32: the P1 row covers only fastapi, fetch and fastmcp, so no kafka, paper-inhouse-bus, infra-signatures or severity-rules leaf is
# classified here: every one is KILLED (topic facts: test_rule_pack_topic_facts.py; signature pins: test_rule_pack_signature_pins.py
# and data/infra_probes.tsv; severity: the spec test's vectors and mutants).
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
    "infra-signatures:id",
    "infra-signatures:kind",
    "infra-signatures:packFormat",
    "infra-signatures:runtimes[0].regex",
    "infra-signatures:runtimes[0].runtime",
    "infra-signatures:runtimes[1].regex",
    "infra-signatures:runtimes[1].runtime",
    "infra-signatures:runtimes[2].regex",
    "infra-signatures:runtimes[2].runtime",
    "infra-signatures:runtimes[3].regex",
    "infra-signatures:runtimes[3].runtime",
    "infra-signatures:runtimes[4].regex",
    "infra-signatures:runtimes[4].runtime",
    "infra-signatures:signatures[0].class",
    "infra-signatures:signatures[0].id",
    "infra-signatures:signatures[0].regex",
    "infra-signatures:signatures[0].runtime",
    "infra-signatures:signatures[0].sample",
    "infra-signatures:signatures[0].sampleSource",
    "infra-signatures:signatures[10].class",
    "infra-signatures:signatures[10].id",
    "infra-signatures:signatures[10].library",
    "infra-signatures:signatures[10].regex",
    "infra-signatures:signatures[10].runtime",
    "infra-signatures:signatures[10].sample",
    "infra-signatures:signatures[11].class",
    "infra-signatures:signatures[11].id",
    "infra-signatures:signatures[11].library",
    "infra-signatures:signatures[11].regex",
    "infra-signatures:signatures[11].runtime",
    "infra-signatures:signatures[11].sample",
    "infra-signatures:signatures[12].class",
    "infra-signatures:signatures[12].id",
    "infra-signatures:signatures[12].library",
    "infra-signatures:signatures[12].regex",
    "infra-signatures:signatures[12].runtime",
    "infra-signatures:signatures[12].sample",
    "infra-signatures:signatures[12].sampleSource",
    "infra-signatures:signatures[13].class",
    "infra-signatures:signatures[13].id",
    "infra-signatures:signatures[13].regex",
    "infra-signatures:signatures[13].runtime",
    "infra-signatures:signatures[13].sample",
    "infra-signatures:signatures[14].class",
    "infra-signatures:signatures[14].id",
    "infra-signatures:signatures[14].regex",
    "infra-signatures:signatures[14].runtime",
    "infra-signatures:signatures[14].sample",
    "infra-signatures:signatures[15].class",
    "infra-signatures:signatures[15].id",
    "infra-signatures:signatures[15].regex",
    "infra-signatures:signatures[15].runtime",
    "infra-signatures:signatures[15].sample",
    "infra-signatures:signatures[16].class",
    "infra-signatures:signatures[16].id",
    "infra-signatures:signatures[16].regex",
    "infra-signatures:signatures[16].runtime",
    "infra-signatures:signatures[16].sample",
    "infra-signatures:signatures[17].class",
    "infra-signatures:signatures[17].id",
    "infra-signatures:signatures[17].regex",
    "infra-signatures:signatures[17].runtime",
    "infra-signatures:signatures[17].sample",
    "infra-signatures:signatures[18].class",
    "infra-signatures:signatures[18].id",
    "infra-signatures:signatures[18].regex",
    "infra-signatures:signatures[18].runtime",
    "infra-signatures:signatures[18].sample",
    "infra-signatures:signatures[19].class",
    "infra-signatures:signatures[19].id",
    "infra-signatures:signatures[19].regex",
    "infra-signatures:signatures[19].runtime",
    "infra-signatures:signatures[19].sample",
    "infra-signatures:signatures[19].sampleSource",
    "infra-signatures:signatures[1].class",
    "infra-signatures:signatures[1].id",
    "infra-signatures:signatures[1].regex",
    "infra-signatures:signatures[1].runtime",
    "infra-signatures:signatures[1].sample",
    "infra-signatures:signatures[1].sampleSource",
    "infra-signatures:signatures[20].class",
    "infra-signatures:signatures[20].id",
    "infra-signatures:signatures[20].regex",
    "infra-signatures:signatures[20].runtime",
    "infra-signatures:signatures[20].sample",
    "infra-signatures:signatures[20].sampleSource",
    "infra-signatures:signatures[21].class",
    "infra-signatures:signatures[21].id",
    "infra-signatures:signatures[21].regex",
    "infra-signatures:signatures[21].runtime",
    "infra-signatures:signatures[21].sample",
    "infra-signatures:signatures[21].sampleSource",
    "infra-signatures:signatures[22].class",
    "infra-signatures:signatures[22].id",
    "infra-signatures:signatures[22].regex",
    "infra-signatures:signatures[22].runtime",
    "infra-signatures:signatures[22].sample",
    "infra-signatures:signatures[22].sampleSource",
    "infra-signatures:signatures[23].class",
    "infra-signatures:signatures[23].id",
    "infra-signatures:signatures[23].regex",
    "infra-signatures:signatures[23].runtime",
    "infra-signatures:signatures[23].sample",
    "infra-signatures:signatures[24].class",
    "infra-signatures:signatures[24].id",
    "infra-signatures:signatures[24].library",
    "infra-signatures:signatures[24].regex",
    "infra-signatures:signatures[24].runtime",
    "infra-signatures:signatures[24].sample",
    "infra-signatures:signatures[25].class",
    "infra-signatures:signatures[25].id",
    "infra-signatures:signatures[25].library",
    "infra-signatures:signatures[25].regex",
    "infra-signatures:signatures[25].runtime",
    "infra-signatures:signatures[25].sample",
    "infra-signatures:signatures[26].class",
    "infra-signatures:signatures[26].id",
    "infra-signatures:signatures[26].library",
    "infra-signatures:signatures[26].regex",
    "infra-signatures:signatures[26].runtime",
    "infra-signatures:signatures[26].sample",
    "infra-signatures:signatures[27].class",
    "infra-signatures:signatures[27].id",
    "infra-signatures:signatures[27].library",
    "infra-signatures:signatures[27].regex",
    "infra-signatures:signatures[27].runtime",
    "infra-signatures:signatures[27].sample",
    "infra-signatures:signatures[28].class",
    "infra-signatures:signatures[28].id",
    "infra-signatures:signatures[28].regex",
    "infra-signatures:signatures[28].runtime",
    "infra-signatures:signatures[28].sample",
    "infra-signatures:signatures[29].class",
    "infra-signatures:signatures[29].id",
    "infra-signatures:signatures[29].regex",
    "infra-signatures:signatures[29].runtime",
    "infra-signatures:signatures[29].sample",
    "infra-signatures:signatures[2].class",
    "infra-signatures:signatures[2].id",
    "infra-signatures:signatures[2].regex",
    "infra-signatures:signatures[2].runtime",
    "infra-signatures:signatures[2].sample",
    "infra-signatures:signatures[2].sampleSource",
    "infra-signatures:signatures[30].class",
    "infra-signatures:signatures[30].id",
    "infra-signatures:signatures[30].regex",
    "infra-signatures:signatures[30].runtime",
    "infra-signatures:signatures[30].sample",
    "infra-signatures:signatures[31].class",
    "infra-signatures:signatures[31].id",
    "infra-signatures:signatures[31].regex",
    "infra-signatures:signatures[31].runtime",
    "infra-signatures:signatures[31].sample",
    "infra-signatures:signatures[32].class",
    "infra-signatures:signatures[32].id",
    "infra-signatures:signatures[32].regex",
    "infra-signatures:signatures[32].runtime",
    "infra-signatures:signatures[32].sample",
    "infra-signatures:signatures[33].class",
    "infra-signatures:signatures[33].id",
    "infra-signatures:signatures[33].regex",
    "infra-signatures:signatures[33].runtime",
    "infra-signatures:signatures[33].sample",
    "infra-signatures:signatures[34].class",
    "infra-signatures:signatures[34].id",
    "infra-signatures:signatures[34].regex",
    "infra-signatures:signatures[34].runtime",
    "infra-signatures:signatures[34].sample",
    "infra-signatures:signatures[35].class",
    "infra-signatures:signatures[35].id",
    "infra-signatures:signatures[35].regex",
    "infra-signatures:signatures[35].runtime",
    "infra-signatures:signatures[35].sample",
    "infra-signatures:signatures[36].class",
    "infra-signatures:signatures[36].id",
    "infra-signatures:signatures[36].regex",
    "infra-signatures:signatures[36].runtime",
    "infra-signatures:signatures[36].sample",
    "infra-signatures:signatures[37].class",
    "infra-signatures:signatures[37].id",
    "infra-signatures:signatures[37].regex",
    "infra-signatures:signatures[37].runtime",
    "infra-signatures:signatures[37].sample",
    "infra-signatures:signatures[38].class",
    "infra-signatures:signatures[38].id",
    "infra-signatures:signatures[38].regex",
    "infra-signatures:signatures[38].runtime",
    "infra-signatures:signatures[38].sample",
    "infra-signatures:signatures[39].class",
    "infra-signatures:signatures[39].id",
    "infra-signatures:signatures[39].regex",
    "infra-signatures:signatures[39].runtime",
    "infra-signatures:signatures[39].sample",
    "infra-signatures:signatures[3].class",
    "infra-signatures:signatures[3].id",
    "infra-signatures:signatures[3].regex",
    "infra-signatures:signatures[3].runtime",
    "infra-signatures:signatures[3].sample",
    "infra-signatures:signatures[3].sampleSource",
    "infra-signatures:signatures[40].class",
    "infra-signatures:signatures[40].id",
    "infra-signatures:signatures[40].regex",
    "infra-signatures:signatures[40].runtime",
    "infra-signatures:signatures[40].sample",
    "infra-signatures:signatures[41].class",
    "infra-signatures:signatures[41].id",
    "infra-signatures:signatures[41].regex",
    "infra-signatures:signatures[41].runtime",
    "infra-signatures:signatures[41].sample",
    "infra-signatures:signatures[42].class",
    "infra-signatures:signatures[42].id",
    "infra-signatures:signatures[42].regex",
    "infra-signatures:signatures[42].runtime",
    "infra-signatures:signatures[42].sample",
    "infra-signatures:signatures[43].class",
    "infra-signatures:signatures[43].id",
    "infra-signatures:signatures[43].regex",
    "infra-signatures:signatures[43].runtime",
    "infra-signatures:signatures[43].sample",
    "infra-signatures:signatures[44].class",
    "infra-signatures:signatures[44].id",
    "infra-signatures:signatures[44].regex",
    "infra-signatures:signatures[44].runtime",
    "infra-signatures:signatures[44].sample",
    "infra-signatures:signatures[4].class",
    "infra-signatures:signatures[4].id",
    "infra-signatures:signatures[4].regex",
    "infra-signatures:signatures[4].runtime",
    "infra-signatures:signatures[4].sample",
    "infra-signatures:signatures[4].sampleSource",
    "infra-signatures:signatures[5].class",
    "infra-signatures:signatures[5].id",
    "infra-signatures:signatures[5].library",
    "infra-signatures:signatures[5].regex",
    "infra-signatures:signatures[5].runtime",
    "infra-signatures:signatures[5].sample",
    "infra-signatures:signatures[5].sampleSource",
    "infra-signatures:signatures[6].class",
    "infra-signatures:signatures[6].id",
    "infra-signatures:signatures[6].library",
    "infra-signatures:signatures[6].regex",
    "infra-signatures:signatures[6].runtime",
    "infra-signatures:signatures[6].sample",
    "infra-signatures:signatures[7].class",
    "infra-signatures:signatures[7].id",
    "infra-signatures:signatures[7].library",
    "infra-signatures:signatures[7].regex",
    "infra-signatures:signatures[7].runtime",
    "infra-signatures:signatures[7].sample",
    "infra-signatures:signatures[8].class",
    "infra-signatures:signatures[8].id",
    "infra-signatures:signatures[8].library",
    "infra-signatures:signatures[8].regex",
    "infra-signatures:signatures[8].runtime",
    "infra-signatures:signatures[8].sample",
    "infra-signatures:signatures[9].class",
    "infra-signatures:signatures[9].id",
    "infra-signatures:signatures[9].library",
    "infra-signatures:signatures[9].regex",
    "infra-signatures:signatures[9].runtime",
    "infra-signatures:signatures[9].sample",
    "infra-signatures:version",
    "infra-signatures:vocabVersion",
    "kafka:id",
    "kafka:kind",
    "kafka:link[0].kind",
    "kafka:link[0].matcher",
    "kafka:packFormat",
    "kafka:rules[0].emit[0].provides.key",
    "kafka:rules[0].emit[0].provides.kind",
    "kafka:rules[0].extract.topic.arg",
    "kafka:rules[0].id",
    "kafka:rules[0].match.callsite.callee.regex",
    "kafka:rules[0].scope.imports[0].module.prefix",
    "kafka:rules[0].scope.imports[1].module.prefix",
    "kafka:rules[0].scope.imports[2].module.prefix",
    "kafka:rules[1].emit[0].consumes.key",
    "kafka:rules[1].emit[0].consumes.kind",
    "kafka:rules[1].extract.topic.arg",
    "kafka:rules[1].extract.topic.onUnknown",
    "kafka:rules[1].id",
    "kafka:rules[1].match.callsite.callee.regex",
    "kafka:rules[1].scope.imports[0].module.prefix",
    "kafka:rules[1].scope.imports[1].module.prefix",
    "kafka:rules[2].emit[0].consumes.key",
    "kafka:rules[2].emit[0].consumes.keyForm",
    "kafka:rules[2].emit[0].consumes.kind",
    "kafka:rules[2].extract.pattern.arg",
    "kafka:rules[2].id",
    "kafka:rules[2].match.callsite.args[0].arg",
    "kafka:rules[2].match.callsite.callee",
    "kafka:rules[2].scope.imports[0].module.prefix",
    "kafka:rules[2].scope.imports[1].module.prefix",
    "kafka:version",
    "kafka:vocabVersion",
    "kafka:vocab[0].example",
    "kafka:vocab[0].keyNorm",
    "kafka:vocab[0].kind",
    "kafka:vocab[0].transport",
    "paper-inhouse-bus:id",
    "paper-inhouse-bus:kind",
    "paper-inhouse-bus:link[0].kind",
    "paper-inhouse-bus:link[0].matcher",
    "paper-inhouse-bus:packFormat",
    "paper-inhouse-bus:queries[0].from[0].as",
    "paper-inhouse-bus:queries[0].from[0].record",
    "paper-inhouse-bus:queries[0].from[0].where.name.regex",
    "paper-inhouse-bus:queries[0].from[1].as",
    "paper-inhouse-bus:queries[0].from[1].record",
    "paper-inhouse-bus:queries[0].from[1].where.id",
    "paper-inhouse-bus:queries[0].from[2].as",
    "paper-inhouse-bus:queries[0].from[2].record",
    "paper-inhouse-bus:queries[0].from[2].where.id",
    "paper-inhouse-bus:queries[0].from[3].as",
    "paper-inhouse-bus:queries[0].from[3].record",
    "paper-inhouse-bus:queries[0].from[3].where.name.regex",
    "paper-inhouse-bus:queries[0].from[3].where.targetId",
    "paper-inhouse-bus:queries[0].id",
    "paper-inhouse-bus:queries[0].yield.handler",
    "paper-inhouse-bus:queries[0].yield.ns",
    "paper-inhouse-bus:queries[0].yield.topic",
    "paper-inhouse-bus:rules[0].emit[0].provides.key",
    "paper-inhouse-bus:rules[0].emit[0].provides.kind",
    "paper-inhouse-bus:rules[0].extract.topic.arg",
    "paper-inhouse-bus:rules[0].id",
    "paper-inhouse-bus:rules[0].match.callsite.callee",
    "paper-inhouse-bus:rules[0].scope.imports[0].module.exact",
    "paper-inhouse-bus:rules[1].emit[0].consumes.key",
    "paper-inhouse-bus:rules[1].emit[0].consumes.keyForm",
    "paper-inhouse-bus:rules[1].emit[0].consumes.kind",
    "paper-inhouse-bus:rules[1].extract.ns.yield",
    "paper-inhouse-bus:rules[1].extract.topic.yield",
    "paper-inhouse-bus:rules[1].id",
    "paper-inhouse-bus:rules[1].match.query",
    "paper-inhouse-bus:rules[1].scope.imports[0].module.exact",
    "paper-inhouse-bus:version",
    "paper-inhouse-bus:vocabVersion",
    "paper-inhouse-bus:vocab[0].example",
    "paper-inhouse-bus:vocab[0].keyNorm",
    "paper-inhouse-bus:vocab[0].kind",
    "paper-inhouse-bus:vocab[0].transport",
    "severity-rules:formula.authorization.base[0].value",
    "severity-rules:formula.authorization.base[0].when.entry",
    "severity-rules:formula.authorization.base[0].when.operation",
    "severity-rules:formula.authorization.base[1].value",
    "severity-rules:formula.authorization.base[1].when.entry",
    "severity-rules:formula.authorization.base[1].when.operation",
    "severity-rules:formula.authorization.base[2].value",
    "severity-rules:formula.authorization.base[2].when.entry",
    "severity-rules:formula.authorization.base[2].when.operation",
    "severity-rules:formula.authorization.base[3].value",
    "severity-rules:formula.authorization.base[3].when.entry",
    "severity-rules:formula.authorization.base[3].when.operation",
    "severity-rules:formula.authorization.penalties[0].value",
    "severity-rules:formula.authorization.penalties[0].when.chainConfirmed",
    "severity-rules:formula.authorization.penalties[1].value",
    "severity-rules:formula.authorization.penalties[1].when.authCheck",
    "severity-rules:formula.caps[0].max",
    "severity-rules:formula.caps[0].when.fileRole",
    "severity-rules:formula.configuration.base[0].value",
    "severity-rules:formula.configuration.base[0].when.class",
    "severity-rules:formula.configuration.base[1].value",
    "severity-rules:formula.configuration.base[1].when.class",
    "severity-rules:formula.configuration.base[2].value",
    "severity-rules:formula.configuration.base[2].when.class",
    "severity-rules:formula.configuration.base[3].value",
    "severity-rules:formula.configuration.base[3].when.class",
    "severity-rules:formula.configuration.penalties[0].value",
    "severity-rules:formula.configuration.penalties[0].when.basename.regex",
    "severity-rules:formula.injection.base[0].value",
    "severity-rules:formula.injection.base[0].when.sink",
    "severity-rules:formula.injection.base[0].when.source",
    "severity-rules:formula.injection.base[1].value",
    "severity-rules:formula.injection.base[1].when.sink",
    "severity-rules:formula.injection.base[1].when.source",
    "severity-rules:formula.injection.base[2].value",
    "severity-rules:formula.injection.base[2].when.sink",
    "severity-rules:formula.injection.base[2].when.source",
    "severity-rules:formula.injection.base[3].value",
    "severity-rules:formula.injection.base[3].when.sink",
    "severity-rules:formula.injection.base[3].when.source",
    "severity-rules:formula.injection.penalties[0].value",
    "severity-rules:formula.injection.penalties[0].when.chainConfirmed",
    "severity-rules:formula.injection.penalties[1].value",
    "severity-rules:formula.injection.penalties[1].when.sanitizer",
    "severity-rules:formula.secrets.adjust[0].value",
    "severity-rules:formula.secrets.adjust[0].when.flow",
    "severity-rules:formula.secrets.adjust[0].when.state",
    "severity-rules:formula.secrets.base[0].value",
    "severity-rules:formula.secrets.base[0].when.class",
    "severity-rules:formula.secrets.base[0].when.state",
    "severity-rules:formula.secrets.base[1].value",
    "severity-rules:formula.secrets.base[1].when.class",
    "severity-rules:formula.secrets.base[1].when.state",
    "severity-rules:formula.secrets.base[2].value",
    "severity-rules:formula.secrets.base[2].when.class",
    "severity-rules:formula.secrets.base[2].when.state",
    "severity-rules:formula.secrets.base[3].value",
    "severity-rules:formula.secrets.base[3].when.class",
    "severity-rules:formula.secrets.base[3].when.state",
    "severity-rules:formula.secrets.base[4].value",
    "severity-rules:formula.secrets.base[4].when.class",
    "severity-rules:formula.secrets.base[4].when.state",
    "severity-rules:formula.secrets.base[5].value",
    "severity-rules:formula.secrets.base[5].when.class",
    "severity-rules:formula.secrets.base[5].when.state",
    "severity-rules:formula.secrets.penalties[0].value",
    "severity-rules:formula.secrets.penalties[0].when.path.regex",
    "severity-rules:id",
    "severity-rules:kind",
    "severity-rules:packFormat",
    "severity-rules:rows[0].finding",
    "severity-rules:rows[0].id",
    "severity-rules:rows[0].level",
    "severity-rules:rows[0].rank",
    "severity-rules:rows[10].finding",
    "severity-rules:rows[10].id",
    "severity-rules:rows[10].level",
    "severity-rules:rows[10].rank",
    "severity-rules:rows[11].finding",
    "severity-rules:rows[11].id",
    "severity-rules:rows[11].level",
    "severity-rules:rows[11].rank",
    "severity-rules:rows[12].finding",
    "severity-rules:rows[12].id",
    "severity-rules:rows[12].level",
    "severity-rules:rows[12].rank",
    "severity-rules:rows[13].finding",
    "severity-rules:rows[13].id",
    "severity-rules:rows[13].level",
    "severity-rules:rows[13].rank",
    "severity-rules:rows[14].finding",
    "severity-rules:rows[14].id",
    "severity-rules:rows[14].level",
    "severity-rules:rows[14].rank",
    "severity-rules:rows[15].finding",
    "severity-rules:rows[15].id",
    "severity-rules:rows[15].level",
    "severity-rules:rows[15].rank",
    "severity-rules:rows[1].finding",
    "severity-rules:rows[1].id",
    "severity-rules:rows[1].level",
    "severity-rules:rows[1].rank",
    "severity-rules:rows[2].finding",
    "severity-rules:rows[2].id",
    "severity-rules:rows[2].level",
    "severity-rules:rows[2].rank",
    "severity-rules:rows[3].finding",
    "severity-rules:rows[3].id",
    "severity-rules:rows[3].level",
    "severity-rules:rows[3].rank",
    "severity-rules:rows[4].finding",
    "severity-rules:rows[4].id",
    "severity-rules:rows[4].level",
    "severity-rules:rows[4].rank",
    "severity-rules:rows[5].finding",
    "severity-rules:rows[5].id",
    "severity-rules:rows[5].level",
    "severity-rules:rows[5].rank",
    "severity-rules:rows[6].finding",
    "severity-rules:rows[6].id",
    "severity-rules:rows[6].level",
    "severity-rules:rows[6].rank",
    "severity-rules:rows[7].finding",
    "severity-rules:rows[7].id",
    "severity-rules:rows[7].level",
    "severity-rules:rows[7].rank",
    "severity-rules:rows[8].finding",
    "severity-rules:rows[8].id",
    "severity-rules:rows[8].level",
    "severity-rules:rows[8].rank",
    "severity-rules:rows[9].finding",
    "severity-rules:rows[9].id",
    "severity-rules:rows[9].level",
    "severity-rules:rows[9].rank",
    "severity-rules:version",
    "severity-rules:vocabVersion",
}


def test_the_mutation_harness_can_tell_a_killed_leaf_from_a_survivor(tmp_path):
    """The harness is not vacuous: an edit the spec test depends on dies, an edit to prose-free text survives."""
    root = scratch_root(tmp_path, "w")
    found = leaves()
    s, e, tag, style = found["fastapi:id"]
    assert not survives(root, mutate(spec.TEXT, s, e, tag, style))
    assert survives(root, spec.TEXT)  # the unmutated copy passes: the failure above is the edit's, not the scratch repo's


def test_every_leaf_of_the_seven_packs_is_classified():
    found = set(leaves())
    assert len(found) > 500
    assert found == KILLED | UNEXERCISED_UNTIL_P1, sorted(found ^ (KILLED | UNEXERCISED_UNTIL_P1))
    assert not KILLED & UNEXERCISED_UNTIL_P1


@pytest.mark.skipif(not os.environ.get("CCG_LEAF_MUTATION"), reason="slow: set CCG_LEAF_MUTATION=1 (about five and a half minutes on 6 workers)")
def test_each_leaf_behaves_as_classified():
    result = run_all()
    survivors = {k for k, alive in result.items() if alive}
    assert survivors == UNEXERCISED_UNTIL_P1, (sorted(survivors - UNEXERCISED_UNTIL_P1), sorted(UNEXERCISED_UNTIL_P1 - survivors))
