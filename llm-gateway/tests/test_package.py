"""The package is stdlib-only and importable from `src/`."""

import ast
import pathlib
import sys

import llmgateway

SRC = pathlib.Path(llmgateway.__file__).parent


def test_version_present():
    assert llmgateway.__version__ == "0.1.0"


def test_package_imports_only_stdlib_or_itself():
    allowed = set(sys.stdlib_module_names) | {"llmgateway"}
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module.split(".")[0]]
            assert set(names) <= allowed, f"{path.name}: {names}"
