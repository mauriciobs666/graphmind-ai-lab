"""The package imports, and `decisionmodels.__version__` matches `pyproject.toml`'s version.

`setup.sh` already imports the package as its last step, but only when a human runs it. Reading
the version from `pyproject.toml` rather than the installed metadata keeps the check valid with or
without the editable install (`pythonpath` in `pyproject.toml`).
"""

import tomllib
from pathlib import Path

import decisionmodels

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert decisionmodels.__version__ == project["version"]


def test_no_runtime_dependencies() -> None:
    """docs/plans/typed-decision-models.md §3.4: a stdlib core. A dependency is a plan change."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert project["dependencies"] == []
