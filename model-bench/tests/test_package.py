"""The component's install smoke test, as a real test.

`setup.sh` already imports the package as its last step, but that check only runs when a human
invokes the script. This makes the same assertion part of the suite, and pins
`modelbench.__version__` to the installed distribution's metadata: bumping one and forgetting the
other fails here, rather than silently mislabelling `benchVersion` in every stored run record
(`docs/plans/small-model-benchmarking.md` §3.4).
"""

from importlib.metadata import version

import modelbench


def test_version_matches_distribution_metadata() -> None:
    assert modelbench.__version__ == version("model-bench")
