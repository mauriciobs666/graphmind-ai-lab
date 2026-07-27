"""Regression cover for `build.sh --verify-inputs`.

`--verify-inputs` is the *only* mechanism enforcing the invariant that three separate
files state as absolute (`Dockerfile`, `image-tag.sh`, `README.md`): **every path the
Dockerfile COPYs is covered by the content hash.** A false *pass* is the one direction
that costs correctness — the file lands in the image, the tag does not move,
`docker image inspect` hits, and the launch path serves an image that does not contain
the change, which is precisely what the content hash exists to make unrepresentable.

Such a false pass is undetectable by eye, so it is checked here rather than remembered:
the review that found it (`docs/reviews/cpg-mcp-containerization.md` M-7) broke the check
with an ordinary line-continued `COPY`, and got "OK".

Every case runs against a **copy** of `cpg/mcp/` in pytest's tmp_path. The tracked tree is
never modified, no image is built, and Docker is not required — `--verify-inputs` returns
before `build.sh`'s daemon preflight.

Host-only: these tests are not collected inside the image (see `conftest.py`) because
`.dockerignore` keeps the build tooling out of the build context on purpose.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
_IGNORED = shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".git")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="build.sh needs bash on PATH"
)


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """A throwaway copy of `cpg/mcp/`, safe to mutate."""
    dest = tmp_path / "mcp"
    shutil.copytree(MODULE_DIR, dest, ignore=_IGNORED, symlinks=True)
    return dest


def verify_inputs(sandbox: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(sandbox / "build.sh"), "--verify-inputs"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def append_to_dockerfile(sandbox: Path, snippet: str) -> None:
    with (sandbox / "Dockerfile").open("a", encoding="utf-8") as fh:
        fh.write("\n" + snippet.rstrip("\n") + "\n")


# --- the tree as committed -------------------------------------------------------


def test_unmodified_tree_passes(sandbox: Path) -> None:
    """The real Dockerfile's COPY operands are all covered by image-tag.sh."""
    proc = verify_inputs(sandbox)
    assert proc.returncode == 0, proc.stderr
    assert "--verify-inputs OK" in proc.stderr


def test_all_output_goes_to_stderr(sandbox: Path) -> None:
    """Invariant 1: nothing on stdout, ever — docker-run.sh calls this on the MCP pipe."""
    assert verify_inputs(sandbox).stdout == ""


# --- an uncovered COPY must FAIL, in every Dockerfile spelling -------------------


@pytest.mark.parametrize(
    ("label", "snippet", "expected_operands"),
    [
        # The control: caught by the original line-oriented parse too.
        ("single line", "COPY setup.sh setup.sh", ["setup.sh"]),
        # M-7: line-continued COPY. The original parse answered "OK" here.
        ("continued, second operand", "COPY requirements.txt \\\n     setup.sh /app/", ["setup.sh"]),
        # M-7: nothing but flags/continuations on the COPY line itself.
        ("continued, all operands", "COPY \\\n     setup.sh \\\n     run.sh /app/", ["setup.sh", "run.sh"]),
    ],
)
def test_uncovered_copy_fails(
    sandbox: Path, label: str, snippet: str, expected_operands: list[str]
) -> None:
    append_to_dockerfile(sandbox, snippet)
    proc = verify_inputs(sandbox)
    assert proc.returncode != 0, f"{label}: --verify-inputs passed on an uncovered COPY\n{proc.stderr}"
    assert "Refusing to build" in proc.stderr
    for operand in expected_operands:
        assert f"'{operand}'" in proc.stderr, f"{label}: {operand} not named in the diagnostic"


def test_covered_copy_still_passes_when_continued(sandbox: Path) -> None:
    """Joining continuations must not turn a COVERED multi-line COPY into a failure."""
    append_to_dockerfile(sandbox, "COPY requirements.txt \\\n     server.py \\\n     /app/")
    proc = verify_inputs(sandbox)
    assert proc.returncode == 0, proc.stderr


# --- stage-internal and non-context operands ------------------------------------


def test_copy_from_stage_is_not_treated_as_a_context_path(sandbox: Path) -> None:
    """`COPY --from=<stage>` reads from a build stage, so it is not a hash input.

    It used to be reported as a missing build-context file, which is a wrong diagnostic
    on the natural next edit to a multi-stage Dockerfile.
    """
    append_to_dockerfile(sandbox, "COPY --from=base /app/server.py /app/server2.py")
    proc = verify_inputs(sandbox)
    assert proc.returncode == 0, proc.stderr
    assert "/app/server.py" not in proc.stderr


def test_comment_inside_a_continuation_does_not_become_an_operand(sandbox: Path) -> None:
    append_to_dockerfile(sandbox, "COPY requirements.txt \\\n# a comment Docker strips\n     server.py /app/")
    proc = verify_inputs(sandbox)
    assert proc.returncode == 0, proc.stderr


# --- the directory rule (M-4), guarded here so the two rules stay covered together


def test_uncovered_directory_copy_fails(sandbox: Path) -> None:
    (sandbox / "fixtures").mkdir()
    (sandbox / "fixtures" / "sample.cypher").write_text("RETURN 1\n", encoding="utf-8")
    append_to_dockerfile(sandbox, "COPY fixtures fixtures")
    proc = verify_inputs(sandbox)
    assert proc.returncode != 0, proc.stderr
    assert "DIRECTORY 'fixtures'" in proc.stderr
