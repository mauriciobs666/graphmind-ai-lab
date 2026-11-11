"""`SourceRefused` is one class: defined in `extractors/common/errors.py`, re-exported (not copied) by `normalize.core`."""
from __future__ import annotations

import pytest

from extractors.common import errors
from extractors.joern.normalize import core


def test_there_is_one_source_refused_class():
    assert core.SourceRefused is errors.SourceRefused
    assert errors.SourceRefused.__module__ == "extractors.common.errors"


def test_semantics_are_unchanged_a_runtime_error():
    assert issubclass(errors.SourceRefused, RuntimeError) and not issubclass(errors.SourceRefused, ValueError)


def test_each_import_path_catches_the_exception_raised_through_the_other():
    with pytest.raises(core.SourceRefused):
        raise errors.SourceRefused("x")
    with pytest.raises(errors.SourceRefused):
        raise core.SourceRefused("x")


def test_the_normalizers_own_refusals_are_the_shared_class(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe")
    s = core.Source(tmp_path)
    with pytest.raises(errors.SourceRefused, match="not UTF-8"):
        s.text("bad.py")
    with pytest.raises(errors.SourceRefused):
        s.raw("../escape.py")
