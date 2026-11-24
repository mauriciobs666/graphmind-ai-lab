"""D10 — `ccg.gateway.manifestmarkers.ManifestMarkers`: the `ccg:` scope source reads the installed manifest on every
call and refuses (raises) whenever it cannot say (docs/plans/llm-gateway.md §3.6; FR-3, AC-2, AC-21). Expected values are
literals; the installed copy sits under a temp anchor owned by the test uid (`trusted_uids` injected)."""
from __future__ import annotations

import os

import pytest

from broker_helpers import write_manifest
from ccg.gateway.manifestmarkers import ManifestMarkers, MarkerUnreadable

ME = os.getuid()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    (r / "etc").mkdir(parents=True)
    r.chmod(0o755)
    (r / "etc").chmod(0o755)
    return r


def install(root, **kw):
    p = write_manifest(root / "etc", **kw)
    p.chmod(0o644)
    return p


def source(root, p):
    return ManifestMarkers(p, {ME}, anchor=root)


def test_a_local_only_repo_is_restricted_and_a_hosted_one_is_not(root):
    s = source(root, install(root))  # front local-only, docs and lib hosted
    assert s.marker("ccg:repo:front") == "restricted"
    assert s.marker("ccg:repo:docs") == "unrestricted"


def test_the_system_is_restricted_when_any_repo_is_local_only_and_unrestricted_when_none_is(root):
    p = install(root)
    assert source(root, p).marker("ccg:system:shop") == "restricted"
    p = install(root, repos=(("docs", "hosted"), ("lib", "hosted")))
    assert source(root, p).marker("ccg:system:shop") == "unrestricted"


def test_the_manifest_is_read_on_every_call(root):
    p = install(root, repos=(("docs", "hosted"),))
    s = source(root, p)
    assert s.marker("ccg:repo:docs") == "unrestricted"
    install(root, repos=(("docs", "local-only"),))
    assert s.marker("ccg:repo:docs") == "restricted"  # tightening applies on the next call


def test_a_repo_id_is_matched_normalized(root):
    s = source(root, install(root, repos=(("My Repo", "local-only"),)))
    assert s.marker("ccg:repo:my-repo") == "restricted"


@pytest.mark.parametrize("scope", [
    "ccg:repo:absent", "ccg:system:other", "ccg:repo:", "ccg:other:front", "ccg:repo:Front", "plain",
])
def test_a_scope_the_manifest_does_not_describe_is_unreadable(root, scope):
    with pytest.raises(MarkerUnreadable):
        source(root, install(root)).marker(scope)


def test_an_ill_typed_or_unknown_egress_value_is_unreadable(root):
    p = root / "etc" / "system.yaml"
    for egress in ("maybe", "[hosted]", "null", "true"):
        p.write_text(f"schemaVersion: 1\nsystem: shop\negress: {{}}\nrepos:\n  - repoId: a\n    path: .\n    policy:\n      egress: {egress}\n")
        p.chmod(0o644)
        with pytest.raises(MarkerUnreadable):
            source(root, p).marker("ccg:repo:a")


@pytest.mark.parametrize("text", [
    "", "not: [valid", "- a\n- b\n", "system: shop\n", "system: 3\nrepos: []\n",
])
def test_a_missing_malformed_or_ambiguous_manifest_is_unreadable(root, text):
    p = root / "etc" / "system.yaml"
    p.write_text(text)
    p.chmod(0o644)
    with pytest.raises(MarkerUnreadable):
        source(root, p).marker("ccg:system:shop")


def test_a_missing_file_is_unreadable(root):
    with pytest.raises(MarkerUnreadable):
        source(root, root / "etc" / "absent.yaml").marker("ccg:system:shop")


def test_a_writable_or_wrongly_owned_installed_copy_is_unreadable(root):
    p = install(root)
    p.chmod(0o664)
    with pytest.raises(MarkerUnreadable):
        source(root, p).marker("ccg:repo:front")
    p.chmod(0o644)
    with pytest.raises(MarkerUnreadable):
        ManifestMarkers(p, {ME + 1}, anchor=root).marker("ccg:repo:front")


def test_a_symlinked_installed_copy_is_unreadable(root):
    real = install(root)
    link = root / "etc" / "link.yaml"
    link.symlink_to(real)
    with pytest.raises(MarkerUnreadable):
        source(root, link).marker("ccg:repo:front")


def test_a_duplicate_key_that_would_loosen_the_manifest_is_unreadable(root):
    p = root / "etc" / "system.yaml"
    p.write_text("system: shop\nrepos:\n  - repoId: a\n    path: .\n    policy:\n      egress: local-only\n      egress: hosted\n")
    p.chmod(0o644)
    with pytest.raises(MarkerUnreadable):
        source(root, p).marker("ccg:repo:a")
