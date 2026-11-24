"""Tier 2: the root-owned code root and owner config (plan §3.4 table; FR-3, FR-8, AC-21).

Run as the agent account after the install in `llm-gateway/deploy/RUNBOOK.md`."""

import hashlib
import os
import pathlib

import deploylib as dl

REPO = pathlib.Path(__file__).resolve().parents[3]
CODE_FILES = (
    "bin/keyholder.py",
    "bin/service.py",
    "deploy/install.sh",
    "src/llmgateway/keyholder.py",
    "src/llmgateway/framing.py",
    "src/llmgateway/client.py",
)


def test_00_install_is_present():
    """The one test that FAILS (not skips) on a host with no install: it names what is missing."""
    missing = [
        p
        for p in (dl.CODE_ROOT, *(f"{dl.CODE_ROOT}/{rel}" for rel in CODE_FILES))
        if not os.path.exists(p)
    ]
    assert not missing, (
        f"no install: {missing} missing. Follow {dl.RUNBOOK} (bootstrap, holder-account, "
        "owner-files, caller, service-unit) before running `pytest -m deploy`"
    )


def test_code_root_and_ancestors_are_root_owned_and_not_writable(install):
    assert dl.code_root_problems(install) == []


def test_agent_cannot_write_the_code_root_or_its_ancestors(install):
    for directory in ("/opt", install, f"{install}/src/llmgateway", f"{install}/bin"):
        assert dl.create_attempt(directory).status == "denied", directory
    for rel in CODE_FILES:
        assert dl.write_attempt(f"{install}/{rel}").status == "denied", rel


def test_code_root_has_no_bytecode_or_path_files(install):
    stray = [
        os.path.join(d, f)
        for d, _dirs, files in os.walk(install)
        for f in files
        if f.endswith((".pyc", ".pth"))
    ]
    stray += [
        os.path.join(d, n) for d, dirs, _f in os.walk(install) for n in dirs if n == "__pycache__"
    ]
    assert stray == []


def test_etc_artifacts_match_the_table_and_are_not_agent_writable(install):
    t = dl.lstat_problems
    assert t(dl.ETC_DIR, uid=0, mode=0o755, kind="dir") == []
    assert t(f"{dl.ETC_DIR}/ccg", uid=0, mode=0o755, kind="dir") == []
    assert t(f"{dl.ETC_DIR}/gateway.toml", uid=0, mode=0o644, kind="file") == []
    assert t(f"{dl.ETC_DIR}/ccg/system.yaml", uid=0, mode=0o644, kind="file") == []
    assert t(dl.UNIT_PATH, uid=0, mode=0o644, kind="file") == []
    for d in (dl.ETC_DIR, f"{dl.ETC_DIR}/ccg"):
        assert dl.create_attempt(d).status == "denied", d
    for f in (f"{dl.ETC_DIR}/gateway.toml", f"{dl.ETC_DIR}/ccg/system.yaml", dl.UNIT_PATH):
        assert dl.write_attempt(f).status == "denied", f


def test_installed_manifest_is_a_byte_copy_of_the_tracked_one(install):
    """AC-21: the broker reads the installed copy; a difference means `owner-files` is stale."""
    tracked = (REPO / "cpg" / "ccg" / "system.yaml").read_bytes()
    installed = pathlib.Path(f"{dl.ETC_DIR}/ccg/system.yaml").read_bytes()
    assert hashlib.sha256(installed).hexdigest() == hashlib.sha256(tracked).hexdigest(), (
        "the installed manifest differs from cpg/ccg/system.yaml: re-run `install.sh owner-files` "
        "(and restart the broker) if the tracked file changed on purpose"
    )


def test_installed_gateway_toml_names_the_holder_uid_that_exists(install, holder_account):
    """The service and the broker refuse a `holder_uid` equal to their own; it is llmholder's."""
    import tomllib

    cfg = tomllib.loads(pathlib.Path(f"{dl.ETC_DIR}/gateway.toml").read_text())
    assert cfg.get("deployment", {}).get("holder_uid") == holder_account.pw_uid
