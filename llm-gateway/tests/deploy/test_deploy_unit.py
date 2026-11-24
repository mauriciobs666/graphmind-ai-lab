"""Tier 2: the unit files and what systemd made of them (plan §3.4 table; security Pass 5/7)."""

import os
import pathlib
import subprocess

import deploylib as dl

HOLDER_ENABLE = "`sudo systemctl enable --now llmholder`"


def test_holder_unit_has_exactly_the_planned_hardening(install):
    text = pathlib.Path(dl.UNIT_PATH).read_text()
    assert dl.unit_problems(text) == []


def test_systemd_analyze_verifies_the_holder_unit(install):
    out = subprocess.run(
        ["systemd-analyze", "--man=no", "verify", dl.UNIT_PATH],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0 and "Failed" not in out.stderr, out.stderr


def test_systemd_applied_the_hardening_directives(install):
    p = dl.systemctl_show(
        "llmholder",
        "User",
        "RuntimeDirectory",
        "RuntimeDirectoryMode",
        "NoNewPrivileges",
        "ProtectSystem",
        "ReadWritePaths",
        "PrivateTmp",
        "LimitCORE",
        "ProtectProc",
        "RestrictAddressFamilies",
        "CapabilityBoundingSet",
        "ExecStart",
        "UnitFileState",
    )
    assert p["User"] == "llmholder"
    assert p["RuntimeDirectory"] == "llmholder"
    assert p["RuntimeDirectoryMode"] in ("0711", "711")
    assert (p["NoNewPrivileges"], p["PrivateTmp"]) == ("yes", "yes")
    assert (p["ProtectSystem"], p["ProtectProc"]) == ("strict", "invisible")
    assert p["ReadWritePaths"].split() == ["/var/lib/llmholder"]
    assert p["LimitCORE"] == "0"
    assert sorted(p["RestrictAddressFamilies"].split()) == ["AF_INET", "AF_INET6", "AF_UNIX"]
    assert p["CapabilityBoundingSet"] == ""
    assert "argv[]=/usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py ;" in p["ExecStart"]
    assert p["UnitFileState"] == "enabled", f"the unit is not enabled: {HOLDER_ENABLE}"


def test_the_running_holder_carries_the_effect_of_the_unit(holder_proc, holder_account):
    status = dl.proc_status(holder_proc)
    assert status["NoNewPrivs"] == "1"
    assert int(status["CapBnd"], 16) == 0 and int(status["CapEff"], 16) == 0
    assert status["Seccomp"] == "2", "no seccomp filter: RestrictAddressFamilies not in effect"
    uid = str(holder_account.pw_uid)
    assert (
        status["Uid"].split() == [uid] * 4
        and status["Gid"].split() == [str(holder_account.pw_gid)] * 4
    )
    assert status["Groups"].split() in ([], [str(holder_account.pw_gid)])


def test_the_holder_socket_is_recreated_by_systemd_not_left_over(holder_proc, holder_account):
    """RuntimeDirectory= is tmpfs: the socket's birth is this holder's start (not a stale file)."""
    assert dl.lstat_problems(dl.SOCKET, uid=holder_account.pw_uid, mode=0o666, kind="sock") == []
    started = dl.systemctl_show("llmholder", "ActiveEnterTimestampMonotonic")
    assert int(started["ActiveEnterTimestampMonotonic"]) > 0


# --- the service's user unit --------------------------------------------------------------------

USER_UNIT = {
    "ExecStart": "/usr/bin/python3 -I /opt/llmgateway/bin/service.py",
    "Restart": "on-failure",
}


def test_user_unit_runs_the_service_from_the_root_owned_code_root(install):
    path = dl.user_unit_path()
    assert os.path.exists(path), f"{path} missing: `install.sh service-unit` (RUNBOOK)"
    svc = dl.parse_unit(pathlib.Path(path).read_text()).get("Service", {})
    assert {k: v for k, v in svc.items()} == {k: [v] for k, v in USER_UNIT.items()}
    assert dl.lstat_problems(path, uid=os.getuid(), mode=0o644, kind="file") == []


def test_systemd_analyze_verifies_the_user_unit(install):
    out = subprocess.run(
        ["systemd-analyze", "--user", "--man=no", "verify", dl.user_unit_path()],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr


def test_user_unit_is_enabled_and_lingering_for_unattended_runs(install):
    p = subprocess.run(
        ["systemctl", "--user", "is-enabled", "llmgateway"], capture_output=True, text=True
    )
    assert p.stdout.strip() == "enabled", (
        f"{p.stdout.strip()!r} {p.stderr.strip()!r}: `systemctl --user enable --now llmgateway`"
    )
    linger = subprocess.run(
        ["loginctl", "show-user", str(os.getuid()), "-p", "Linger"],
        capture_output=True,
        text=True,
    )
    assert linger.stdout.strip() == "Linger=yes", (
        "no linger: the service stops with the last login: `sudo loginctl enable-linger $USER`"
    )
