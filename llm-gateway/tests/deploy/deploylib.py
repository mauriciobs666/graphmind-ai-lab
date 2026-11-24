"""Helpers for the tier-2 (`-m deploy`) suite: probes of a REAL install, written as pure functions
over a path/pid so `test_deploy_selfcheck.py` can prove them against controls on a host with no
install (docs/plans/llm-gateway.md §5.2 FR-8 tier 2).

No function here reads, returns or prints a provider key. A caller credential (`broker.cred`,
`service.cred`) is read only to sign probe requests; it never appears in a message or an assert.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import pathlib
import pwd
import re
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import NamedTuple

CODE_ROOT = "/opt/llmgateway"
ETC_DIR = "/etc/llmgateway"
UNIT_PATH = "/etc/systemd/system/llmholder.service"
STATE_ROOT = "/var/lib/llmholder"
RUN_DIR = "/run/llmholder"
SOCKET = RUN_DIR + "/holder.sock"
HOLDER = "llmholder"
PY = "/usr/bin/python3"
DOCKER_SOCKET = "/var/run/docker.sock"
PROBE_ENDPOINT = "https://deploy-probe.invalid:443"  # never a configured provider
RUNBOOK = "llm-gateway/deploy/RUNBOOK.md"


def home() -> str:
    return pwd.getpwuid(os.getuid()).pw_dir


def cred_path(name: str) -> str:
    return os.path.join(home(), ".config", "llmgateway", "callers", name + ".cred")


def user_unit_path() -> str:
    return os.path.join(home(), ".config", "systemd", "user", "llmgateway.service")


def fake_marker() -> str:
    """A planted FAKE key-shaped marker; its only job is to be searched for."""
    return "sk-ant-FAKE-" + uuid.uuid4().hex


# --- file facts --------------------------------------------------------------------------------


def lstat_problems(path: str, *, uid: int, mode: int, kind: str) -> list[str]:
    """`lstat` one artifact against owner, permission bits and kind (`file|dir|sock`)."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        return [f"{path}: {exc.strerror}"]
    test = {"file": stat.S_ISREG, "dir": stat.S_ISDIR, "sock": stat.S_ISSOCK}[kind]
    out = []
    if not test(st.st_mode):
        out.append(f"{path}: not a {kind}")
    if st.st_uid != uid:
        out.append(f"{path}: owner uid {st.st_uid}, want {uid}")
    if stat.S_IMODE(st.st_mode) != mode:
        out.append(f"{path}: mode {stat.S_IMODE(st.st_mode):04o}, want {mode:04o}")
    return out


def tree_differences(ref: str, installed: str) -> list[str]:
    """Paths whose presence, bytes or executable bit differ between two trees (the verified
    extract of the reviewed sha vs the installed code root)."""

    def listing(root: str) -> dict[str, tuple[bytes | None, bool]]:
        out: dict[str, tuple[bytes | None, bool]] = {}
        for dirpath, dirs, files in os.walk(root):
            for name in dirs:
                out[os.path.relpath(os.path.join(dirpath, name), root) + "/"] = (None, True)
            for name in files:
                p = os.path.join(dirpath, name)
                st = os.lstat(p)
                data = pathlib.Path(p).read_bytes() if stat.S_ISREG(st.st_mode) else b"<special>"
                out[os.path.relpath(p, root)] = (data, bool(st.st_mode & 0o111))
        return out

    a, b = listing(ref), listing(installed)
    diff = [f"only in the sha: {k}" for k in sorted(a.keys() - b.keys())]
    diff += [f"only installed: {k}" for k in sorted(b.keys() - a.keys())]
    diff += [f"differs: {k}" for k in sorted(a.keys() & b.keys()) if a[k] != b[k]]
    return diff


def code_root_problems(root: str = CODE_ROOT, *, trusted: int = 0, anchor: str = "/") -> list[str]:
    """Every ancestor (from `anchor`) and every entry: trusted-owned, not group/other-writable,
    regular file or directory only."""
    out = []
    parts = [p for p in root.split("/") if p]
    for i in range(len([p for p in anchor.split("/") if p]), len(parts) + 1):
        path = "/" + "/".join(parts[:i])
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode):
            out.append(f"{path}: ancestor is not a plain directory")
        if st.st_uid != trusted:
            out.append(f"{path}: owner uid {st.st_uid}")
        if st.st_mode & 0o022:
            out.append(f"{path}: group/other-writable")
    for dirpath, dirs, files in os.walk(root):
        for name in dirs + files:
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
                out.append(f"{p}: not a regular file or directory")
            if st.st_uid != trusted:
                out.append(f"{p}: owner uid {st.st_uid}")
            if st.st_mode & 0o022:
                out.append(f"{p}: group/other-writable")
    return out


@dataclass(frozen=True)
class Access:
    """The outcome of one read attempt: `read` (data), `denied` (EACCES/EPERM), `gone`, `error`."""

    status: str
    data: bytes = b""
    err: str = ""


def read_attempt(path: str, limit: int = 1 << 20) -> Access:
    try:
        with open(path, "rb", buffering=0) as fh:
            return Access("read", fh.read(limit))
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            return Access("denied", err=errno.errorcode[exc.errno])
        if exc.errno in (errno.ENOENT, errno.ESRCH, errno.ENOTDIR):
            return Access("gone", err=errno.errorcode[exc.errno])
        return Access("error", err=errno.errorcode.get(exc.errno or 0, "?"))


def write_attempt(path: str) -> Access:
    """Try to open an existing file for append (never creates, writes nothing)."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            return Access("denied", err=errno.errorcode[exc.errno])
        return Access("error", err=errno.errorcode.get(exc.errno or 0, "?"))
    os.close(fd)
    return Access("read", err="opened for writing")


def create_attempt(directory: str) -> Access:
    """Try to create a new file in `directory`; remove it at once if it unexpectedly worked."""
    p = os.path.join(directory, f".deploy-probe-{uuid.uuid4().hex}")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            return Access("denied", err=errno.errorcode[exc.errno])
        return Access("error", err=errno.errorcode.get(exc.errno or 0, "?"))
    os.close(fd)
    os.unlink(p)
    return Access("read", err="created a file")


def tree_hits(root: str, needle: bytes) -> tuple[list[str], list[str]]:
    """(files under `root` holding `needle`, paths the probe was denied); hidden files included."""
    hits, denied = [], []

    def onerror(exc: OSError) -> None:
        denied.append(str(exc.filename))

    for dirpath, _dirs, files in os.walk(root, onerror=onerror):
        for name in files:
            p = os.path.join(dirpath, name)
            a = read_attempt(p)
            if a.status == "read" and needle in a.data:
                hits.append(p)
            elif a.status == "denied":
                denied.append(p)
    if not os.path.isdir(root):
        a = read_attempt(root)
        if a.status == "denied":
            denied.append(root)
    try:
        os.listdir(root)
    except PermissionError:
        denied.append(root)
    except OSError:
        pass
    return hits, denied


# --- process probes ----------------------------------------------------------------------------


def proc_file(pid: int, name: str, limit: int = 1 << 20) -> Access:
    return read_attempt(f"/proc/{pid}/{name}", limit)


def scan_memory(pid: int, needle: bytes, budget: int = 128 << 20) -> Access:
    """The agent's memory probe: `/proc/<pid>/maps` then `/proc/<pid>/mem` over every readable
    region. `read` with data b"1" means found, b"0" not found; `denied` when maps or mem is."""
    maps = proc_file(pid, "maps")
    if maps.status != "read":
        return maps
    try:
        mem = open(f"/proc/{pid}/mem", "rb", buffering=0)
    except OSError as exc:
        status = "denied" if exc.errno in (errno.EACCES, errno.EPERM) else "error"
        return Access(status, err=errno.errorcode.get(exc.errno or 0, "?"))
    found, total = False, 0
    with mem:
        for line in maps.data.decode("ascii", "replace").splitlines():
            fields = line.split()
            if len(fields) < 2 or "r" not in fields[1]:
                continue
            if len(fields) >= 6 and fields[5] in ("[vvar]", "[vsyscall]", "[vdso]"):
                continue
            lo, hi = (int(x, 16) for x in fields[0].split("-"))
            size = min(hi - lo, budget - total)
            if size <= 0:
                break
            try:
                mem.seek(lo)
                chunk = mem.read(size)
            except (OSError, OverflowError, ValueError):
                continue
            total += len(chunk)
            if needle in chunk:
                found = True
                break
    return Access("read", b"1" if found else b"0")


_PTRACE_HELPER = r"""
import ctypes, json, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
libc.ptrace.restype = ctypes.c_long
out = {}
for arg in sys.argv[1:]:
    r = libc.ptrace(0x4206, int(arg), None, None)  # PTRACE_SEIZE: stops nothing
    out[arg] = [r, ctypes.get_errno()]
print(json.dumps(out))
"""


def ptrace_seize(pid: int) -> tuple[int, int]:
    """(return value, errno) of `PTRACE_SEIZE` on `pid` from a short-lived helper process that
    exits at once, which detaches. 0 means the agent may trace it; -1/EPERM is the refusal."""
    out = subprocess.run(
        [sys.executable, "-I", "-c", _PTRACE_HELPER, str(pid)],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin"},
    )
    r, err = json.loads(out.stdout)[str(pid)]
    return r, err


class MarkerChild:
    """An own-uid process carrying a planted FAKE marker in its argv, environment and memory:
    the positive control for every agent probe (planted where the agent can reach)."""

    SCRIPT = (
        "import sys, time\n"
        "keep = bytearray(sys.stdin.buffer.readline().strip())\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )

    def __init__(self, marker: str | None = None) -> None:
        self.marker = marker or fake_marker()
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-c", self.SCRIPT, "--marker=" + self.marker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "PROBE_MARKER": self.marker},
        )
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(self.marker.encode() + b"\n")
        self.proc.stdin.flush()
        assert self.proc.stdout.readline().strip() == b"ready"
        self.pid = self.proc.pid

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()
        for f in (self.proc.stdin, self.proc.stdout):
            if f:
                f.close()


def probe_process(pid: int, needle: bytes) -> dict[str, Access]:
    """What the agent can learn from `pid` through cmdline, environ, memory and fds. A `read`
    carries whether the needle was found in `data` (b"1"/b"0")."""

    def found(a: Access) -> Access:
        return Access("read", b"1" if needle in a.data else b"0") if a.status == "read" else a

    return {
        "cmdline": found(proc_file(pid, "cmdline")),
        "environ": found(proc_file(pid, "environ")),
        "mem": scan_memory(pid, needle),
        "maps": Access(proc_file(pid, "maps").status),
        "fd": _listdir_access(f"/proc/{pid}/fd"),
        "cwd": _readlink_access(f"/proc/{pid}/cwd"),
    }


def _listdir_access(path: str) -> Access:
    try:
        os.listdir(path)
        return Access("read")
    except PermissionError:
        return Access("denied", err="EACCES")
    except OSError as exc:
        return Access("gone", err=errno.errorcode.get(exc.errno or 0, "?"))


def _readlink_access(path: str) -> Access:
    try:
        os.readlink(path)
        return Access("read")
    except PermissionError:
        return Access("denied", err="EACCES")
    except OSError as exc:
        return Access("gone", err=errno.errorcode.get(exc.errno or 0, "?"))


def proc_status(pid: int) -> dict[str, str]:
    a = proc_file(pid, "status")
    assert a.status == "read", f"/proc/{pid}/status: {a.status} {a.err}"
    out = {}
    for line in a.data.decode().splitlines():
        k, _, v = line.partition(":")
        out[k] = v.strip()
    return out


def docker_reachable(path: str = DOCKER_SOCKET) -> bool:
    """True when this account can connect to the Docker socket (root-equivalent)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect(path)
    except OSError:
        return False
    finally:
        s.close()
    return True


CONTAINER_SOCKETS = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/run/containerd/containerd.sock",
    "/run/podman/podman.sock",
    "/var/run/podman/podman.sock",
    "/mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock",
    "/mnt/wsl/docker-desktop/docker-desktop-proxy",
    "/var/lib/lxd/unix.socket",
    "/var/snap/lxd/common/lxd/unix.socket",
)


def reachable_container_sockets(
    env: dict[str, str] | None = None,
    extra: tuple[str, ...] = (),
    system: tuple[str, ...] = CONTAINER_SOCKETS,
):
    """Every root-equivalent container endpoint this account can connect to: the usual system
    sockets, Docker Desktop's WSL sockets, and `DOCKER_HOST` (a `unix://` path, or a `tcp://`
    host:port, which counts when it accepts a connection)."""
    env = os.environ if env is None else env
    found = [p for p in (*system, *extra) if docker_reachable(p)]
    host = env.get("DOCKER_HOST", "")
    if host.startswith("unix://"):
        if docker_reachable(host[len("unix://") :]):
            found.append(host)
    elif host.startswith("tcp://"):
        h, _, port = host[len("tcp://") :].rpartition(":")
        try:
            with socket.create_connection((h or "127.0.0.1", int(port or 2375)), timeout=3):
                found.append(host)
        except (OSError, ValueError):
            pass
    return found


def container_exposure(
    env: dict[str, str] | None = None,
    system: tuple[str, ...] = CONTAINER_SOCKETS,
    groups: list[int] | None = None,
    group_lookup=None,
) -> list[str]:
    """Why this account can act as root through a container runtime (empty = none): membership
    of group `docker` (live process groups or `/etc/group`), or a reachable runtime socket."""
    import grp

    out = []
    lookup = group_lookup or (lambda name: grp.getgrnam(name))
    try:
        docker = lookup("docker")
    except KeyError:
        docker = None
    if docker is not None:
        live = os.getgroups() if groups is None else groups
        if docker.gr_gid in live:
            out.append("this session's account is in group docker")
        if pwd.getpwuid(os.getuid()).pw_name in docker.gr_mem:
            out.append("/etc/group lists this account in group docker")
    out += [
        f"container socket reachable: {s}" for s in reachable_container_sockets(env, (), system)
    ]
    return out


def sudo_refused(*argv: str) -> tuple[bool, str]:
    """`sudo -n -k -u llmholder ...`: `-k` ignores a cached sudo ticket, so a terminal that just
    ran `sudo` cannot make the refusal look like success. True when sudo refused."""
    out = subprocess.run(
        ["sudo", "-n", "-k", "-u", HOLDER, *argv],
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    return out.returncode != 0 and out.stdout == "", out.stderr.strip()


# --- the unit file ------------------------------------------------------------------------------

UNIT_EXPECTED: dict[str, str] = {
    "User": "llmholder",
    "RuntimeDirectory": "llmholder",
    "RuntimeDirectoryMode": "0711",
    "ExecStart": "/usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py",
    "NoNewPrivileges": "yes",
    "ProtectSystem": "strict",
    "ReadWritePaths": "/var/lib/llmholder",
    "PrivateTmp": "yes",
    "LimitCORE": "0",
    "ProtectProc": "invisible",
    "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6",
    "CapabilityBoundingSet": "",
}
UNIT_ALLOWED_EXTRA = {"Restart"}


def parse_unit(text: str) -> dict[str, dict[str, list[str]]]:
    sections: dict[str, dict[str, list[str]]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
            continue
        key, _, value = line.partition("=")
        sections.setdefault(current, {}).setdefault(key.strip(), []).append(value.strip())
    return sections


def unit_problems(text: str) -> list[str]:
    """The plan §3.4 hardening list, exactly: every directive present once with its value, and
    no other `[Service]` directive (an `Environment=`, `ExecStartPre=` or capability grant would
    be one)."""
    service = parse_unit(text).get("Service", {})
    out = []
    for key, want in UNIT_EXPECTED.items():
        values = service.get(key)
        if values is None:
            out.append(f"{key}: missing")
        elif len(values) != 1:
            out.append(f"{key}: set {len(values)} times")
        elif values[0] != want:
            out.append(f"{key}: {values[0]!r}, want {want!r}")
    for key in sorted(set(service) - set(UNIT_EXPECTED) - UNIT_ALLOWED_EXTRA):
        out.append(f"{key}: not in the plan's hardening list")
    return out


def systemctl_show(unit: str, *props: str, user: bool = False) -> dict[str, str]:
    argv = ["systemctl", *(["--user"] if user else []), "show", unit]
    for p in props:
        argv += ["-p", p]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, f"{' '.join(argv)}: {out.stderr.strip()}"
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)


# --- `check-deployment` as the agent ---------------------------------------------------------


def check_output_problems(lines: list[str], state_root: str = STATE_ROOT) -> list[str]:
    """Problems an agent-run `check-deployment` may report: only `Permission denied` under the
    holder's 0700 state directory (the agent cannot stat inside it: that IS the isolation).
    Everything else is a real finding."""
    allowed = state_root.rstrip("/") + "/"
    return [
        ln for ln in lines if not (ln.startswith(allowed) and ln.endswith(": Permission denied"))
    ]


def run_check_deployment(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, "-I", CODE_ROOT + "/bin/keyholder.py", "check-deployment", *extra],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin"},
    )


class _RedactedHex(str):
    """A `str` whose `repr`/`str`/format are fixed text; it still concatenates and encodes."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<redacted credential>"

    def __format__(self, spec: str) -> str:
        return "<redacted credential>"

    __str__ = __repr__


class Secret(bytes):
    """Credential bytes that cannot be printed by accident. pytest shows the arguments of the
    last frame of a failure (`--tb=long`/`auto`) and the operands of a failed `assert`; both go
    through `repr`, which here is fixed text. Everything else is plain `bytes`. Covered: `repr`,
    `str`, f-strings, `in`/`!=` operands, `.hex()`. NOT covered: an `==` between two bytes makes
    pytest diff them byte by byte; compare with `assert_secrets_differ`, never `assert a == b`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<redacted credential>"

    __str__ = __repr__

    def hex(self, *args, **kwargs) -> str:  # type: ignore[override]
        return _RedactedHex(super().hex(*args, **kwargs))


def secret_needle(secret: bytes) -> Secret:
    """The hex form of a credential, as a search needle, kept redacted the same way."""
    return Secret(secret.hex().encode())


def assert_secrets_differ(a: bytes, b: bytes, message: str) -> None:
    """Fail with `message` only. A bare `assert a != b` makes pytest print both operands, which
    for a credential lands in the paste-back; this compares digests and shows neither."""
    da, db = hashlib.sha256(a).digest(), hashlib.sha256(b).digest()
    if hmac.compare_digest(da, db):
        raise AssertionError(message)


# --- the holder wire, as the agent speaks it -----------------------------------------------------

_CRED = re.compile(rb"[0-9a-f]{64}\n?")


def load_credential(path: str) -> bytes:
    """A caller credential: 32 random bytes as hex. Raises `ValueError` without echoing content."""
    from llmgateway import framing

    try:
        return Secret(framing.load_secret(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: not a usable credential ({type(exc).__name__})") from None


def credential_shape_ok(path: str) -> bool:
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or stat.S_IMODE(st.st_mode) != 0o600:
        return False
    if st.st_uid != os.getuid():
        return False
    return bool(_CRED.fullmatch(pathlib.Path(path).read_bytes()))


def wire(
    sock_path: str,
    secret: bytes,
    *,
    caller: str,
    ts: int | None = None,
    nonce: str | None = None,
    endpoint: str = PROBE_ENDPOINT,
    run_id: str = "deploy-probe-1",
    timeout: float = 10.0,
):
    """Send one signed request; return the parsed `HolderReply` (unsigned allowed: the holder's
    pre-authentication refusals cannot be signed). Never sends a configured endpoint."""
    from llmgateway import framing

    frame, n = framing.build_request(
        secret,
        caller=caller,
        method="POST",
        endpoint=endpoint,
        path="/v1/messages",
        run_id=run_id,
        body=b'{"model":"deploy-probe","max_tokens":1}',
        ts=ts,
        nonce=nonce,
    )
    return roundtrip(sock_path, frame, secret, n, timeout=timeout)


def roundtrip(sock_path: str, frame: bytes, secret: bytes, nonce: str, *, timeout: float = 10.0):
    from llmgateway import framing

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(frame)
        payload = framing.read_frame(s, max_bytes=1 << 20, timeout=timeout)
    finally:
        s.close()
    return framing.parse_reply(payload, secret, nonce, require_mac=False)


def silent_close(sock_path: str, data: bytes, *, timeout: float = 10.0) -> bool:
    """Send junk; True when the holder closes without answering (no hang, no reply)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(data)
        try:
            return s.recv(1) == b""
        except TimeoutError:
            return False
    finally:
        s.close()


def windows_clock_skew() -> float | None:
    """Seconds this WSL clock is ahead of the Windows host's, or None when interop is absent."""
    exe = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not os.path.exists(exe):
        return None
    try:
        t0 = time.time()
        out = subprocess.run(
            [exe, "-NoProfile", "-Command", "[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        t1 = time.time()
        host = int(out.stdout.strip()) / 1000.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return (t0 + t1) / 2 - host


# --- the host-wide search for a planted FAKE key (AC-5) ----------------------------------------

SKIP_DIRS = frozenset({".venv", "venv", "node_modules", "__pycache__", ".git"})


class Search(NamedTuple):
    """A host-wide search outcome: `hits == []` means "found nowhere" ONLY when `complete`."""

    hits: list[str]
    complete: bool
    files: int  # files read
    problems: list[str]  # why it is not complete: budget exhausted, unreadable or missing root


CHUNK = 4 << 20


def _contains(path: str, needle: bytes) -> bool:
    """Stream `path` for `needle` in chunks that overlap by len(needle)-1, whatever its size."""
    with open(path, "rb") as fh:
        tail = b""
        while chunk := fh.read(CHUNK):
            if needle in tail + chunk:
                return True
            tail = chunk[-(len(needle) - 1) :] if len(needle) > 1 else b""
    return False


def surface_hits(
    needle: bytes,
    roots: list[str],
    *,
    budget_seconds: float = 1800.0,
    exclude: tuple[str, ...] = (),
) -> Search:
    """Search every file readable by this account under `roots` for `needle` (the needle is never
    returned). Coverage limits, stated: a directory named in `SKIP_DIRS` (`.git`, `.venv`, `venv`,
    `node_modules`, `__pycache__`) is pruned (git history is `git_history_hits`); a file or
    directory this account cannot read is by definition not reachable by an agent here; special
    files are skipped; there is no size cap (every readable file is streamed; the time budget
    bounds the run). The search is `complete` only if every root could be entered and the
    budget was not exhausted."""
    hits: list[str] = []
    problems: list[str] = []
    files = 0
    end = time.monotonic() + budget_seconds
    for root in roots:
        try:
            os.listdir(root)
        except OSError as exc:
            problems.append(f"root {root}: {exc.strerror}")
            continue
        for dirpath, dirs, names in os.walk(root, onerror=lambda _e: None):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in names:
                if time.monotonic() > end:
                    problems.append(f"budget of {budget_seconds:.0f}s exhausted in {root}")
                    return Search(hits, False, files, problems)
                path = os.path.join(dirpath, name)
                if path in exclude:
                    continue
                try:
                    if not stat.S_ISREG(os.lstat(path).st_mode):
                        continue
                    found = _contains(path, needle)
                except OSError:
                    continue
                files += 1
                if found:
                    hits.append(path)
    return Search(hits, not problems, files, problems)


def journal_search(needle: bytes, *, user: bool = False, budget_seconds: float = 300.0) -> Search:
    """The journal (system, or `--user`) as this account can read it; an unreadable journal or a
    `journalctl` failure makes the result incomplete. `files` counts 4 MiB chunks read."""
    import threading

    argv = ["journalctl", "--no-pager", "-q", "-o", "cat"] + (["--user"] if user else [])
    hits: list[str] = []
    problems: list[str] = []
    chunks = 0
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return Search([], False, 0, [f"{argv[0]}: {exc.strerror}"])
    assert proc.stdout and proc.stderr
    err_chunks: list[bytes] = []

    def drain() -> None:  # stderr is read at the same time, so a full pipe cannot stall stdout
        assert proc.stderr
        while data := proc.stderr.read(65536):
            err_chunks.append(data)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    tail, end = b"", time.monotonic() + budget_seconds
    while chunk := proc.stdout.read(CHUNK):
        chunks += 1
        if needle in tail + chunk:
            hits.append("journal --user" if user else "journal")
            break
        tail = chunk[-(len(needle) - 1) :]
        if time.monotonic() > end:
            problems.append("journal budget exhausted")
            break
    proc.kill()
    proc.wait()
    reader.join(5)
    err = b"".join(err_chunks).decode(errors="replace").strip()
    if proc.returncode not in (0, -9) and not hits:
        problems.append(f"journalctl exit {proc.returncode}: {err[:200]}")
    elif "No journal files" in err or "not seeing messages" in err:
        problems.append(f"journal not readable: {err[:200]}")
    return Search(hits, not problems, chunks, problems)


def process_hits(needle: bytes) -> list[int]:
    """Pids whose argv or environment (as this account can read them) contain `needle`."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        for name in ("cmdline", "environ"):
            a = proc_file(int(entry), name)
            if a.status == "read" and needle in a.data:
                pids.append(int(entry))
                break
    return pids


def git_history_hits(repo: str, needle: str) -> bool:
    """True when any commit in `repo` (all refs) adds or removes `needle` (pickaxe)."""
    out = subprocess.run(
        ["git", "-C", repo, "log", "--all", "--oneline", "-S" + needle],
        capture_output=True,
        text=True,
        timeout=300,
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return bool(out.stdout.strip())
